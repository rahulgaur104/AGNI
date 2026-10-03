#!/usr/bin/env python3
"""agnimhd as a DESC optimisation objective, in one file. NOT part of the package.

Requires DESC. Shows the DESC-side wrapper described in docs/desc_sync_plan.md
section 5: build the PEST grid once, let DESC evaluate the contract fields at
every call as the ``params -> EquilibriumData`` map of ``agnimhd.growth_rate_of``
(optimize mode), and get d(lambda)/d(R_lmn, Z_lmn, L_lmn, p_l, c_l, Psi) from
``jax.grad`` through DESC's compute chain. The eigensolve sits behind a
custom_vjp with zero cotangent, so the gradient is the Hellmann-Feynman
contraction v^T (dA/dq) v.

What this sketch does NOT do (see the plan): re-solve theta from theta_PEST
inside ``compute`` when L_lmn is free (the grid is mapped once, in ``build``),
and it uses the host ARPACK solve, fine for the 24x12x8 test case only.

Run on CPU::

    JAX_PLATFORMS=cpu python desc_objective.py
"""

import time

import numpy as np
from desc import set_device

set_device("cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from desc.compute.utils import _compute as compute_fun  # noqa: E402
from desc.compute.utils import get_profiles, get_transforms  # noqa: E402
from desc.grid import Grid, LinearGrid, QuadratureGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives.objective_funs import _Objective  # noqa: E402

from agnimhd import (  # noqa: E402
    AssemblyConfig,
    EquilibriumData,
    SolverConfig,
    growth_rate_of,
)
from agnimhd.basis import standard_grid  # noqa: E402

# ----------------------------------------------------------------------------- settings
EQ_PATH = "/pscratch/sd/r/rgaur/DESC2/DESC/tests/inputs/AGNI_QH_lowres.h5"
RES = (24, 12, 8)
AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
GAMMA = 5.0 / 3.0
# -----------------------------------------------------------------------------

#: DESC compute key -> EquilibriumData field (same table as tools/export_fixture.py)
KEY_MAP = {
    "g_rr|PEST": "g_rr",
    "g_rv|PEST": "g_rv",
    "g_rp|PEST": "g_rp",
    "g_vv|PEST": "g_vv",
    "g_vp|PEST": "g_vp",
    "g_pp|PEST": "g_pp",
    "g^rr": "g_sup_rr",
    "sqrt(g)_PEST": "sqrtg",
    "(sqrt(g)_PEST_r)|PEST": "sqrtg_r",
    "(sqrt(g)_PEST_v)|PEST": "sqrtg_v",
    "(sqrt(g)_PEST_p)|PEST": "sqrtg_p",
    "J^zeta": "J_sup_zeta",
    "|J|": "abs_J",
    "iota": "iota",
    "psi_r": "psi_r",
    "psi_rr": "psi_rr",
    "p": "p",
    "p_r": "p_r",
}
VECTOR_KEYS = {
    "J x grad(rho)": "J_cross_grad_rho",
    "(B*grad) grad(rho)": "B_dot_grad_grad_rho",
}
KEYS = list(KEY_MAP) + list(VECTOR_KEYS)
#: flux-surface quantities: computed on a LinearGrid (quadrature weights) and copied
FLUX_KEYS = [
    "iota",
    "iota_r",
    "iota_den",
    "iota_den_r",
    "iota_num",
    "iota_num_r",
    "iota_num current",
    "iota_num_r current",
    "iota_num vacuum",
    "iota_num_r vacuum",
    "psi_r",
    "psi_rr",
    "p",
    "p_r",
]


class AgniStability(_Objective):
    """lambda of the most unstable finite-n mode, from agnimhd, as a DESC objective."""

    _coordinates = ""  # scalar objective: the base build must not look for grid weights
    _units = "(dimensionless)"
    _print_value_fmt = "finite-n lambda (agnimhd): "
    _static_attrs = _Objective._static_attrs + [
        "_res",
        "_automorphism",
        "_assembly",
        "_solver",
    ]

    def __init__(
        self,
        eq,
        res=RES,
        automorphism=AUTOMORPHISM,
        assembly=None,
        solver=None,
        target=0.0,
        bounds=None,
        weight=1.0,
        name="agni finite-n",
    ):
        self._res = tuple(int(r) for r in res)
        self._automorphism = dict(automorphism)
        self._assembly = assembly or AssemblyConfig(gamma=GAMMA)
        self._solver = solver or SolverConfig()
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=False,
            normalize_target=False,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        eq = self.things[0]
        n_rho, n_theta, n_zeta = self._res
        nodes, diffmat = standard_grid(
            n_rho, n_theta, n_zeta, NFP=eq.NFP, automorphism=self._automorphism
        )
        rho, theta, zeta = (np.asarray(nodes[k]) for k in ("rho", "theta", "zeta"))
        R, T, Z = np.meshgrid(
            rho, theta, zeta, indexing="ij"
        )  # rho-major, as agnimhd wants
        pest = jnp.asarray(np.stack([R.ravel(), T.ravel(), Z.ravel()], axis=-1))
        # theta_PEST -> theta ONCE. Valid while L_lmn is fixed; a free lambda needs
        # this inside compute (DESC's FinitenStability does that with _mapped_grid).
        rtz = eq.map_coordinates(
            pest,
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(np.inf, 2 * np.pi, 2 * np.pi),
            tol=1e-12,
            maxiter=50,
        )
        grid = Grid(rtz, sort=False)
        quad = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
        # Flux-surface quantities need quadrature weights the custom PEST grid does
        # not have: compute them on a LinearGrid at the same rho and copy over.
        flux_grid = LinearGrid(
            rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
        )
        self._dim_f = 1
        self._constants = {
            "diffmat": diffmat,
            "grid": grid,
            "flux_grid": flux_grid,
            "transforms": get_transforms(KEYS, obj=eq, grid=grid),
            "profiles": get_profiles(KEYS, obj=eq, grid=grid),
            "flux_transforms": get_transforms(FLUX_KEYS, obj=eq, grid=flux_grid),
            "flux_profiles": get_profiles(FLUX_KEYS, obj=eq, grid=flux_grid),
            "a_transforms": get_transforms(["a"], obj=eq, grid=quad),
            "a_profiles": get_profiles(["a"], obj=eq, grid=quad),
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        c = constants or self._constants
        lam = growth_rate_of(
            params,
            lambda p: self._equilibrium_data(p, c),
            c["diffmat"],
            self._assembly,
            self._solver,
        )
        return jnp.atleast_1d(lam)

    def _equilibrium_data(self, params, c):
        """The ``params -> EquilibriumData`` map: DESC's compute chain, no solve."""
        eq = self.things[0]
        n_rho, n_theta, n_zeta = self._res
        # minor radius from the QuadratureGrid definition (docs/interface.md),
        # not from the PEST grid
        a = compute_fun(
            eq,
            ["a"],
            params=params,
            transforms=c["a_transforms"],
            profiles=c["a_profiles"],
        )["a"]
        flux = compute_fun(
            eq,
            FLUX_KEYS,
            params=params,
            transforms=c["flux_transforms"],
            profiles=c["flux_profiles"],
            data={"a": a},
        )
        prefill = {"a": a}
        for k in FLUX_KEYS:
            prefill[k] = c["grid"].copy_data_from_other(
                flux[k], c["flux_grid"], surface_label="rho"
            )
        data = compute_fun(
            eq,
            KEYS,
            params=params,
            transforms=c["transforms"],
            profiles=c["profiles"],
            data=prefill,
        )
        fields = {v: data[k] for k, v in KEY_MAP.items()}
        fields.update({v: data[k] for k, v in VECTOR_KEYS.items()})
        # validate=False: validation uses NumPy and cannot run on traced arrays.
        # Validate
        # once eagerly (e.g. in build with the starting params) instead.
        return EquilibriumData(
            n_rho=n_rho,
            n_theta=n_theta,
            n_zeta=n_zeta,
            NFP=eq.NFP,
            Psi=params["Psi"],
            a=a,
            validate=False,
            **fields,
        )


if __name__ == "__main__":
    eq = load(EQ_PATH)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    obj = AgniStability(eq)
    t0 = time.time()
    obj.build(verbose=0)
    print(f"[build] {time.time() - t0:.1f} s", flush=True)

    params = eq.params_dict
    t0 = time.time()
    lam = obj.compute(params)
    print(
        f"[compute] lambda = {float(lam[0]):.10e}  ({time.time() - t0:.1f} s)",
        flush=True,
    )

    # Gradient w.r.t. the full DESC parameter dict through DESC's compute chain.
    t0 = time.time()
    g = jax.grad(lambda p: obj.compute(p)[0])(params)
    print(f"[grad] {time.time() - t0:.1f} s", flush=True)
    for k in ("R_lmn", "Z_lmn", "L_lmn", "p_l", "c_l", "Psi"):
        if k in g and np.size(g[k]) > 0:
            gk = np.asarray(g[k])
            print(
                f"[grad] d lambda / d {k}: max|.| {np.abs(gk).max():.3e}, "
                f"size {gk.size}",
                flush=True,
            )

    # Finite-difference check on Psi, the cheapest parameter.
    psi0 = float(np.reshape(params["Psi"], -1)[0])
    h = 1e-6 * psi0
    pp = dict(params)
    pp["Psi"] = params["Psi"] + h
    pm = dict(params)
    pm["Psi"] = params["Psi"] - h
    fd = (float(obj.compute(pp)[0]) - float(obj.compute(pm)[0])) / (2 * h)
    ad = float(np.reshape(g["Psi"], -1)[0])
    print(
        f"[check] d lambda / d Psi: AD {ad:.6e}, FD {fd:.6e}, "
        f"rel diff {abs(fd - ad) / max(abs(fd), 1e-300):.2e}",
        flush=True,
    )
