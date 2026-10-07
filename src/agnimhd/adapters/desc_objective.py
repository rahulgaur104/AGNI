"""``AgniStability``: agnimhd's growth rate as a DESC optimization objective.

Imports DESC at module import, so ``import agnimhd`` never loads this file; use
``from agnimhd.adapters.desc_objective import AgniStability``. Inside DESC's
optimizer (e.g. ``ProximalProjection``) the equilibrium is re-solved every step
and the chain rule through force balance gives d(gamma^2)/d(boundary, profiles).
"""

import numpy as np
from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective

from ..backend import errorif
from ..config import AssemblyConfig, SolverConfig
from ..equilibrium import EquilibriumData
from ..objective import growth_rate_of
from .desc import KEY_MAP

__all__ = ["AgniStability"]

#: Flux functions: computed on a LinearGrid at the PEST rho values (it has the
#: quadrature weights the PEST grid lacks) and copied onto the PEST nodes.
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
    """Finite-n ideal MHD squared growth rate ``gamma^2 = -lambda`` from agnimhd.

    Positive means unstable; the default target is 0. The PEST nodes are
    mapped to DESC's ``theta`` at the current parameters on every call, so the
    objective and its gradient follow the equilibrium as it moves. ``a`` is
    computed on a ``QuadratureGrid``. With ``eigensolver="jd"`` the equilibrium
    is also evaluated on the coarse level's nodes at every call, and that level
    seeds and deflates the solve; no gradient flows through it.
    """

    _coordinates = ""
    _units = "(dimensionless)"
    _print_value_fmt = "finite-n gamma^2 (agnimhd): "
    _static_attrs = _Objective._static_attrs + [
        "_basis",
        "_family",
        "_assembly",
        "_solver",
        "_coarse_basis",
    ]

    def __init__(
        self,
        eq,
        basis,
        family=0,
        assembly=None,
        solver=None,
        coarse=None,
        target=None,
        bounds=None,
        weight=1.0,
        name="agni finite-n",
    ):
        """Store the choices; nodes and DESC transforms are made in :meth:`build`.

        Parameters
        ----------
        eq : desc.equilibrium.Equilibrium
        basis : agnimhd.Basis
        family : int
            Toroidal mode family ``n = family + k NFP`` whose ``gamma^2`` is
            returned (:meth:`agnimhd.Basis.families`); one objective per family.
        assembly, solver : AssemblyConfig, SolverConfig, optional
        coarse : agnimhd.Basis, optional
            ``eigensolver="jd"`` only: the coarse level,
            ``basis.coarse(n_theta, n_zeta)``; default ``basis.coarse()``.
        target, bounds, weight, name
            As for every DESC objective; the default target is 0.
        """
        if target is None and bounds is None:
            target = 0.0
        self._basis = basis
        self._family = int(family)
        self._assembly = assembly or AssemblyConfig()
        self._solver = solver or SolverConfig()
        self._coarse_basis = None
        if self._solver.eigensolver == "jd":
            coarse = basis.coarse() if coarse is None else coarse
            errorif(
                coarse != basis.coarse(coarse.n_theta, coarse.n_zeta),
                ValueError,
                "coarse must be basis.coarse(n_theta, n_zeta): the same radial "
                "nodes, map, mpol and ntor on fewer angular nodes.",
            )
            self._coarse_basis = coarse
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
        """Fixed PEST nodes, DiffMat and DESC transforms (and the coarse level's)."""
        eq = self.things[0]
        rho, diffmat, nodes = self._pest_nodes(self._basis)
        flux_grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        quad = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
        self._dim_f = 1
        self._constants = {
            "diffmat": diffmat,
            "nodes": nodes,
            "flux_grid": flux_grid,
            "flux_transforms": get_transforms(FLUX_KEYS, obj=eq, grid=flux_grid),
            "flux_profiles": get_profiles(FLUX_KEYS, obj=eq, grid=flux_grid),
            "a_transforms": get_transforms(["a"], obj=eq, grid=quad),
            "a_profiles": get_profiles(["a"], obj=eq, grid=quad),
            "quad_weights": 1.0,  # scalar objective: no grid weights
        }
        if self._coarse_basis is not None:  # same radial nodes: same flux grid
            c = self._constants
            c["coarse_nodes"] = self._pest_nodes(self._coarse_basis)[2]
            eq_c = self._equilibrium_data(eq.params_dict, c, coarse=True)
            _, c["coarse_diffmat"], c["coarse_transfer"] = self._basis.coarse_level(
                eq_c, self._family
            )
        super().build(use_jit=use_jit, verbose=verbose)

    def _pest_nodes(self, basis):
        """``(rho, diffmat, nodes)``: ``basis``' PEST nodes, rho-major, with
        DESC's unique-rho index maps."""
        eq = self.things[0]
        nodes, diffmat = basis.nodes_and_diffmat(eq.NFP, self._family)
        rho, theta, zeta = (np.asarray(nodes[k]) for k in ("rho", "theta", "zeta"))
        R, T, Z = np.meshgrid(rho, theta, zeta, indexing="ij")
        pest = np.stack([R.ravel(), T.ravel(), Z.ravel()], axis=-1)
        _, u_idx, i_idx = np.unique(pest[:, 0], return_index=True, return_inverse=True)
        nodes = {"pest": pest, "u_idx": u_idx, "i_idx": i_idx}
        return rho, diffmat, {k: jnp.asarray(v) for k, v in nodes.items()}

    def compute(self, params, constants=None):
        """``gamma^2`` at ``params``, differentiable with respect to them."""
        c = constants or self._constants
        coarse = None
        if self._coarse_basis is not None:
            eq_c = self._equilibrium_data(params, c, coarse=True)
            coarse = (eq_c, c["coarse_diffmat"], c["coarse_transfer"])
        gamma2 = growth_rate_of(
            params,
            lambda p: self._equilibrium_data(p, c),
            c["diffmat"],
            self._assembly,
            self._solver,
            coarse=coarse,
        )
        return jnp.atleast_1d(gamma2)

    def _equilibrium_data(self, params, c, coarse=False):
        """``params -> EquilibriumData`` through DESC's compute functions, on
        the PEST nodes of the basis or, with ``coarse``, of the coarse level."""
        eq = self.things[0]
        nodes = c["coarse_nodes"] if coarse else c["nodes"]
        rtz = eq.map_coordinates(
            nodes["pest"],
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
            tol=1e-12,
            maxiter=50,
            params=params,
        )
        grid = Grid(
            rtz,
            coordinates="rtz",
            sort=False,
            jitable=True,
            _unique_rho_idx=nodes["u_idx"],
            _inverse_rho_idx=nodes["i_idx"],
        )
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
        data = {"a": a}
        for k in FLUX_KEYS:
            data[k] = grid.copy_data_from_other(
                flux[k], c["flux_grid"], surface_label="rho"
            )
        data = eq.compute(
            list(KEY_MAP), grid=grid, params=params, data=data, override_grid=False
        )
        basis = self._coarse_basis if coarse else self._basis
        return EquilibriumData(
            n_rho=basis.n_rho,
            n_theta=basis.n_theta,
            n_zeta=basis.n_zeta,
            NFP=eq.NFP,
            Psi=params["Psi"],
            a=a,
            validate=False,
            **{v: data[k] for k, v in KEY_MAP.items()},
        )
