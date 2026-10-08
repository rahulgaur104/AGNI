"""``AgniStability``: agnimhd's growth rate as a DESC optimization objective.

Imports DESC at module import, so ``import agnimhd`` never loads this file; use
``from agnimhd.adapters.desc_objective import AgniStability``. Inside DESC's
optimizer (e.g. ``ProximalProjection``) the equilibrium is re-solved every step
and the chain rule through force balance gives d(gamma^2)/d(boundary, profiles).
"""

import jax
import numpy as np
from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective
from jax.experimental import io_callback

from ..assemble import keep_indices, operator_dtype
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


class _WarmStart:
    """The last converged eigenvector, on the host, between DESC's calls.

    ``read`` and ``write`` are called from inside DESC's jitted compute through
    ``io_callback``: the read feeds the solve and the write needs its result,
    so within a call they run before and after it, and between calls the
    device finishes one program, callbacks included, before the next starts.
    A vector that is not finite is not kept.
    """

    def __init__(self, n, dtype):
        self.vector_spec = jax.ShapeDtypeStruct((n,), dtype)
        self.v = np.zeros(n, dtype)
        self.valid = False
        self.gamma2 = 0.0
        self.reads = self.hits = 0  # calls, and calls that started from a kept vector

    def __deepcopy__(self, memo):
        """DESC's optimizers copy the objective; the copies share this cache, so
        the original sees the solves made through them and a copy does not
        recompile (the store is static to DESC's jit, by identity)."""
        return self

    def read(self):
        """``((vector, valid), gamma^2)`` as traced arrays."""
        flag = jax.ShapeDtypeStruct((), bool)
        g = jax.ShapeDtypeStruct((), np.float64)
        v, valid, gamma2 = io_callback(self._read, (self.vector_spec, flag, g))
        return (v, valid), gamma2

    def _read(self):
        self.reads += 1
        self.hits += self.valid
        return self.v, np.bool_(self.valid), np.float64(self.gamma2)

    def write(self, v, gamma2):
        """Keep the converged eigenvector and its ``gamma^2`` for the next call."""
        io_callback(self._write, None, v, gamma2)

    def _write(self, v, gamma2):
        v = np.asarray(v)
        if np.isfinite(v).all() and np.isfinite(gamma2):
            self.v, self.gamma2, self.valid = v, float(gamma2), True


class AgniStability(_Objective):
    """Finite-n ideal MHD squared growth rate ``gamma^2 = -lambda`` from agnimhd.

    Positive means unstable; the default target is 0. The PEST nodes are
    mapped to DESC's ``theta`` at the current parameters on every call, so the
    objective and its gradient follow the equilibrium as it moves. ``a`` is
    computed on a ``QuadratureGrid``. With ``eigensolver="jd"`` the equilibrium
    is also evaluated on the coarse level's nodes at every call, and that level
    seeds and deflates the solve; no gradient flows through it. With
    ``density=True`` the kinetic energy is weighted by DESC's ``ni`` normalized
    to its maximum, as ``from_desc(..., density=True)``, on both levels.
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
        "_density",
        "_warm_start",
        "_sigma_factor",
        "_sigma_floor",
        "_warm",  # the host-side store: static to DESC's jit, by identity
    ]

    def __init__(
        self,
        eq,
        basis,
        family=0,
        assembly=None,
        solver=None,
        coarse=None,
        density=False,
        warm_start=False,
        sigma_factor=None,
        sigma_floor=1e-7,
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
        density : bool
            Weight the kinetic energy with ``ni / max(ni)``; needs
            ``eq.electron_density``.
        warm_start : bool
            ``eigensolver="jd"`` only: start each solve from the previous
            call's converged eigenvector, kept on the host. The solve still
            runs to ``jd_tol`` at every call, so the value and the derivative
            are those of a converged vector; at the same point (DESC's
            Jacobian after its value) the start is already converged and JD
            stops within a round or two. The first call starts from the coarse
            level as usual.
        sigma_factor : float, optional
            Needs ``warm_start``. Adapt the shift of the JD solve: after the
            first call, ``sigma = sigma_factor * gamma^2`` of the previous call
            (at least ``sigma_floor``) instead of the fixed ``solver.sigma``. JD
            needs fewer rounds the closer the shift is to ``gamma^2``: measured at
            ``gamma^2 = 3.8e-6``, 894 rounds at ``sigma = 1e-3``, 295 at 3e-5.
            The shift must stay above the largest ``gamma^2``; a solve whose
            ``gamma^2`` ends within a factor 1.3 of its shift (the optimizer's
            trial points can raise ``gamma^2`` by a factor 3 or more) is redone at
            ``solver.sigma``. ``2`` is a good factor.
        sigma_floor : float
            Lower bound of the adapted shift.
        target, bounds, weight, name
            As for every DESC objective; the default target is 0.
        """
        if target is None and bounds is None:
            target = 0.0
        self._basis = basis
        self._family = int(family)
        self._assembly = assembly or AssemblyConfig()
        self._solver = solver or SolverConfig()
        self._density = bool(density)
        errorif(
            self._density and eq.electron_density is None,
            ValueError,
            "density=True needs a density profile, eq.electron_density.",
        )
        self._warm_start = bool(warm_start)
        errorif(
            self._warm_start and self._solver.eigensolver != "jd",
            ValueError,
            'warm_start=True needs eigensolver="jd" (the dense solvers factor '
            "the matrix and gain nothing from a start vector).",
        )
        errorif(
            sigma_factor is not None and not self._warm_start,
            ValueError,
            "sigma_factor needs warm_start=True (it uses the kept gamma^2)",
        )
        errorif(
            sigma_factor is not None and sigma_factor <= 1,
            ValueError,
            "sigma_factor must exceed 1: the shift lies above gamma^2",
        )
        self._sigma_factor = sigma_factor
        self._sigma_floor = float(sigma_floor)
        self._warm = None
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
        if self._warm_start and self._warm is None:  # a rebuilt copy keeps the store
            b = self._basis
            n = int(keep_indices(b.n_rho, b.n_theta, b.n_zeta).size)
            self._warm = _WarmStart(n, operator_dtype(self._assembly, diffmat))
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
        warm, guess, sigma = self._warm, None, None
        if warm is not None:
            guess, last = warm.read()
            if self._sigma_factor is not None:  # the first call: the fixed sigma
                kept = guess[1] & (last > 0)
                adapted = jnp.maximum(self._sigma_factor * last, self._sigma_floor)
                sigma = jnp.where(kept, adapted, self._solver.sigma)
        gamma2 = growth_rate_of(
            params,
            lambda p: self._equilibrium_data(p, c),
            c["diffmat"],
            self._assembly,
            self._solver,
            v_guess=guess,
            coarse=coarse,
            on_vector=None if warm is None else warm.write,
            sigma=sigma,
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
        keys = list(KEY_MAP) + ["ni"] * self._density
        data = eq.compute(
            keys, grid=grid, params=params, data=data, override_grid=False
        )
        basis = self._coarse_basis if coarse else self._basis
        return EquilibriumData(
            n_rho=basis.n_rho,
            n_theta=basis.n_theta,
            n_zeta=basis.n_zeta,
            NFP=eq.NFP,
            Psi=params["Psi"],
            a=a,
            density=data["ni"] / jnp.max(data["ni"]) if self._density else None,
            validate=False,
            **{v: data[k] for k, v in KEY_MAP.items()},
        )
