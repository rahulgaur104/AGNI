"""DESC's boundary vacuum response, ``phi_matrix_pest``, on an AGNI boundary shell.

Two halves, as in DESC's ``FinitenStability``:

- :func:`phi_scaffold`: everything that depends only on the resolution -- the
  PEST eval and source grids, the ``Phi`` basis, the singular-integral support
  ``(st, sz, q)`` and its frozen interpolator, the chunk size. Built once, held
  as static data (a frozen :class:`PhiScaffold`).
- :func:`phi_matrix`: the matrix itself at given DESC ``params``, jit- and
  grad-safe. The quadrature *topology* (the frozen interpolator) gets no
  gradient; the boundary geometry it integrates does -- the same tradeoff
  DESC's ``BoundaryError``/``FreeSurfaceError`` make.

``from_desc(..., free_boundary=True)`` calls both at ``eq.params_dict``. DESC is
imported inside the functions only.

Node orders: AGNI's shell is theta-outer, zeta-fastest; BIEST's (DESC's
singular integrals) is zeta-outer, theta-fastest. :func:`phi_matrix` returns
AGNI order on both axes.
"""

from dataclasses import dataclass

import numpy as np

from ..backend import jax, jnp

__all__ = ["PhiScaffold", "phi_matrix", "phi_scaffold"]

#: Memory the boundary singular integral may use, per level, when the caller does
#: not pick a chunk size. Deliberately modest: the integral runs alongside the
#: ring-block preconditioner build, so it cannot have the device to itself.
PHI_CHUNK_BUDGET_GIB = 4.0

# Simultaneous live intermediates per eval point, MEASURED rather than assumed:
# chunk_size=256 on a 35x35 eval grid against a 54x108 source asked for 44.21 GiB, and
# 54*108*1225*8 = 57.2 MB per eval point, so 44.21 GiB / 256 / 57.2 MB = 3.24.
_PHI_CHUNK_LIVE = 3.24


def _phi_chunk_from_budget(
    n_eval, n_src, n_modes, budget_gib=PHI_CHUNK_BUDGET_GIB, live=_PHI_CHUNK_LIVE
):
    """Eval points per singular-integral launch that fit in ``budget_gib``.

    The peak is roughly ``chunk * live * n_src * n_modes * 8`` bytes.
    """
    per_eval = live * n_src * n_modes * 8
    return max(1, min(int(n_eval), int(budget_gib * (1024**3) // max(per_eval, 1))))


def _agni_to_biest(a, n_theta, n_zeta):
    """Reorder a shell array, ``(n_surf,)`` or ``(n_surf, k)``, AGNI -> BIEST.

    NumPy: only ever applied to the static PEST grid's spacing and weights,
    which must stay concrete under ``jit``.
    """
    a = np.asarray(a)
    out = np.swapaxes(a.reshape(n_theta, n_zeta, -1), 0, 1).reshape(
        n_theta * n_zeta, -1
    )
    return out.reshape(-1) if a.ndim == 1 else out


def _phi_best_ratio(eq, n_theta_src, n_zeta_src):
    r"""DESC's ``_best_ratio`` for the source grid, without mapping the source grid.

    ``_best_ratio`` wants the surface-area-weighted mean of ``|e_zeta| /
    |e_theta|``, a property of the BOUNDARY SURFACE, not of the node set sampling
    it. So evaluate on a plain ``LinearGrid`` in DESC angles, at the Nyquist
    resolution of the boundary, and correct for the measure: the source grid is
    uniform in ``theta_PEST``, so every node carries the weight ``theta_PEST_t``.
    Without it the ratio is biased by a resolution-independent 4-8%, enough to
    move ``(st, sz)``. Assumes PEST zeta IS DESC zeta (DESC's zeta is phi).

    Returns ``(ratio, (n_theta, n_zeta))``, the second so the caller can say what
    it sampled.
    """
    from desc.compute import get_profiles, get_transforms
    from desc.compute.utils import _compute as compute_fun
    from desc.grid import LinearGrid
    from desc.integrals.quad_utils import _best_ratio

    # Nyquist for the boundary over one of the EQUILIBRIUM's field periods, capped at
    # the source resolution so this never asks for more than the mapped path would.
    n_theta_r = max(min(2 * int(eq.M) + 1, n_theta_src), 2)
    n_zeta_r = max(min(2 * int(eq.N) + 1, n_zeta_src), 1)
    ratio_grid = LinearGrid(
        rho=1.0, theta=n_theta_r, zeta=n_zeta_r, NFP=eq.NFP, sym=False
    )
    keys = ["|e_theta_PEST x e_phi|r,v|", "e_theta_PEST", "e_phi|r,v", "theta_PEST_t"]
    d = compute_fun(
        eq,
        keys,
        params=eq.params_dict,
        transforms=get_transforms(keys, obj=eq, grid=ratio_grid),
        profiles=get_profiles(keys, obj=eq, grid=ratio_grid),
    )
    jac = np.abs(np.asarray(d["theta_PEST_t"]))
    ratio = float(
        _best_ratio(
            {
                "e_theta": d["e_theta_PEST"],
                "e_zeta": d["e_phi|r,v"],
                "|e_theta x e_zeta|": jac * np.asarray(d["|e_theta_PEST x e_phi|r,v|"]),
            }
        )
    )
    return ratio, (n_theta_r, n_zeta_r)


@dataclass(frozen=True, eq=False)
class PhiScaffold:
    """Static, resolution-only data of one level's ``phi_matrix``.

    Built by :func:`phi_scaffold`; nothing in it moves with the equilibrium, so
    it is safe to hold as a static attribute of a jitted objective. Compared by
    identity.
    """

    n_theta: int
    n_zeta: int
    eval_pest_grid: object  # rho = 1 shell of the AGNI nodes, uniform PEST
    src_pest_grid: object  # the quadrature: eval_pest_grid unless refined
    src_nodes: object  # PEST nodes of the refined source, rtz order; None if not
    Phi_basis: object
    support: tuple  # (st, sz, q)
    interpolator: object  # frozen singular-integral interpolator
    chunk_size: int
    scale: float = None

    @property
    def upscaled(self):
        """bool : The source quadrature is finer than the eval grid."""
        return self.src_nodes is not None


def phi_scaffold(
    eq,
    theta,
    zeta,
    NFP_surf=None,
    n_theta_src=None,
    n_zeta_src=None,
    st=None,
    sz=None,
    q=None,
    chunk_size=None,
    scale=None,
    cache=None,
    verbose=True,
):
    """The :class:`PhiScaffold` of the boundary shell with angles ``theta``, ``zeta``.

    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium
    theta, zeta : array_like
        The basis's PEST angular nodes, as ``basis.nodes_and_diffmat`` returns
        them: equispaced from 0, ``zeta`` over one field period.
    NFP_surf : int, optional
        Field periods of the surface grids. Default ``eq.NFP``; an axisymmetric
        basis (one toroidal node) uses 128, as DESC does.
    n_theta_src, n_zeta_src : int, optional
        Resolution of the SOURCE quadrature of the singular integral. Raising it
        resolves the integral -- what makes the discrete operator self-adjoint
        and sign-definite in the surface measure -- without changing
        ``phi_matrix``'s size. Default: twice the eval grid's. On precise_QA a
        source equal to the eval grid gives an indefinite ``phi_matrix`` at
        9x7 and 13x11 shells and a definite one at twice that; the assembly
        projects the vacuum kernel onto the PSD cone either way.
    st, sz, q : int, optional
        Support of the local singular grid and polar quadrature order. All three,
        or none to pick them with DESC's ``_best_params`` heuristic.
    chunk_size : int, optional
        Eval points per singular-integral launch; ``0`` for unchunked. Default:
        the most that fit in :data:`PHI_CHUNK_BUDGET_GIB`.
    scale : float, optional
        DIAGNOSTIC: multiply ``phi_matrix`` by this. ``> 1`` stiffens the
        boundary toward the fixed-boundary limit, ``0`` removes the vacuum
        response (still free-boundary), ``< 0`` makes it destabilizing.
    cache : dict, optional
        Shared between the levels of one solve, so the heuristic runs once per
        source grid.
    verbose : bool
        Print the support the heuristic picked.

    Returns
    -------
    PhiScaffold
    """
    from desc.basis import DoubleFourierSeries
    from desc.grid import LinearGrid
    from desc.integrals.quad_utils import _best_params
    from desc.integrals.singularities import get_interpolator

    theta, zeta = np.asarray(theta), np.asarray(zeta)
    n_theta, n_zeta = theta.size, zeta.size
    if NFP_surf is None:
        NFP_surf = 128 if n_zeta == 1 else int(eq.NFP)
    eval_grid = LinearGrid(rho=1.0, theta=theta, zeta=zeta, NFP=NFP_surf, sym=False)
    # Capped to what this grid resolves: `_lsmr_compute_phi_matrix` asserts
    # basis.M <= potential_grid.M.
    Phi_basis = DoubleFourierSeries(
        M=eval_grid.M, N=eval_grid.N, NFP=eval_grid.NFP, sym=eval_grid.sym
    )

    n_theta_src = int(2 * n_theta if n_theta_src is None else n_theta_src)
    n_zeta_src = int(
        (1 if n_zeta == 1 else 2 * n_zeta) if n_zeta_src is None else n_zeta_src
    )
    upscaled = (n_theta_src, n_zeta_src) != (n_theta, n_zeta)
    if upscaled:
        src_grid = LinearGrid(
            rho=1.0, theta=n_theta_src, zeta=n_zeta_src, NFP=NFP_surf, sym=False
        )
    else:
        src_grid = eval_grid
    nodes0 = np.reshape(
        np.asarray(src_grid.meshgrid_reshape(src_grid.nodes, "rtz")),
        (n_theta_src * n_zeta_src, 3),
    )

    if chunk_size is None:
        chunk_size = _phi_chunk_from_budget(
            n_theta * n_zeta, n_theta_src * n_zeta_src, Phi_basis.num_modes
        )
    else:
        chunk_size = int(chunk_size) or None

    if st is not None and sz is not None and q is not None:
        support = (int(st), int(sz), int(q))
    else:
        prev = None if cache is None else cache.get("support")
        if prev is not None and np.array_equal(prev[0], nodes0):
            support = prev[1]
        else:
            ratio, rshape = _phi_best_ratio(eq, n_theta_src, n_zeta_src)
            support = tuple(int(x) for x in _best_params(src_grid, ratio))
            if cache is not None:
                cache["support"] = (nodes0, support)
            if verbose:
                print(
                    f"[phi] interpolator support  ratio={ratio:.8f}"
                    f"  (ratio grid {rshape[0]}x{rshape[1]}"
                    f" vs source {n_theta_src}x{n_zeta_src})  "
                    f"phi_st={support[0]} phi_sz={support[1]} phi_q={support[2]}"
                    "  -- pass these to skip this step entirely",
                    flush=True,
                )
    st, sz, q = support
    interpolator = get_interpolator(eval_grid, src_grid, {}, st=st, sz=sz, q=q)
    return PhiScaffold(
        n_theta=n_theta,
        n_zeta=n_zeta,
        eval_pest_grid=eval_grid,
        src_pest_grid=src_grid,
        src_nodes=nodes0 if upscaled else None,
        Phi_basis=Phi_basis,
        support=support,
        interpolator=interpolator,
        chunk_size=chunk_size,
        scale=None if scale is None else float(scale),
    )


def phi_matrix(eq, params, scaffold, bnd_rtz, transforms=None):
    """``phi_matrix_pest`` at ``params``, in AGNI shell order on both axes.

    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium
    params : dict
        DESC parameters, e.g. ``eq.params_dict``; may be traced.
    scaffold : PhiScaffold
    bnd_rtz : array_like, shape (n_theta * n_zeta, 3)
        DESC ``(rho, theta, zeta)`` of the AGNI boundary shell's PEST nodes, in
        AGNI order: the last ``n_theta * n_zeta`` rows of the volume nodes
        already mapped at ``params``. Used as the source grid unless the
        quadrature is refined, which maps its own nodes at ``params``.
    transforms : dict, optional
        Prebuilt DESC transforms for this source grid; built here otherwise.

    Returns
    -------
    jax.Array, shape (n_theta * n_zeta, n_theta * n_zeta)
        In meters: ``Phi = phi_matrix @ B_n`` on the shell.
    """
    from desc.compute import get_profiles, get_transforms
    from desc.compute.utils import _compute as compute_fun
    from desc.grid import Grid

    s = scaffold
    n_theta, n_zeta = s.n_theta, s.n_zeta
    n_surf = n_theta * n_zeta
    src_pest_grid = s.src_pest_grid
    if s.upscaled:
        n_theta_s, n_zeta_s = src_pest_grid.num_theta, src_pest_grid.num_zeta
        rtz = eq.map_coordinates(
            jnp.asarray(s.src_nodes),
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
            tol=1e-12,
            maxiter=50,
            params=params,
        )
        surf_nodes = jnp.transpose(rtz.reshape(n_theta_s, n_zeta_s, 3), (1, 0, 2))
    else:
        n_theta_s, n_zeta_s = n_theta, n_zeta
        surf_nodes = jnp.transpose(
            jnp.asarray(bnd_rtz).reshape(n_theta, n_zeta, 3), (1, 0, 2)
        )
    n_surf_s = n_theta_s * n_zeta_s
    surf_nodes = surf_nodes.reshape(n_surf_s, 3)  # BIEST order

    # The index arrays are static grid metadata and must stay CONCRETE (NumPy): built
    # with jnp they are tracers under jit, and DESC's `pest` transform converts them
    # to NumPy to group the nodes by zeta plane.
    surf_grid = Grid(
        nodes=surf_nodes,
        jitable=True,
        is_meshgrid=True,
        spacing=_agni_to_biest(src_pest_grid.spacing, n_theta_s, n_zeta_s),
        weights=_agni_to_biest(src_pest_grid.weights, n_theta_s, n_zeta_s),
        NFP=s.eval_pest_grid.NFP,
        _unique_rho_idx=np.array([0]),
        _unique_poloidal_idx=np.arange(n_theta_s),
        _unique_zeta_idx=np.arange(n_zeta_s) * n_theta_s,
        _inverse_rho_idx=np.zeros(n_surf_s, dtype=int),
        _inverse_poloidal_idx=np.tile(np.arange(n_theta_s), n_zeta_s),
        _inverse_zeta_idx=np.repeat(np.arange(n_zeta_s), n_theta_s),
    )

    # `pest_grid` is the SOURCE side: it also sizes transforms["Phi_PEST"], whose
    # Vandermonde is the source-grid `Phi_src`. `potential_grid` is the EVAL side and
    # fixes the matrix at (n_surf, n_surf). Both must match the frozen interpolator.
    keys = ["phi_matrix_pest"]
    kw = dict(
        pest_grid=src_pest_grid,
        potential_grid=s.eval_pest_grid,
        problem="exterior Neumann",
        chunk_size=s.chunk_size,
        Phi_basis=s.Phi_basis,
    )
    if transforms is None:
        # `pest` evaluates only the (l, m) modes on this non-tensor-product grid,
        # ~14x less memory than `direct1`; `Phi`/`Phi_PEST` stay on `direct1`. It
        # groups the nodes by zeta plane in NumPy, and a jitable `Grid` makes its
        # index arrays tracers under `jit`, so under a trace DESC's default method
        # is used instead.
        traced = isinstance(surf_grid.nodes, jax.core.Tracer)
        method = {} if traced else {"method": "pest"}
        transforms = get_transforms(keys, obj=eq, grid=surf_grid, **method, **kw)
    data = compute_fun(
        eq,
        keys,
        params=params,
        transforms=transforms,
        profiles=get_profiles(keys, obj=eq, grid=surf_grid),
        # Prefilled, so DESC never rebuilds the interpolator under a trace.
        data={"interpolator_pest": s.interpolator},
        **kw,
    )
    P = data["phi_matrix_pest"]
    if s.scale is not None:
        print(
            f"[phi] *** DIAGNOSTIC: phi_matrix scaled by {s.scale:g}. "
            "NOT A PHYSICAL RUN. ***",
            flush=True,
        )
        P = s.scale * P
    # BIEST -> AGNI on both axes.
    return jnp.transpose(
        P.reshape(n_zeta, n_theta, n_zeta, n_theta), (1, 0, 3, 2)
    ).reshape(n_surf, n_surf)
