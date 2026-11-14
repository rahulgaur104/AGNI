"""DESC -> :class:`agnimhd.EquilibriumData`.

``from_desc`` takes a DESC ``Equilibrium`` or the path of a DESC ``.h5`` file,
evaluates the contract fields on the PEST nodes of a :class:`~agnimhd.Basis` and
returns the ``EquilibriumData`` together with the ``DiffMat`` on the same nodes.
DESC is imported inside the function only.
"""

import numpy as np

from ..backend import errorif
from ..equilibrium import EquilibriumData
from ._stellsym import PARITY, half_and_mirror, mirror_field, mirror_rtz

__all__ = ["desc_equilibrium", "from_desc", "is_desc_file"]

#: DESC compute key -> EquilibriumData field.
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
    "J x grad(rho)": "J_cross_grad_rho",
    "(B*grad) grad(rho)": "B_dot_grad_grad_rho",
}


def is_desc_file(path):
    """True if ``path`` is an HDF5 file written by DESC (``desc.*`` class tag)."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            return (
                str(f.attrs.get("__class__", f.get("__class__", [b""])[()])).find(
                    "desc."
                )
                >= 0
            )
    except Exception:  # not HDF5, no h5py, or no class tag
        return False


def desc_equilibrium(eq):
    """``eq`` itself, or the last equilibrium in the DESC ``.h5`` file ``eq``."""
    if not isinstance(eq, str):
        return eq
    from desc.io import load

    eq = load(eq)
    return eq[-1] if isinstance(eq, (list, tuple)) or hasattr(eq, "__getitem__") else eq


def _pest_transforms(eq, keys, grid):
    """DESC transforms of ``keys`` on ``grid``, the mapped PEST nodes, ``pest`` method.

    The mapped nodes are uniform in zeta (DESC's zeta is phi, which the map leaves
    alone) but not a tensor product in (rho, theta), so DESC's automatic choice is
    ``direct1``: the full basis evaluated at every node. ``pest`` evaluates only the
    (l, m) modes at the nodes and the toroidal modes at the unique zeta values, at
    machine precision; DESC's own ``FinitenStability`` does the same. Only the main
    grid's transforms: ``eq.compute`` still builds its 0-D and flux-surface grids'.

    ``method="pest"`` is DESC PR ``ma/pest_transform``. Until it is merged, a DESC
    without it gets agnimhd's forward-only copy (``_desc_pest``, temporary);
    without that, DESC would silently fall back to ``direct1``.
    """
    from desc.compute import get_transforms

    from ._desc_pest import desc_has_pest, pest_transforms_fallback

    if not desc_has_pest():  # TEMPORARY, until DESC has method="pest"
        return pest_transforms_fallback(eq, keys, grid)
    # `pest` groups the nodes by zeta plane with the grid's zeta indices, which a
    # plain Grid finds from the node values (zeta is exact: the map only moves theta).
    return get_transforms(keys, obj=eq, grid=grid, method="pest")


def from_desc(
    eq,
    basis,
    family=0,
    density=False,
    coarse=None,
    stellsym=None,
    free_boundary=None,
    phi_n_theta=None,
    phi_n_zeta=None,
    phi_st=None,
    phi_sz=None,
    phi_q=None,
    phi_chunk_size=None,
    _phi_cache=None,
):
    """Evaluate a DESC equilibrium on the PEST nodes of ``basis``.

    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium or str
        A DESC ``Equilibrium``, or the path of a DESC ``.h5`` file (the last
        equilibrium of a family is used).
    basis : agnimhd.Basis
        Nodes and derivative matrices; see ``docs/options.md``.
    family : int
        Toroidal mode family of the returned ``DiffMat``, ``n = family + k NFP``
        (:meth:`~agnimhd.Basis.nodes_and_diffmat`). The geometry does not depend
        on it: another family's ``DiffMat`` is
        ``basis.nodes_and_diffmat(eq_data.NFP, family=x)[1]``.
    density : bool
        Store DESC's ``ni`` on the nodes, normalized to its maximum (ones if the
        equilibrium has no density profile), as ``eq_data.density``: the mass
        weighting every solver then uses.
    coarse : Basis, optional
        ``basis.coarse(n_theta, n_zeta)``: also return the coarse level of
        ``eigensolver="jd"``, the equilibrium on its nodes as
        :meth:`~agnimhd.Basis.coarse_level` packs it for
        ``growth_rate(..., coarse=coarse)``. Free-boundary exactly when the fine
        level is, with the same ``phi_*`` options.
    stellsym : bool, optional
        Use stellarator symmetry: map and evaluate only one node of each mirror pair
        ``(theta, zeta) -> (-theta, -zeta)``, here and on the vacuum response's
        source grid, and fill in the rest with each quantity's parity
        (:mod:`agnimhd.adapters._stellsym`). About half the work; the result
        matches the full evaluation to the map tolerance (~1e-12). The source grid
        itself stays the full lattice: the singular integral needs every node.
        Default ``eq.sym``; ``True`` on an asymmetric equilibrium raises.
    free_boundary : bool, optional
        Also store DESC's vacuum response of the boundary, ``phi_matrix_pest``
        on the outer shell of the nodes, as ``eq_data.phi_matrix``: every solver
        is then free-boundary. Default ``basis.free_boundary``, the basis that
        puts its outer radial node at ``rho = 1``; any other basis must already
        end there (``radial="lobatto"``).
    phi_n_theta, phi_n_zeta : int, optional
        Resolution of the singular integral's source quadrature; default twice
        the shell's. See :func:`~agnimhd.adapters.desc_vacuum.phi_scaffold`.
    phi_st, phi_sz, phi_q : int, optional
        Its support and polar order, all three or none (DESC's heuristic).
    phi_chunk_size : int, optional
        Eval points per launch; ``0`` unchunked; default from a 4 GiB budget.

    Returns
    -------
    eq_data : EquilibriumData
    diffmat : DiffMat
        Differentiation matrices on exactly the nodes the geometry was
        evaluated at. Use these two together.
    coarse : tuple, only if ``coarse`` is given
    """
    from desc.grid import Grid

    eq = desc_equilibrium(eq)
    n_rho, n_theta, n_zeta = basis.n_rho, basis.n_theta, basis.n_zeta
    nfp = int(eq.NFP)
    nodes, diffmat = basis.nodes_and_diffmat(nfp, family)
    rho, theta, zeta = (np.asarray(nodes[k]) for k in ("rho", "theta", "zeta"))
    R, T, Z = np.meshgrid(rho, theta, zeta, indexing="ij")  # rho-major
    pest = np.stack([R.ravel(), T.ravel(), Z.ravel()], axis=-1)
    stellsym = bool(eq.sym) if stellsym is None else bool(stellsym)
    errorif(
        stellsym and not eq.sym,
        ValueError,
        "stellsym=True needs a stellarator-symmetric equilibrium (eq.sym).",
    )
    if stellsym:  # one node per mirror pair; the rest by parity below
        half, m_src, m_flip = half_and_mirror(n_theta, n_zeta, n_rho)
        to_map = pest[half]
    else:
        to_map = pest
    rtz = eq.map_coordinates(
        to_map,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
        tol=1e-12,
        maxiter=50,
    )
    keys = list(KEY_MAP) + ["a"] + (["ni"] if density else [])
    grid = Grid(rtz)
    data = eq.compute(keys, grid=grid, transforms=_pest_transforms(eq, keys, grid))
    n_eval = grid.num_nodes

    def field(key):
        val = np.asarray(data[key]).reshape(n_eval, -1).squeeze()
        if not stellsym:
            return val
        return np.asarray(mirror_field(val, m_src, m_flip, PARITY[key]))

    if stellsym:
        rtz = mirror_rtz(rtz, m_src, m_flip, pest)
    fields = {dst: field(src) for src, dst in KEY_MAP.items()}
    if density:
        ni = field("ni").reshape(-1)
        ok = np.isfinite(ni).any() and np.nanmax(ni) > 0
        ni = np.nan_to_num(ni / np.nanmax(ni), nan=1.0) if ok else np.ones_like(ni)
        fields["density"] = ni
    if free_boundary is None:
        free_boundary = basis.free_boundary
    phi_opts = dict(
        free_boundary=free_boundary,
        phi_n_theta=phi_n_theta,
        phi_n_zeta=phi_n_zeta,
        phi_st=phi_st,
        phi_sz=phi_sz,
        phi_q=phi_q,
        phi_chunk_size=phi_chunk_size,
    )
    if free_boundary:
        from .desc_vacuum import phi_matrix, phi_scaffold

        errorif(
            abs(float(rho[-1]) - 1.0) > 1e-12,
            ValueError,
            "free_boundary=True needs the outer radial node on the plasma boundary, "
            f"rho = 1; this basis ends at rho = {float(rho[-1]):.6g}. Use "
            "Basis(..., free_boundary=True), which rescales the radial nodes, "
            "derivative matrix and weights so that it does.",
        )
        _phi_cache = {} if _phi_cache is None else _phi_cache
        scaffold = phi_scaffold(
            eq,
            theta,
            zeta,
            NFP_surf=None if n_zeta == 1 else nfp,
            n_theta_src=phi_n_theta,
            n_zeta_src=phi_n_zeta,
            st=phi_st,
            sz=phi_sz,
            q=phi_q,
            chunk_size=phi_chunk_size,
            stellsym=stellsym,
            cache=_phi_cache,
        )
        n_surf = n_theta * n_zeta
        fields["phi_matrix"] = np.asarray(
            phi_matrix(eq, eq.params_dict, scaffold, np.asarray(rtz)[-n_surf:])
        )
    eq_data = EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        NFP=nfp,
        a=float(np.asarray(data["a"]).reshape(-1)[0]),
        **fields,
    )
    if coarse is None:
        return eq_data, diffmat
    eq_coarse = from_desc(
        eq,
        coarse,
        density=density,
        stellsym=stellsym,
        _phi_cache=_phi_cache,
        **phi_opts,
    )[0]
    return eq_data, diffmat, basis.coarse_level(eq_coarse, family)
