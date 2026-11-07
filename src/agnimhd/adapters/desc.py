"""DESC -> :class:`agnimhd.EquilibriumData`.

``from_desc`` takes a DESC ``Equilibrium`` or the path of a DESC ``.h5`` file,
evaluates the contract fields on the PEST nodes of a :class:`~agnimhd.Basis` and
returns the ``EquilibriumData`` together with the ``DiffMat`` on the same nodes.
DESC is imported inside the function only.
"""

import numpy as np

from ..backend import errorif
from ..equilibrium import EquilibriumData

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


def from_desc(
    eq,
    basis,
    family=0,
    density=False,
    coarse=None,
    full_torus=False,
    free_boundary=None,
    phi_n_theta=None,
    phi_n_zeta=None,
    phi_st=None,
    phi_sz=None,
    phi_q=None,
    phi_chunk_size=None,
    phi_scale=None,
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
    full_torus : bool
        Put the toroidal nodes on the whole torus, ``[0, 2 pi)``, instead of one
        field period: the nodes of ``basis.nodes_and_diffmat(1)``, every toroidal
        mode number in family 0, ``eq_data.NFP = 1``. How DESC's
        ``FinitenStability`` scripts run (``FULL_TORUS = True``). Default False.
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
    phi_scale : float, optional
        DIAGNOSTIC: scale ``phi_matrix``. Large values approach the
        fixed-boundary limit; not a physical run.

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
    nfp = 1 if full_torus else int(eq.NFP)
    nodes, diffmat = basis.nodes_and_diffmat(nfp, family)
    rho, theta, zeta = (np.asarray(nodes[k]) for k in ("rho", "theta", "zeta"))
    R, T, Z = np.meshgrid(rho, theta, zeta, indexing="ij")  # rho-major
    pest = np.stack([R.ravel(), T.ravel(), Z.ravel()], axis=-1)
    rtz = eq.map_coordinates(
        pest,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
        tol=1e-12,
        maxiter=50,
    )
    keys = list(KEY_MAP) + ["a"] + (["ni"] if density else [])
    data = eq.compute(keys, grid=Grid(rtz))
    n = n_rho * n_theta * n_zeta
    fields = {
        dst: np.asarray(data[src]).reshape(n, -1).squeeze()
        for src, dst in KEY_MAP.items()
    }
    if density:
        ni = np.asarray(data["ni"]).reshape(-1)
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
        phi_scale=phi_scale,
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
            scale=phi_scale,
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
        full_torus=full_torus,
        _phi_cache=_phi_cache,
        **phi_opts,
    )[0]
    return eq_data, diffmat, basis.coarse_level(eq_coarse, family)
