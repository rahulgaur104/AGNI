"""DESC -> :class:`agnimhd.EquilibriumData`.

``from_desc`` takes a DESC ``Equilibrium`` or the path of a DESC ``.h5`` file,
evaluates the contract fields on the nodes of a :class:`~agnimhd.Basis` and
returns the ``EquilibriumData`` together with the ``DiffMat`` on the same nodes.
The nodes are PEST nodes by default; with ``coordinates="native"`` they are
nodes of DESC's own ``(rho, theta, zeta)``, which needs no coordinate map.
DESC is imported inside the function only.
"""

import numpy as np

from ..equilibrium import EquilibriumData
from ._stellsym import PARITY, half_and_mirror, mirror_field, mirror_rtz, use_stellsym

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


#: The same contract in DESC's own coordinates ``(rho, theta, zeta)``: the
#: metric, the Jacobian and its partial derivatives at fixed DESC coordinates,
#: and the two contravariant field components that replace ``(iota, 1)``.
#: ``(r, v, p)`` in the field names then stand for DESC's ``(rho, theta, zeta)``.
KEY_MAP_NATIVE = {
    "g_rr": "g_rr",
    "g_rt": "g_rv",
    "g_rz": "g_rp",
    "g_tt": "g_vv",
    "g_tz": "g_vp",
    "g_zz": "g_pp",
    "g^rr": "g_sup_rr",
    "sqrt(g)": "sqrtg",
    "sqrt(g)_r": "sqrtg_r",
    "sqrt(g)_t": "sqrtg_v",
    "sqrt(g)_z": "sqrtg_p",
    "B^theta": "B_sup_theta",
    "B^zeta": "B_sup_zeta",
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

COORDINATES = ("pest", "native")

#: Derivatives of DESC's ``lambda`` that :func:`shift_theta` needs.
LAMBDA_KEYS = (
    "lambda_r",
    "lambda_t",
    "lambda_z",
    "lambda_rt",
    "lambda_tt",
    "lambda_tz",
)


def key_map(coordinates, choices=COORDINATES):
    """The DESC compute keys of ``coordinates``: PEST's for ``"pest"``, those of
    DESC's own coordinates for anything else in ``choices``."""
    if coordinates not in choices:
        raise ValueError(f"coordinates must be one of {choices}, got {coordinates!r}.")
    return KEY_MAP if coordinates == "pest" else KEY_MAP_NATIVE


def shift_theta(fields, lam):
    """Fields of ``(rho, theta, zeta)`` in ``(rho, theta + Lambda, zeta)``.

    ``fields`` are the :data:`KEY_MAP_NATIVE` fields of some coordinates
    ``(rho, theta, zeta)`` and ``lam`` the six derivatives :data:`LAMBDA_KEYS` of
    a function ``Lambda(rho, theta, zeta)`` on the same nodes. Returns the same
    fields in the coordinates with poloidal angle ``theta' = theta + Lambda``.
    ``Lambda`` is any fixed function, not necessarily the ``lambda`` of the
    equilibrium the fields belong to.

    With ``s = 1 + Lambda_theta``, ``a_rho = Lambda_rho / s`` and ``a_zeta =
    Lambda_zeta / s``, the position vector gives

    .. math::

        \\mathbf{e}'_\\theta = \\mathbf{e}_\\theta / s, \\quad
        \\mathbf{e}'_\\rho = \\mathbf{e}_\\rho - a_\\rho \\mathbf{e}_\\theta, \\quad
        \\mathbf{e}'_\\zeta = \\mathbf{e}_\\zeta - a_\\zeta \\mathbf{e}_\\theta,
        \\qquad \\sqrt{g}' = \\sqrt{g} / s,

    the partial derivatives at fixed primed coordinates are ``d'_theta = d_theta
    / s``, ``d'_rho = d_rho - a_rho d_theta``, ``d'_zeta = d_zeta - a_zeta
    d_theta``, and ``B . grad(theta') = s B^theta + Lambda_zeta B^zeta``.
    ``grad(rho)``, ``grad(zeta)`` and every scalar are unchanged.
    """
    s = 1.0 + lam["lambda_t"]
    a_r, a_z = lam["lambda_r"] / s, lam["lambda_z"] / s
    g_rr, g_rv, g_rp = fields["g_rr"], fields["g_rv"], fields["g_rp"]
    g_vv, g_vp, g_pp = fields["g_vv"], fields["g_vp"], fields["g_pp"]
    sqrtg = fields["sqrtg"]
    # derivatives of sqrt(g) / s at fixed (rho, theta, zeta), then to primed
    d_r = fields["sqrtg_r"] / s - sqrtg * lam["lambda_rt"] / s**2
    d_v = fields["sqrtg_v"] / s - sqrtg * lam["lambda_tt"] / s**2
    d_p = fields["sqrtg_p"] / s - sqrtg * lam["lambda_tz"] / s**2
    out = dict(fields)
    out.update(
        g_rr=g_rr - 2 * a_r * g_rv + a_r**2 * g_vv,
        g_rv=(g_rv - a_r * g_vv) / s,
        g_rp=g_rp - a_r * g_vp - a_z * g_rv + a_r * a_z * g_vv,
        g_vv=g_vv / s**2,
        g_vp=(g_vp - a_z * g_vv) / s,
        g_pp=g_pp - 2 * a_z * g_vp + a_z**2 * g_vv,
        sqrtg=sqrtg / s,
        sqrtg_r=d_r - a_r * d_v,
        sqrtg_v=d_v / s,
        sqrtg_p=d_p - a_z * d_v,
        B_sup_theta=s * fields["B_sup_theta"] + lam["lambda_z"] * fields["B_sup_zeta"],
    )
    return out


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
    """Check if the current version of DESC has the PEST transform method;
    otherwise, use the fallback."""
    from desc.compute import get_transforms

    from ._desc_pest import desc_has_pest, pest_transforms_fallback

    if not desc_has_pest():
        return pest_transforms_fallback(eq, keys, grid)
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
    n_mode_axisym=1,
    coordinates="pest",
    _phi_cache=None,
):
    """Evaluate a DESC equilibrium on the nodes of ``basis``.

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
        ``basis.nodes_and_diffmat(eq_data.NFP, family=x)[1]``. The free-boundary
        ``phi_matrix`` does: it is the vacuum response of this family's boundary
        data (``exp(2 pi i x p / NFP)`` on field period ``p``), so a free-boundary
        ``eq_data`` belongs to its family only.
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
    n_mode_axisym : int
        On an axisymmetric basis (``n_zeta = 1``), the toroidal mode number the
        vacuum response is for; must equal the solve's
        ``AssemblyConfig.n_mode_axisym`` (default 1, as there). Ignored in 3D,
        where the family sets it.
    coordinates : {"pest", "native"}
        The flux coordinates the angular nodes of ``basis`` are uniform in.
        ``"pest"`` (default): PEST ``(rho, theta_PEST, zeta)``, reached by a
        Newton solve for DESC's ``theta`` at every node. ``"native"``: DESC's
        own ``(rho, theta, zeta)``, with no map; the field then enters through
        ``B_sup_theta`` and ``B_sup_zeta`` (``docs/general_coordinates.md``).
        Both discretize the same eigenvalue problem and converge to the same
        ``gamma^2``, but on different nodes, so at a finite resolution the two
        values differ by their discretization errors. Not available together
        with ``free_boundary``.

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
    keymap = key_map(coordinates)
    stellsym = use_stellsym(eq, stellsym)
    if stellsym:  # one node per mirror pair; the rest by parity below
        half, m_src, m_flip = half_and_mirror(n_theta, n_zeta, n_rho)
        to_map = pest[half]
    else:
        to_map = pest
    if coordinates == "native":  # the nodes are DESC's own: nothing to solve for
        rtz = to_map
    else:
        rtz = eq.map_coordinates(
            to_map,
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(np.inf, 2 * np.pi, np.inf),
            tol=1e-12,
            maxiter=50,
        )
    keys = list(keymap) + ["a"] + (["ni"] if density else [])
    grid = Grid(rtz)
    transforms = _pest_transforms(eq, keys, grid) if coordinates == "pest" else None
    data = eq.compute(keys, grid=grid, transforms=transforms)
    n_eval = grid.num_nodes

    def field(key):
        val = np.asarray(data[key]).reshape(n_eval, -1).squeeze()
        if not stellsym:
            return val
        return np.asarray(mirror_field(val, m_src, m_flip, PARITY[key]))

    if stellsym:
        rtz = mirror_rtz(rtz, m_src, m_flip, pest)
    fields = {dst: field(src) for src, dst in keymap.items()}
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
        from .desc_vacuum import check_boundary_node, phi_matrix, phi_scaffold

        check_boundary_node(rho)
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
        phi_family = int(n_mode_axisym) if n_zeta == 1 else int(family)
        fields["phi_matrix"] = np.asarray(
            phi_matrix(
                eq,
                eq.params_dict,
                scaffold,
                np.asarray(rtz)[-n_surf:],
                family=phi_family,
            )
        )
        fields["phi_family"] = phi_family
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
        family=family,  # the coarse level's vacuum response is this family's too
        density=density,
        stellsym=stellsym,
        n_mode_axisym=n_mode_axisym,
        _phi_cache=_phi_cache,
        coordinates=coordinates,
        **phi_opts,
    )[0]
    return eq_data, diffmat, basis.coarse_level(eq_coarse, family)
