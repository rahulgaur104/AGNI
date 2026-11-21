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
from ._stellsym import PARITY, half_and_mirror, mirror_field, use_stellsym

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


def from_desc(
    eq, basis, family=0, density=False, coarse=None, stellsym=None, coordinates="pest"
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
        ``basis.nodes_and_diffmat(eq_data.NFP, family=x)[1]``.
    density : bool
        Store DESC's ``ni`` on the nodes, normalized to its maximum (ones if the
        equilibrium has no density profile), as ``eq_data.density``: the mass
        weighting every solver then uses.
    coarse : Basis, optional
        ``basis.coarse(n_theta, n_zeta)``: also return the coarse level of
        ``eigensolver="jd"``, the equilibrium on its nodes as
        :meth:`~agnimhd.Basis.coarse_level` packs it for
        ``growth_rate(..., coarse=coarse)``.
    stellsym : bool, optional
        Use stellarator symmetry: map and evaluate only one node of each mirror pair
        ``(theta, zeta) -> (-theta, -zeta)`` and fill in the rest with each
        quantity's parity (:mod:`agnimhd.adapters._stellsym`). About half the work.
        The result matches the full evaluation to the map tolerance (~1e-12).
        Default ``eq.sym``. ``True`` on an asymmetric equilibrium raises.
    coordinates : {"pest", "native"}
        The flux coordinates the angular nodes of ``basis`` are uniform in.
        ``"pest"`` (default): PEST ``(rho, theta_PEST, zeta)``, reached by a
        Newton solve for DESC's ``theta`` at every node. ``"native"``: DESC's
        own ``(rho, theta, zeta)``, with no map; the field then enters through
        ``B_sup_theta`` and ``B_sup_zeta`` (``docs/general_coordinates.md``).
        Both discretize the same eigenvalue problem and converge to the same
        ``gamma^2``, but on different nodes, so at a finite resolution the two
        values differ by their discretization errors.

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
    nodes, diffmat = basis.nodes_and_diffmat(eq.NFP, family)
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
    data = eq.compute(keys, grid=grid)
    n_eval = grid.num_nodes

    def field(key):
        val = np.asarray(data[key]).reshape(n_eval, -1).squeeze()
        if not stellsym:
            return val
        return np.asarray(mirror_field(val, m_src, m_flip, PARITY[key]))

    fields = {dst: field(src) for src, dst in keymap.items()}
    if density:
        ni = field("ni").reshape(-1)
        ok = np.isfinite(ni).any() and np.nanmax(ni) > 0
        ni = np.nan_to_num(ni / np.nanmax(ni), nan=1.0) if ok else np.ones_like(ni)
        fields["density"] = ni
    eq_data = EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        NFP=int(eq.NFP),
        a=float(np.asarray(data["a"]).reshape(-1)[0]),
        **fields,
    )
    if coarse is None:
        return eq_data, diffmat
    eq_coarse = from_desc(
        eq, coarse, density=density, stellsym=stellsym, coordinates=coordinates
    )[0]
    return eq_data, diffmat, basis.coarse_level(eq_coarse, family)
