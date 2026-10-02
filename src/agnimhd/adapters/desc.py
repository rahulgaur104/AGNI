"""DESC -> :class:`agnimhd.EquilibriumData`.

``from_desc`` takes a DESC ``Equilibrium`` or the path of a DESC ``.h5`` file,
evaluates the contract fields on a PEST tensor-product grid and returns the
``EquilibriumData`` together with the ``DiffMat`` built on the same nodes.
DESC is imported inside the function only.
"""

import numpy as np

from ..basis import standard_grid
from ..equilibrium import EquilibriumData

__all__ = ["from_desc", "is_desc_file"]

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

#: Radial node clustering used by every reference run (DESC tests/test_AGNI.py).
AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)


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


def from_desc(eq, n_rho, n_theta, n_zeta, automorphism=AUTOMORPHISM):
    """Evaluate a DESC equilibrium on a PEST grid.

    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium or str
        A DESC ``Equilibrium``, or the path of a DESC ``.h5`` file (the last
        equilibrium of a family is used).
    n_rho, n_theta, n_zeta : int
        PEST grid resolution.
    automorphism : dict or None
        Staircase clustering of the radial Lobatto nodes; None for none.

    Returns
    -------
    eq_data : EquilibriumData
    diffmat : DiffMat
        Differentiation matrices on exactly the nodes the geometry was
        evaluated at. Use these two together.
    """
    from desc.grid import Grid
    from desc.io import load

    if isinstance(eq, str):
        eq = load(eq)
        eq = (
            eq[-1]
            if isinstance(eq, (list, tuple)) or hasattr(eq, "__getitem__")
            else eq
        )

    nodes, diffmat = standard_grid(
        n_rho, n_theta, n_zeta, NFP=eq.NFP, automorphism=automorphism
    )
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
    data = eq.compute(list(KEY_MAP) + ["a"], grid=Grid(rtz))
    n = n_rho * n_theta * n_zeta
    fields = {
        dst: np.asarray(data[src]).reshape(n, -1).squeeze()
        for src, dst in KEY_MAP.items()
    }
    eq_data = EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        NFP=int(eq.NFP),
        Psi=float(np.asarray(eq.Psi)),
        a=float(np.asarray(data["a"]).reshape(-1)[0]),
        **fields,
    )
    return eq_data, diffmat
