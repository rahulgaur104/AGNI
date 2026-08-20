#!/usr/bin/env python3
"""A DESC adapter, in about fifty lines. NOT part of the package.

This is the reference for what a consumer writes. ``agnimhd`` ships no
adapters: an adapter imports an equilibrium code, and the package must not.
The dependency runs the other way -- DESC installs ``agnimhd``, and code like
this lives in DESC.

Nothing in the package or the test suite imports this file. It requires DESC,
so it does not run in the DESC-free environment the tests use.

Usage::

    python examples/desc_adapter.py path/to/equilibrium.h5 --res 24,12,8

See docs/adapters.md for the checklist this implements, and
tools/export_fixture.py for the fuller version that produced the test fixture.
"""

import argparse

import numpy as np

from agnimhd import EquilibriumData, eigenpair
from agnimhd.basis import standard_grid

#: DESC compute key -> EquilibriumData field. This table is the whole adapter.
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
#: Supply these two instead of `finite_n_instability_drive` and agnimhd forms
#: the drive itself, which is how you avoid re-deriving the s -> rho
#: substitution by hand.
VECTOR_KEY_MAP = {
    "J x grad(rho)": "J_cross_grad_rho",
    "(B*grad) grad(rho)": "B_dot_grad_grad_rho",
}

AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)


def desc_to_agnimhd(eq, n_rho, n_theta, n_zeta, automorphism=AUTOMORPHISM):
    """Convert a DESC ``Equilibrium`` into an ``EquilibriumData``.

    Returns ``(equilibrium_data, diffmat)`` -- both, because the grid operators
    must be built on exactly the nodes the geometry was evaluated at, and
    handing them back separately invites the mismatch.
    """
    from desc.backend import jnp
    from desc.grid import Grid, QuadratureGrid

    nodes, diffmat = standard_grid(
        n_rho, n_theta, n_zeta, NFP=eq.NFP, automorphism=automorphism
    )
    rho, theta, zeta = (np.asarray(nodes[k]) for k in ("rho", "theta", "zeta"))

    # rho-major tensor product: index (i*n_theta + j)*n_zeta + k.
    R, T, Z = np.meshgrid(rho, theta, zeta, indexing="ij")
    pest_nodes = jnp.asarray(np.stack([R.ravel(), T.ravel(), Z.ravel()], axis=-1))

    # DESC's `theta` is NOT theta_PEST: map coordinates first. Use a tight tol;
    # the geometry inherits this root-find's error.
    rtz = eq.map_coordinates(
        pest_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    data = eq.compute(list(KEY_MAP) + list(VECTOR_KEY_MAP), grid=Grid(rtz))

    n = n_rho * n_theta * n_zeta
    fields = {dst: np.asarray(data[src]).reshape(-1) for src, dst in KEY_MAP.items()}
    fields.update(
        {
            dst: np.asarray(data[src]).reshape(n, 3)
            for src, dst in VECTOR_KEY_MAP.items()
        }
    )

    # `a` must be the QuadratureGrid cross-section area integral. The LinearGrid
    # value differs by 3.76% and the eigenvalue is hypersensitive to it.
    a = float(
        np.asarray(eq.compute("a", grid=QuadratureGrid(eq.L, eq.M, eq.N, eq.NFP))["a"])
    )

    eq_data = EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        NFP=int(eq.NFP),
        Psi=float(np.asarray(eq.params_dict["Psi"])),
        a=a,
        **fields,
    )
    return eq_data, diffmat


def main():
    """Convert one equilibrium and solve it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="DESC .h5 equilibrium")
    ap.add_argument("--res", default="24,12,8", help="n_rho,n_theta,n_zeta")
    ap.add_argument("--save", default=None, help="optional .npz to write")
    args = ap.parse_args()

    from desc.equilibrium import Equilibrium

    eq = Equilibrium.load(args.path)
    eq_data, diffmat = desc_to_agnimhd(eq, *(int(v) for v in args.res.split(",")))
    if args.save:
        print("wrote", eq_data.save(args.save))

    lam, _, resid = eigenpair(eq_data, diffmat)
    print(
        f"lambda {float(lam):+.10e}  residual {float(resid):.3e}  "
        f"{'UNSTABLE' if float(lam) < 0 else 'stable'}"
    )


if __name__ == "__main__":
    main()
