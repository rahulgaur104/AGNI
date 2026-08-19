#!/usr/bin/env python3
"""The matrix-free path: operator, ring preconditioner, preconditioned CG.

At production resolution the dense matrix does not fit -- on an 80 GB A100 that
happens somewhere between 32x32x12 and 32x48x16 -- and the solve has to run
without ever forming it. This example exercises that machinery on the shipped
low-resolution case, where the dense matrix *is* available and can be used as
ground truth.

What it shows:

1. ``matfree_operator`` applies the same operator ``assemble_dense`` builds,
   without materializing it (measured agreement: 4.8e-16 relative, per column).
2. The **ring** (theta-line) block-Jacobi blocks are exact sub-blocks of that
   matrix, built by a ``vmap`` rather than extracted from it.
3. Ring preconditioning cuts the CG iteration count on the real operator.

What it does not show is the coarse level -- see the note at the bottom.

Run::

    python examples/matrix_free_solve.py
"""

from pathlib import Path

import numpy as np

from agnimhd import AssemblyConfig, EquilibriumData
from agnimhd.assemble import assemble_dense, keep_indices, matfree_operator
from agnimhd.backend import jnp
from agnimhd.basis import standard_grid
from agnimhd.solvers import (
    build_ring_blocks,
    factor_ring_blocks,
    make_block_precond,
    pcg,
    ring_index_maps,
)

AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
FIXTURE = Path(__file__).resolve().parents[1] / "tests/data/qh_lowres_24x12x8.npz"


def main():
    """Compare matrix-free against dense, then precondition a CG solve."""
    eq = EquilibriumData.load(FIXTURE)
    res = eq.resolution
    _, diffmat = standard_grid(*res, NFP=eq.NFP, automorphism=AUTOMORPHISM)
    config = AssemblyConfig()

    op = matfree_operator(eq, diffmat, config)
    n = op["n_keep"]
    print(f"resolution {res}, {n} retained degrees of freedom")

    # ---- 1. matrix-free == dense --------------------------------------------
    A = np.asarray(assemble_dense(eq, diffmat, config)["A"])
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(n))
    err = np.max(np.abs(np.asarray(op["Ax"](x)) - A @ np.asarray(x)))
    print(
        f"|A x - Ax(x)|_max = {err:.3e}   (relative to |A x|: "
        f"{err / np.max(np.abs(A @ np.asarray(x))):.2e})"
    )

    # ---- 2. the shift -------------------------------------------------------
    # CG is only a legal Krylov method when H = A - sigma I is positive
    # definite, i.e. sigma below the whole spectrum. Here that is measured
    # exactly; in production it comes from SolverConfig.sigma, and choosing it
    # is discussed in docs/resolution.md.
    sigma = float(np.min(np.linalg.eigvalsh(A))) - 1.0
    H = A - sigma * np.eye(n)
    print(f"sigma = {sigma:.6f}, H is SPD: {np.min(np.linalg.eigvalsh(H)) > 0}")

    # ---- 3. ring blocks and the preconditioner ------------------------------
    keep = keep_indices(*res)
    sel, pad, G = ring_index_maps(keep, res)
    blocks = build_ring_blocks(eq, diffmat, config, res, sel, pad, sigma=sigma)
    L, ok, ridge = factor_ring_blocks(blocks)
    print(
        f"ring blocks: {blocks.shape[0]} of size {blocks.shape[1]}, "
        f"factored ok={bool(ok)}, ridge={ridge:g}"
    )
    M = make_block_precond(L, G, n)

    # ---- 4. CG, with and without ---------------------------------------------
    Hf = lambda v: jnp.asarray(H) @ v  # noqa: E731
    rhs = jnp.asarray(rng.standard_normal(n))
    _, it_plain, res_plain = pcg(Hf, rhs, lambda v: v, 1e-8, 4000)
    x_prec, it_prec, res_prec = pcg(Hf, rhs, M, 1e-8, 4000)

    want = np.linalg.solve(H, np.asarray(rhs))
    rel = np.max(np.abs(np.asarray(x_prec) - want)) / np.max(np.abs(want))
    print(
        f"plain CG:         {int(it_plain):5d} iterations, relres "
        f"{float(res_plain):.2e}"
    )
    print(
        f"ring-preconditioned: {int(it_prec):5d} iterations, relres "
        f"{float(res_prec):.2e}"
    )
    print(f"preconditioned answer vs a direct solve: {rel:.2e} relative")

    print()
    print("Note on the coarse level. A production two-level solve deflates")
    print("against the softest modes of the SAME problem on a coarser grid,")
    print("prolonged up (agnimhd.solvers.coarse_seed_and_deflation). That needs")
    print("the equilibrium re-evaluated at the coarse nodes, which only your")
    print("equilibrium code can do -- it is not an interpolation of this data.")
    print("The coarse RADIAL resolution has a hard floor of 16: below it the")
    print("solve returns the wrong mode with the opposite sign, and the floor")
    print("costs nothing. See docs/resolution.md.")


if __name__ == "__main__":
    main()
