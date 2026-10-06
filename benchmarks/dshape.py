#!/usr/bin/env python3
"""DSHAPE tokamak benchmark: the AGNI paper's tokamak case at its resolution.

arXiv:2608.01750v3, section 5.2, figure 5: each toroidal mode ``n = 1 ... 5``
solved on its own, coupled Zernike basis on 96x96 nodes, ``MPOL = 4 n``,
``Gamma = 5/3``. Zernike penalty 0.08 for ``n = 1`` and 0.01 for ``n = 2 ... 5``:
the paper's runs used 0.01 for every ``n`` (the paper does not state it), which
leaves a spurious unstable ``n = 1`` mode. Shift-invert Lanczos with a dense LU
on the GPU.

Run on a node with one 80 GB GPU (needs DESC to load the equilibrium)::

    python benchmarks/dshape.py

Edit the settings below; there are no arguments. Prints, per ``n``, ``gamma^2``,
the eigenpair residual, the growth rate in rad/s and the paper's value, and
writes them to ``OUTPUT``. The residual is large (4e2 to 3e3 for ``n = 2 ... 5``)
because the whitened penalty makes the matrix stiff, even where ``gamma^2``
matches the paper's ARPACK value to seven digits. Measured values and the
penalty's effect: ``docs/benchmarks.md``.
"""

from desc import set_device

set_device("gpu")  # before JAX starts: DESC otherwise selects the CPU

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.constants import mu_0, proton_mass  # noqa: E402

from agnimhd import (  # noqa: E402
    AssemblyConfig,
    Basis,
    SolverConfig,
    eigenpair,
    from_desc,
)

# ---- settings ----------------------------------------------------------------
EQUILIBRIUM = Path(__file__).resolve().parents[1] / "tests/data/dshape_imax0.98_1608.h5"
N_RHO, N_THETA = 96, 96
MODES = (1, 2, 3, 4, 5)
MPOL_PER_MODE = 4  # MPOL = 4 n
ZERNIKE_PENALTY = {1: 0.08, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01}
GAMMA = 5.0 / 3.0
# Shift per mode, above the largest gamma^2 (the measured values above).
SIGMA = {1: 1e-4, 2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 5e-4}
NUM_MATVECS = 300
# Ion density [m^-3] that converts gamma^2 to rad/s, as in figure 5(e).
ION_DENSITY = 2.2e20
# Figure 5(e), growth rate [rad/s]; n = 1 read off the plot.
PAPER_RAD_PER_S = {1: 5.13e2, 2: 2.24146e4, 3: 2.24973e4, 4: 1.79953e4, 5: 1.08227e4}
OUTPUT = Path("dshape_benchmark.json")
# ------------------------------------------------------------------------------


def rad_per_s(gamma2, eq):
    """Growth rate in rad/s: ``sqrt(gamma^2) v_A / a``, ``B_N = Psi / (pi a^2)``."""
    a, Psi = float(eq.a), float(eq.Psi)
    v_alfven = abs(Psi) / (np.pi * a**2) / np.sqrt(mu_0 * ION_DENSITY * proton_mass)
    return float(np.sqrt(max(gamma2, 0.0)) * v_alfven / a)


def solve_mode(eq, n):
    """``(gamma2, residual, seconds)`` of toroidal mode ``n``."""
    start = time.time()
    basis = Basis(
        N_RHO,
        N_THETA,
        1,
        radial="zernike",
        mpol=MPOL_PER_MODE * n,
        zernike_penalty=ZERNIKE_PENALTY[n],
    )
    _, diffmat = basis.nodes_and_diffmat(eq.NFP)
    config = AssemblyConfig(
        gamma=GAMMA,
        axisym=True,
        n_mode_axisym=n,
        coupled_rt=True,
        n_rho_coupled=N_RHO,
        n_theta_coupled=N_THETA,
    )
    solver = SolverConfig(
        eigensolver="jax_lanczos", factor="lu", sigma=SIGMA[n], num_matvecs=NUM_MATVECS
    )
    gamma2, _, residual = eigenpair(eq, diffmat, config, solver)
    return float(gamma2), float(residual), time.time() - start


def main():
    """Load DSHAPE on the Zernike nodes, solve every mode, report."""
    start = time.time()
    # The geometry does not depend on MPOL or the penalty; each mode builds its own.
    nodes_only = Basis(N_RHO, N_THETA, 1, radial="zernike", mpol=1, zernike_penalty=0)
    eq, _ = from_desc(str(EQUILIBRIUM), nodes_only)
    print(
        f"loaded {EQUILIBRIUM.name} on {N_RHO}x{N_THETA} in {time.time() - start:.0f} s"
    )
    rows = []
    for n in MODES:
        gamma2, residual, seconds = solve_mode(eq, n)
        rate = rad_per_s(gamma2, eq)
        rows.append(
            dict(n=n, gamma2=gamma2, residual=residual, rad_per_s=rate, seconds=seconds)
        )
        print(
            f"n={n}  gamma^2 {gamma2:+.6e}  residual {residual:.1e}  {rate:.4e} rad/s  "
            f"paper {PAPER_RAD_PER_S[n]:.4e}  ratio {rate / PAPER_RAD_PER_S[n]:.4f}  "
            f"{seconds:.0f} s",
            flush=True,
        )
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
