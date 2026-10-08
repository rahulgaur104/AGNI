#!/usr/bin/env python3
"""Stabilize a DESC equilibrium with agnimhd's growth rate as the objective.

The shipped low-resolution QH case (``tests/data/AGNI_QH_lowres.h5``): three
steps of DESC's ``proximal-lsq-exact`` in the eight lowest boundary modes, with
force balance kept by the optimizer, the plasma profiles and ``Psi`` fixed.
The stability term is ``AgniStability`` with ``SOLVER`` below, the dense
single-GPU solver or the matrix-free Jacobi-Davidson one; the measured run of
each is in ``docs/examples.md``.

Needs DESC. About 6 min on one A100 (``DEVICE = "gpu"``); on a CPU, lower
``MAXITER`` or the basis. Run from the repository root::

    python examples/desc_optimization.py

Writes ``examples/figures/desc_optimization.png``: ``gamma^2`` at every
evaluation the optimizer made, and the boundary before and after.
"""

import time

from desc import set_device

DEVICE = "gpu"  # or "cpu"
set_device(DEVICE)  # before any other DESC or JAX import

import jax  # noqa: E402
import numpy as np  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    FixAtomicNumber,
    FixBoundaryR,
    FixBoundaryZ,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.optimize import Optimizer  # noqa: E402

from agnimhd import AssemblyConfig, Basis, SolverConfig  # noqa: E402
from agnimhd.adapters.desc_objective import AgniStability  # noqa: E402

# ----------------------------------------------------------------------------- settings
EQ_PATH = "tests/data/AGNI_QH_lowres.h5"
BASIS = Basis(
    24,
    12,
    8,
    radial="lobatto",
    automorphism=dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0),
    mpol=5,
    ntor=3,
)
ASSEMBLY = AssemblyConfig(gamma=5.0 / 3.0)
SOLVER = "jd"  # "jd" (matrix-free, coarse level BASIS.coarse()) or "dense" (one GPU)
SIGMA = 1e-3  # above every gamma^2 the optimizer meets (trial points reached 3.9e-4)
WEIGHT = 100.0  # stability term, target 0
FB_WEIGHT = 500.0  # force balance in the objective; also a constraint
UNFIX_K = 1  # boundary modes with max(|m|, |n|) <= UNFIX_K are free: 8 modes
MAXITER = 3
OUT = "examples/figures/desc_optimization.png"
# -----------------------------------------------------------------------------

HISTORY = []  # gamma^2 at every evaluation, accepted or trial
T_START = time.time()


class LoggedAgniStability(AgniStability):
    """``AgniStability`` that records ``gamma^2`` at every evaluation."""

    def compute(self, params, constants=None):
        """``gamma^2``, appended to ``HISTORY`` from inside DESC's jit."""
        value = super().compute(params, constants)
        jax.debug.callback(lambda g: HISTORY.append(float(g)), value[0])
        return value


def solver_config():
    """The ``SolverConfig`` of ``SOLVER``."""
    if SOLVER == "jd":
        return SolverConfig(
            eigensolver="jd", sigma=SIGMA, jd_tol=1e-4, jd_theta_tol=0.0
        )
    return SolverConfig(eigensolver="dense", sigma=SIGMA)


def constraints(eq):
    """Force balance, the fixed boundary modes, fixed profiles and ``Psi``."""
    R = np.asarray(eq.surface.R_basis.modes)
    Z = np.asarray(eq.surface.Z_basis.modes)
    return (
        ForceBalance(eq=eq),
        FixBoundaryR(
            eq=eq, modes=np.vstack(([0, 0, 0], R[np.abs(R).max(1) > UNFIX_K]))
        ),
        FixBoundaryZ(eq=eq, modes=Z[np.abs(Z).max(1) > UNFIX_K]),
        FixPsi(eq=eq),
        FixIota(eq=eq),
        FixElectronDensity(eq=eq),
        FixElectronTemperature(eq=eq),
        FixIonTemperature(eq=eq),
        FixAtomicNumber(eq=eq),
    )


def boundary(eq, zeta):
    """``(R, Z)`` of the boundary at toroidal angle ``zeta``, closed."""
    grid = LinearGrid(rho=1.0, theta=np.linspace(0, 2 * np.pi, 181), zeta=zeta)
    data = eq.compute(["R", "Z"], grid=grid)
    return np.asarray(data["R"]), np.asarray(data["Z"])


def draw(history, eq_before, eq_after, out):
    """``gamma^2`` per evaluation and the boundary before and after, to ``out``.

    ``history``: the list of one run, or ``{label: list}`` of several.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = history if isinstance(history, dict) else {SOLVER: history}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), width_ratios=[1.4, 1, 1])
    ax = axes[0]
    for label, values in runs.items():
        ax.semilogy(range(1, len(values) + 1), values, "o-", label=label, alpha=0.8)
    first = next(iter(runs.values()))
    ax.axhline(first[0], color="k", lw=0.8, ls="--")
    ax.axhline(first[-1], color="k", lw=0.8, ls=":")
    ax.set_xlabel("evaluation (accepted and trial points)")
    ax.set_ylabel(r"$\gamma^2$ (positive: unstable)")
    ax.set_title(f"{first[0]:.3e} -> {first[-1]:.3e}")
    ax.legend()
    for ax, zeta, name in zip(axes[1:], (0.0, np.pi / eq_after.NFP), ("0", "pi / NFP")):
        for eq, style, lab in ((eq_before, "--", "before"), (eq_after, "-", "after")):
            R, Z = boundary(eq, zeta)
            ax.plot(R, Z, style, label=lab)
        ax.set_aspect("equal")
        ax.set_title(f"boundary, zeta = {name}")
        ax.set_xlabel("R (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[{time.time() - T_START:6.0f} s] wrote {out}", flush=True)


def main():
    """Optimize, report every accepted step, draw the figure."""
    eq0 = Equilibrium.load(EQ_PATH)
    stability = LoggedAgniStability(
        eq0, BASIS, assembly=ASSEMBLY, solver=solver_config(), target=0.0, weight=WEIGHT
    )
    objective = ObjectiveFunction(
        (stability, ForceBalance(eq=eq0, weight=FB_WEIGHT)), deriv_mode="blocked"
    )
    (eq1,), result = Optimizer("proximal-lsq-exact").optimize(
        eq0,
        objective,
        constraints(eq0),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        maxiter=MAXITER,
        verbose=3,
        options={"solve_options": {"maxiter": 10, "verbose": 0}},
    )
    print(
        f"[{time.time() - T_START:6.0f} s] gamma^2 {HISTORY[0]:.6e} -> "
        f"{HISTORY[-1]:.6e} in {result['nit']} steps, {len(HISTORY)} evaluations "
        f"({SOLVER})",
        flush=True,
    )
    draw(HISTORY, eq0, eq1, OUT)


if __name__ == "__main__":
    main()
