#!/usr/bin/env python3
"""Stabilize a DESC equilibrium with agnimhd's growth rate. Requires DESC.

``AgniStability`` is a DESC objective. ``ProximalProjection`` (optimizer
``"proximal-lsq-exact"``) re-solves the equilibrium every step, so the step is
taken in the boundary coefficients with force balance kept.

Run on CPU::

    JAX_PLATFORMS=cpu python examples/desc_objective.py
"""

from pathlib import Path

from desc import set_device

set_device("cpu")

from desc.io import load  # noqa: E402
from desc.objectives import (  # noqa: E402
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)

from agnimhd.adapters.desc_objective import AgniStability  # noqa: E402

EQ_PATH = Path(__file__).resolve().parents[1] / "tests/data/AGNI_QH_lowres.h5"
RES = (24, 12, 8)  # PEST grid of the stability solve
MAX_MODE = 1  # free boundary modes: max(|m|, |n|) <= MAX_MODE
MAXITER = 2


def fixed_modes(basis, keep_axis):
    """Boundary modes held fixed: everything above MAX_MODE (and R_00)."""
    return [
        (lm, m, n)
        for lm, m, n in basis.modes
        if max(abs(m), abs(n)) > MAX_MODE or (keep_axis and m == 0 and n == 0)
    ]


eq = load(str(EQ_PATH))
eq = eq[-1] if hasattr(eq, "__getitem__") else eq
constraints = (
    ForceBalance(eq),
    FixBoundaryR(eq, modes=fixed_modes(eq.surface.R_basis, True)),
    FixBoundaryZ(eq, modes=fixed_modes(eq.surface.Z_basis, False)),
    FixPressure(eq),
    FixCurrent(eq),
    FixPsi(eq),
)
objective = ObjectiveFunction((AgniStability(eq, res=RES),))
eq_new, _ = eq.optimize(
    objective, constraints, optimizer="proximal-lsq-exact", maxiter=MAXITER, copy=True
)
