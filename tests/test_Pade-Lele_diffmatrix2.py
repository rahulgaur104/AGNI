#!/usr/bin/env python3
"""Test file for higher-order finite-difference mixed derivatives."""
from __future__ import annotations

import itertools
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from create_Pade_Lele_diffmatrix import (
    apply_compact_derivative,
    create_lele_D1_6_matrix,
    create_lele_D2_6_matrix,
    create_pade_D1_4_matrix,
    create_pade_D2_4_matrix,
)

jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------------ #
#  static parameters                                                 #
# ------------------------------------------------------------------ #
GRID_NS: list[int] = [51, 101, 201, 401]  # resolutions to scan
CASES: list[tuple[str, int]] = [  # (function, derivative order)
    ("sine", 1),
    ("sine", 2),
    ("gaussian", 1),
    ("gaussian", 2),
    ("tanh", 1),
    ("tanh", 2),  # 2nd-derivative of tanh skipped
]

# Empirical must-pass thresholds on the *reference* grid n = 101
STRICT_TOL = {
    ("sine", 1, "pade"): 7e-3,
    ("sine", 1, "lele"): 2e-5,
    ("sine", 2, "pade"): 6e-4,
    ("sine", 2, "lele"): 4e-4,
    ("gaussian", 1, "pade"): 2e-3,
    ("gaussian", 1, "lele"): 1e-5,
    ("gaussian", 2, "pade"): 5e-4,
    ("gaussian", 2, "lele"): 3e-5,
    ("tanh", 1, "pade"): 3e-3,
    ("tanh", 1, "lele"): 5e-7,
    ("tanh", 2, "pade"): 1e-2,
    ("tanh", 2, "lele"): 5e-3,
}


# ------------------------------------------------------------------ #
#  reference profiles                                                #
# ------------------------------------------------------------------ #
def _test_functions():
    def sine(x):
        f = jnp.sin(2 * jnp.pi * x)
        d1 = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
        d2 = -((2 * jnp.pi) ** 2) * jnp.sin(2 * jnp.pi * x)
        return f, d1, d2

    def gaussian(x):
        f = jnp.exp(-5 * x**2)
        d1 = -10 * x * f
        d2 = (-10 + 100 * x**2) * f
        return f, d1, d2

    def tanh(x):
        f = jnp.tanh(5 * x)
        sech2 = 1.0 / jnp.cosh(5 * x) ** 2
        d1 = 5 * sech2
        d2 = -50 * jnp.tanh(5 * x) * sech2
        return f, d1, d2

    return {"sine": sine, "gaussian": gaussian, "tanh": tanh}


FUNCS = _test_functions()

# ------------------------------------------------------------------ #
#  container for plot data                                           #
# ------------------------------------------------------------------ #
CollectedT = Tuple[str, int, float, float, float]  # (label, order, h, errP, errL)
_COLLECTED: list[CollectedT] = []


# ------------------------------------------------------------------ #
#  parametrised test                                                 #
# ------------------------------------------------------------------ #
@pytest.mark.regression
@pytest.mark.parametrize("n", GRID_NS)
@pytest.mark.parametrize("label, order", CASES)
def test_pade_lele_resolution(label: str, order: int, n: int):
    """Compare Padé-4 and Lele-6 accuracy on several grids."""
    x = jnp.linspace(-1.0, 1.0, n)
    h = float(x[1] - x[0])

    # build matrices for this resolution
    D1_pade = create_pade_D1_4_matrix(n, h)
    D2_pade = create_pade_D2_4_matrix(n, h)
    D1_lele = create_lele_D1_6_matrix(n, h)
    D2_lele = create_lele_D2_6_matrix(n, h)

    f, d1_true, d2_true = FUNCS[label](x)

    if order == 1:
        d_pade = apply_compact_derivative(*D1_pade, f)
        d_lele = apply_compact_derivative(*D1_lele, f)
        truth = d1_true
    else:  # order == 2
        d_pade = apply_compact_derivative(*D2_pade, f)
        d_lele = apply_compact_derivative(*D2_lele, f)
        truth = d2_true

    err_pade = jnp.linalg.norm(d_pade - truth) / jnp.linalg.norm(truth)
    err_lele = jnp.linalg.norm(d_lele - truth) / jnp.linalg.norm(truth)

    _COLLECTED.append((label, order, h, float(err_pade), float(err_lele)))

    # --- assertions ---------------------------------------------------
    # always: Lele beats Padé & both are < 1
    assert err_pade < 1.0
    assert err_lele < err_pade

    # strict bounds only on the 101-point grid
    if n == 101:
        assert err_pade < STRICT_TOL[(label, order, "pade")]
        assert err_lele < STRICT_TOL[(label, order, "lele")]


# ------------------------------------------------------------------ #
#  plotting helper                                                   #
# ------------------------------------------------------------------ #
def _plot():
    # organise: {(label, order): [(h, errP, errL), …]}
    buckets: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for rec in _COLLECTED:
        buckets.setdefault((rec[0], rec[1]), []).append(rec[2:])

    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour_iter = itertools.cycle(colour_cycle)

    for (label, order), rows in buckets.items():
        rows.sort(key=lambda t: t[0])  # by h
        h_vals, errP, errL = zip(*rows)

        col = next(colour_iter)  # Padé colour — reused for Lele

        plt.loglog(
            h_vals,
            errP,
            marker="o",
            linestyle="-",
            color=col,
            label=f"{label}  Padé  D{order}",
        )
        plt.loglog(
            h_vals,
            errL,
            marker="s",
            linestyle="--",
            color=col,
            label=f"{label}  Lele  D{order}",
        )

    # --- reference slope lines ---------------------------------------
    def slope_line(ax, p: int, h_anchor: float, y_anchor: float, **kw):
        h = jnp.array(ax.get_xlim())
        y = y_anchor * (h / h_anchor) ** p
        ax.loglog(h, y, **kw)

    ax = plt.gca()
    # anchor at the largest h in the data
    h0 = max(h for _, _, h, _, _ in _COLLECTED)
    # choose a visible anchor error
    y0 = max(err for _, _, _, err, _ in _COLLECTED) * 0.6
    slope_line(ax, 2, h0, y0, color="grey", linestyle=":", label="slope 2")
    slope_line(ax, 4, h0, y0, color="grey", linestyle="-.", label="slope 4")

    ax.invert_xaxis()
    ax.set_xlabel("grid spacing  h")
    ax.set_ylabel("relative L₂ error")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    ax.set_title("Padé-4  vs  Lele-6   convergence")
    plt.show()


# ------------------------------------------------------------------ #
#  autouse fixture to draw the figure                                #
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def _plot_after_module(request):
    """Draw convergence plot once all tests in this file finish."""
    yield
    if request.config.getoption("-s"):
        _plot()
