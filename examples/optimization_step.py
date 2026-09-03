#!/usr/bin/env python3
"""Optimize mode: the interface, and one gradient step.

Solve mode is ``growth_rate(eq, diffmat)`` on a stored equilibrium, and is not
differentiable. Optimize mode is ``growth_rate_of(params, equilibrium_map,
diffmat)``, differentiable in ``params``, where ``equilibrium_map`` is your
differentiable map from parameters to an equilibrium. This script shows the
second. Note that the map used here is **not an equilibrium solve**: the
package contains no equilibrium code and cannot supply one.

Run::

    python examples/optimization_step.py
"""

from pathlib import Path

import jax

from agnimhd import (
    AssemblyConfig,
    EquilibriumData,
    growth_rate,
    growth_rate_and_grad,
    growth_rate_of,
)
from agnimhd.basis import standard_grid

AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
CASE = Path(__file__).resolve().parents[1] / "tests/data/qh_lowres_24x12x8.npz"


def rescale_a(eq):
    """Build a ``params -> EquilibriumData`` map: ``{"a": value}``.

    **A demonstration map, not a physical one.** It moves the minor radius the
    operator is normalized by and leaves every other array alone, which does
    not produce a new equilibrium -- metric, Jacobian, current and profiles
    would all have to move together, and only an equilibrium solve knows how.
    A real map is that solve plus an adapter::

        def equilibrium_map(params):
            eq_desc = solve_equilibrium(params)     # DESC, differentiable
            return to_equilibrium_data(eq_desc)     # the adapter

    with ``params`` the boundary or profile coefficients you are designing.
    ``examples/desc_adapter.py`` is the adapter half, and is numpy-based and so
    solve-mode only. See docs/adapters.md.
    """
    return lambda params: eq.replace(a=params["a"])


def main():
    """Take one ascent step in the parameters and report the change."""
    eq = EquilibriumData.load(CASE)
    _, diffmat = standard_grid(*eq.resolution, NFP=eq.NFP, automorphism=AUTOMORPHISM)
    config = AssemblyConfig()

    # ---- solve mode: one equilibrium, one answer, no derivative ----------
    lam_solve = float(growth_rate(eq, diffmat, config))
    state = "UNSTABLE" if lam_solve < 0 else "stable"
    print(f"solve mode: lambda {lam_solve:+.6e} ({state})")
    try:
        jax.grad(growth_rate)(eq, diffmat, config)
    except TypeError as err:
        print(f"solve mode: jax.grad refused -- {str(err).splitlines()[0]}")
    print()

    # ---- optimize mode: parameters in, d(lambda)/d(parameters) out -------
    equilibrium_map = rescale_a(eq)
    params = {"a": eq.a}

    # Value and gradient from one eigensolve. `grad` has the structure of
    # `params`, not of the equilibrium.
    lam0, grad = growth_rate_and_grad(params, equilibrium_map, diffmat, config)
    lam0 = float(lam0)
    dlam_da = float(grad["a"])
    print(f"optimize mode: lambda {lam0:+.6e}   (same solve, same number)")
    print(f"               dlambda/da {dlam_da:+.6e}")
    print()

    # ASCENT, not descent. Instability is lambda < 0, so stabilizing means
    # raising lambda toward zero. Getting this backwards is the single easiest
    # way to run a long optimization in the wrong direction.
    a0 = float(params["a"])
    step = 1e-4 * a0 / abs(dlam_da)  # sized to a small relative change in a
    a1 = a0 + step * dlam_da
    lam1 = float(growth_rate_of({"a": a1}, equilibrium_map, diffmat, config))

    print(f"a: {a0:.9f} -> {a1:.9f}   ({(a1 - a0) / a0:+.3e} relative)")
    print(f"lambda: {lam0:+.6e} -> {lam1:+.6e}   ({lam1 - lam0:+.3e})")
    print("step direction:", "correct" if lam1 > lam0 else "WRONG")
    print()

    # A caller minimizing rather than maximizing negates, and may wrap the
    # whole thing in jax.jit with the map and both configs static.
    obj = jax.jit(
        jax.grad(lambda p: -growth_rate_of(p, equilibrium_map, diffmat, config))
    )
    g_obj = float(obj(params)["a"])
    print(f"d(-lambda)/da from a user-defined objective: {g_obj:+.6e}")


if __name__ == "__main__":
    main()
