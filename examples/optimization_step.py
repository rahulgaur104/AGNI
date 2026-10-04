#!/usr/bin/env python3
"""Optimize mode: the interface, and one gradient step.

Solve mode is ``growth_rate(eq, diffmat)`` on a stored equilibrium, and is not
differentiable. Optimize mode is ``growth_rate_of(params, equilibrium_map,
diffmat)``, differentiable in ``params``, where ``equilibrium_map`` evaluates
geometry and profiles from the equilibrium's parameters. This script shows the
second, with a demonstration map that is not a physical parameterization.

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

    A demonstration map, not a physical one. It moves the minor radius the
    operator is normalized by and leaves every other array unchanged, which
    does not give a new equilibrium: metric, Jacobian, current and profiles all
    move together under a change of parameters.

    A real map evaluates geometry and profiles from the equilibrium's spectral
    coefficients and packs them::

        def equilibrium_map(params):
            data = evaluate_on_pest_grid(params)    # DESC, differentiable
            return to_equilibrium_data(data)        # the adapter

    It contains no equilibrium solve. Force balance is a constraint on the
    optimization, enforced by the optimizer. ``examples/desc_objective.py`` is
    such a map for DESC. See docs/index.md.
    """
    return lambda params: eq.replace(a=params["a"])


def main():
    """Take one descent step in the parameters and report the change."""
    eq = EquilibriumData.load(CASE)
    _, diffmat = standard_grid(*eq.resolution, NFP=eq.NFP, automorphism=AUTOMORPHISM)
    config = AssemblyConfig()

    # ---- solve mode: one equilibrium, one answer, no derivative ----------
    gamma2_solve = float(growth_rate(eq, diffmat, config))
    state = "UNSTABLE" if gamma2_solve > 0 else "stable"
    print(f"solve mode: gamma^2 {gamma2_solve:+.6e} ({state})")
    try:
        jax.grad(growth_rate)(eq, diffmat, config)
    except TypeError as err:
        print(f"solve mode: jax.grad refused -- {str(err).splitlines()[0]}")
    print()

    # ---- optimize mode: parameters in, d(gamma^2)/d(parameters) out ------
    equilibrium_map = rescale_a(eq)
    params = {"a": eq.a}

    # Value and gradient from one eigensolve. `grad` has the structure of
    # `params`, not of the equilibrium.
    gamma2_0, grad = growth_rate_and_grad(params, equilibrium_map, diffmat, config)
    gamma2_0 = float(gamma2_0)
    dgamma2_da = float(grad["a"])
    print(f"optimize mode: gamma^2 {gamma2_0:+.6e}   (same solve, same number)")
    print(f"               dgamma^2/da {dgamma2_da:+.6e}")
    print()

    # Descent: instability is gamma^2 > 0, so stabilizing means lowering
    # gamma^2 toward zero, and a minimizer uses the value as it is.
    a0 = float(params["a"])
    step = 1e-4 * a0 / abs(dgamma2_da)  # sized to a small relative change in a
    a1 = a0 - step * dgamma2_da
    gamma2_1 = float(growth_rate_of({"a": a1}, equilibrium_map, diffmat, config))

    print(f"a: {a0:.9f} -> {a1:.9f}   ({(a1 - a0) / a0:+.3e} relative)")
    print(f"gamma^2: {gamma2_0:+.6e} -> {gamma2_1:+.6e}   ({gamma2_1 - gamma2_0:+.3e})")
    print("step direction:", "correct" if gamma2_1 < gamma2_0 else "WRONG")
    print()

    # A minimizer differentiates the value as it is, and may wrap the whole
    # thing in jax.jit with the map and both configs static.
    obj = jax.jit(
        jax.grad(lambda p: growth_rate_of(p, equilibrium_map, diffmat, config))
    )
    g_obj = float(obj(params)["a"])
    print(f"dgamma^2/da from a user-defined objective: {g_obj:+.6e}")


if __name__ == "__main__":
    main()
