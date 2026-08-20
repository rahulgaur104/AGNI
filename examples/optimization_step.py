#!/usr/bin/env python3
"""One gradient step against ideal MHD instability.

Shows the whole optimization interface: ``jax.grad`` applied to
:func:`agnimhd.growth_rate` from outside the package, and a step that moves the
growth rate in the stabilizing direction.

The parameter stepped here is the minor radius ``a``. It is a single scalar the
entire operator is normalized by, which makes it the cheapest honest
demonstration -- and it is also the input the eigenvalue is most sensitive to.
A real optimization would differentiate through an adapter, all the way back to
boundary-shape or profile parameters; the composition is the same.

Run::

    python examples/optimization_step.py
"""

from pathlib import Path

import jax

from agnimhd import AssemblyConfig, EquilibriumData, growth_rate, growth_rate_and_grad
from agnimhd.basis import standard_grid

AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
FIXTURE = Path(__file__).resolve().parents[1] / "tests/data/qh_lowres_24x12x8.npz"


def main():
    """Take one ascent step in `a` and report the change."""
    eq = EquilibriumData.load(FIXTURE)
    _, diffmat = standard_grid(*eq.resolution, NFP=eq.NFP, automorphism=AUTOMORPHISM)
    config = AssemblyConfig()

    # Value and gradient in one eigensolve. `grad` is an EquilibriumData whose
    # every leaf holds d(lambda)/d(that leaf) -- arrays and both scalars.
    lam0, grad = growth_rate_and_grad(eq, diffmat, config)
    lam0 = float(lam0)
    dlam_da = float(grad.a)

    print(f"lambda        {lam0:+.6e}   ({'UNSTABLE' if lam0 < 0 else 'stable'})")
    print(f"dlambda/da    {dlam_da:+.6e}")
    print(f"dlambda/dPsi  {float(grad.Psi):+.6e}")
    print(
        "|dlambda/d(finite_n_instability_drive)|_max  "
        f"{abs(grad.finite_n_instability_drive).max():.6e}"
    )
    print()

    # ASCENT, not descent. Instability is lambda < 0, so stabilizing means
    # raising lambda toward zero. Getting this backwards is the single easiest
    # way to run a long optimization in the wrong direction.
    a0 = float(eq.a)
    step = 1e-4 * a0 / abs(dlam_da)  # sized to a small relative change in a
    a1 = a0 + step * dlam_da
    lam1 = float(growth_rate(eq.replace(a=a1), diffmat, config))

    print(f"a: {a0:.9f} -> {a1:.9f}   ({(a1 - a0) / a0:+.3e} relative)")
    print(f"lambda: {lam0:+.6e} -> {lam1:+.6e}   ({lam1 - lam0:+.3e})")
    print("step direction:", "correct" if lam1 > lam0 else "WRONG")

    # A caller wanting the gradient of their own objective composes normally:
    #
    #     def objective(a):
    #         return -growth_rate(eq.replace(a=a), diffmat, config)
    #     jax.grad(objective)(a0)
    #
    # and may wrap the whole thing in jax.jit with the two configs static.
    obj_grad = jax.grad(lambda a: -growth_rate(eq.replace(a=a), diffmat, config))(a0)
    print(f"d(-lambda)/da from a user-defined objective: {float(obj_grad):+.6e}")


if __name__ == "__main__":
    main()
