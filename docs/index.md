# agnimhd

**AGNI** — Analysis of Global Normal modes in Ideal MHD. A differentiable,
GPU-capable, finite-*n* ideal MHD stability solver and optimizer, packaged as a
standalone Python library.

AGNI discretizes the ideal MHD **energy principle** pseudospectrally in real
space, on a straight-field-line grid, and solves the resulting generalized
symmetric eigenvalue problem for the most unstable global mode. Because the
whole assembly is written in JAX, the growth rate is a differentiable function
of the equilibrium: `jax.grad` returns `dlambda/d(equilibrium)` analytically,
without re-solving the equilibrium and without differentiating the eigensolve.

```python
import jax
from agnimhd import EquilibriumData, growth_rate

eq = EquilibriumData.load("my_equilibrium.npz")   # plain arrays, no equilibrium code
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
g = jax.grad(growth_rate)(eq, diffmat)             # d lambda / d(every input)
```

## What this package is not

It is **not** an equilibrium code, and it does not depend on one.

`agnimhd` depends on `jax`, `numpy`, `scipy` and `matfree`, and on nothing else.
DESC in particular appears nowhere — not as a dependency, not as an optional
extra, not in the tests, not behind a lazy import. The dependency is designed to
run the other way: DESC (or VMEC, GVEC, or anything else) installs `agnimhd`,
converts its own equilibrium into an [`EquilibriumData`](interface.md), and
wraps [`growth_rate`](api.md) as an objective. Adapters live in the consumer's
repository; `examples/desc_adapter.py` is a ~50-line reference one, and it is
not part of the package.

## The five-minute version

1. Your equilibrium code evaluates the metric, Jacobian, current and profiles on
   a tensor-product PEST grid `(rho, theta_PEST, phi)`, flattened **rho-major**.
2. You pack them into an `EquilibriumData`. That object is the *entire*
   interface — see [the interface contract](interface.md) for the field-by-field
   spec, and run `agnimhd validate my_equilibrium.npz` to check your adapter.
3. You build a `DiffMat` — differentiation and quadrature operators on the same
   nodes. Legendre-Lobatto radially with a clustering map, Fourier in the two
   angles, is the default and the best-converging choice
   (`agnimhd.basis.standard_grid` builds both together, on purpose).
4. `growth_rate(eq, diffmat)` returns `lambda`. **Negative means unstable.**
5. `jax.grad(growth_rate)(eq, diffmat)` returns the Hellmann-Feynman gradient
   with respect to every array and both scalars, at the cost of one extra
   operator application.

A complete runnable version is `examples/growth_rate.py`; a one-step
optimization is `examples/optimization_step.py`.

## Sign convention, once

`agnimhd` returns `lambda = <xi|A|xi> / <xi|B|xi>`, the **energy** quotient:

| | unstable | marginal | stable |
|---|---|---|---|
| `agnimhd` `growth_rate` | `lambda < 0` | `lambda = 0` | `lambda > 0` |
| AGNI paper, Eq. (16) | `lambda > 0` | `lambda = 0` | `lambda < 0` |

The two differ by an overall sign, and an optimizer that minimizes what it
should maximize will run happily in the wrong direction for a long time. In this
package an optimizer **raises** `lambda` toward zero. Full derivation:
[Theory → Sign convention](theory.md#sign-convention).

## Status and provenance

This package is an extraction of the AGNI solver developed inside
[DESC](https://github.com/PlasmaControl/DESC) (PR #1789). The physics,
discretization, benchmarks against `NIMSTELL`, and the numerical scheme are
described in:

> R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
> stability solver & optimizer for magnetic confinement fusion devices* (2026).

Two bugs in the original implementation were found by the test suite during the
extraction and are fixed here — `fourier_interp_matrix` ignoring its `period`,
and `pcg_deflated` double-counting a seed given alongside a deflation space. See
[Migrating from DESC](migration.md#things-that-changed-because-they-were-wrong)
for the measurements that caught them.

## Where to go next

- **New to the package?** Start with [Theory](theory.md) for the physics and
  the discretization, then [Interface contract](interface.md) for what
  `EquilibriumData` actually needs.
- **Writing an adapter** for a new equilibrium code? [Writing an adapter](adapters.md).
- **Choosing a resolution, a shift, or an eigensolver?**
  [Resolution and solvers](resolution.md).
- **Coming from AGNI-inside-DESC?** [Migrating from DESC](migration.md).
- **Just need a function signature?** [API reference](api.md).
