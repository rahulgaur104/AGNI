# agnimhd

**AGNI** — Analysis of Global Normal modes in Ideal MHD. A finite-*n* ideal MHD
stability solver, GPU-capable and differentiable, packaged as a standalone
Python library. It computes a stability objective and its gradient; it does not
perform the optimization.

AGNI discretizes the ideal MHD **energy principle** pseudospectrally in real
space, on a straight-field-line grid, and solves the resulting generalized
symmetric eigenvalue problem for the most unstable global mode. The assembly is
written in JAX, so the growth rate is differentiable with respect to the
equilibrium analytically, without re-solving the equilibrium and without
differentiating the eigensolve.

```python
from agnimhd import EquilibriumData, growth_rate

eq = EquilibriumData.load("my_equilibrium.npz")   # plain arrays, no equilibrium code
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
```

## Two modes

An `EquilibriumData` that did not come out of an equilibrium solve is not an
equilibrium. Its arrays — metric, Jacobian, current, profiles, sampled on the
grid — are not independent; they satisfy force balance because a solve made
them satisfy it. The package neither checks this nor can restore it, which
separates two uses.

**Solve mode** requires only this package. One equilibrium in, one stability
answer out:

```python
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
lam, v, residual = eigenpair(eq, diffmat)         # and the mode itself
jax.grad(growth_rate)(eq, diffmat)                # raises TypeError
```

`dlambda/d(EquilibriumData)` is a sensitivity to grid samples, which are not
free parameters. A step along it gives arrays that are not in force balance, so
the `lambda` evaluated there does not correspond to any equilibrium. The call
raises rather than returning zero, since a zero gradient is indistinguishable
from an optimization that has converged.

**Optimize mode** requires a differentiable equilibrium solver and an
optimizer. The entry point takes the parameters and the map from them to an
equilibrium; that map is the equilibrium solve:

```python
def equilibrium_map(params):                      # must be differentiable
    return to_equilibrium_data(solve_equilibrium(params))

g = jax.grad(growth_rate_of)(params, equilibrium_map, diffmat)   # like params
```

`params` are the design variables: boundary Fourier coefficients, profile
coefficients, coil currents. AGNI supplies `dlambda/d(EquilibriumData)` and
`equilibrium_map` supplies `d(EquilibriumData)/d(params)`; the chain rule
closes only if both exist. [DESC](https://github.com/PlasmaControl/DESC)
provides both the map and the optimizer: it is written in JAX, differentiates
through force balance, and evaluates geometry on the PEST grid the
[interface contract](interface.md) specifies. See
[Consuming the gradient](adapters.md#consuming-the-gradient).

The same argument applies to anything beyond a single evaluation. Changing the
resolution or sweeping a profile requires a new equilibrium solve, and plotting
the eigenvector in real space requires the equilibrium code's geometry.

### Dependencies

`agnimhd` requires `jax`, `numpy`, `scipy` and `matfree`, and nothing else.
DESC is not a dependency, an optional extra, a test requirement, or a lazy
import. The dependency runs the other way: DESC (or VMEC, GVEC) installs
`agnimhd`, converts its own equilibrium, and wraps [`growth_rate_of`](api.md)
as an objective, with the adapter in the consumer's repository. Solve mode
therefore needs only an `.npz` and four packages, while optimize mode is
necessarily a coupled calculation.

## Procedure

1. Evaluate the metric, Jacobian, current and profiles on a tensor-product PEST
   grid `(rho, theta_PEST, phi)`, flattened **rho-major**.
2. Pack them into an `EquilibriumData`. That object is the entire interface;
   [the interface contract](interface.md) gives the field-by-field
   specification, and `agnimhd validate my_equilibrium.npz` checks an adapter
   against it.
3. Build a `DiffMat`, the differentiation and quadrature operators on the same
   nodes. The default is Legendre-Lobatto radially through a clustering map and
   Fourier in the two angles, which converges fastest;
   `agnimhd.basis.standard_grid` constructs both together.
4. `growth_rate(eq, diffmat)` returns `lambda`. **Negative means unstable.**
5. For optimize mode, supply the map from the design parameters and call
   `growth_rate_of(params, equilibrium_map, diffmat)`, which is differentiable
   in `params` at the cost of one additional operator application. See
   [Consuming the gradient](adapters.md#consuming-the-gradient).

`examples/growth_rate.py` is a runnable version of steps 1-4;
`examples/optimization_step.py` covers step 5.

## Sign convention

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
