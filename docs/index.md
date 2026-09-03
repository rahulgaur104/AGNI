# agnimhd

**AGNI** — Analysis of Global Normal modes in Ideal MHD. A differentiable,
GPU-capable, finite-*n* ideal MHD stability solver, packaged as a standalone
Python library. Differentiable so that it can serve as an objective inside
someone else's stellarator optimization; it is not itself an optimizer.

AGNI discretizes the ideal MHD **energy principle** pseudospectrally in real
space, on a straight-field-line grid, and solves the resulting generalized
symmetric eigenvalue problem for the most unstable global mode. The whole
assembly is written in JAX, so the growth rate is differentiable with respect
to the equilibrium — analytically, without re-solving the equilibrium and
without differentiating the eigensolve.

```python
from agnimhd import EquilibriumData, growth_rate

eq = EquilibriumData.load("my_equilibrium.npz")   # plain arrays, no equilibrium code
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
```

## Two modes

Everything follows from one fact: **an `EquilibriumData` that did not come out
of an equilibrium solve is not an equilibrium.** Its arrays — metric, Jacobian,
current, profiles, sampled on the grid — are not independent. They satisfy
force balance, or they describe nothing physical. Nothing here checks that, and
nothing here can restore it.

**Solve mode**, this package alone. One equilibrium in, one stability answer
out, no equilibrium code — the equilibrium already happened, somebody else ran
it and saved the result:

```python
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
lam, v, residual = eigenpair(eq, diffmat)         # and the mode itself
jax.grad(growth_rate)(eq, diffmat)                # TypeError, on purpose
```

`dlambda/d(EquilibriumData)` is a sensitivity to grid samples, and those are
not free parameters — nothing you can design is one of them. Step along it and
you land on arrays that are not an equilibrium, so the `lambda` there is the
growth rate of no plasma at all. It raises rather than returning zero: a silent
zero is indistinguishable from an optimization that converged without moving.

**Optimize mode** needs a differentiable equilibrium solver *and* an optimizer,
not optionally. The entry point takes your parameters and the map from them to
an equilibrium — that map is the solve:

```python
def equilibrium_map(params):                      # must be differentiable
    return to_equilibrium_data(solve_equilibrium(params))

g = jax.grad(growth_rate_of)(params, equilibrium_map, diffmat)   # like params
```

`params` are boundary Fourier coefficients, profile coefficients, coil currents
— what you can change. AGNI supplies `dlambda/d(EquilibriumData)`;
`equilibrium_map` supplies `d(EquilibriumData)/d(params)`; the chain closes
only if both exist. **[DESC](https://github.com/PlasmaControl/DESC) is the
natural partner** for the map and the optimizer — in JAX, a gradient through
force balance, geometry on the PEST grid the
[interface contract](interface.md) asks for. See
[Consuming the gradient](adapters.md#consuming-the-gradient).

The same reasoning covers everything past a single number: regenerating the
input at all — a resolution change, a profile sweep — is a new equilibrium
solve, and plotting the eigenvector in real space is the equilibrium code's
geometry, not this package's.

### And yet there is no dependency

`agnimhd` depends on `jax`, `numpy`, `scipy` and `matfree`, and nothing else.
DESC appears nowhere — not a dependency, not an optional extra, not in the
tests, not behind a lazy import. That is about installation, not use: the
dependency runs the other way, and DESC (or VMEC, GVEC, anything else) installs
`agnimhd`, converts its own equilibrium, and wraps
[`growth_rate_of`](api.md) as an objective, with the adapter living in the
consumer's repository. So solve mode stays reachable from anywhere — an `.npz`
and four packages — while optimize mode remains, unavoidably, coupled.

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
5. That is solve mode, and it is done. To optimize, supply the map from your
   parameters and call `growth_rate_of(params, equilibrium_map, diffmat)`,
   which is differentiable in `params` at the cost of one extra operator
   application. See [Consuming the gradient](adapters.md#consuming-the-gradient).

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
