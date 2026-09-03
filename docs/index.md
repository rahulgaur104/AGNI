# agnimhd

!!! warning "Under active development"

    This package is not stable. The API, the file format and the numerics are
    all still changing, and changes are not announced. For work that needs a
    settled version, use the AGNI implementation inside DESC,
    [PR #1893](https://github.com/PlasmaControl/DESC/pull/1893), branch
    `rg/AGNI_var`.

**AGNI**, Analysis of Global Normal modes in Ideal MHD, is a finite-*n* ideal MHD
stability solver, GPU-capable and differentiable, packaged as a standalone
Python library. It computes a stability objective and its gradient. It does not
perform the optimization.

AGNI discretizes the ideal MHD **energy principle** pseudospectrally in real
space on a straight-field-line grid. The discretization gives a generalized
symmetric problem `A x = lambda B x` with `B` positive definite. `B` is
block-diagonal in the three components at each node, so it is Cholesky-factored
node by node and the congruence `L^-1 A L^-T` reduces the pencil to a
**standard** symmetric problem. That standard problem is what every solver in
the package works on. The assembly is written in JAX, so the growth rate is
differentiable with respect to the equilibrium analytically, without re-solving
the equilibrium and without differentiating the eigensolve.

```python
from agnimhd import EquilibriumData, growth_rate

eq = EquilibriumData.load("my_equilibrium.npz")   # plain arrays, no equilibrium code
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
```

## Two modes

An `EquilibriumData` holds the metric, Jacobian, current and profiles sampled
on a grid. Those arrays are not independent. They satisfy force balance because
an equilibrium solve produced them, and the package can neither verify that nor
restore it. Two uses follow.

**Solve mode** requires only this package. One equilibrium in, one stability
answer out:

```python
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
lam, v, residual = eigenpair(eq, diffmat)         # and the mode itself
jax.grad(growth_rate)(eq, diffmat)                # raises TypeError
```

`dlambda/d(EquilibriumData)` is a sensitivity to grid samples, which are not
free parameters. A step along it produces arrays that violate force balance, so
the `lambda` evaluated there corresponds to no equilibrium. The call raises
rather than returning zero, because a zero gradient cannot be distinguished
from an optimization that has converged.

**Optimize mode** takes the equilibrium's parameters and a map from them to an
`EquilibriumData`:

```python
def equilibrium_map(params):                      # geometry and profiles only
    return to_equilibrium_data(evaluate_on_pest_grid(params))

g = jax.grad(growth_rate_of)(params, equilibrium_map, diffmat)
```

`equilibrium_map` evaluates geometry and profiles from the equilibrium's
spectral coefficients and packs the result. **It contains no equilibrium
solve.** Differentiating through a Newton iteration is not how this derivative
is computed in practice, and nothing here asks for it.

What AGNI and the map produce together is a partial derivative, taken at a
fixed force balance residual. Force balance is a constraint on the
optimization, and enforcing it is the optimizer's task. In DESC this is
`ProximalProjection`. After each step the equilibrium is perturbed and
re-solved to return the iterate to the constraint surface, and the reduced
derivative

```
d lambda / dc  =  @lambda/@c  -  (@lambda/@x) (@F/@x)^-1 (@F/@c)
```

is formed, where `F` is the force balance residual, `x` the equilibrium state
`(R_lmn, Z_lmn, L_lmn)` and `c` the free parameters such as boundary
coefficients, profile coefficients and `Psi`. AGNI supplies the `@lambda`
factors through `equilibrium_map`. DESC supplies `F`, its Jacobians, and the
projection. See [Consuming the gradient](adapters.md#consuming-the-gradient).

Anything beyond a single evaluation needs the equilibrium code for the same
reason. Changing the resolution or sweeping a profile requires a new
equilibrium solve, and plotting the eigenvector in real space requires the
equilibrium code's geometry.

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
2. Pack them into an `EquilibriumData`. That object is the entire interface.
   [the `EquilibriumData` page](interface.md) gives the field-by-field
   specification, and `agnimhd validate my_equilibrium.npz` checks an adapter
   against it.
3. Build a `DiffMat`, the differentiation and quadrature operators on the same
   nodes. The default is Legendre-Lobatto radially through a clustering map and
   Fourier in the two angles, which converges fastest.
   `agnimhd.basis.standard_grid` constructs both together.
4. `growth_rate(eq, diffmat)` returns `lambda`. **Negative means unstable.**
5. For optimize mode, supply the map from the design parameters and call
   `growth_rate_of(params, equilibrium_map, diffmat)`, which is differentiable
   in `params` at the cost of one additional operator application. See
   [Consuming the gradient](adapters.md#consuming-the-gradient).

`examples/growth_rate.py` is a runnable version of steps 1-4, and
`examples/optimization_step.py` covers step 5.

## Sign convention

`agnimhd` returns `lambda = <xi|A|xi> / <xi|B|xi>`, the **energy** quotient:

| | unstable | marginal | stable |
|---|---|---|---|
| `agnimhd` `growth_rate` | `lambda < 0` | `lambda = 0` | `lambda > 0` |
| AGNI paper, Eq. (19) | `lambda > 0` | `lambda = 0` | `lambda < 0` |

The paper writes `dW_p = -lambda dK`, so its `lambda` is the normalized squared
growth rate and carries the opposite sign. An optimizer that minimizes where it
should maximize will run in the wrong direction without failing. In this package
an optimizer **raises** `lambda` toward zero. Full derivation:
[Theory, Sign convention](theory.md#sign-convention).

## Status and provenance

This package is an extraction of the AGNI solver developed inside
[DESC](https://github.com/PlasmaControl/DESC)
([PR #1893](https://github.com/PlasmaControl/DESC/pull/1893), which builds on
the differentiation matrices of PR #1789). The physics,
discretization, benchmarks against `NIMSTELL`, and the numerical scheme are
described in:

> R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
> stability solver & optimizer for magnetic confinement fusion devices* (2026).

Two bugs in the original implementation were found by the test suite during the
extraction and are fixed here: `fourier_interp_matrix` ignoring its `period`,
and `pcg_deflated` double-counting a seed given alongside a deflation space. See
[Migrating from DESC](migration.md#things-that-changed-because-they-were-wrong)
for the measurements that caught them.

## Where to go next

- **Running a case for the first time?**
  [Getting the data and running a case](running.md) gives the export and solve
  procedure for a stellarator and for a tokamak.
- **The physics and the discretization?** [Theory](theory.md), then
  [`EquilibriumData`](interface.md) for what the solver needs.
- **A code other than DESC?** [Writing an adapter](adapters.md).
- **Resolution, shift, or eigensolver?** [Resolution and solvers](resolution.md).
- **Coming from AGNI-inside-DESC?** [Migrating from DESC](migration.md).
- **A function signature?** [API reference](api.md).
