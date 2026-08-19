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
g = jax.grad(growth_rate)(eq, diffmat)            # d lambda / d(every input)
```

## What this package is not

It is **not** an equilibrium code, and it does not depend on one.

`agnimhd` depends on `jax`, `numpy`, `scipy` and `matfree`, and on nothing else.
DESC in particular appears nowhere — not as a dependency, not as an optional
extra, not in the tests, not behind a lazy import. The dependency is designed to
run the other way: DESC (or VMEC, GVEC, or anything else) installs `agnimhd`,
converts its own equilibrium into an [`EquilibriumData`](docs/interface.md), and
wraps [`growth_rate`](docs/api.md) as an objective. Adapters live in the
consumer's repository; `examples/desc_adapter.py` is a ~50-line reference one,
and it is not part of the package.

## Install

```bash
pip install agnimhd                 # runtime
pip install "agnimhd[hdf5,test]"    # HDF5 serialization and the test suite
```

Python >= 3.10. Everything runs on CPU; JAX picks up a GPU if one is present and
a CUDA-enabled `jaxlib` is installed, which is worth roughly an order of
magnitude on both the eigenvalue and the gradient (paper, tables 3–4).

## The five-minute version

1. Your equilibrium code evaluates the metric, Jacobian, current and profiles on
   a tensor-product PEST grid `(rho, theta_PEST, phi)`, flattened **rho-major**.
2. You pack them into an `EquilibriumData`. That object is the *entire*
   interface — see [docs/interface.md](docs/interface.md) for the field-by-field
   contract, and run `agnimhd validate my_equilibrium.npz` to check your adapter.
3. You build a `DiffMat` — differentiation and quadrature operators on the same
   nodes. Legendre-Lobatto radially with a clustering map, Fourier in the two
   angles, is the default and the best-converging choice.
4. `growth_rate(eq, diffmat)` returns `lambda`. **Negative means unstable.**
5. `jax.grad(growth_rate)(eq, diffmat)` returns the Hellmann-Feynman gradient
   with respect to every array and both scalars, at the cost of one extra
   operator application.

A complete runnable version is `examples/growth_rate.py`.

## Sign convention, once

`agnimhd` returns `lambda = <xi|A|xi> / <xi|B|xi>`, the **energy** quotient:

| | unstable | marginal | stable |
|---|---|---|---|
| `agnimhd` `growth_rate` | `lambda < 0` | `lambda = 0` | `lambda > 0` |
| AGNI paper, Eq. (16) | `lambda > 0` | `lambda = 0` | `lambda < 0` |

The two differ by an overall sign, and an optimizer that minimizes what it
should maximize will run happily in the wrong direction for a long time. In this
package an optimizer **raises** `lambda` toward zero. See
[docs/theory.md](docs/theory.md#sign-convention).

## Documentation

| | |
|---|---|
| [docs/theory.md](docs/theory.md) | the energy functional, the discretization, the sign convention, and what `lambda` means physically |
| [docs/interface.md](docs/interface.md) | the `EquilibriumData` contract: every field, its units, its coordinate basis |
| [docs/adapters.md](docs/adapters.md) | writing an adapter for a new equilibrium code, with a checklist and the two traps that produce silently wrong answers |
| [docs/resolution.md](docs/resolution.md) | choosing resolution, the shift, and the solver; what is converged and what is noise |
| [docs/api.md](docs/api.md) | public API reference |

## Status and provenance

This package is an extraction of the AGNI solver developed inside
[DESC](https://github.com/PlasmaControl/DESC) (PR #1789). The physics,
discretization, benchmarks against `NIMSTELL`, and the numerical scheme are
described in:

> R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
> stability solver & optimizer for magnetic confinement fusion devices* (2026).

Two bugs in the original implementation were found by the test suite during the
extraction and are fixed here (`fourier_interp_matrix` ignoring its `period`;
`pcg_deflated` double-counting a seed given alongside a deflation space). Both
are written up in `STATUS.md` with the measurements that caught them.

## License

MIT.
