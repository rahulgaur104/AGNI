# agnimhd

[![CI](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml)
[![docs](https://github.com/rahulgaur104/AGNI/actions/workflows/docs.yml/badge.svg)](https://rahulgaur104.github.io/AGNI/)
[![codecov](https://codecov.io/gh/rahulgaur104/AGNI/branch/master/graph/badge.svg)](https://codecov.io/gh/rahulgaur104/AGNI)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**AGNI** is a finite-*n* ideal MHD stability solver, GPU-capable and
differentiable, packaged as a standalone Python library with no dependency on
any equilibrium code. It computes a stability objective and its gradient. It
does not perform the optimization.

```python
from agnimhd import EquilibriumData, growth_rate

eq = EquilibriumData.load("my_equilibrium.npz")   # plain arrays, no equilibrium code
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
```

The package exposes two distinct modes:

* **Solve mode**: `growth_rate(eq, diffmat)`, `eigenpair(eq, diffmat)`. One
  stored equilibrium in, one stability answer out. Not differentiable:
  `jax.grad` raises, since `dlambda/d(EquilibriumData)` is a sensitivity to
  grid samples, which are not design variables and are in force balance only
  because an equilibrium solve made them so.
* **Optimize mode**: `growth_rate_of(params, equilibrium_map, diffmat)`,
  differentiable in `params`. `equilibrium_map` evaluates geometry and profiles
  from the equilibrium's parameters and contains no equilibrium solve, so the
  result is a partial derivative at a fixed force balance residual. Enforcing
  force balance is the optimizer's task, done in
  [DESC](https://github.com/PlasmaControl/DESC) by `ProximalProjection`.

See [Two modes](https://rahulgaur104.github.io/AGNI/#two-modes).

## Install

```bash
pip install agnimhd                 # runtime
pip install "agnimhd[hdf5,test]"    # HDF5 serialization and the test suite
```

Requires Python 3.12. Dependencies are `jax`, `numpy`, `scipy` and `matfree`,
and no equilibrium code. This concerns installation only: solve mode requires
nothing further, while optimize mode requires an equilibrium solver and an
optimizer coupled to it. See
[**Full documentation**](https://rahulgaur104.github.io/AGNI/) for the
dependency direction and for how to write an adapter.

## Documentation

**➡ [rahulgaur104.github.io/AGNI](https://rahulgaur104.github.io/AGNI/)**

Start with
[Getting the data and running a case](https://rahulgaur104.github.io/AGNI/running/),
which covers the export and solve procedure for a stellarator and a tokamak.
The physics, the input specification, adapter guidance, resolution and solver
choices, the full API reference, and migration notes from AGNI-inside-DESC all
live there. This file is a landing page, not the manual.

## Development

```bash
git clone https://github.com/rahulgaur104/AGNI && cd AGNI
pip install -e ".[dev]"
pre-commit install                  # black, isort, flake8, the no-DESC check
pytest tests -q --cov=agnimhd       # ~13 min on CPU; nothing here needs a GPU
```

## Algorithm and references

Extracted from the AGNI solver developed inside
[DESC](https://github.com/PlasmaControl/DESC)
([PR #1893](https://github.com/PlasmaControl/DESC/pull/1893)):

> R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
> stability solver & optimizer for magnetic confinement fusion devices* (2026).

## License

MIT.
