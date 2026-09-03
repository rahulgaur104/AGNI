# agnimhd

[![CI](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml)
[![docs](https://github.com/rahulgaur104/AGNI/actions/workflows/docs.yml/badge.svg)](https://rahulgaur104.github.io/AGNI/)
[![codecov](https://codecov.io/gh/rahulgaur104/AGNI/branch/master/graph/badge.svg)](https://codecov.io/gh/rahulgaur104/AGNI)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**AGNI** — a differentiable, GPU-capable, finite-*n* ideal MHD stability
solver, packaged as a standalone Python library with no dependency on any
equilibrium code. Differentiable so that it can serve as an objective inside
someone else's stellarator optimization; it is not itself an optimizer.

```python
from agnimhd import EquilibriumData, growth_rate

eq = EquilibriumData.load("my_equilibrium.npz")   # plain arrays, no equilibrium code
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
```

There are two modes, and they are different functions:

* **Solve mode** — `growth_rate(eq, diffmat)`, `eigenpair(eq, diffmat)`. One
  stored equilibrium in, one stability answer out, no equilibrium code. **Not
  differentiable**: `jax.grad` raises, because `dlambda/d(EquilibriumData)` is
  a sensitivity to grid samples, which are not design variables and are in
  force balance only because a solve put them there.
* **Optimize mode** — `growth_rate_of(params, equilibrium_map, diffmat)`,
  differentiable in `params`. Requires a differentiable equilibrium solver
  *and* an optimizer, not optionally: `equilibrium_map` is that solve plus your
  adapter, and it supplies the outer factor of the chain rule.
  [DESC](https://github.com/PlasmaControl/DESC) is the natural partner.

See [Two modes](https://rahulgaur104.github.io/AGNI/#two-modes).

## Install

```bash
pip install agnimhd                 # runtime
pip install "agnimhd[hdf5,test]"    # HDF5 serialization and the test suite
```

Requires Python 3.12. Dependencies are exactly `jax`, `numpy`, `scipy`,
`matfree` — nothing else, and never an equilibrium code. That is a statement
about installation, not about use: solve mode needs nothing more, optimize mode
needs an equilibrium solver and an optimizer coupled to it. See
[**Full documentation**](https://rahulgaur104.github.io/AGNI/) for why the
dependency runs that direction, and how to write an adapter.

## Documentation

**➡ [rahulgaur104.github.io/AGNI](https://rahulgaur104.github.io/AGNI/)**

The physics, the interface contract, adapter guidance, resolution/solver
choices, the full API reference, and migration notes from AGNI-inside-DESC all
live there — this file is a landing page, not the manual.

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
