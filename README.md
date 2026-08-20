# agnimhd

[![CI](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml)
[![docs](https://github.com/rahulgaur104/AGNI/actions/workflows/docs.yml/badge.svg)](https://rahulgaur104.github.io/AGNI/)
[![codecov](https://codecov.io/gh/rahulgaur104/AGNI/branch/master/graph/badge.svg)](https://codecov.io/gh/rahulgaur104/AGNI)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**AGNI** — a differentiable, GPU-capable, finite-*n* ideal MHD stability solver
and optimizer, packaged as a standalone Python library with no dependency on
any equilibrium code.

```python
import jax
from agnimhd import EquilibriumData, growth_rate

eq = EquilibriumData.load("my_equilibrium.npz")   # plain arrays, no equilibrium code
lam = growth_rate(eq, diffmat)                    # lambda < 0 means UNSTABLE
g = jax.grad(growth_rate)(eq, diffmat)            # d lambda / d(every input)
```

## Install

```bash
pip install agnimhd                 # runtime
pip install "agnimhd[hdf5,test]"    # HDF5 serialization and the test suite
```

Requires Python 3.12. Dependencies are exactly `jax`, `numpy`, `scipy`,
`matfree` — nothing else, and never an equilibrium code; see
[**Full documentation**](https://rahulgaur104.github.io/AGNI/) for why that
matters and how to write an adapter.

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

## Provenance

Extracted from the AGNI solver developed inside
[DESC](https://github.com/PlasmaControl/DESC)
([PR #1789](https://github.com/PlasmaControl/DESC/pull/1789)):

> R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
> stability solver & optimizer for magnetic confinement fusion devices* (2026).

## License

MIT.
