# agnimhd

[![CI](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulgaur104/AGNI/actions/workflows/ci.yml)
[![docs](https://github.com/rahulgaur104/AGNI/actions/workflows/docs.yml/badge.svg)](https://rahulgaur104.github.io/AGNI/)

AGNI is a differentiable finite-n ideal MHD stability solver. It discretizes the
ideal MHD energy principle on a straight-field-line grid and returns the most
unstable eigenvalue and its gradient. It computes the stability objective; it
does not run an optimization itself.

The package is under active development, and the API, the file format and the
numerics change without notice. For a stable version use the AGNI
implementation inside DESC,
[PR #1893](https://github.com/PlasmaControl/DESC/pull/1893), branch
`rg/AGNI_var`.

## Quick start

Stability of a DESC equilibrium (needs DESC installed):

```bash
agnimhd solve my_equilibrium.h5 --res 24,12,8
```

From Python:

```python
import agnimhd

eq, diffmat = agnimhd.from_desc("my_equilibrium.h5", 24, 12, 8)
lam = agnimhd.growth_rate(eq, diffmat)           # lam < 0: unstable
```

`growth_rate` is solve mode and is not differentiable. The gradient comes from
optimize mode, `jax.grad(agnimhd.growth_rate_of)(params, equilibrium_map,
diffmat)`, with `equilibrium_map` the equilibrium code's map from its parameters
to an `EquilibriumData`.

Two shipped cases run without DESC: `python examples/cross_sections.py`. Other
codes build an `EquilibriumData` from arrays, or load a saved one with
`EquilibriumData.load("eq.npz")`. See the
[documentation](https://rahulgaur104.github.io/AGNI/).

## Install

```bash
pip install agnimhd               # jax, numpy, scipy, matfree
pip install "agnimhd[hdf5,test]"  # HDF5 files and the test suite
```

Python 3.12. DESC is optional and only needed for `from_desc`.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests -q          # about 25 min on one CPU
```

## Reference

R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
stability solver & optimizer for magnetic confinement fusion devices* (2026).
Originally developed inside [DESC](https://github.com/PlasmaControl/DESC)
([PR #1893](https://github.com/PlasmaControl/DESC/pull/1893), on the
differentiation matrices of PR #1789). MIT license.
