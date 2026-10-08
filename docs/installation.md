# Installation

Python 3.12. The package depends on `jax`, `numpy`, `scipy` and `matfree` only.

```bash
pip install agnimhd                      # CPU
pip install "agnimhd[hdf5,plot]"         # + HDF5 files, + figures
```

From a clone:

```bash
git clone https://github.com/rahulgaur104/AGNI.git
cd AGNI
pip install -e ".[hdf5,plot]"
```

Check it:

```bash
agnimhd solve tests/data/qh_lowres_24x12x8.npz --radial lobatto \
    --automorphism '{"eps": 0.01, "x_0": 0.65, "m_1": 2.0, "m_2": 3.0}'
```

prints `gamma^2 = +1.3376268705e-04` for family 0 and `UNSTABLE`
([Home](index.md#from-any-other-code)), in a few minutes on a CPU.

## GPU

Install the CUDA build of jax first, then agnimhd:

```bash
pip install "jax[cuda12]"
pip install agnimhd
```

`eigensolver="dense"` and `"jd"` run on the GPU as they are; `"eigsh"` (the
default) calls SciPy on the host. 64-bit floats are switched on by the package
at import; see [Choosing options](options.md) for which solver fits where.

## DESC (optional)

`from_desc`, `AgniStability` and `agnimhd solve eq.h5` need DESC. DESC pins jax
below 0.10; the tested combination is

```bash
pip install "desc-opt==0.17.3" agnimhd
```

(CI runs `tests/test_adapters.py` with it). Nothing else in agnimhd imports
DESC, and a DESC import anywhere in the package fails CI.

## Several GPUs (optional)

`eigensolver="dense_mg"` needs `jaxmg`. With DESC in the same environment the
tested pair is `jaxmg==0.0.9` and jax 0.6.2; jaxmg 1.0 and later need jax 0.11,
which DESC does not accept yet ([Several GPUs](multigpu.md)).

## Development

```bash
pip install -e ".[dev]"      # test, hdf5, plot, black, isort, flake8, pre-commit
pre-commit install
pytest tests -q              # about 25 min on one CPU
pip install -e ".[docs]" && mkdocs serve
```

What a change needs before it is merged: [Developer guide](contributing.md).
