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
([Interface](interface.md#saving-and-loading)), in a few minutes on a CPU.

## GPU

Install the CUDA build of jax first, then agnimhd:

```bash
pip install "jax[cuda12]"
pip install agnimhd
```

`eigensolver="dense"`, the default, and `"jd"` run on the GPU as they are,
while `"eigsh"` calls SciPy on the host. 64-bit floats are switched on by the
package at import. See [Choosing options](options.md) for which solver fits
where.

## TPU

Install the TPU build of jax first, then agnimhd:

```bash
pip install "jax[tpu]"
pip install agnimhd
```

`eigensolver="dense"` runs on a TPU and matched the CPU value of the 8×8×3 QH
case to 8e-10 on a TPU v5e. `"jd"` has not been tested on a TPU, and `"eigsh"`
and `"dense_mg"` do not run on one.

## DESC (optional)

`from_desc`, `AgniStability` and `agnimhd solve eq.h5` need DESC. DESC pins jax
below 0.10, and the tested combination is

```bash
pip install "desc-opt==0.17.3" agnimhd
```

(CI runs `tests/test_adapters.py` with it). Nothing else in agnimhd imports
DESC, and a DESC import anywhere in the package fails CI.

## Several GPUs (optional)

`eigensolver="dense_mg"` needs `jaxmg`. `jaxmg==0.0.9` with jax 0.6.2 runs in
one process on one node and fits within DESC's jax bound. `jaxmg==1.2.0` runs
one process per GPU on one or several nodes and was tested with jax 0.11.0,
above that bound. jaxmg 1.3 needs jax 0.11.1 ([Several GPUs](multigpu.md)).

## Development

```bash
pip install -e ".[dev]"      # test, hdf5, plot, black, isort, flake8, pre-commit
pre-commit install
pytest tests -q              # about 25 min on one CPU
pip install -e ".[docs]" && mkdocs serve
```

What a change needs before it is merged: [Developer guide](contributing.md).
