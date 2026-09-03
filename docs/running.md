# Getting the data and running a case

The solver takes one object: an `EquilibriumData` holding geometry, current and
profiles on a PEST grid. Producing it is the only work between an equilibrium
and a growth rate. This page gives the procedure for a stellarator and for a
tokamak, using DESC as the equilibrium code.

Only DESC is used here because it computes every required quantity on a PEST
grid directly. Any code that can do the same works; see
[docs/adapters.md](adapters.md).

---

## What is needed

* DESC, in the environment where the export runs. `agnimhd` does not need it,
  and the export can be done once on another machine.
* A solved equilibrium, as a DESC `.h5`.
* `agnimhd` installed in the environment where the solve runs.

---

## A stellarator

The equilibrium quantities go straight into an `EquilibriumData` and then into
the solver, in one process:

```python
from desc.equilibrium import Equilibrium
from agnimhd import EquilibriumData, AssemblyConfig, eigenpair

from tools.export_fixture import KEY_MAP, build_pest_level   # the adapter

eq_desc = Equilibrium.load("equilibrium.h5")
n_rho, n_theta, n_zeta = 24, 12, 8

# 1. Nodes and operators, built together so they cannot disagree.
pest_grid, diffmat, _ = build_pest_level(eq_desc, n_rho, n_theta, n_zeta)

# 2. Geometry, current and profiles on those nodes.
data = eq_desc.compute(list(KEY_MAP) + ["a"], grid=..., diffmat=diffmat,
                       gamma=5 / 3, incompressible=False)

# 3. Pack, flattened rho-major.
eq = EquilibriumData(
    n_rho=n_rho, n_theta=n_theta, n_zeta=n_zeta, NFP=int(eq_desc.NFP),
    Psi=..., a=...,
    **{dst: np.asarray(data[src]).reshape(-1) for src, dst in KEY_MAP.items()},
)

# 4. Solve.
lam, v, residual = eigenpair(eq, diffmat, AssemblyConfig(gamma=5 / 3))
print(float(lam), float(residual))
```

`tools/export_fixture.py` is that sequence written out in full and working,
including the step this sketch elides: `theta_PEST` is not DESC's `theta`, so
the PEST nodes must be mapped with `eq.map_coordinates` before `eq.compute`.
Read it as the reference. Its `KEY_MAP` is the whole adapter.

The toroidal nodes cover one field period, `zeta in [0, 2*pi/NFP)`, which
restricts the calculation to toroidal mode numbers `n = 0 mod NFP`. Modes
outside that family need nodes over the full torus.

### Solving it again later

Writing the case to disk is worthwhile when the equilibrium is computed on a
machine that has DESC and solved on one that does not, or when the same case is
solved repeatedly. `tools/export_fixture.py` does the export end to end:

```bash
python tools/export_fixture.py \
    --eq /path/to/equilibrium.h5 \
    --res 24,12,8 \
    --out my_case.npz \
    --meta my_case.json
```

then, with no equilibrium code installed:

```bash
agnimhd validate my_case.npz -v
agnimhd solve my_case.npz \
    --automorphism '{"eps": 0.01, "x_0": 0.65, "m_1": 2.0, "m_2": 3.0}'
```

which prints the eigenvalue, the Rayleigh residual, and a verdict.
**`lambda < 0` means unstable.** On the shipped case this is

```
lambda   -1.3376268705e-04
residual 5.558e-06
verdict  UNSTABLE
```

**The clustering parameters must match the export.** They place the radial
nodes. If they differ, the differentiation matrices and the geometry sit on
different node sets, and the result is a wrong eigenvalue with no error. From
Python the same rule applies to `standard_grid`:

```python
from agnimhd.basis import standard_grid
_, diffmat = standard_grid(
    *eq.resolution, NFP=eq.NFP,
    automorphism=dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0),
)
```

Trust the eigenvalue only when the residual is small; see
[docs/resolution.md](resolution.md#the-shift).

## A tokamak

A tokamak is axisymmetric, so the toroidal direction carries a single mode
number `n` rather than a grid. Build **one toroidal plane** and tell the
assembler which `n` to solve for. The operator becomes complex Hermitian,
because `d/dphi` becomes `i n`.

### 1. Build one plane

Follow the stellarator procedure with `n_zeta = 1` on an axisymmetric
equilibrium (`NFP = 1`). Every array then has `n_rho * n_theta` entries.

To take a plane out of an existing multi-plane case, slice each array and
rebuild with `n_zeta=1`, keeping `Psi` and `a` unchanged. `_zeta_plane` in
`tests/conftest.py` is that slice, written out.

### 2. Solve one mode number

```python
from agnimhd import AssemblyConfig, SolverConfig, eigenpair

for n in (1, 2, 3, 4):
    cfg = AssemblyConfig(gamma=5 / 3, axisym=True, n_mode_axisym=n)
    lam, _, residual = eigenpair(eq, diffmat, cfg, SolverConfig(sigma=-1e-3))
    print(f"n = {n}: lambda = {float(lam):+.6e}  residual = {float(residual):.2e}")
```

Each `n` is a separate eigenvalue problem, so scan the range of interest and
take the most negative `lambda`. The `--automorphism` rule from the stellarator
case applies unchanged.

Two notes specific to this path:

* `A` is complex Hermitian here, not real symmetric. Both `eigsh` and
  `jax_lanczos` handle it, and `matfree >= 0.6.2` is required: earlier versions
  orthonormalize with `Q.T @ Q` and return a wrong eigenvector with a
  plausible eigenvalue. See
  [docs/migration.md](migration.md#things-that-changed-because-they-were-wrong).
* `lambda` is real, and the returned dtype is real. A complex `lambda` means the
  Rayleigh quotient was formed without conjugation.

What is verified in this repository is the single-plane complex Hermitian path,
taken from the shipped stellarator case. A real tokamak equilibrium has not been
run through it here.

---

## Choosing the numbers

| | start with | why |
|---|---|---|
| `n_rho` | 24 | radial resolution buys the most; see [resolution.md](resolution.md) |
| `n_theta`, `n_zeta` | 12, 8 | they set which poloidal and toroidal modes exist |
| `x_0` | near the resonant surface | node clustering matters more than raw resolution |
| `sigma` | `-1e-3` | must lie below the whole spectrum, and not far below |
| `gamma` | `5/3` | raise it toward the incompressible limit; see [theory.md](theory.md#8-incompressibility) |

The reliable procedure for `sigma` is the paper's: compute the spectrum once at
low resolution, then place the shift just below it.

---

## Next

* [`EquilibriumData`](interface.md) for the field-by-field specification.
* [docs/adapters.md](adapters.md) for a code other than DESC, and for the
  gradient.
* [docs/resolution.md](resolution.md) for convergence, timings and the shift.
