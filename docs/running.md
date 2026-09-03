# Getting the data and running a case

The solver takes one object: an `EquilibriumData` holding geometry, current and
profiles on a PEST grid. Producing it is the only work between an equilibrium
and a growth rate. This page gives the procedure for a stellarator and for a
tokamak, using DESC as the equilibrium code.

Only DESC is used here because it computes every required quantity on a PEST
grid directly. Any code that can do the same works. See
[docs/adapters.md](adapters.md).

---

## Two cases that run now

Both input files are in the repository, so neither needs DESC:

```bash
python examples/cross_sections.py
```

This solves each case with the dense Lanczos-LU eigensolver, prints the
eigenvalue against the dense reference stored in the case's sidecar, and writes
the eigenfunction cross sections to `examples/figures`.

| case | basis | grid | mode |
|---|---|---|---|
| modified LBD QH | Legendre-Lobatto, `x_0 = 0.6` | 24x12x8, one field period | the toroidal grid carries `n` |
| modified DSHAPE, `iota_max = 0.98` | coupled Zernike-Fourier, `M = 12`, penalty 0.02 | 64x48x1, one plane | `n` chosen per solve |

The tokamak is drawn at `zeta = 0`, since an axisymmetric equilibrium looks the
same at every toroidal angle. The stellarator is drawn at `zeta = 0` and
`zeta = pi / NFP`, the two planes where the shaping differs most.

Both files can be regenerated with `tools/export_desc_example.py`; the exact
command for each is at the top of that script, and each case's clustering,
basis and reference eigenvalue are recorded in `examples/data/<case>.json`.

---

## Using your own equilibrium

What is needed:

* DESC, in the environment where the export runs. `agnimhd` does not need it,
  and the export can be done once on another machine.
* A solved equilibrium, either a DESC example name or an `.h5` file.
* `agnimhd` installed in the environment where the solve runs.

---

### A stellarator, from scratch

Everything runs in one process. Nothing is written to disk.

```python
import numpy as np
from desc.equilibrium import Equilibrium
from desc.grid import Grid

from agnimhd import EquilibriumData, eigenpair
from agnimhd.basis import standard_grid

desc_eq = Equilibrium.load("equilibrium.h5")

n_rho, n_theta, n_zeta = 24, 12, 8
cluster = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)

# Radial Legendre-Lobatto nodes through the clustering map, Fourier in the two
# angles, and the differentiation and quadrature operators on those same nodes.
nodes, diffmat = standard_grid(
    n_rho, n_theta, n_zeta, NFP=desc_eq.NFP, automorphism=cluster
)

# The PEST nodes, flattened rho-major, then mapped to DESC's own poloidal angle.
# theta_PEST is not DESC's theta, so this root find is not optional.
rho, theta, zeta = np.meshgrid(
    nodes["rho"], nodes["theta"], nodes["zeta"], indexing="ij"
)
pest_nodes = np.column_stack([rho.ravel(), theta.ravel(), zeta.ravel()])
grid = Grid(
    desc_eq.map_coordinates(
        pest_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
        tol=1e-12,
    )
)

data = desc_eq.compute(
    [
        "g_rr|PEST", "g_rv|PEST", "g_rp|PEST",      # covariant PEST metric
        "g_vv|PEST", "g_vp|PEST", "g_pp|PEST",
        "g^rr",                                     # grad(rho) . grad(rho)
        "sqrt(g)_PEST",                             # Jacobian and its derivatives
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "J^zeta", "|J|",                            # current
        "iota", "psi_r", "psi_rr", "p", "p_r",      # profiles
        "finite-n instability drive",               # the drive
        "a",                                        # minor radius
    ],
    grid=grid,
)
flat = lambda key: np.asarray(data[key]).reshape(-1)

agni_input = EquilibriumData(
    n_rho=n_rho,
    n_theta=n_theta,
    n_zeta=n_zeta,
    NFP=int(desc_eq.NFP),
    Psi=float(np.asarray(desc_eq.params_dict["Psi"]).reshape(-1)[0]),
    a=float(flat("a")[0]),
    g_rr=flat("g_rr|PEST"),
    g_rv=flat("g_rv|PEST"),
    g_rp=flat("g_rp|PEST"),
    g_vv=flat("g_vv|PEST"),
    g_vp=flat("g_vp|PEST"),
    g_pp=flat("g_pp|PEST"),
    g_sup_rr=flat("g^rr"),
    sqrtg=flat("sqrt(g)_PEST"),
    sqrtg_r=flat("(sqrt(g)_PEST_r)|PEST"),
    sqrtg_v=flat("(sqrt(g)_PEST_v)|PEST"),
    sqrtg_p=flat("(sqrt(g)_PEST_p)|PEST"),
    J_sup_zeta=flat("J^zeta"),
    abs_J=flat("|J|"),
    iota=flat("iota"),
    psi_r=flat("psi_r"),
    psi_rr=flat("psi_rr"),
    p=flat("p"),
    p_r=flat("p_r"),
    finite_n_instability_drive=flat("finite-n instability drive"),
)

lam, xi, residual = eigenpair(agni_input, diffmat)
print(f"lambda = {float(lam):+.6e}   residual = {float(residual):.2e}")
```

That is the whole procedure. The left column of the `EquilibriumData` call is
the specification in [interface.md](interface.md), and the right column is the
DESC key that supplies it. An adapter for another code replaces the right
column only.

`gamma` defaults to `5/3` and `sigma` to `-1e-1`, so neither
`AssemblyConfig` nor `SolverConfig` is needed until one of them has to change.

The toroidal nodes cover one field period, `zeta in [0, 2*pi/NFP)`, which
restricts the calculation to toroidal mode numbers `n = 0 mod NFP`. Modes
outside that family need nodes over the full torus.

`tools/export_fixture.py` is this same sequence, tested, with the file writing
attached.

### Solving the same case again later

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

**The clustering parameters must match the ones used at export.** They place
the radial nodes. If they differ, the differentiation matrices and the geometry
sit on different node sets, and the result is a wrong eigenvalue with no error.

Trust the eigenvalue only when the residual is small. See
[docs/resolution.md](resolution.md#the-shift).

### A tokamak

A tokamak is axisymmetric, so the toroidal direction carries a single mode
number `n` rather than a grid. Build **one toroidal plane** and tell the
assembler which `n` to solve for. The operator becomes complex Hermitian,
because `d/dphi` becomes `i n`.

#### One plane

Follow the stellarator procedure with `n_zeta = 1` on an axisymmetric
equilibrium (`NFP = 1`). Every array then has `n_rho * n_theta` entries.

To take a plane out of an existing multi-plane case, slice each array and
rebuild with `n_zeta=1`, keeping `Psi` and `a` unchanged. `_zeta_plane` in
`tests/conftest.py` is that slice, written out.

#### One mode number at a time

```python
from agnimhd import AssemblyConfig, SolverConfig, eigenpair

for n in (1, 2, 3, 4):
    cfg = AssemblyConfig(axisym=True, n_mode_axisym=n)
    lam, _, residual = eigenpair(agni_input, diffmat, cfg, SolverConfig(sigma=-1e-3))
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

| | start with | reason |
|---|---|---|
| `n_rho` | 24 | the eigenvalue converges fastest in the radial direction. See [resolution.md](resolution.md) |
| `n_theta`, `n_zeta` | 12, 8 | these set which poloidal and toroidal mode numbers the discretization can represent |
| `x_0` | at the resonant surface | node placement changes the eigenvalue more than added resolution does |
| `sigma` | `-1e-3` | must lie below the whole spectrum, and not far below it |
| `gamma` | `5/3` | raise it to approach the incompressible limit. See [theory.md](theory.md#8-incompressibility) |

The reliable procedure for `sigma` is the paper's: compute the spectrum once at
low resolution, then place the shift just below it.

---

## Next

* [`EquilibriumData`](interface.md) for the field-by-field specification.
* [docs/adapters.md](adapters.md) for a code other than DESC, and for the
  gradient.
* [docs/resolution.md](resolution.md) for convergence, timings and the shift.
