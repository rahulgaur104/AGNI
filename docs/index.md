# agnimhd

AGNI computes the most unstable finite-n ideal MHD mode of a 3D equilibrium. It
discretizes the energy principle pseudospectrally on a PEST grid, solves
`A xi = lambda B xi` for the lowest eigenvalue, and returns the squared growth
rate `gamma^2 = -lambda` (positive: unstable) and its derivative with respect
to the equilibrium's parameters. It supplies the stability objective
and its gradient; the optimization itself is done by the equilibrium code.

The package is under active development, and the API, the file format and the
numerics change without notice. For a stable version use the AGNI
implementation inside DESC,
[PR #1893](https://github.com/PlasmaControl/DESC/pull/1893), branch
`rg/AGNI_var`.

## From a DESC file

```bash
agnimhd solve my_equilibrium.h5 --res 24,12,8
```

prints `gamma^2` and the eigenpair residual of each toroidal mode family, the
most unstable family, and `UNSTABLE` or `stable`. The grid spans one field
period (`n_zeta = 8` nodes); family `x` holds the toroidal modes
`n = x + k NFP`, and the families together hold every `n`
([Toroidal mode families](options.md#toroidal-mode-families)). The same in
Python:

```python
import agnimhd

basis = agnimhd.Basis(24, 12, 8)                    # see Choosing options
eq, diffmat = agnimhd.from_desc("my_equilibrium.h5", basis)       # family 0
gamma2 = agnimhd.growth_rate(eq, diffmat)               # > 0: unstable
gamma2, v, residual = agnimhd.eigenpair(eq, diffmat)   # and the mode itself
for x in basis.families(eq.NFP):                       # every family
    print(x, agnimhd.growth_rate(eq, basis.nodes_and_diffmat(eq.NFP, family=x)[1]))
```

`from_desc` also accepts a DESC `Equilibrium` object. It needs DESC installed;
nothing else in the package does.

## Two modes

An `EquilibriumData` holds the metric, Jacobian, current and profiles sampled on
a grid. These arrays are in force balance only because an equilibrium solve made
them so, and the package can neither check nor restore that.

Solve mode, `growth_rate(eq, diffmat)` and `eigenpair(eq, diffmat)`, needs only
this package: one equilibrium in, one stability answer out. It is not
differentiable. `jax.grad(growth_rate)` raises a `TypeError`, because a step
along `d lambda / d(EquilibriumData)` gives arrays that violate force balance,
and returning zero instead would look like a converged optimization.

Optimize mode takes the equilibrium's parameters and a map from them to an
`EquilibriumData`:

```python
def equilibrium_map(params):         # geometry and profiles, no equilibrium solve
    return to_equilibrium_data(evaluate_on_pest_grid(params))

g = jax.grad(agnimhd.growth_rate_of)(params, equilibrium_map, diffmat)
```

`g` has the structure of `params`. It is a partial derivative at a fixed force
balance residual; enforcing force balance is the optimizer's task. In DESC,
`ProximalProjection` re-solves the equilibrium after each step and forms the
reduced derivative

```
d gamma^2 / dc = @gamma^2/@c - (@gamma^2/@x) (@F/@x)^-1 (@F/@c)
```

with `F` the force balance residual, `x = (R_lmn, Z_lmn, L_lmn)` and `c` the
free parameters (boundary and profile coefficients, `Psi`). agnimhd supplies the
`@gamma^2` factors.

For DESC this is packaged as an objective:

```python
from agnimhd.adapters.desc_objective import AgniStability

basis = agnimhd.Basis(24, 12, 8)
objective = ObjectiveFunction((AgniStability(eq, basis=basis, family=0),))
eq.optimize(objective, constraints, optimizer="proximal-lsq-exact")
```

`constraints` holds `ForceBalance` and the fixed boundary modes and profiles;
the step is taken in the free boundary coefficients. `examples/desc_objective.py`
runs it.

## Shipped examples

```bash
python examples/cross_sections.py
```

solves two cases stored in `examples/data`, prints each eigenvalue against the
dense reference in the case's `.json` sidecar, and writes eigenfunction cross
sections to `examples/figures` (needs matplotlib, not DESC).

| case | basis | grid |
|---|---|---|
| modified LBD QH | Legendre-Lobatto, `x_0 = 0.6` | 24x12x8, one field period |
| modified DSHAPE, `iota_max = 0.98` | coupled Zernike-Fourier, `M = 12`, penalty 0.02 | 64x48x1, one plane |

`tools/export_desc_example.py` regenerates both; the command for each is
recorded in `examples/data/<case>.json`.

## From any other code

Fill an [`EquilibriumData`](interface.md) with the metric, Jacobian, current and
profiles on the nodes of an `agnimhd.Basis`, take the matching `DiffMat` from
`basis.nodes_and_diffmat(NFP, family=x)`, and call `growth_rate`. Check the arrays with
`agnimhd validate eq.npz -v`.

To solve on a machine without the equilibrium code, export once and solve the
file:

```bash
python tools/export_fixture.py --eq equilibrium.h5 --res 24,12,8 \
    --out case.npz --meta case.json                       # needs DESC
agnimhd solve case.npz --radial lobatto \
    --automorphism '{"eps": 0.01, "x_0": 0.65, "m_1": 2.0, "m_2": 3.0}'
```

The radial basis and clustering parameters must be the ones used at export
(`export_fixture.py` uses these): they place the radial nodes, and a mismatch
gives a wrong eigenvalue with no error. On the shipped 24x12x8 case (`NFP = 4`)
this prints, in 3 min 20 s on a CPU:

```
family 0  gamma^2 +1.3376268705e-04  residual 5.580e-06
family 1  gamma^2 +1.3526553716e-04  residual 4.943e-02
family 2  gamma^2 +1.6437827281e-04  residual 4.072e-06
most unstable: family 2
verdict  UNSTABLE
```

Family 1 is complex. Its `gamma^2` equals the lowest eigenvalue of the dense
matrix to 3.6e-9; ARPACK's complex driver returns a less converged vector at the
default `eigsh_tol=1e-8` (residual 6.8e-5 at `1e-11`).

## Tokamaks

An axisymmetric equilibrium uses one toroidal plane (`n_zeta = 1`, `NFP = 1`)
and one toroidal mode number per solve:

```python
for n in (1, 2, 3, 4):
    cfg = agnimhd.AssemblyConfig(axisym=True, n_mode_axisym=n)
    gamma2, _, residual = agnimhd.eigenpair(eq, diffmat, cfg, agnimhd.SolverConfig(sigma=1e-3))
```

Take the largest `gamma^2` over the scan. `d/dphi` becomes `i n`, so the
operator is complex Hermitian; both `"eigsh"` and `"jax_lanczos"` solve it, and
`gamma^2` is real. This path is tested on one plane of the shipped stellarator
(`_zeta_plane` in `tests/conftest.py`), not on a real tokamak equilibrium.

## Sign convention

Everything agnimhd returns is the squared growth rate `gamma^2 = -lambda`, with
`lambda` the lowest eigenvalue of the energy quotient `<xi|A|xi> / <xi|B|xi>`:
**positive means unstable**. This is the sign of the AGNI paper's `lambda`
(Eq. 19, `dW_p = -lambda dK`). An optimizer seeking stability lowers `gamma^2`
toward zero. The shift `SolverConfig.sigma` is given in the same convention: a
value above the largest `gamma^2`, for example `1.05` times an estimate of it.

## Pages

- [Choosing options](options.md): grid, radial basis, MPOL, NTOR, eigensolver,
  shift, gamma.
- [Interface](interface.md): the fields of `EquilibriumData` and how to produce
  them from DESC, VMEC or GVEC.
- [Dense solves on several GPUs](multigpu.md): `eigensolver="dense_mg"`, with
  measured sizes and times.
- [API](api.md): functions and configuration.
- [Theory](theory.md): what is discretized and how.
- [Migrating from DESC](migration.md).
