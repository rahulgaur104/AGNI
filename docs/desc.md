# DESC coupling

agnimhd takes a DESC equilibrium in two ways:

| | `from_desc` | `AgniStability` |
|---|---|---|
| gives | the arrays of one equilibrium | a DESC objective |
| for | `growth_rate`, `eigenpair`, `agnimhd solve` | `ObjectiveFunction`, `eq.optimize` |
| evaluated | once, through NumPy | at every call, inside DESC's `jax.jit` |
| derivative | none | `d gamma^2 / d(DESC parameters)` |

Both evaluate the same DESC quantities on the same nodes. On the shipped QH case
(an iota profile, `density=True`) their `gamma^2` agree to 2.1e-11.

## Nodes

The `Basis` fixes the nodes in PEST coordinates `(rho, theta_PEST, zeta)`:
`n_rho` radial nodes (Gauss-Radau-Jacobi by default), `n_theta` equally spaced
`theta_PEST` in `[0, 2 pi)`, `n_zeta` equally spaced `zeta` in one field period
`[0, 2 pi / NFP)`. The nodes never move in these coordinates, and the derivative
matrices (`DiffMat`) belong to them; both are built once.

DESC's poloidal angle is `theta`, with `theta_PEST = theta + lambda(rho, theta,
zeta)`. For every node DESC solves this equation for `theta`
(`eq.map_coordinates`, tolerance 1e-12, at most 50 iterations). Where the nodes
sit in DESC's `theta` therefore depends on `lambda`'s coefficients `L_lmn`.

## Quantities

At the nodes DESC computes these keys; each becomes a field of
[`EquilibriumData`](interface.md):

| DESC key | field |
|---|---|
| `g_rr\|PEST` ... `g_pp\|PEST` | `g_rr` ... `g_pp` (covariant PEST metric) |
| `g^rr` | `g_sup_rr` |
| `sqrt(g)_PEST`, `(sqrt(g)_PEST_r)\|PEST`, `_v`, `_p` | `sqrtg`, `sqrtg_r`, `sqrtg_v`, `sqrtg_p` |
| `J^zeta`, `\|J\|` | `J_sup_zeta`, `abs_J` |
| `iota`, `psi_r`, `psi_rr`, `p`, `p_r` | same |
| `J x grad(rho)`, `(B*grad) grad(rho)` | `J_cross_grad_rho`, `B_dot_grad_grad_rho` |
| `ni` (`density=True` only) | `density = ni / max(ni)` |

and two scalars: `Psi` (DESC's toroidal flux) and `a`, DESC's minor radius
(the cross-section area definition; `gamma^2` is sensitive to it).

Quantities that need integrals are computed on DESC's own grids and copied onto
the nodes: `a` on `QuadratureGrid(L_grid, M_grid, N_grid)` (a `LinearGrid`
value differs by 3.76 % on the test case), the flux functions (`iota` and the
parts DESC builds it from, `psi_r`, `psi_rr`, `p`, `p_r`) on a `LinearGrid`
at the nodes' `rho` values with `M_grid`, `N_grid`. `from_desc` gets these grids
from `eq.compute` (its default `override_grid=True`); `AgniStability` builds the
same two grids in `build()` and computes on them itself.

`density=True` needs a density profile (`eq.electron_density`). Without one,
`from_desc` gives every node weight one and `AgniStability` raises.

## The objective

```python
from agnimhd.adapters.desc_objective import AgniStability

stability = AgniStability(
    eq, basis, family=0, assembly=None, solver=None, coarse=None, density=False,
    target=0.0, weight=1.0,
)
```

`build()` runs once: the nodes and `DiffMat` of `basis` for toroidal family
`family`, and DESC's transforms for the flux-function grid and the `a` grid.
With `solver.eigensolver="jd"` also the coarse level: the nodes and `DiffMat` of
`coarse = basis.coarse(n_theta, n_zeta)` (same radial nodes, `mpol` and `ntor`,
fewer angles; any other basis is refused) and the angular interpolation between
the two levels.

`compute(params)` runs at every value and every Jacobian DESC asks for:

1. map the nodes to DESC's `theta` at `params`;
2. compute the quantities above at `params` and pack an `EquilibriumData`;
3. with `"jd"`, steps 1 and 2 on the coarse nodes;
4. solve for the most unstable eigenpair (the lowest eigenvalue of `A`) and
   return `gamma^2 = -v^H A v / v^H v` at the eigenvector `v`.

DESC compiles `compute` and its Jacobian with `jax.jit`, and the eigensolve runs
inside: `"jd"` and the dense JAX solver on the device, `"eigsh"` through
`jax.pure_callback` to ARPACK on the host. An objective returns the `gamma^2`
of one family; several families need one objective each.

## The derivative

DESC's Jacobian of `AgniStability` is the chain rule through the fields `q` of
the `EquilibriumData`:

```
d gamma^2 / d params = sum over q of  (d gamma^2 / d q) (d q / d params)
```

- `d gamma^2 / d q`, by Hellmann-Feynman: at an eigenvector, the eigenvalue's
  derivative is the derivative of the quotient with `v` held fixed,
  `-v^H (dA/dq) v / v^H v`. agnimhd gives the eigensolve a zero derivative
  (`jax.custom_vjp`) and lets JAX differentiate the quotient, that is the
  assembly of `A`, density weighting included. This is exact only at a
  converged eigenvector: stop `"jd"` on the residual (`jd_tol`), not on the
  Ritz value (measured in DESC, that stop left residuals of 0.5 to 1.7 and
  gradients up to 3.3e-2 off).
- `d q / d params`: JAX differentiates DESC's compute functions at the nodes,
  and the node positions as well. DESC solves for `theta` with
  `jax.lax.custom_root`, which differentiates the solution implicitly, so a
  change of `L_lmn` that moves the nodes in DESC's `theta` is in the derivative.
- Not differentiated: the eigenvector (above), the JD coarse level (it only
  supplies the start vector and the deflation space; `jax.lax.stop_gradient`),
  the nodes in PEST coordinates and the `DiffMat`, and `sigma`.

For this one-number objective DESC uses reverse mode: one backward pass gives
the whole row of the Jacobian. Checked: against central differences in the
three coefficients `gamma^2` moves most with (`h = 1e-6`, 1e-4 relative); JD
against `"eigsh"` with density, value 1.2e-11 and Jacobian 7.2e-8 of its
largest entry apart (`tests/test_adapters.py`).

This is a partial derivative at a fixed force-balance residual: `params` are
DESC's state `x = (R_lmn, Z_lmn, L_lmn)` and the free parameters `c` (boundary
and profile coefficients, `Psi`), varied independently. Force balance
`F(x, c) = 0` is kept by the optimizer. With `ProximalProjection` (optimizers
`"proximal-*"`) DESC re-solves the equilibrium after every step and forms the
derivative along the constraint,

```
d gamma^2 / dc = @gamma^2/@c - (@gamma^2/@x) (@F/@x)^-1 (@F/@c)
```

from the two `@gamma^2` factors above and the force-balance Jacobian.

## An optimization

The shipped low-resolution QH case (`tests/data/AGNI_QH_lowres.h5`, `NFP = 4`),
eight free boundary modes (`max(|m|, |n|) <= 1`):

```python
import agnimhd
import numpy as np
from desc.equilibrium import Equilibrium
from desc.objectives import (
    FixAtomicNumber, FixBoundaryR, FixBoundaryZ, FixElectronDensity,
    FixElectronTemperature, FixIonTemperature, FixIota, FixPsi, ForceBalance,
    ObjectiveFunction,
)
from desc.optimize import Optimizer

from agnimhd.adapters.desc_objective import AgniStability

eq = Equilibrium.load("tests/data/AGNI_QH_lowres.h5")
basis = agnimhd.Basis(24, 12, 8, radial="lobatto",
                      automorphism=dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0),
                      mpol=5, ntor=3)
solver = agnimhd.SolverConfig(eigensolver="jd", sigma=1e-3, jd_tol=1e-4,
                              jd_theta_tol=0.0, jd_outer=1000)
objective = ObjectiveFunction(
    (AgniStability(eq, basis, assembly=agnimhd.AssemblyConfig(gamma=5 / 3),
                   solver=solver, coarse=basis.coarse(), weight=100.0),
     ForceBalance(eq, weight=500.0)),
    deriv_mode="blocked",
)
R, Z = eq.surface.R_basis.modes, eq.surface.Z_basis.modes
constraints = (
    ForceBalance(eq),
    FixBoundaryR(eq, modes=np.vstack(([0, 0, 0], R[np.abs(R).max(1) > 1]))),
    FixBoundaryZ(eq, modes=Z[np.abs(Z).max(1) > 1]),
    FixPsi(eq), FixIota(eq), FixElectronDensity(eq), FixElectronTemperature(eq),
    FixIonTemperature(eq), FixAtomicNumber(eq),
)
(eq,), result = Optimizer("proximal-lsq-exact").optimize(
    eq, objective, constraints, ftol=1e-6, xtol=1e-6, gtol=1e-6, maxiter=3,
    options={"solve_options": {"maxiter": 10, "verbose": 0}},
)
```

Measured on one A100, `"jd"` against the dense JAX solver (`sigma=1e-3`) with the same
basis:

| accepted step | cost, dense | cost, JD | `gamma^2`, dense | `gamma^2`, JD |
|---|---|---|---|---|
| 0 | 9.752e-5 | 9.752e-5 | 1.33763e-4 | 1.33763e-4 |
| 1 | 2.291e-5 | 2.291e-5 | 5.534594e-5 | 5.534594e-5 |
| 2 | 1.677e-5 | 1.677e-5 | 4.385812e-5 | 4.385941e-5 |
| 3 | 8.288e-6 | 8.288e-6 | 1.525200e-5 | 1.525657e-5 |

Wall time 344 s dense, 555 s JD; at this size the dense matrix is small.

## Choices that matter

- `sigma` stays fixed for the whole optimization. It must lie above the largest
  `gamma^2` of every point the optimizer tries, accepted or not, and for
  `"jd"` and the dense JAX solver not far above it ([Shift](options.md#shift-sigma)).
- The basis is fixed for the whole optimization; the equilibrium's own
  resolution (`L`, `M`, `N` and their grids) is DESC's.
- `"dense_mg"` (several GPUs) is not yet usable inside the objective.

## One equilibrium: `from_desc`

```python
eq_data, diffmat = agnimhd.from_desc("eq.h5", basis, family=0, density=False)
eq_data, diffmat, coarse = agnimhd.from_desc("eq.h5", basis, coarse=basis.coarse())
jd = agnimhd.SolverConfig(eigensolver="jd", sigma=1e-3)
gamma2 = agnimhd.growth_rate(eq_data, diffmat, solver=jd, coarse=coarse)
```

The first argument is a DESC `Equilibrium` or the path of a DESC `.h5` file
(the last equilibrium of a family). With `coarse` it also evaluates the equilibrium on the
coarse nodes and returns the JD coarse level. It converts through NumPy, so its
output has no derivative; `jax.grad(growth_rate)` raises
([Two modes](index.md#two-modes)). `agnimhd solve eq.h5` calls it.
