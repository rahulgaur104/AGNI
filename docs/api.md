# API reference

Everything a consumer needs is importable from the top level:

```python
from agnimhd import (
    EquilibriumData, DiffMat, AssemblyConfig, SolverConfig,
    growth_rate, eigenpair,                      # solve mode
    growth_rate_of, growth_rate_and_grad,        # optimize mode
)
```

Full docstrings live in the source and are the authority; this page is the map.

---

## Solve mode

One stored equilibrium in, one stability answer out. Neither function is
differentiable; `jax.grad` raises. See [Two modes](index.md#two-modes).

### `growth_rate(eq, diffmat, assembly=None, solver=None)`

A scalar `jax.Array`: the Rayleigh quotient of the energy operator at the
computed eigenvector. **Negative means unstable**; an optimizer must raise it
toward zero.

`jax.jit` may be applied from outside the package, with the two configs
static:

```python
f = jax.jit(growth_rate, static_argnums=(2, 3))
lam = f(eq, diffmat, AssemblyConfig(), SolverConfig())
```

### `eigenpair(eq, diffmat, assembly=None, solver=None)`

`(lambda, v, residual)`, where `residual` is
`||A v - lambda v|| / (|lambda| ||v||)`. This is the quantity to check; the
inner CG's relative residual is not, being anti-correlated with accuracy on
this operator.

`lambda` is the Rayleigh quotient, identical to `growth_rate`'s, not the
eigensolver's own reported eigenvalue.

---

## Optimize mode

### `growth_rate_of(params, equilibrium_map, diffmat, assembly=None, solver=None)`

The same `lambda`, as a function of the design parameters.

* `params` — pytree. Boundary Fourier coefficients, profile coefficients, coil
  currents: whatever the equilibrium solver takes.
* `equilibrium_map` — callable `params -> EquilibriumData`, differentiable in
  JAX: the equilibrium solve together with the adapter. Static under `jit`.
* `diffmat` — fixed across the optimization. Resolution is not a parameter.

`jax.grad` returns a pytree shaped like `params`. AGNI supplies the analytic
`dlambda/d(EquilibriumData)` ([Theory § 7](theory.md#7-the-gradient)) and
`equilibrium_map` supplies the remaining factor; see
[Consuming the gradient](adapters.md#consuming-the-gradient). Under `jit` the
map joins the configs as static:
`jax.jit(jax.grad(growth_rate_of), static_argnums=(1, 3, 4))`.

Whether `equilibrium_map` solves force balance is not verified here and is the
caller's responsibility. Two calls are refused, both of which are solve mode in
this signature: an `EquilibriumData` passed as `params`, and a non-callable
`equilibrium_map`.

### `growth_rate_and_grad(params, equilibrium_map, diffmat, assembly=None, solver=None)`

`(lambda, gradient)` from a single eigensolve. `gradient` has the structure of
`params`.

---

## The equilibrium

### `EquilibriumData(...)`

The complete interface between AGNI and any equilibrium code. Field-by-field
contract: [docs/interface.md](interface.md).

| | |
|---|---|
| `.resolution` | `(n_rho, n_theta, n_zeta)` |
| `.n_nodes` | their product |
| `.reshape(arr)` | flat array to `(n_rho, n_theta, n_zeta)` |
| `.replace(**changes)` | a copy with fields replaced |
| `.instability_drive()` | the drive, supplied or derived |
| `.validate()` | structural and finiteness checks; raises with a message that names the field |
| `.save(path)` / `.load(path)` | `.npz` plus a JSON provenance sidecar |
| `.save_hdf5(path)` / `.load_hdf5(path)` | optional, needs `h5py` |

A registered JAX pytree: arrays and both scalars are dynamic leaves; the
resolution and `NFP` are static.

`agnimhd.FORMAT_VERSION` is the on-disk format version; `load` refuses a newer
one.

---

## Grid operators

### `agnimhd.basis.standard_grid(n_rho, n_theta, n_zeta, NFP=1, automorphism=None)`

Returns `(nodes, diffmat)` — AGNI's default discretization: Legendre-Lobatto
radially through the clustering map, Fourier in both angles with the toroidal
pair scaled for one field period.

Use this rather than assembling a `DiffMat` by hand. The nodes and the matrices
must come from the same construction, and nothing downstream can check that they
did.

### `DiffMat(D_rho=..., W_rho=..., ...)`

Holds one `(D, W)` pair per coordinate. This is the seam for anyone who wants
their own basis: it takes plain arrays.

### Individual bases — `agnimhd.basis`

`legendre_diffmat`, `fourier_diffmat`, `fourier_diffmat_truncated`,
`jacobi_diffmat`, `bspline_diffmat`, `finite_difference_diffmat`,
`zernike_fourier_diffmat`, plus the Zernike primitives (`zernike_modes`,
`zernike_radial`, `zernike_eval_matrix`,
`zernike_penalty_projector_from_diffmat`).

Each returns a `(D, W)` pair on the *same* nodes. Most satisfy summation by
parts, `D^T W + W D = B`, which is what makes the discrete energy match the
continuous one — and is checked for every basis that should have it.

### Nodes and weights — `agnimhd.quadrature`

`leggauss_lob`, `gauss_radau_jacobi`, `bspline_nodes_weights`,
`zernike_nodes_weights`, and the radial clustering maps
`automorphism_staircase1` / `automorphism_staircase2`.

---

## Configuration

Both are frozen, hashable dataclasses — **static**, because they drive Python
branches and array shapes. Passing a dict is refused rather than accepted, since
it would retrace on every call. Options resolve **keyword first, environment
variable second, default last**; an exported variable never overrides an
explicit argument.

### `AssemblyConfig`

`gamma`, `incompressible`, `axisym`, `n_mode_axisym`, `coupled_rt`,
`n_rho_coupled`, `n_theta_coupled`.

### `SolverConfig`

`eigensolver`, `sigma`, `num_matvecs`, `coarse_num_matvecs`, `cg_tol`,
`cg_maxiter`, `cg_maxiter_cold`, `k_defl`, `rr_refine`, `factor`, `sigma_mode`,
`sigma_factor`, `eigsh_tol`, `seed`.

Read `sigma`'s docstring before changing it: the constraint is two-sided, and
the failure mode at a bad shift is a wrong mode rather than a slow one. See
[docs/resolution.md](resolution.md#the-shift).

---

## Assembly — `agnimhd.assemble`

| | |
|---|---|
| `assemble_dense(eq, diffmat, config)` | the reduced whitened matrix, plus the pieces the other paths reuse |
| `matfree_operator(eq, diffmat, config)` | the same operator as a callable, never materialized |
| `ring_block` / `finish_ring_block` | one poloidal ring's exact sub-block |
| `keep_indices(n_rho, n_theta, n_zeta)` | the degrees of freedom left after the Dirichlet mask, as concrete NumPy |

All three assembly paths are checked against each other: matrix-free reproduces
dense to 4.8e-16 relative per column, and ring blocks reproduce the dense
sub-blocks to <1e-14.

---

## Solvers — `agnimhd.solvers`

The two-level machinery, for resolutions where the dense matrix does not fit.

**Coordinates**: `to_phys`, `from_phys`, `to_phys_h`, `from_phys_h`,
`level_meta`.

**Transfer between levels**: `barycentric_matrix`, `fourier_interp_matrix`,
`transfer_matrices`, `make_transfer`, `adjoint_defect`. `PT` is the exact
transpose of `P`, not a re-derived restriction — check it with `adjoint_defect`
before trusting a deflated solve.

**Preconditioning**: `ring_nodes`, `ring_index_maps`, `group_index_matrix`,
`build_ring_blocks`, `factor_ring_blocks`, `factor_ring_blocks_traced`,
`make_block_precond`.

**Iterations**: `pcg`, `pcg_deflated`, `deflation_Y`, `coarse_gen_modes`,
`coarse_seed_and_deflation`.

The coarse **radial** resolution has a floor of **16**. Below it the two-level
solve returns the wrong mode with the opposite sign, and the floor is free.

---

## Plotting — `agnimhd.plotting`

`mode_components`, `mode_displacement`, `mode_speed` return arrays and need
nothing beyond the four dependencies. `plot_mode_cross_section`,
`plot_radial_profile`, `plot_spectrum` import matplotlib lazily and raise a named
`ImportError` if it is absent (`pip install "agnimhd[plot]"`).

---

## Command line

```
agnimhd info                     # print the interface contract
agnimhd validate FILE [-v]       # check an adapter's output; nonzero exit if invalid
agnimhd solve FILE [--gamma ...] [--sigma ...] [--eigensolver ...]
               [--automorphism '{"eps": 0.01, "x_0": 0.65, "m_1": 2, "m_2": 3}']
```

`--automorphism` **must** match what the export used, or the differentiation
matrices are built on different nodes than the geometry.
