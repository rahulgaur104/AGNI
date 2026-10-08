# API

```python
from agnimhd import (
    EquilibriumData, Basis, DiffMat, AssemblyConfig, SolverConfig,
    from_desc,
    growth_rate, eigenpair,                 # solve mode
    growth_rate_of, growth_rate_and_grad,   # optimize mode
)
```

Docstrings in the source are the full reference.

## Solve mode

`growth_rate(eq, diffmat, assembly=None, solver=None, v_fixed=None, v_guess=None, coarse=None)`
returns the squared growth rate `gamma^2 = -lambda` (positive: unstable) for
one stored equilibrium. `jax.grad` raises (see
[Two modes](index.md#two-modes)). Under `jax.jit`, make the two configs static:
`jax.jit(growth_rate, static_argnums=(2, 3))`.

- `v_guess`: start vector, for example the previous step's eigenvector.
- `v_fixed`: skip the eigensolve and use this vector. Valid only at the same
  equilibrium it came from.
- `coarse`: the coarse level `eigensolver="jd"` requires,
  `basis.coarse_level(eq_coarse, family)` (or `from_desc(..., coarse=...)`), or
  the `(v0, Z)` that `agnimhd.objective.coarse_space(eq, diffmat, coarse,
  assembly, solver)` returned for it.

`eigenpair(eq, diffmat, assembly=None, solver=None, v_guess=None, coarse=None)`
returns `(gamma^2, v, residual)` with `residual = ||A v + gamma^2 v|| / |gamma^2|`.
Check the residual; it is the quality measure.

## Optimize mode

`growth_rate_of(params, equilibrium_map, diffmat, assembly=None, solver=None, v_fixed=None, v_guess=None, coarse=None)`
is the same `gamma^2` as a function of `params`, with `equilibrium_map` a JAX
function `params -> EquilibriumData` that evaluates geometry and profiles and
contains no equilibrium solve. `jax.grad` returns a pytree shaped like `params`
(Hellmann-Feynman: the eigensolve has a zero backward rule, the Rayleigh
quotient carries the derivative). An `EquilibriumData` as `params`, or a
non-callable `equilibrium_map`, is refused. Under `jax.jit` the map is static
too: `jax.jit(jax.grad(growth_rate_of), static_argnums=(1, 3, 4))`.

`growth_rate_and_grad(params, equilibrium_map, diffmat, ...)`: value and
gradient from one eigensolve.

`from_desc(eq_or_path, basis, family=0, density=False, coarse=None)` returns
`(EquilibriumData, DiffMat)` on the nodes of `basis`, the `DiffMat` of toroidal
mode family `family`, then the JD coarse level on the nodes of
`coarse=basis.coarse(...)`. With `density=True` the normalized `ni` is stored as
`EquilibriumData.density`, the mass weighting every solver uses. Needs DESC.
`AgniStability(eq, basis, family=0, assembly=None, solver=None, coarse=None, ...)`
(`agnimhd.adapters.desc_objective`) is the DESC objective for one family. With
`solver.eigensolver="jd"` it also evaluates the equilibrium on the coarse level
`coarse=basis.coarse(n_theta, n_zeta)` (default `basis.coarse()`) at every call.

## Configuration

Frozen dataclasses, passed as static arguments.

`AssemblyConfig`: `gamma=5/3`, `incompressible=False`, `axisym=False`,
`n_mode_axisym=1`, `coupled_rt=False`, `n_rho_coupled`, `n_theta_coupled`.

`SolverConfig`:

| field | default | used by |
|---|---|---|
| `eigensolver` | `"eigsh"` | `"eigsh"`, `"dense"`, `"jd"`, `"dense_mg"` |
| `sigma` | `0.1` | all; above the largest `gamma^2` (the solvers shift `A` by `-sigma`) |
| `eigsh_tol` | `1e-8` | eigsh |
| `num_matvecs`, `factor`, `seed` | `50`, `"cholesky"`, `0` | dense |
| `sigma_mode`, `sigma_factor` | `"fixed"`, `2.5` | dense |
| `jd_outer, jd_inner, jd_maxdim, jd_keep` | `200, 100, 60, 10` | jd |
| `jd_tol, jd_theta_tol` | `0.0, 1e-8` | jd stop tests (residual of the returned vector, Ritz change) |
| `ring_batch` | `24` | jd: rings assembled at once, both levels |
| `coarse_num_matvecs`, `k_defl` | `100`, `50` | jd coarse level |
| `mg_tile`, `mg_block`, `mg_iters`, `mg_tol` | `1024`, `16`, `6`, `1e-6` | dense_mg: tile width, block size, iterations, residual stop |

See [Choosing options](options.md) for how to set them.

## Grid operators: `agnimhd.basis`, `agnimhd.quadrature`

- `Basis(n_rho, n_theta, n_zeta, *, radial="gauss_radau_jacobi",
  alpha=-0.35, beta=-0.65, automorphism=AUTOMORPHISM, mpol=None, ntor=None,
  zernike_penalty=DEFAULT_ZERNIKE_PENALTY_ALPHA)`,
  a frozen dataclass on one field period
  ([Choosing the basis](options.md#choosing-the-basis)).
  `nodes_and_diffmat(nfp, family=0)` returns `(nodes, diffmat)` for the
  toroidal modes `n = family + k nfp`, `families(nfp)` the families to solve
  (`0 ... nfp // 2`), `coarse(n_theta=None, n_zeta=None)` the Jacobi-Davidson
  coarse level (fewer angular nodes; needs `mpol` and `ntor`) and
  `coarse_level(eq_coarse, family=0)` the `coarse` argument made from the
  equilibrium on its nodes
  ([Toroidal mode families](options.md#toroidal-mode-families)).
- One-dimensional bases, each returning `(D, W)` on the same nodes:
  `legendre_diffmat`, `jacobi_diffmat`, `fourier_diffmat`,
  `fourier_diffmat_truncated`, `bspline_diffmat`, `finite_difference_diffmat`,
  `zernike_fourier_diffmat`.
- Nodes and maps: `leggauss_lob`, `gauss_radau_jacobi`, `zernike_nodes_weights`,
  `automorphism_staircase1`.
- `DiffMat(D_rho=, W_rho=, D_theta=, W_theta=, D_zeta=, W_zeta=)` holds the
  pairs; `w_rho`, `w_theta`, `w_zeta` are the weights as 1-D vectors.
- Mode caps are checked: `fourier_diffmat_truncated` and
  `zernike_fourier_diffmat` raise if `M > (n - 1) // 2`.

## Lower level

- `agnimhd.assemble`: `assemble_dense` (the reduced whitened matrix),
  `assemble_rows` (any block of its rows, from the matrix-free operator),
  `matfree_operator` (the same operator as a function), `keep_indices`,
  `operator_dtype(config, diffmat)` (complex for `axisym=True` or a complex
  `D_zeta`).
- `agnimhd.solvers`: `jacobi_davidson`, the ring preconditioner
  (`build_ring_blocks`, `factor_ring_blocks_traced`, `make_block_precond`), the
  coarse level (`coarse_seed_and_deflation`, `fourier_interp_matrix`).
- `agnimhd.multigpu`: `shifted_rows`, `solve_shifted`, `dense_mg`, the pieces of
  `eigensolver="dense_mg"` ([Dense solves on several GPUs](multigpu.md)); real
  operators only (families 0 and `NFP / 2`).
- `agnimhd.plotting`: `mode_components`, `mode_displacement`,
  `mode_plot_displacement`, `mode_delta_v`, `mode_speed` return arrays; `plot_*`
  need matplotlib. `plot_mode_cross_section` and
  `plot_eigenfunction_cross_sections` draw `(R, Z)` contours at `zeta = 0`, and
  for 3D cases also at `zeta = pi / NFP`.

## Command line

```
agnimhd info                              # list the EquilibriumData fields
agnimhd validate FILE [BASIS] [-v]       # check a saved or DESC equilibrium
agnimhd solve FILE [BASIS] [--family X] [--gamma G] [--sigma S] [--eigensolver E]
              [--coarse T,Z]             # E: eigsh, dense, jd (DESC file)

BASIS: [--res R,T,Z]
       [--radial gauss_radau_jacobi|lobatto] [--mpol M] [--ntor N]
       [--automorphism '{"eps": 0.01, "x_0": 0.6, "m_1": 2.5, "m_2": 3.0}']
```

`FILE` is an agnimhd `.npz` or `.h5`, or a DESC `.h5` (then `--res` is
required; `Z` nodes per field period). For an agnimhd file, `--radial` and
`--automorphism` must match the nodes the file was exported on;
`--automorphism null` is no map. `solve` reports every toroidal mode family
`x = 0 ... NFP // 2` and the most unstable, or only family `X` with `--family`.
A family whose eigensolve fails (ARPACK finds no converged eigenpair when no
eigenvalue lies below round-off) is reported as such, the others are still
solved, and the exit status is 1.
