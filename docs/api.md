# API

```python
from agnimhd import (
    EquilibriumData, DiffMat, AssemblyConfig, SolverConfig,
    from_desc,
    growth_rate, eigenpair,                 # solve mode
    growth_rate_of, growth_rate_and_grad,   # optimize mode
)
```

Docstrings in the source are the full reference.

## Solve mode

`growth_rate(eq, diffmat, assembly=None, solver=None, v_fixed=None, v_guess=None, coarse=None)`
returns `lambda` for one stored equilibrium. `jax.grad` raises (see
[Two modes](index.md#two-modes)). Under `jax.jit`, make the two configs static:
`jax.jit(growth_rate, static_argnums=(2, 3))`.

- `v_guess`: start vector, for example the previous step's eigenvector.
- `v_fixed`: skip the eigensolve and use this vector. Valid only at the same
  equilibrium it came from.
- `coarse=(eq_c, diffmat_c)`: coarse level for `eigensolver="jd"`.

`eigenpair(eq, diffmat, assembly=None, solver=None, v_guess=None, coarse=None)`
returns `(lambda, v, residual)` with `residual = ||A v - lambda v|| / |lambda|`.
Check the residual; it is the quality measure.

## Optimize mode

`growth_rate_of(params, equilibrium_map, diffmat, assembly=None, solver=None, v_fixed=None, v_guess=None, coarse=None)`
is the same `lambda` as a function of `params`, with `equilibrium_map` a JAX
function `params -> EquilibriumData` that evaluates geometry and profiles and
contains no equilibrium solve. `jax.grad` returns a pytree shaped like `params`
(Hellmann-Feynman: the eigensolve has a zero backward rule, the Rayleigh
quotient carries the derivative). An `EquilibriumData` as `params`, or a
non-callable `equilibrium_map`, is refused. Under `jax.jit` the map is static
too: `jax.jit(jax.grad(growth_rate_of), static_argnums=(1, 3, 4))`.

`growth_rate_and_grad(params, equilibrium_map, diffmat, ...)`: value and
gradient from one eigensolve.

`from_desc(eq_or_path, n_rho, n_theta, n_zeta, automorphism=AUTOMORPHISM)`
returns `(EquilibriumData, DiffMat)`. Needs DESC.

## Configuration

Frozen dataclasses, passed as static arguments.

`AssemblyConfig`: `gamma=5/3`, `incompressible=False`, `axisym=False`,
`n_mode_axisym=1`, `coupled_rt=False`, `n_rho_coupled`, `n_theta_coupled`.

`SolverConfig`:

| field | default | used by |
|---|---|---|
| `eigensolver` | `"eigsh"` | `"eigsh"`, `"jax_lanczos"`, `"jd"` |
| `sigma` | `-0.1` | all; must be negative |
| `eigsh_tol` | `1e-8` | eigsh |
| `num_matvecs`, `factor`, `seed` | `50`, `"lu"`, `0` | jax_lanczos |
| `sigma_mode`, `sigma_factor` | `"fixed"`, `2.5` | jax_lanczos |
| `jd_outer, jd_inner, jd_maxdim, jd_keep` | `200, 100, 60, 10` | jd |
| `jd_tol, jd_theta_tol` | `0.0, 1e-8` | jd stop tests (residual, Ritz change) |
| `coarse_num_matvecs`, `k_defl` | `100`, `50` | jd coarse level |

See [Choosing options](options.md) for how to set them.

## Grid operators: `agnimhd.basis`, `agnimhd.quadrature`

- `standard_grid(n_rho, n_theta, n_zeta, NFP=1, automorphism=None)` returns
  `(nodes, diffmat)`: Lobatto radially through the clustering map, Fourier in
  both angles.
- One-dimensional bases, each returning `(D, W)` on the same nodes:
  `legendre_diffmat`, `jacobi_diffmat`, `fourier_diffmat`,
  `fourier_diffmat_truncated`, `bspline_diffmat`, `finite_difference_diffmat`,
  `zernike_fourier_diffmat`.
- Nodes and maps: `leggauss_lob`, `gauss_radau_jacobi`, `zernike_nodes_weights`,
  `automorphism_staircase1`, `automorphism_staircase2`.
- `DiffMat(D_rho=, W_rho=, D_theta=, W_theta=, D_zeta=, W_zeta=)` holds the
  pairs; `w_rho`, `w_theta`, `w_zeta` are the weights as 1-D vectors.
- Mode caps are checked: `fourier_diffmat_truncated` and
  `zernike_fourier_diffmat` raise if `M > (n - 1) // 2`.

## Lower level

- `agnimhd.assemble`: `assemble_dense` (the reduced whitened matrix),
  `matfree_operator` (the same operator as a function), `ring_block`,
  `keep_indices`, `operator_dtype` (complex for `axisym=True`).
- `agnimhd.solvers`: `jacobi_davidson`, the ring preconditioner
  (`build_ring_blocks`, `factor_ring_blocks`, `make_block_precond`), the coarse
  level (`coarse_seed_and_deflation`, `transfer_matrices`), `pcg`,
  `pcg_deflated`.
- `agnimhd.plotting`: `mode_components`, `mode_displacement`,
  `mode_plot_displacement`, `mode_delta_v`, `mode_speed` return arrays; `plot_*`
  need matplotlib. `plot_mode_cross_section` and
  `plot_eigenfunction_cross_sections` draw `(R, Z)` contours at `zeta = 0`, and
  for 3D cases also at `zeta = pi / NFP`.

## Command line

```
agnimhd info                              # list the EquilibriumData fields
agnimhd validate FILE [--res R,T,Z] [-v]  # check a saved or DESC equilibrium
agnimhd solve FILE [--res R,T,Z] [--gamma G] [--sigma S] [--eigensolver E]
              [--automorphism '{"eps": 0.01, "x_0": 0.65, "m_1": 2, "m_2": 3}']
```

`FILE` is an agnimhd `.npz` or `.h5`, or a DESC `.h5` (then `--res` is
required). `--automorphism` must match the clustering the file was exported
with.
