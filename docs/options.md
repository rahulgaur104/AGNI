# Choosing options

All of these are convergence choices. Scan them at low resolution first, then
make one production run.

## Grid resolution

`n_rho x n_theta x n_zeta` on the PEST grid; 24x12x8 is a reasonable start. The
eigenvalue converges fastest radially, and the placement of the radial nodes
changes it more than added resolution does: put the clustering point `x_0` of
`automorphism_staircase1` where the mode peaks (usually the resonant surface).
The innermost node sits at `rho = eps`, with `eps` between 1e-3 and 1e-2,
because several coefficients are singular on axis.

The toroidal nodes span one field period, so only the `n = 0 mod NFP` family is
resolved. Other families need a full-torus grid.

## Radial basis

| basis | how to build | notes |
|---|---|---|
| Legendre-Lobatto with staircase map | `basis.standard_grid` (what `from_desc` uses) | the test fixtures and the paper's convergence study |
| Gauss-Radau-Jacobi (GJ) | `basis.jacobi_diffmat(n, alpha, beta)`; nodes from `quadrature.gauss_radau_jacobi` | no node on the axis; production choice for the QH runs with `alpha = -0.35`, `beta = -0.65` |
| Zernike, coupled (rho, theta) | `basis.zernike_fourier_diffmat`, `AssemblyConfig(coupled_rt=True)` | regular at the axis by construction; dense operator, slower |

`from_desc` currently builds only the first. For GJ or Zernike, build the
`DiffMat` yourself on the same nodes the equilibrium is evaluated at. The
Zernike radial degree defaults to `L = 2 (n_rho // 2 - 1)`, as in DESC; the
near-interpolating `2 (n_rho - 1)` gives a badly conditioned radial
pseudo-inverse.

## MPOL and NTOR

`basis.fourier_diffmat_truncated(n, M)` keeps poloidal (or toroidal) modes up to
`M`. The grid must hold them, `MPOL <= (n_theta - 1) // 2` and
`NTOR <= (n_zeta - 1) // 2`, and a larger cap (also `M` of
`zernike_fourier_diffmat`) raises a `ValueError`.

- For a comparison with another code, cap them deliberately so one mode
  dominates. The paper used `m <= 8`, `n <= 8`; in its NIMSTELL benchmark the
  remaining mode is `m = n = 4`, an interchange mode near `iota = 1.02` where
  the shear vanishes.
- For a physics answer, raise them until `lambda` stops changing.
- On a one-period grid NTOR counts modes per period; on a full torus it is the
  full-torus `n`.
- With the two-level `"jd"` solver, coarse and fine levels need the same MPOL
  and NTOR.

## Eigensolver (`SolverConfig.eigensolver`)

| | forms the matrix | runs on | use when |
|---|---|---|---|
| `"eigsh"` (default) | yes | host (SciPy ARPACK) | the dense matrix fits in memory |
| `"jax_lanczos"` | yes | CPU or GPU | you need to stay on the device inside `jit` |
| `"jd"` | no | CPU or GPU | the dense matrix does not fit |
| `"dense_mg"` | yes, split over all visible GPUs | several GPUs | the dense matrix fits in their combined memory; needs `jaxmg` |

On an 80 GB A100 the dense matrix stops fitting between 24x40x12 and 32x48x16
(paper, table 3). While it fits, the GPU is about ten times faster than a
128-core CPU node; past it, the paper's matrix-free GPU path is less than twice
as fast and loses accuracy against the CPU reference. The gradient costs a small
fraction of the eigensolve: 0.04 to 0.26 s on the GPU for all 7444 parameters
up to 40x48x16 (paper, table 4).

`"dense_mg"` splits the dense matrix over the GPUs and runs block inverse
iteration with JAXMg's Cholesky solve. On one node of four 80 GB A100s it solved
the Patil QH case up to 80x48x16, 182,784 unknowns, in 8.5 minutes. See
[Dense solves on several GPUs](multigpu.md).

`"jd"` is Jacobi-Davidson with a ring block preconditioner. Pass a coarse level,
`growth_rate(eq, diffmat, solver=..., coarse=(eq_c, diffmat_c))`; on the
24x12x8 test case it did not converge without one. For gradients set a residual
stop (`jd_tol=1e-5`): with the default Ritz-value stop the gradient was off by up
to 3 %.

## Shift (`sigma`)

`sigma` must lie below the lowest eigenvalue, and not too far below it for the
fixed-budget solvers (`"jax_lanczos"`, `"jd"`). On the 24x12x8 test case
(`lambda = -1.34e-4`) a 50-step Lanczos returned the wrong mode at
`sigma = -0.1` and at `-1e-2`, and the right one at `sigma = -1e-3`. At
`sigma = -0.1`, 200 steps recover the eigenvalue at four times the cost, but
the vector is still not converged.

Procedure: estimate `lambda` from a cheap low-resolution run, then set
`sigma = 1.3` to `2.5` times that estimate. `sigma_mode="adapt"` does the
re-shift automatically for `"jax_lanczos"`. Always check the residual returned
by `eigenpair`, not the eigenvalue: the wrong mode above had residual 4.6e4, the
200-step run 2.9e2, the right one 4.8e-4.

## gamma

`AssemblyConfig(gamma=5/3)` is the compressible default. A large `gamma` drives
the solution toward incompressibility and stays differentiable;
`incompressible=True` imposes the constraint exactly but is too expensive inside
an optimization. DESC's own objective defaults to `gamma = 0`, so state the
value with every result.

## Accuracy

- Eigenvalues with `|lambda|` below about 1e-10 are at roundoff and mean
  marginal.
- Two correct runs agree to 2.8e-5 relative; the test tolerances use this.
- A finite-difference gradient check needs a step near `h = 1e-7` (0.45 %
  agreement here; the paper reports below 0.2 %): larger steps see curvature,
  smaller ones see the noise floor. The equilibrium must be converged at both
  points and the mode must not swap between them, or the check fails for
  reasons that are not the gradient's.
