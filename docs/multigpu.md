# Dense solves on several GPUs

`SolverConfig(eigensolver="dense_mg")` solves the dense problem with the matrix
split over all GPUs visible to one process. It reaches sizes whose matrix does
not fit on one GPU, with the accuracy of the dense solve.

## Method

1. Each GPU builds its own block of rows of `A + sigma I` from the matrix-free
   operator (`assemble.assemble_rows`). No GPU ever holds the whole matrix or the
   Kronecker derivative matrices. The matrix is padded with an identity block to
   a multiple of `mg_tile` times the GPU count.
2. Block inverse iteration: one JAXMg Cholesky solve (`potrs`) with a block of
   `mg_block` vectors, then Rayleigh-Ritz on those vectors with the exact
   matrix-free operator, which gives `gamma^2 = -lambda` and the residual
   `||A v - lambda v|| / |lambda|`.
3. Step 2 repeats until the residual is below `mg_tol` or after `mg_iters`
   iterations. JAXMg cannot reuse a factorization between calls, so every
   iteration rebuilds and refactors the matrix.

An iteration is one factorization, one solve with 16 right-hand sides and 16
matrix-free products. It is not a Lanczos step; `num_matvecs` is not used.

`sigma` must lie above the largest `gamma^2`, or the Cholesky factorization
fails (the first iteration returns NaN), and close to it, because the
convergence rate depends on it. On the 24x12x8 test case the residual reached
1e-6 after 5 iterations with `sigma = 1.05 gamma^2` and was still 1.7e-6 after
10 with `sigma = 1.3 gamma^2`. Take `gamma^2` from a smaller grid.

## Measured: Patil QH case on one node

Quasi-helical equilibrium with beta 1.5 % and iota_min 1.02; Gauss-Radau-Jacobi
radial nodes (alpha -0.35, beta -0.65) through the staircase map (eps 1e-2,
x_0 0.6, m_1 2.5, m_2 3.0), Fourier truncated at MPOL 8, NTOR 2, one field
period, gamma 5/3, mass weighted by the normalized ion density,
`sigma = 1.5116e-4` (1.05 times the 40x48x16 value), `mg_block = 16`. One node,
four A100 80 GB GPUs, 2026-10-03.

| grid | unknowns | gamma^2 | residual | iterations | time per iteration | total | memory per GPU |
|---|---|---|---|---|---|---|---|
| 40x48x16 | 90,624 | 1.4396332860e-4 | 7.6e-7 | 6 | 17 s | 143 s | 19 GB |
| 56x48x16 | 127,488 | 1.4404497423e-4 | 1.7e-6 | 8 | 31 s | 294 s | 36 GB |
| 64x48x16 | 145,920 | 1.4406145412e-4 | 1.2e-5 | 8 | 40 s | 366 s | 44 GB |
| 72x48x16 | 164,352 | 1.4407170414e-4 | 4.9e-6 | 8 | 53 s | 472 s | 57 GB |
| 80x48x16 | 182,784 | 1.4407842416e-4 | 1.8e-5 | 7 | 65 s | 508 s | 68 GB |

Totals include about 30 s of DESC export. At 40x48x16 the result matches the
dense CPU solve to 7e-13; that CPU solve needed 402 GB of host memory. The
residual levels off between 1e-6 and 2e-5, set by the accuracy of each solve;
`gamma^2` stops changing in all printed digits by iteration 5. 80x48x16 is about
the largest grid of this shape on one node: its share of the matrix is 68 GB
per GPU.

## Running it

- **Environment.** Tested with `jaxmg==0.0.9` and jax 0.6.2, alongside DESC; it runs
  in one process that sees all GPUs of one node. jaxmg 1.0 and later need jax
  0.11, one process per GPU, and can span several nodes; agnimhd calls the same
  `potrs` there, but that path has not been run yet.
- **All GPUs visible.** Do not restrict `CUDA_VISIBLE_DEVICES`. DESC's
  `set_device("gpu")` sets it to a single GPU, so drop the variable again before
  JAX starts: `os.environ.pop("CUDA_VISIBLE_DEVICES", None)`.
- **Memory cap.** JAX's allocator stops at 75 % of each GPU by default. Set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=.93` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **Density.** `from_desc(..., density=True)` also returns the normalized
  `ni`; pass it as `density` to `multigpu.dense_mg`.

```python
from agnimhd import AssemblyConfig, SolverConfig, from_desc, multigpu

eq, diffmat, density = from_desc(eq_desc, 80, 48, 16, density=True)
solver = SolverConfig(eigensolver="dense_mg", sigma=1.05 * gamma2_estimate)
v, gamma2 = multigpu.dense_mg(eq, diffmat, AssemblyConfig(), solver, density=density,
                              log=lambda it, gamma2, res, v: print(it, float(gamma2), float(res)))
```

`log` is called after every iteration; returning `True` stops the iteration, for
example before a job's time limit. `from_desc(..., grid=(nodes, diffmat))` takes
another radial basis, such as the Gauss-Radau-Jacobi grid of the table.
