# Dense solves on several GPUs

`SolverConfig(eigensolver="dense_mg")` solves the dense problem with the matrix
split over all GPUs visible to one process. It reaches sizes whose matrix does
not fit on one GPU, with the accuracy of the dense solve. Every toroidal mode
family works: families 0 and `NFP / 2` on a real symmetric matrix, the others
(and `axisym=True`) on a complex Hermitian one.

## Method

1. Each GPU builds its own block of rows of `A + sigma I` from the matrix-free
   operator (`assemble.assemble_rows`). No GPU ever holds the whole matrix or the
   Kronecker derivative matrices. The matrix is padded with an identity block to
   a multiple of `mg_tile` times the GPU count.
2. Block inverse iteration: one JAXMg Cholesky solve (`potrs`) with a block of
   `mg_block` vectors, then Rayleigh-Ritz on those vectors with the exact
   matrix-free operator, which gives `gamma^2 = -lambda` and the residual
   `||A v - lambda v|| / |lambda|`.
3. JAXMg cannot reuse a factorization between calls, so every iteration
   rebuilds and refactors the matrix, and the shift can change for free. It
   starts at `sigma` and moves halfway towards the Ritz value `gamma^2` after
   every factorization that succeeds. The Ritz value never exceeds the largest
   `gamma^2`, which lies below every shift that factors. A shift that lands
   below the largest `gamma^2` fails to factor (JAXMg's status) and is replaced
   by the midpoint between it and the last shift that factored.
4. Steps 2 and 3 repeat until `gamma^2` changes by at most `mg_theta_tol`
   (relative) over an iteration, the residual is below `mg_tol`, or after
   `mg_iters` iterations. At a large Zernike penalty the residual scales with
   the penalty and never reaches `mg_tol`: the `gamma^2` change ends the run.

An iteration is one factorization, one solve with 16 right-hand sides and 16
matrix-free products. It is not a Lanczos step, and `num_matvecs` is not used.

`sigma` must lie above the largest `gamma^2`, or the first Cholesky
factorization fails and the result is NaN, as in the one-GPU dense solve.

## Measured: Patil QH case on one node

Quasi-helical equilibrium with beta 1.5 % and iota_min 1.02. Gauss-Radau-Jacobi
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
dense CPU solve to 7e-13, and that CPU solve needed 402 GB of host memory. The
residual levels off between 1e-6 and 2e-5, set by the accuracy of each solve.
`gamma^2` stops changing in all printed digits by iteration 5. 80x48x16 is about
the largest grid of this shape on one node: its share of the matrix is 68 GB
per GPU.

## Running it

- **Environment.** Tested with DESC alongside `jaxmg==0.0.9` (jax 0.6.2) and
  `jaxmg==1.2.0` (jax 0.11.0, DESC's upper bound; jaxmg 1.3 needs jax 0.11.1).
  jaxmg 0.0.9 runs in one process that sees all GPUs of one node. jaxmg 1.x runs
  one process per GPU, on one node or several, and lets `potrs` overwrite the
  matrix: the 48x48x16 QH family 1 dense solve, out of memory on four 80 GB GPUs
  with 0.0.9, peaks at 59.4 GB per GPU on one node and 36.7 GB on two.
- **Several processes.** Call `jax.distributed.initialize()` before anything
  else touches JAX, one process per GPU, and call `dense_mg` directly rather
  than through `eigenpair`. DESC builds its transforms on the CPU device of
  process 0, which the other processes cannot address: write the equilibrium
  with `EquilibriumData.save` from one process and load it in all of them
  (`benchmarks/zernike_jd_vs_dense.py --saved ... --distributed`).
- **All GPUs visible.** Do not restrict `CUDA_VISIBLE_DEVICES`. DESC's
  `set_device("gpu")` sets it to a single GPU, so drop the variable again before
  JAX starts: `os.environ.pop("CUDA_VISIBLE_DEVICES", None)`.
- **Memory cap.** JAX's allocator stops at 75 % of each GPU by default. Set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=.93` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **Density.** `from_desc(..., density=True)` stores the normalized `ni` as
  `eq.density`, and every solver uses it.

```python
from agnimhd import AssemblyConfig, Basis, SolverConfig, from_desc, multigpu

basis = Basis(80, 48, 16, mpol=8, ntor=2)  # the table's; toroidal family 0
eq, diffmat = from_desc(eq_desc, basis, density=True)
solver = SolverConfig(eigensolver="dense_mg", sigma=1.05 * gamma2_estimate)
v, gamma2 = multigpu.dense_mg(eq, diffmat, AssemblyConfig(), solver,
                              log=lambda it, gamma2, res, v: print(it, float(gamma2), float(res)))
```

`log` is called after every iteration. Returning `True` stops the iteration, for
example before a job's time limit.
