# Resolution, shifts, and what counts as converged

Everything on this page is a measurement, either from the AGNI paper or from
this repository's test suite. Where the two differ in convention, the source is
named. Numbers without a source are not on this page.

---

## The accuracy floor

`A_hat` has a very wide spectral range, so the eigenvalue has a hard
finite-precision floor:

| | value | meaning |
|---|---|---|
| absolute noise floor | **1e-10** | `~ eps * \|\|A_hat\|\|_2` with `\|\|A_hat\|\|_2 ~ 1e6` for a typical stellarator |
| relative noise floor | **2.8e-5** | two correct runs of this implementation may differ by this much |

Consequences:

* An eigenvalue with `|lambda| <~ 1e-10` is **not resolved**. It is
  indistinguishable from marginal stability, and reporting it as a growth rate
  is reporting roundoff. The paper demonstrates the floor directly: with the
  instability drive switched off, every computed eigenvalue of the benchmark
  case lands below `4e-10`.
* Test tolerances in this repository are `2.8e-5` relative, and that is the
  reason. Tightening one does not make anything more correct.
* Raising `AssemblyConfig.gamma` toward the incompressible limit moves the
  eigenvalue **away** from the floor, because the compressible branch approaches
  the incompressible one from the unstable side. That is a numerical argument
  for the `gamma` route on top of the physics one.

---

## Grid resolution

The eigenvalue converges fastest with radial resolution, and node placement
changes it more than added resolution does. The radial nodes should be
concentrated where the eigenfunction peaks, which for an interchange mode is at
the resonant surface. Set `x_0` in `automorphism_staircase1` there.

Poloidal and toroidal resolution set which modes exist at all. For a benchmark
against another code, cap the mode content deliberately. The paper limits the
maximum poloidal mode number to `m = 8` and the toroidal to `n = 8` when
building the differentiation matrices, which filters the higher-`n` modes and
leaves one dominant mode both codes can be compared on. In the NIMSTELL
benchmark that mode is `m = n = 4`, an interchange mode peaking near
`iota = 1.02` where the magnetic shear vanishes.

Those caps must stay below the collocation Nyquist limits:
`MPOL <= (NT - 1) // 2` for the coupled Zernike-Fourier poloidal basis and
`NTOR <= (NZ - 1) // 2` for toroidal Fourier truncation. The coupled Zernike
radial cap follows DESC AGNI by default,
`L_RAD = 2 * (NR // 2 - 1)`, rather than the near-interpolatory
`2 * (NR - 1)`, which makes the radial pseudo-inverse poorly conditioned at
practical node counts.

Reported timings and eigenvalues, paper table 3 (a modified LBD QH case,
paper sign convention, so `lambda > 0` is the unstable mode). All runs used
`sigma = 1e-3`. Rows marked `mf` used the preconditioned matrix-free path on the
GPU with `m_CG = 6000`, `n_mv = 100`, `k_defl = 50`. The rest used the dense
Lanczos-LU path. The CPU column is `scipy.sparse.linalg.eigsh`.

| `n_rho x n_theta x n_zeta` | GPU time | CPU time | `lambda` (GPU) | `lambda` (CPU) |
|---|---|---|---|---|
| 8x24x8 | 11.3 s | 12.7 s | 1.45e-3 | 1.45e-3 |
| 16x32x8 | 12.2 s | 45.0 s | 6.91e-5 | 6.91e-5 |
| 24x40x12 | 26.4 s | 154.9 s | 8.35e-5 | 8.35e-5 |
| 32x48x16 | 629.5 s `mf` | 1129.6 s | 8.33e-5 | 8.34e-5 |
| 40x48x16 | 670.0 s `mf` | 1667.5 s | 8.31e-5 | 8.34e-5 |

Hardware was Perlmutter: one A100 with 80 GB for the GPU column, and one node
with two AMD EPYC 7713 processors, 128 cores and 450 GB for the CPU column. The
GPU is roughly an order of magnitude faster while the dense matrix fits. Past
that point the dense path is unavailable on the GPU and the matrix-free path
takes over, at which point the GPU advantage narrows to under a factor of two
and the GPU eigenvalue begins to lose accuracy against the CPU reference, which
is why `sigma`, `m_CG`, `n_mv` and the coarse resolution have to be chosen with
care there.

Reverse-mode gradient timings, paper table 4, for `dlambda/dx` against all 7444
equilibrium parameters, excluding compilation:

| `n_rho x n_theta x n_zeta` | GPU | CPU |
|---|---|---|
| 8x24x8 | 0.04 s | 1.82 s |
| 16x32x8 | 0.07 s | 5.91 s |
| 24x40x12 | 0.12 s | 10.00 s |
| 32x48x16 | 0.23 s | 34.90 s |
| 40x48x16 | 0.26 s | 35.32 s |

The gradient costs a small fraction of the eigensolve it follows, which is the
argument for gradient-based optimization on a GPU.

---

## Choosing the eigensolver

| `SolverConfig.eigensolver` | forms the matrix? | device | use when |
|---|---|---|---|
| `"eigsh"` (default) | yes, dense | CPU host callback | the dense matrix fits. **Measured 1.53x faster than our JAX Lanczos on CPU** |
| `"jax_lanczos"` | yes, dense | CPU/GPU | you want to stay on the accelerator, or inside a `jit` with no host callback |
| `"pcg_deflated"` | no | CPU/GPU | the dense matrix does not fit |

`eigsh` and `jax_lanczos` have the same asymptotic cost. The difference between
them and a full `eigh` is that they compute the top `k` modes rather than the
whole spectrum, which is most of the win.

Note that `"eigsh"` runs on the host through `jax.pure_callback`. It survives
`jit`, and the tests apply `jax.jit` from outside the package, but it is a host
synchronization point, and it is not differentiable. Neither matters here,
because the gradient rule discards the eigensolve entirely.

---

## The shift

`sigma` selects which part of the spectrum the shift-invert reaches, and the
constraint on it is **two-sided**.

**Not above the spectrum.** Above the smallest eigenvalue the solve converges to
the wrong mode, and `H = A_hat - sigma I` stops being positive definite, so the
preconditioned CG in the matrix-free path is no longer a legal Krylov method.
`SolverConfig` refuses `sigma >= 0` outright.

**Not arbitrarily far below it either**, for a solver that stops at a fixed
matvec count. This applies to `jax_lanczos` and `pcg_deflated`, not to `eigsh`,
which iterates to `eigsh_tol`. Shift-invert maps `lambda` to
`mu = 1/(lambda - sigma)`, and Lanczos separates two modes at a rate set by the
ratio of their `mu`. As `sigma` recedes, every `mu` approaches `-1/sigma` and
the ratio approaches one, so more iterations are needed for the same separation.

Measured on the shipped 24x12x8 case with `jax_lanczos` and
`sigma_mode="fixed"`, against the dense reference `-1.337627e-04`:

| `sigma` | 50 matvecs | residual | 200 matvecs | residual |
|---|---|---|---|---|
| `-1e-1` (the default) | `+1.598084e-04` | 4.6e+04 | `-1.3375914e-04` | 2.9e+02 |
| `-1e-2` | `-1.222757e-04` | 1.6e+04 | `-1.3376269e-04` | 4.2e-05 |
| `-1e-3` | `-1.3376269e-04` | 4.8e-04 | `-1.3376269e-04` | 4.8e-04 |

Shift placement and iteration count trade against each other. Either a shift
near the spectrum at 50 matvecs or the default shift at 200 matvecs gives the
right eigenvalue. The default `sigma = -1e-1` is placed conservatively on the
side that has no recovery, and is safe for the default `eigsh`, which iterates
to `eigsh_tol` rather than stopping at a fixed count.

Read the residual, not the eigenvalue. It is the quantity that separates the
converged rows from the rest, including the `-1e-1`, 200-matvec row, whose
eigenvalue is correct to five digits while its residual shows the eigenvector is
not converged.

Two remedies, either of which is enough:

1. **Move the shift.** The paper's benchmarks used `|sigma| = 1e-3`, obtained
   from a low-resolution full-spectrum calculation in preprocessing. Measure the
   spectrum once at low resolution, then place the shift.
2. **Raise `num_matvecs`.** 200 recovers the same eigenvalue at the far shift,
   at four times the cost of moving `sigma`.

A third option, `sigma_mode="adapt"`, runs a cheap first pass and re-shifts to
`sigma_factor * lambda` (paper Algorithm 1, `c = 2.5`). It is the best measured
strategy when the first pass lands on the right side. It does not rescue a shift
as far out as `-1e-1`, where the first pass returns a positive `lambda`, the
guard rejects it, and the second pass repeats the first.

A fourth mode, `track`, re-bases the shift on the previous optimization step's
eigenvalue. It is deliberately **not implemented**. It degrades as `lambda`
approaches zero, and a tracked excursion can end worse than it started.

`num_matvecs` itself is **fixed and untuned** at 50, matching the paper. It was
never swept. It is a plausible knob for further speedup, not a converged choice.

---

## The matrix-free path

`"pcg_deflated"` never forms `A_hat`. It replaces the dense LU application of
`H_sigma^-1` by an inner preconditioned CG solve (paper Eq. 57), leaving the
outer shift-invert Lanczos unchanged. The preconditioner is the **ring**
(`theta`-line) block-Jacobi factorization of `H_sigma` (paper Eqs. 51-52),
assembled directly from the same discretized energy terms used to apply
`A_hat`, without forming it (paper Eqs. 53-54).

The condition number of `H_sigma` is `O(1e10)`, which for plain CG implies
`m_CG = O(sqrt(kappa)) = O(1e5)` and makes it unusable here. The ring
preconditioner brings this down by about two orders of magnitude, to `O(1e8)`.
The remaining small eigenvalues are removed by an additive coarse correction
(paper Eq. 55),

```
M^-1 = M_ring^-1 + Z (Z^T H_sigma Z)^-1 Z^T
```

where `Z` holds `k_defl` vectors interpolated from a coarse grid, obtained by
solving the coarse generalized problem
`H_sigma^coarse z = eta M_ring^coarse z` (paper Eq. 56) and keeping the smallest
`eta`. The vector for the smallest `eta` also seeds the fine Lanczos iteration.

**The deflation enters through the preconditioner, additively. It is not a
projection of the operator.** `agnimhd.solvers.deflation_Y` builds the `Y` with
`Y Y^T = Z (Z^T H Z)^-1 Z^T` for exactly this, and using `pcg_deflated`'s
projection form instead produces a converged-looking eigenvalue of the wrong
sign.

Cost is `m_CG * n_mv * C_mv` with memory `O(n_mv N + S_M)`, so this is the only
path that reaches `N ~ 1e6`. Its speed depends strongly on the preconditioner,
and better preconditioners for the matrix-free path are named in the paper's
conclusions as the first direction of future work, which is why the dense-LU
Lanczos remains the default here.

Two measured facts govern its use here:

* **The coarse radial resolution has a floor of 16.** Below it the two-level
  solve produces a **sign flip**: an unstable equilibrium reported as stable.
  The floor is free: the coarse solve is negligible next to the fine one.
* **Neither the coarse eigenvalue's sign nor the CG relative residual predicts
  quality.** On this operator the CG residual is anti-correlated with accuracy:
  a run at relative residual 1.42 was 0.10% from truth while one at 0.91 was
  7.9% off. Use the Rayleigh residual, or `rr_refine`.

`rr_refine` (Rayleigh-Ritz re-extraction of the eigenvector against `A_hat`
itself, rather than taking the Lanczos tridiagonal's eigenvector) closes cases
that no budget increase could: the Krylov *space* stays orthonormal to machine
precision even when CG's residual has corrupted the *selection within it*.
**The `trusted` flag is meaningless when `rr_refine` is on**, and is not
reported.

---

## What this repository has actually measured

All on CPU, all reproduced by the test suite:

| what | measured |
|---|---|
| dense `lambda3`, 24x12x8 shipped QH case | -1.337626871e-04 |
| package vs its own DESC export | 2.3e-10 relative |
| matrix-free operator vs dense, per column | 4.8e-16 relative (0.16 ulp of `\|\|A\|\|_max`) |
| ring blocks vs the dense matrix's sub-blocks | < 1e-14 relative |
| `growth_rate` vs the sidecar reference | 7.2e-10 relative |
| analytic `dlambda/da` vs central difference | agrees, at `h = 1e-7` |

## The finite-difference step

If the gradient is checked against finite differences, and it should be, the step
is not free. Recorded agreement is **0.45%, and only at `h = 1e-7`**. Larger
steps are dominated by the curvature of the eigenvalue landscape, and smaller ones
fall into the 2.8e-5 relative noise floor, where the difference quotient is
measuring noise divided by a small number. A disagreement at some other step is
the finite difference's problem, not the gradient's.

The paper's Fig. 5c shows the same window from the other side, and adds a
warning worth repeating: where the equilibrium itself is out of force balance
(`max|F|_normalized >~ 1%`), or where the most unstable mode switches branches
and the landscape stops being smooth, the FD gradient degrades while the AD
gradient stays sound. A mode swap between the two perturbed points looks exactly
like a wrong gradient.
