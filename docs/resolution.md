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

Radial resolution buys the most, and the clustering map buys more than
resolution does: the radial nodes should be concentrated where the eigenfunction
peaks, which for an interchange mode is near the resonant surface. Set
`x_0` in `automorphism_staircase1` there.

Poloidal and toroidal resolution set which modes exist at all. For a benchmark
against another code, cap the mode content deliberately — the paper's comparison
limited `m <= 8` and `n <= 4` when building the differentiation matrices, in
order to filter high-`n` modes and leave one dominant mode that both codes could
be compared on.

Reported timings and eigenvalues, paper table 3 (a modified LBD QH case,
paper sign convention, so `lambda > 0` is the unstable mode):

| `n_rho x n_theta x n_zeta` | GPU time | CPU time | `lambda` |
|---|---|---|---|
| 8x24x4 | 11.0 s | 9.1 s | 3.97e-6 |
| 16x32x8 | 12.1 s | 31.2 s | 3.02e-5 |
| 24x40x12 | 26.2 s | 191.5 s | 4.82e-5 |
| 32x32x12 | 29.3 s | 240.8 s | 6.11e-5 |
| 32x48x16 | OOM | 1117 s | 5.81e-5 |
| 40x48x16 | OOM | 1736 s | 5.85e-5 |

The GPU is roughly an order of magnitude faster until the dense matrix stops
fitting in 80 GB, which on an A100 happens between `32x32x12` and `32x48x16`.
Above that the choices are a CPU node or the matrix-free path. Gradient timings
(paper table 4) follow the same pattern and the same ceiling.

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
`jit` — the tests apply `jax.jit` from outside the package — but it is a host
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

**Not arbitrarily far below it either** — for any solver that stops at a fixed
matvec count, which means `jax_lanczos` and `pcg_deflated` but not `eigsh`.
Shift-invert maps `lambda` to `mu = 1/(lambda - sigma)`, and Lanczos separates
two modes at a rate set by the *ratio* of their `mu`. As `sigma` recedes, every
`mu` collapses onto `-1/sigma` and the ratio goes to one. Measured on the
shipped 24x12x8 case, whose spectrum starts `-1.34e-4, -6.25e-5`, then a cluster
of numerically null modes near `1e-11`:

| `sigma` | `mu[0]/mu[1]` | `jax_lanczos`, 50 matvecs |
|---|---|---|
| `-1e-1` (the default) | 1.0007 | **wrong mode**, `lambda = +1.598e-04` |
| `-1e-2` | 1.0075 | `-1.337435e-04` (1.4e-5 relative off) |
| `-1e-3` | 1.0823 | `-1.337627e-04` (exact) |

The default `-1e-1` is conservative about the side that has no recovery, and it
is safe for the default `eigsh`, which iterates to `eigsh_tol` instead of
stopping at a fixed count. It is **not** safe for a 50-matvec Lanczos on this
case.

The failure is not silent. Check the Rayleigh residual from
`agnimhd.eigenpair`: on the case above it is **4.6e+04** for the wrong mode
against **1.6e-04** for the converged one. Nothing else distinguishes them — the
wrong answer is a finite number of an entirely plausible magnitude.

Three ways out, in order of preference:

1. **Move the shift.** The paper's benchmarks used `|sigma| = 1e-3`, obtained
   from a cheap low-resolution full-spectrum calculation as a pre-processing
   step. That is the recommended procedure: measure the spectrum once at low
   resolution, then place the shift.
2. **Raise `num_matvecs`.** 200 also recovers the right mode at the far shift,
   at four times the cost of moving `sigma`.
3. **`sigma_mode="adapt"`**, which runs a cheap first pass and re-shifts to
   `sigma_factor * lambda` (paper Algorithm 1, `c = 2.5`). This is the measured
   best strategy *when the first pass lands on the right side*, and it does not
   rescue a shift as far out as `-1e-1`: the first pass returns a positive
   `lambda`, the guard rejects it, and the second pass repeats the first.

A fourth mode, `track` — re-basing the shift on the previous optimization step's
eigenvalue — is deliberately **not implemented**. It degrades as `lambda`
approaches zero, and a tracked excursion can end worse than it started.

`num_matvecs` itself is **fixed and untuned** at 50, matching the paper. It was
never swept. It is a plausible knob for further speedup, not a converged choice.

---

## The matrix-free path

`"pcg_deflated"` never forms `A_hat`. Each application of the shift-inverted
operator is an inner preconditioned CG solve, preconditioned by the **ring**
(`theta`-line) block-Jacobi blocks and deflated against a coarse space obtained
by solving the generalized problem on a coarsened grid. Its cost is
`m_CG * n_mv * C_mv` and its memory is `O(n_mv N + S_M)`, so it is the only path
that reaches `N ~ 1e6`, but its performance depends strongly on the
preconditioner — the paper is explicit that a robust preconditioner for AGNI is
still open work, which is why the dense-LU Lanczos remains the default.

Two measured facts govern its use here:

* **The coarse radial resolution has a floor of 16.** Below it the two-level
  solve produces a **sign flip** — an unstable equilibrium reported as stable.
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

If you check the gradient against finite differences — and you should — the step
is not free. Recorded agreement is **0.45%, and only at `h = 1e-7`**. Larger
steps are dominated by the curvature of the eigenvalue landscape; smaller ones
fall into the 2.8e-5 relative noise floor, where the difference quotient is
measuring noise divided by a small number. A disagreement at some other step is
the finite difference's problem, not the gradient's.

The paper's Fig. 5c shows the same window from the other side, and adds a
warning worth repeating: where the equilibrium itself is out of force balance
(`max|F|_normalized >~ 1%`), or where the most unstable mode switches branches
and the landscape stops being smooth, the FD gradient degrades while the AD
gradient stays sound. A mode swap between the two perturbed points looks exactly
like a wrong gradient.
