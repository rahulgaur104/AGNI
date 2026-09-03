# Theory: from the energy principle to a matrix

This page is the bridge between the AGNI paper and the code. It states what is
discretized, in which coordinates, with which normalization and which sign, and
it names the file and function where each piece lives. It is deliberately
explicit about conventions, because every convention on this page has a sign or
a factor that will silently produce a plausible wrong number if it is guessed.

Symbols follow the paper (Gaur *et al.*, 2026), with one systematic exception
noted under [Radial coordinate](#radial-coordinate).

---

## 1. The physics being solved

Linearizing ideal MHD about a static equilibrium (`V_0 = 0`,
`grad p_0 = j_0 x B_0`) gives a force operator `F[xi]` acting on the plasma
displacement `xi`, and a normal-mode problem

```
F[xi] = gamma^2 M n_0 xi
```

Rather than discretizing `F`, a second-order differential operator whose
discretization is not symmetric unless one is careful, AGNI discretizes the
**energy functional** obtained by contracting with `-xi` and integrating over
the plasma volume. Dropping vacuum/external modes, the potential energy is

```
dW_p = INT dV [ |C|^2 + Gamma p_0 |div xi|^2 - F |xi . grad rho|^2 ]

  C = Q + (j_0 x grad rho)/|grad rho|^2 (xi . grad rho)
  Q = curl(xi x B_0)                                   (field-line bending)
  F = 2 (j_0 x grad rho) . (B_0 . grad grad rho) / |grad rho|^4
```

(paper Eq. 13 for the force operator, Eqs. 16-17 for the energy integral). Note
the fourth power in `F`: in the code this is `(g^rr)**2`, since
`g^rr = |grad rho|^2`. The three terms are the three that matter physically.
`|C|^2` is stabilizing field-line bending, `Gamma p_0 |div xi|^2` is stabilizing
compression, and `F` is the **instability drive**, the only term that can be
negative and the only one whose sign decides the answer.

The kinetic energy `dK = INT dV |xi|^2` is positive definite. The variational
problem `dW_p = -lambda dK` (paper Eq. 19) becomes, after discretization, a
generalized Hermitian eigenvalue problem (paper Eq. 45)

```
A xi = lambda B xi
```

with `A` the potential-energy matrix and `B` the kinetic-energy (mass) matrix.
`A` and `B` are real symmetric when the full torus is kept, and complex
Hermitian under `AssemblyConfig(axisym=True)`, where a single toroidal harmonic
is retained and `d/dphi` becomes `i n`.

### Sign convention

**This is the one thing to get right before reading anything else.**

`agnimhd.growth_rate` returns the Rayleigh quotient of the *energy* matrix,

```
lambda = <xi|A|xi> / <xi|B|xi>
```

so that

| | `agnimhd` | AGNI paper Eq. (19) |
|---|---|---|
| unstable | `lambda < 0` | `lambda > 0` |
| stable | `lambda > 0` | `lambda < 0` |

The paper writes `dW_p = -lambda dK`, which puts the minus sign on the other
side, so its `lambda` is the normalized squared growth rate. In this
package the returned number is the energy quotient itself, so

```
gamma^2 = -lambda * (B_N / a_N)^2 / (mu_0 M n_0)
```

and an optimizer seeking stability must **increase** `lambda` toward zero.
`tests/test_objective.py::test_a_descent_step_moves_lambda_the_right_way` is the
end-to-end assertion of exactly this, because a globally flipped gradient passes
every magnitude and finiteness check there is.

The implementation of the sign lives in
`agnimhd.assemble._normalized_fields`, where the drive enters as
`F = -mu_0 * eq.instability_drive() / B_N**2`.

---

## 2. Coordinates and normalization

### Radial coordinate

TERPSICHORE, whose notation the energy integral follows, writes its expressions
against the normalized toroidal flux `s = rho^2`. **In `agnimhd`, every `s` in
those expressions is replaced by `rho = sqrt(psi / psi_edge)`, throughout, with
no exceptions.** This is not a change of variable applied consistently. It is a
redefinition of the radial coordinate the whole functional is written in. In particular
the instability drive is

```
drive = 2 * dot(J x grad(rho), (B . grad) grad(rho)) / (g^rr)^2
```

This is TERPSICHORE, doi:10.1007/978-1-4613-0659-7_8, Eq. (5) p. 162, with
`s -> rho`.
This substitution changes the drive by a factor that depends on `dpsi/drho`, so
an adapter that supplies a drive built with `s` will produce a wrong eigenvalue
that looks entirely reasonable. `EquilibriumData.instability_drive` can build it
from the two vector fields instead, and the two routes are checked against each
other to 2.8e-16 in `tests/test_equilibrium.py`.

### Angles

PEST straight-field-line coordinates `(rho, theta_PEST, phi)`, abbreviated
`(r, v, p)` in the field names. `phi` is the geometric toroidal angle. For a
field-period-symmetric device the grid spans one field period,
`phi in [0, 2*pi/NFP)`, and this is what restricts the analysis to the `n = 0
mod NFP` mode family. Modes with `n mod NFP != 0` require the full torus.

The choice of PEST is not physics: any straight-field-line system works, as the
paper notes. PEST is used for robustness and speed.

### Normalization

Lengths by the minor radius `a`, fields by `B_N = |Psi| / (pi a^2)`. Inputs are
supplied **unnormalized, in SI**. The normalization is applied internally by
`_normalized_fields`. Do not pre-normalize.

> **`a` is the sharpest input in the package.** The eigenvalue is
> hypersensitive to it, and two defensible definitions of "the minor radius",
> DESC's `QuadratureGrid` and `LinearGrid` averages, were measured to differ by
> **3.76%** on the same equilibrium. Record which definition an export used.
> `EquilibriumData.save` writes it into the sidecar. See
> [docs/adapters.md](adapters.md).

---

## 3. Discretization

### The quadrature form

Every term of the energy integral is a quadrature of the form

```
INT dV sqrt(g) (d_y xi^x) E (d_y' xi^x')
  ~=  xi^x+  [ D_y+ diag(W sqrt(g) E) D_y' ]  xi^x'
```

with `E` a purely equilibrium-dependent coefficient evaluated at the collocation
nodes, and `D` a first-order spectral differentiation matrix (paper Eqs. 26-27).
The symmetry of the energy integral guarantees a conjugate partner for every
off-diagonal term, so the assembled matrix is Hermitian by construction rather
than by symmetrization. This is why the code assembles `dW`, not `F`.

Because the three directions are separable, the 3D operators are Kronecker
products of the 1D ones:

```
D_rho = D_rho0 (x) I_theta (x) I_zeta,  etc.
```

The full operator is `3N x 3N` with `N = n_rho * n_theta * n_zeta`: nine major
blocks from the component pairs, each block a sum of terms in the derivative
pairs. `agnimhd.assemble.assemble_dense` builds it, and the individual terms appear
there in the same order as Appendix B of the paper.

### Node ordering

**rho-major**, flat index of `(i, j, k)` is `(i * n_theta + j) * n_zeta + k`, and
component `c` lives at `c * n_total + that`. Every index map in the package
assumes it. An adapter emitting a different ordering does not error. It solves a
different problem.

### Regularizing the axis

Terms such as `1/sqrt(g)` and `g_rv` are singular on axis. Two mechanisms handle
it:

1. **Rescaling the displacement** (paper Eq. 24): `xi^rho -> xi^rho / psi'`,
   `xi^zeta -> iota xi^zeta`, and the code works in `(xi^rho, upsilon, xi^zeta)`
   with `upsilon = xi^theta - xi^zeta`. Since `psi' ~ rho` near the axis this
   both enforces `xi^rho -> 0` there and cancels the `1/sqrt(g)` behaviour in
   `Q`.
2. **Not putting a node on the axis.** The innermost surface sits at
   `rho = epsilon`, with `epsilon` in `[1e-3, 1e-2]`. Results are insensitive to
   it once it is small enough.

### Boundary conditions

Perfectly conducting wall: `xi^rho = 0` on the innermost and outermost radial
surfaces. The two tangential components are kept everywhere. So

```
n_keep = 3 * n_total - 2 * n_theta * n_zeta
```

and `agnimhd.assemble.keep_indices` returns exactly those indices, as concrete
NumPy, because shapes cannot come from traced values.

The condition is applied to `B` *before* the Cholesky factorization (by zeroing
the off-diagonal couplings of the constrained rows while keeping their
diagonals, so they cannot influence the result), and to the whitened `A` after
the congruence, by deleting rows and columns. Applying Dirichlet to `v = L^T xi`
is equivalent to applying it to `xi`, because `B` is block-diagonal per node.

### Bases

Fourier in both angles, which is exact for band-limited data on a doubly
periodic domain. Radially, `agnimhd.basis.standard_grid` builds
**Legendre-Lobatto** nodes through the clustering map. The paper's radial
convergence scan (Fig. 7c) finds that basis the most accurate of those tested,
and selects Gauss-Jacobi-Radau as the paper's default only because it is more
modular while converging almost as well. Both are available here, along with
Chebyshev, B-spline, fourth-order SBP finite differences, and a coupled
non-separable Zernike-Fourier radial-poloidal basis, in `agnimhd.basis`.

Two notes from measurement rather than theory:

* For the coupled Zernike path the **Jacobi radial recurrence is the trusted
  ground truth**. A uniform radial variant produces spurious modes.
* Zernike gives a **dense** `A` (paper, Fig. 8d). It has no sparsity structure
  to exploit, unlike the other radial bases.

### The radial mapping function

Legendre-Lobatto nodes cluster at the ends, but the unstable eigenfunction peaks
in the interior. AGNI therefore solves on a mapped grid `rho_s = f(rho)`, which
transforms the differentiation matrix and the weights as

```
D_rho_s = W_s^-1 D_rho,   W_s = diag(f'(rho)) (x) I_theta (x) I_zeta
```

with `f` the smooth two-sided exponential clustering map of paper Eq. (65),
(the transformation itself is paper Eq. 64),
implemented as `agnimhd.quadrature.automorphism_staircase1(x, eps, x_0, m_1,
m_2)`: `x_0` is the target radius, `m_1` and `m_2` control how strongly nodes
are pulled from the inner and outer ends toward it, and `eps` is the axis
offset. `f'` is obtained by `jax.grad`, not by hand.

> **The map is part of the grid.** The equilibrium must be evaluated at the
> mapped points `rho_s = f(rho)` while the quadrature weights live on the
> Legendre grid. If a `DiffMat` is built with different `automorphism` kwargs
> than the export used, the geometry and the derivative operators are on
> different node sets and nothing will complain. The CLI's `--automorphism` flag
> carries this warning for the same reason.

---

## 4. Reduction to a standard problem

`B` is symmetric positive definite, and in node-major ordering it is **block
diagonal with 3x3 blocks**, because without derivatives the three components at
a node couple only to each other (paper Fig. 1). So its Cholesky factorization
costs `O(N)` rather than `O(N^3)`: `N` independent 3x3 factorizations, done with
one `vmap`.

With `B = L L^T` and `v = L^T xi` (paper Eq. 46),

```
A_hat v = lambda v,     A_hat = L^-1 A L^-T
```

which is the standard Hermitian problem the eigensolvers see. **No generalized
eigensolve is performed anywhere in the package.** The pencil is reduced by this
congruence and every solver works on `A_hat`. `A_hat` is what
`assemble_dense` returns, and what `matfree_operator` applies without forming.

---

## 5. What the eigenvalue is allowed to mean

`A_hat` has a very wide spectral range: the ideal-MHD force operator carries
continuous spectra (Alfven and slow continua) with accumulation points near
marginal stability. The discrete unstable modes AGNI targets are separated from
that cluster. The stable spectrum is not resolved and is not meant to be.

The practical consequence is an **accuracy floor**. For a backward-stable
Hermitian eigensolver the absolute roundoff scale is `~ eps * ||A_hat||_2`, and
for a typical stellarator `||A_hat||_2 ~ 1e6`, giving

```
absolute noise floor  ~ 1e-10
relative noise floor  = 2.8e-5   (measured, this implementation)
```

An eigenvalue with `|lambda| <~ 1e-10` is **not numerically resolved**. It is
indistinguishable from marginal. The paper demonstrates this directly: with the
instability drive `F` switched off, every computed eigenvalue of the benchmark
equilibrium falls under `4e-10`, entirely inside the roundoff scale. With
the drive on, one isolated unstable eigenvalue appears far above it.

Two things follow for anyone using this package:

* Report a growth rate only when `|lambda|` is orders of magnitude above 1e-10.
* Two correct runs may differ by 2.8e-5 relative. Test tolerances in this repo
  are set from that number, and a finite-difference gradient check has to step
  outside it. See [docs/resolution.md](resolution.md).

---

## 6. Finding the mode

Shift-invert (paper Eq. 47). Instead of `A_hat`, iterate on

```
(A_hat - sigma I)^-1 v = mu v,    mu = 1 / (lambda - sigma)
```

The paper writes `H_sigma = sigma I - A_hat` and `mu = 1/(sigma - lambda)`. The
two differ by the overall sign of `lambda` between the paper's convention and
this package's, and are the same operation.

so eigenvalues near `sigma` become the largest-magnitude eigenvalues of the
transformed operator, and a Krylov method reaches them in a few tens of
iterations without touching the clustered stable spectrum. Only a small Krylov
basis and one factorization are stored, rather than a full `3N x 3N`
eigenvector matrix, which is what makes the gradient affordable.

`agnimhd` offers three eigensolvers, selected by `SolverConfig.eigensolver`:

| | how | when |
|---|---|---|
| `"eigsh"` | dense assembly, host SciPy ARPACK behind a `pure_callback` | default; **measured 1.53x faster than our JAX Lanczos on CPU** |
| `"jax_lanczos"` | dense assembly in JAX, `matfree` Lanczos on an exact LU/Cholesky shift-invert | stays on the accelerator; what the paper's Algorithm 1 describes |
| `"pcg_deflated"` | never forms the matrix: matrix-free operator, ring block-Jacobi preconditioner, deflation, coarse-level solve | resolutions where the dense matrix does not fit |

Choosing `sigma` is not free, and the constraint is two-sided. See
[docs/resolution.md](resolution.md#the-shift). In short, the
paper's benchmarks used `|sigma| = 1e-3`, obtained from a cheap low-resolution
full-spectrum calculation, and that a shift much further from the spectrum
breaks any solver that stops at a fixed matvec count.

---

## 7. The gradient

The eigenvalue derivative is Hellmann-Feynman (paper Eq. 59):

```
dlambda/dx = <v| (dA_hat/dx) |v> / <v|v>
```

At an eigenvector, the eigenvalue's derivative is the derivative of the Rayleigh
quotient **with the vector held fixed**. `agnimhd.objective.growth_rate`
implements this by wrapping the eigensolve in a `jax.custom_vjp` whose backward
rule returns **zero** cotangents, then returning the ordinary Rayleigh quotient.
Autodiff of that expression then *is* the contraction above.

Two consequences, both load-bearing:

* **The eigensolve need not be differentiable.** That is what allows host ARPACK
  behind a `pure_callback` to be the default solver, and it removes the
  eigenvector-selection `argmax`, which has no useful derivative, from the
  graph entirely.
* **`v` is recomputed at every call.** This is a fixed-vector gradient, not a
  stale-vector one.

The operator-vector product `(dA_hat/dx) v` is never materialized as a matrix.
Reverse-mode differentiation of the matrix-free application supplies it
directly, which is the memory argument in the paper's footnote 5.

**What `x` is.** The paper takes this derivative with respect to boundary shape
or profile parameters, at fixed force balance and with the remaining equilibrium
parameters held (paper Sec. 5.2, scanning `R_b,10`). The equilibrium is not
re-solved inside the derivative. Force balance is a constraint on the
optimization and is imposed by the optimizer, which in DESC is
`ProximalProjection`: it perturbs and re-solves the equilibrium back onto the
constraint after each step and forms the reduced derivative
`dlambda/dc = @lambda/@c - (@lambda/@x)(@F/@x)^-1 (@F/@c)`. What this package
returns is the `@lambda` factor. See
[docs/adapters.md](adapters.md#consuming-the-gradient).

Validation: the analytic gradient agrees with a central finite difference to
**0.45%** in this implementation, and only at `h = 1e-7`. The paper reports a
relative error below 0.002 for a small enough step (Fig. 5c). Larger steps are
dominated by curvature of the eigenvalue landscape and smaller ones fall into
the relative noise floor, so a disagreement at some other step is the finite
difference's problem. Two further failure modes are not the gradient's either:
the equilibrium drifting out of force balance, `max|F|_normalized >~ 1%`, and
the most unstable mode switching branches, which ends the smoothness of the
landscape.

---

## 8. Incompressibility

The most dangerous modes are usually incompressible. The cheap route to that
limit is to **raise `Gamma`** (paper Sec. 6.2). The compressibility term
`Gamma p_0 |div xi|^2` is purely stabilizing, so a large `AssemblyConfig.gamma`
penalizes `div xi` and drives the solution toward `div xi -> 0`. Equivalently,
it raises the sound speed `c_s = sqrt(Gamma p / rho_0)` until the fluid behaves
incompressibly. It needs no extra factorization and is fully differentiable, so
it works inside an optimization loop.

The alternative projects the compressible modes out with
`P = I - C_hat^T (L_G L_G^T)^-1 C_hat` (paper Eqs. 61-62), giving
`div xi ~ O(1e-8)`, far more accurately. It requires a dense Cholesky of the
Gram matrix `G = C_hat C_hat^T` inside the AD loop, which the paper reports as
prohibitively expensive for optimization. The paper shows the two agree as
`Gamma` grows, and that the compressible branch approaches the incompressible
one **from the unstable side**, which keeps the computed eigenvalue further from
the noise floor.

`AssemblyConfig(incompressible=True)` selects the direct constraint, and
`AssemblyConfig(gamma=...)` is the optimization-compatible route.

---

## References

* R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
  stability solver & optimizer for magnetic confinement fusion devices* (2026).
  The paper this package implements. Section numbering above follows it.
* I. B. Bernstein, E. A. Frieman, M. D. Kruskal, R. M. Kulsrud, *An energy
  principle for hydromagnetic stability problems*, Proc. R. Soc. A **244** (1958).
* D. V. Anderson *et al.*, `TERPSICHORE`, doi:10.1007/978-1-4613-0659-7_8:
  the notation the energy integral follows, and the source of the drive
  expression (Eq. 5, p. 162), with `s -> rho`.
* N. Krämer *et al.*, `matfree`: the Lanczos tridiagonalization used by the
  matrix-free paths.
