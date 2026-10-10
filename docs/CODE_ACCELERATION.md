# Code acceleration: toroidal mode families and stellarator symmetry

Written 2026-10-03. Developer notes. Section 1 is implemented (F1, F2 of section
5, branch `toroidal-families`); the rest are ideas, not implemented. Cost figures are
operation counts of the dense Cholesky factorization (about `N^3/3` flops for a
real `N x N` matrix, four times that for complex) and matrix storage, not
measurements. Everything stays in real space: the unknowns are nodal values of
the displacement, and the Fourier differentiation matrices are the same
collocation matrices agnimhd uses today.

Notation: `NFP` field periods, period length `L = 2 pi / NFP`; a full-torus grid
with `NFP * n_zeta` toroidal nodes and a field-period grid with the
first `n_zeta` of them; `N` unknowns on the full torus, `N / NFP` on one period.
Here `m` and `n` are only the poloidal and toroidal mode numbers; `x` labels a
family of toroidal mode numbers.
Displacement unknowns in the code's order (component-major): `xi^rho / psi'`,
`upsilon = xi^theta - xi^zeta`, `iota xi^zeta` (theory.md lists the last two
the other way round).

**Status: proved numerically on 2026-10-03** with agnimhd's own operator (section
7): the full-torus spectrum equals the union of the family spectra to round-off.
The family matrix must be the exact one of section 1, not `D_period + i x I`.

## 1. Toroidal mode families: every family on one field period

### Why it works

The equilibrium repeats every field period, so the operator `A` (and `B`)
commutes with the shift `T: zeta -> zeta + L`. On the full-torus grid `T` moves
every node `n_zeta` places along `zeta`; `A` is block-circulant in the period index.

Eigenvectors can therefore be chosen as eigenvectors of `T` as well (Bloch's
theorem, as for electrons in a crystal):

    f(zeta + L) = exp(2 pi i x / NFP) f(zeta),     x = 0, 1, ..., NFP - 1.

The family `x` holds the toroidal modes `n = x + k NFP`. **A family-`x`
eigenfunction is known everywhere once it is known on one field period**: on
period `p` it is the first period's values times `exp(2 pi i x p / NFP)`. So
each family is solved on one field period, with the same node count as today's
`domain="field_period"` runs. The full torus is not needed for any family.

This is exact, not an approximation: the projectors
`P_x = (1/NFP) sum_p exp(-2 pi i x p / NFP) T^p` commute with `A`, so the
full-torus matrix splits into `NFP` independent blocks, one per family, and the
full-torus spectrum is the union of the family spectra. Measured: to 2e-15 to
1e-14 of the spectral radius, the same as the full-torus matrix against itself
with its rows and columns shuffled (section 7).

### The differentiation matrix of one family

The family block of the full-torus Fourier matrix, acting on the nodal values of
the displacement on the first period:

    D_x[i, j] = sum_{p=0}^{NFP-1} D_full[i, j + p n_zeta] * exp(2 pi i x p / NFP),
    D_full = fourier_diffmat(NFP * n_zeta),     i, j = 0 ... n_zeta - 1.

It is a real-space `n_zeta x n_zeta` collocation matrix, built once. On period
`p` the displacement is the first period's values times `exp(2 pi i x p / NFP)`.
Assembled with it, agnimhd's operator equals the family block of the full-torus
matrix to 2.5e-16, for every `x` and odd or even `n_zeta`.

**The shortcut `NFP * D_period + i x I` (acting on `g = exp(-i x zeta) f`) is
not exact in general.** It keeps a different window of toroidal harmonics:
`n = x + k NFP` with `k` in the period grid's window, instead of the full-torus
window `|n| <= NFP n_zeta / 2`. The two agree only for odd `n_zeta` and
`|x| < NFP/2`. They differ by one harmonic for `x = NFP/2` (the full torus has
`NFP n_zeta` nodes, an even number when `NFP` is even, and its Nyquist harmonic
falls in that family even for odd `n_zeta`), and for every `x != 0` at even
`n_zeta`. On the QH test case that one harmonic moved the most unstable
eigenvalue by 14-26 % and, on one grid, put it in the wrong family.

- Multiplication by equilibrium coefficients keeps a function in its family,
  because the coefficients have period `L`. Nothing else in the assembly changes.
- Quadrature: `|f|^2` repeats every period, so the field-period weights are
  unchanged (the full-torus energy is `NFP` times the one-period energy for
  every family; the factor cancels in `lambda`).
- `ntor` truncates the full-torus matrix at `|n| <= ntor NFP` before its family
  blocks are taken, so every family keeps the modes of its own with
  `|n| <= ntor NFP`. This keeps the union and `x` / `NFP - x` properties exact;
  family 0 keeps `|k| <= ntor` as before.
- `axisym=True` is already the special case of a one-node period grid:
  `D_zeta0 = 1j * n_mode_axisym * [[1]]` (assemble.py:263). The family matrix is
  its generalization to `n_zeta > 1` nodes per period.

### Which families, real or complex

- `D_{NFP-x}` is the complex conjugate of `D_x` (`D_full` is real), so `x` and
  `NFP - x` have the same eigenvalues (measured to 1.6e-15): only
  `x = 0 ... floor(NFP/2)` are needed.
- `x = 0` (periodic) is real symmetric and is exactly today's
  `domain="field_period"` matrix (1e-18). **Today's field-period runs solve only
  this family**: `n = 0, +-NFP, +-2 NFP, ...`. On the QH test case (`NFP = 4`, coarse
  grids) the most unstable full-torus mode was in family `x = 2` on every grid,
  and family 0 alone saw `-2.28e-3` against the full torus's `-5.78e-3`, or
  nothing unstable at all on one grid. These grids are not converged; the point
  is that a field-period run can miss the most unstable mode.
- `x = NFP/2` (even `NFP`): `D_x` is real, the `fourier_diffmat` formula with
  `tan` and `sin` swapped (`NFP * 0.5 (-1)^(i-j) / tan(pi (i-j)/n_zeta)` for odd
  `n_zeta`, `/ sin(...)` for even), so this family is a real symmetric problem.
- Every other `x` gives a complex Hermitian problem: the path `axisym=True`
  already uses (`operator_dtype`, conjugate transposes, complex Lanczos, the
  Hellmann-Feynman gradient `v^H dA v / v^H B v`).

### Cost against one full-torus solve (operation counts)

| NFP | families solved | real / complex | work, all families | memory, largest matrix | work vs one `x = 0` run |
|---|---|---|---|---|---|
| 2 | 0, 1 | 2 / 0 | 4x less | 4x less | 2x |
| 3 | 0, 1 | 1 / 1 | 5.4x less | 4.5x less | 5x |
| 4 | 0, 1, 2 | 2 / 1 | 10.7x less | 8x less | 6x |
| 5 | 0, 1, 2 | 1 / 2 | 13.9x less | 12.5x less | 9x |

One family alone costs `NFP^3 / 4` less than the full torus (complex) or `NFP^3`
less (real). When the physics points at one family, only that family is solved.
For the matrix-free JD path each family's operator applications cost about
`NFP/4` of a full-torus one (complex); iteration counts are not known.

## 2. Stellarator symmetry: even and odd displacements

### What it means

A stellarator-symmetric equilibrium (DESC `sym=True`) is unchanged under the
reflection `(rho, theta, zeta) -> (rho, -theta, -zeta)`. In the lab frame this is
a rotation by `pi` about the `R` axis at `phi = 0`. The PEST angles flip the same
way (`lambda` is odd). `B` and `J` change sign under it, but `dW` and `dK` are
quadratic in them, so the energy is unchanged.

The reflection acts on a displacement as `S`:

    (S xi)^rho(rho, theta, zeta)   =  + xi^rho(rho, -theta, -zeta)
    (S xi)^theta(rho, theta, zeta) =  - xi^theta(rho, -theta, -zeta)
    (S xi)^zeta(rho, theta, zeta)  =  - xi^zeta(rho, -theta, -zeta)

so with the code's unknowns the signs are `(+, -, -)` for
`(xi^rho/psi', upsilon, iota xi^zeta)`. Measured: `||S A S - A|| / ||A||` is
1e-15 with these signs and 0.26-0.91 with any other choice. For family
`x = NFP/2` (antiperiodic) the reflection carries an extra `-1` on every node
that wraps around the period (`k != 0`).

`S` commutes with `A` and `B` and `S^2 = I`, so every eigenvector can be chosen
even (`S xi = xi`) or odd (`S xi = -xi`), and the problem splits into two
independent problems. Solve both and keep the more unstable one.

### In real space

On agnimhd's uniform angle grids starting at 0, node `theta_j = 2 pi j / n_theta`
mirrors to `theta_{(n_theta - j) mod n_theta}`, and the same for `zeta` over the
grid's own period. Radial nodes and the Dirichlet rows (`keep_indices`) are
untouched by the reflection, so mirrored nodes are kept or dropped together.

An even (odd) vector is fixed by its values on half the nodes. With `P_e`, `P_o`
the matrices that build an even (odd) vector from those values (entries
`1/sqrt(2)` on a mirror pair, with the component's sign; `1` or `0` on a node that
is its own mirror, `theta = 0, pi`, `zeta = 0, L/2`), the two problems are

    (P_e^T A P_e) u = lambda (P_e^T B P_e) u,   and the same with P_o.

Assembly fits `assemble_rows`: it builds columns of `A` by applying the
matrix-free operator to unit vectors; applying it to the columns of `P_e`
instead gives `A P_e` with half the applications, and `P_e^T` sums mirror rows.

### Cost

About `N/2` unknowns each: work `2 (N/2)^3 = N^3 / 4` (4x less), memory per
problem 4x less (each problem solved alone). Needs `eq.sym = True`, nothing else.

## 3. Both together

The reflection sends family `x` to family `-x` (it flips `zeta`), so:

- Families `x = 0` and `x = NFP/2` are mapped to themselves and split into even
  and odd halves as in section 2.
- For any other `x`, `S` pairs `x` with `NFP - x`, which is already counted once.
  But `K = S` followed by complex conjugation maps family `x` to itself, is
  antiunitary and has `K^2 = I`. A Hermitian matrix commuting with such a `K`
  has a **real symmetric form of the same size**: in the basis of vectors with
  `K v = v`, built from mirror pairs as `(e_j + s e_j')/sqrt 2` and
  `i (e_j - s e_j')/sqrt 2`. So with stellarator symmetry no family needs complex
  arithmetic. Measured for `x = 1, 2` (in the `g` variable): `K` commutes with the
  family matrix to 1.3e-15, and the real form has the same eigenvalues to
  9.7e-16. In the variable of the exact `D_x` the mirror pairs that wrap around
  the period need the period phase; not tested yet.

Operation counts with both (stellarator-symmetric equilibrium):

| NFP | work, all families, vs full torus | memory, largest matrix |
|---|---|---|
| 2 | 16x less | 16x less |
| 3 | 21.6x less | 9x less |
| 4 | 42.7x less | 16x less |
| 5 | 55.6x less | 25x less |

## 4. Checks before any of it is used (summary; tests in section 6)

Each is a short test on a fixture with the matching symmetry (check `eq.sym` and
`NFP` of the fixtures first; a small `NFP = 2` or `3` stellarator-symmetric
case may have to be exported).

1. Commutators: `||S A - A S|| / ||A||` and the same for `T` at round-off on the
   assembled dense matrix. This fixes the signs of section 2 (done in the
   prototype, section 7).
2. Families: the lowest eigenvalues of the full-torus dense matrix equal the
   union of the family eigenvalues to round-off, with `NFP * n_zeta` nodes on the full torus.
3. Even/odd: the union of the even and odd spectra equals the full spectrum.
4. Real form of section 3: same eigenvalues as the complex family problem.
5. Gradients: the Hellmann-Feynman gradient of a family eigenvalue (complex
   eigenvector) against finite differences.
6. Measured on GPUs, with the user's setup and go: Patil QH (`NFP = 4`) full
   torus against the three families, wall time and memory; this replaces the
   operation counts above.

`jaxmg.potrs` (0.0.9): the library has a complex128 kernel and conjugates the
row-sharded matrix before the column-major call, so complex Hermitian input is
handled, and on four GPUs it agrees with a one-GPU solve to 1e-15. JD runs
every family: `basis.coarse_level(eq_coarse, family=x)` builds the coarse level
of family `x` (Fourier interpolation in zeta with the family's phase), and JD
matches the dense eigenvalue on families 1 and 2.

## 5. How it will be coded

Decision (user, 2026-10-03), once the prototype (section 7) and the tests of
section 6 pass: **the grid is always one field period and `domain` is removed.**
The full torus is the set of all families, so nothing is lost: a full-torus
mode is found in its family at `1/NFP` of the unknowns. Complex families cost
2x memory and about 4x flops against a real problem of the same size (a complex
multiply-add is four real ones), so one complex family against the full torus
is `NFP^2 / 2` less memory and `NFP^3 / 4` less work (NFP 4: 8x and 16x).

After the sign, `Basis` and JD-transfer PRs, and before the renames (so code
that families delete is not renamed first), as small PRs:

**F1. Families in `Basis`, `domain` removed.** `nodes_and_diffmat(nfp,
family=x)` returns the field-period nodes and the exact `D_x` of section 1
(complex for `x != 0, NFP/2`, real for those two); the nodes, the other
matrices and all weights do not depend on `x`. `family=0` is exactly today's
`domain="field_period"` run. As implemented, `family` is an argument of
`nodes_and_diffmat` (and of `from_desc`, `AgniStability`, `agnimhd solve
--family`), not a `Basis` field, so the caller builds both JD levels with the
same `x`; `basis.families(nfp)` lists `0 ... nfp // 2`. The `domain` keyword
and the full-torus node set go.

**F2. Complex operator decided by the data.** `operator_dtype` (assemble.py:53-74)
must return complex when `D_zeta` is complex, instead of reading
`AssemblyConfig.axisym`. This is not optional: the prototype showed that a
complex `D_zeta` with today's `operator_dtype` gives exactly the real part of the
matrix (5.1e-4 relative error) with only warnings, no error. Every reader of
`operator_dtype` changes with it: assemble.py:376 (dense assembly; the one line
the prototype had to bypass), assemble.py:1256 (`assemble_rows`),
objective.py:186 (JD start vector), objective.py:265 (the eigsh callback's
declared output, which would cast a complex eigenpair to real), multigpu.py:58.
`B` stays real (allocate it real at assemble.py:378). The matrix-free operator
needed no change (matches the dense family matrix to 5e-16). The `axisym` path
then becomes the one-node case `Basis(..., n_zeta=1, family=n)`: `axisym` and
`n_mode_axisym` and their branches (diffmat.py:196-205, assemble.py:203,
:261-263, :725-727, :916-917) can go, with the axisymmetric fixture value as the
check. Implemented: `operator_dtype(config, diffmat)` is complex for `axisym` or
a complex `D_zeta`, every reader passes the `DiffMat`, and `B` (with its
per-node blocks, also for `axisym`) is real. The `axisym` folding is left for
its own PR.

**F3. All families in one call.** `ag.solve(src, basis)` solves
`x = 0 ... floor(NFP/2)` by default and returns the most unstable `gamma^2`, its
family and its eigenvector; `families=x` (int or list) solves only those. The
Hellmann-Feynman gradient is the one of the returned family (complex path, as
`axisym`). For optimization (phase 2) `AgniStability` returns one `gamma^2` per
family as a vector, so a change of the most unstable family does not make the
objective jump. Until `ag.solve` exists, `agnimhd solve` does this on the
command line, and `AgniStability(eq, basis, family=x)` is one family's value.

**F4. Eigenvector on the full torus.** One function for plotting and output: on
period `p` the first period's values times `exp(2 pi i x p / NFP)` (the exact
`D_x` acts on the displacement itself, so no `exp(i x zeta)` factor).

**F5. Solvers.** `dense` and `jd` already run complex Hermitian operators
(`axisym`); the ring preconditioner and the coarse level are built from the
same complex operator, and so does `dense_mg`: `jaxmg.potrs` (0.0.9) accepts a
complex Hermitian matrix, and on four GPUs it agrees with a one-GPU solve to
1e-15.

**F6. Stellarator symmetry (later).** `Basis(..., parity="even" | "odd")` builds
the mirror-pair matrix `P`; dense assembly applies the matrix-free operator to
the columns of `P` through `assemble_rows`; `jd` works on `P^T A P` by gathering
and scattering mirror pairs. Then the real form of section 3 for complex
families.

Example of what F1-F3 buy on Patil QH (`NFP = 4`, arithmetic, not measured):
the full-torus problem at `40 x 48 x 64` has `4 x 90,624 = 362,496` unknowns,
too large for one node even with `dense_mg`. With families it is three solves of
`90,624` unknowns each, the size of today's `40 x 48 x 16` field-period run
(19 GB per GPU on four A100s, real); the complex family `x = 1` needs twice that
memory.

## 6. How it will be tested

Tests read like user scripts, on a small fixture with `NFP >= 2` (and
`sym=True` for the parity tests), dense solves, at odd and even `n_zeta` (with
the exact `D_x` both work; odd `n_zeta` does not avoid the Nyquist harmonic when
`NFP` is even). The QH test case (`NFP = 4`, `sym = True`) at `8 x 8 x 3` has
2,112 unknowns on the full torus and runs in seconds:

| test | what it shows |
|---|---|
| `test_full_torus_spectrum_is_the_union_of_the_families` | lowest eigenvalues of the full torus with `NFP * n_zeta` toroidal nodes equal the sorted union over `x = 0 ... NFP-1` on the `n_zeta`-node period grid |
| `test_families_x_and_nfp_minus_x_have_the_same_spectrum` | why only `x <= NFP/2` is solved |
| `test_family_zero_is_the_field_period_run` | `family=0` reproduces today's `domain="field_period"` values exactly |
| `test_axisymmetric_case_is_a_one_node_family` | the `axisym` fixture value from `Basis(n_zeta=1, family=n)` (only if F2 folds `axisym` in) |
| `test_solve_all_families_returns_the_most_unstable` | `ag.solve(..., families="all")` equals the full-torus lowest eigenvalue |
| `test_family_gradient_matches_finite_differences` | gradient of a complex family eigenvalue |
| `test_jd_matches_the_dense_eigenpair` | JD with its coarse level on a complex family |
| `test_reflection_commutes_with_the_operator` | `S A = A S` with the component signs of section 2 (F6) |
| `test_even_and_odd_spectra_make_the_full_spectrum` | the parity split (F6) |

Implemented with F1 and F2 (`tests/test_families.py`, on a one-period 8x8x3
fixture tiled into the full torus): the union and `x` / `NFP - x` tests, family 0
as the field-period matrix and the exported reference, the complex dtype, the
gradient of family 1, and `dense_mg` solving a complex family. JD on family 1:
`tests/test_objective.py`, `tests/test_adapters.py`. The `ag.solve`, `axisym`
and parity tests wait for their code.

Then measured on GPUs, with the user's setup and go: Patil QH full torus at a
size that still fits, against its families (eigenvalues, wall time, memory per
GPU).

**Comparison against the branch it replaces.** The families work lives on its
own branch, `toroidal-families`, made from the branch it will replace. Before it
is merged, both are run on the same cases (the benchmark list in roadmap.md):
the old branch on the full torus with `domain="full_torus"` and on one field
period, the new branch with all families. A comparison script reads both
outputs and reports, per case, the lowest `gamma^2` of each family against the
full-torus spectrum, the old field-period value against family 0, wall time and
peak memory. It merges only if every eigenvalue agrees to the tolerance of the
tests and nothing is slower at equal accuracy.

## 7. Prototype results (measured 2026-10-03)

agnimhd at 9a76004, unchanged except the dtype bypass in the script; CPU, dense
`eigvalsh`. Equilibrium `tests/data/AGNI_QH_lowres.h5` (`NFP = 4`,
`sym = True`). Grids `(n_rho, n_theta, n_zeta)` = (8,8,3), (8,8,5), (12,12,3),
(8,8,4); `N` = 2,112 to 4,896 on the full torus.

| check | result |
|---|---|
| shift `T` commutes with the full-torus `A` | 1.8e-15 to 2.5e-15 (DESC data periodic to 1e-12, its mapping tolerance); 1e-18 with exactly periodic data |
| family `x` assembled with the exact `D_x` vs the family block of the full-torus `A` | <= 2.5e-16, every `x`, odd and even `n_zeta` |
| full-torus spectrum vs union of the 4 family spectra | 1.8e-15 to 8.3e-15 of the spectral radius (shuffled-matrix floor 3e-15 to 1.2e-14); same number of unstable eigenvalues (16, 19, 42, 23) |
| lowest eigenvalue, full torus vs families | 2.2e-10 to 2.2e-9 relative (floor 1.2e-10 to 2.3e-10) |
| family `x = 0` vs today's field-period matrix | 1e-18 |
| `x` vs `NFP - x` | 1.6e-15 |
| shortcut `D_period + i x I` | wrong for `x = NFP/2` (all grids) and for `x != 0` at even `n_zeta`: lowest eigenvalue off by 14-26 %, unstable count 38 instead of 23 at (8,8,4) |
| reflection `S` with signs (+,-,-) | commutes to 1.2e-15 to 3.4e-15; other signs 0.26-0.91 |
| even + odd spectra vs whole | <= 9.6e-15 of the spectral radius; the most unstable mode is even on every grid |
| real form of complex families `x = 1, 2` | eigenvalues equal to <= 9.7e-16 |
| matrix-free operator with a complex family `D_zeta` | equals the dense family matrix to <= 5.1e-16, no code change |
| dtype | the one bypass: `operator_dtype` (see F2) |

Where the most unstable mode sat (coarse grids, not converged in `n_zeta`, so
only the pattern counts): family `x = 2` on all four grids, e.g. (8,8,3):
full torus `-5.777e-3`, family 2 `-5.777e-3`, families `+-1` `-4.012e-3`,
family 0 (today's field-period run) `-2.278e-3`.

## 8. Accelerations already in the code

- Component-major whitening without the permutation round trip (assemble.py).
- `dense_mg`: the dense matrix assembled by row blocks and factored over all GPUs
  of a node (docs/multigpu.md).
- `jd`: matrix-free Jacobi-Davidson with the ring preconditioner and coarse-level
  deflation.
