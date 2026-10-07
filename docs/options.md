# Choosing options

All of these are convergence choices. Scan them at low resolution first, then
make one production run.

## Choosing the basis

One `Basis` places the nodes and builds the derivative matrices on them:

```python
basis = agnimhd.Basis(40, 48, 16, mpol=8, ntor=2)
eq, diffmat = agnimhd.from_desc("eq.h5", basis)          # toroidal family 0
```

| keyword | default | meaning |
|---|---|---|
| `n_rho, n_theta, n_zeta` | positional | PEST grid resolution; `n_zeta` toroidal nodes on one field period |
| `radial` | `"gauss_radau_jacobi"` | radial nodes, `"gauss_radau_jacobi"`, `"lobatto"` or `"zernike"` ([Radial basis](#radial-basis)) |
| `alpha`, `beta` | `-0.35`, `-0.65` | Jacobi exponents of `"gauss_radau_jacobi"` |
| `automorphism` | `dict(eps=1e-2, x_0=0.6, m_1=2.5, m_2=3.0)` | staircase map of the radial nodes onto `[eps, 1]`; `None` for none. Three ARIES-CS drivers used `eps=5e-2` |
| `mpol`, `ntor` | `None`: every mode the grid holds | highest poloidal mode, and highest toroidal mode in field-period harmonics (toroidal `n` up to `ntor NFP` in magnitude), that the derivative matrices keep |
| `zernike_penalty` | `0.05` | `radial="zernike"` only: penalty on the nodal content the Zernike basis does not hold |

The defaults are the choices of the Patil QH benchmark runs; the call above is
that benchmark's grid. The test fixtures use
`Basis(24, 12, 8, radial="lobatto", automorphism=dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0))`.
Codes without an adapter take `nodes, diffmat = basis.nodes_and_diffmat(NFP, family=x)`
and evaluate on the tensor product of `nodes`. `basis.coarse()` is the
Jacobi-Davidson coarse level: fewer angular nodes, the rest unchanged;
`basis.coarse_level(eq_coarse, family)` pairs it with the basis.

## Toroidal mode families

The toroidal nodes always span one field period, `[0, 2 pi / NFP)`, and every
toroidal mode number `n` is still solved for. The equilibrium repeats every
field period, so the modes split into `NFP` independent families: family `x`
holds `n = x + k NFP` (`k` any integer). On field period `p` a family-`x` mode is
the first period's displacement times `exp(2 pi i x p / NFP)`, so it is known on
the whole torus once it is known on one period, and each family is solved on
the one-period grid, at `1/NFP` of the unknowns of the full torus. Together the
families have exactly the eigenvalues of the full torus with `NFP * n_zeta`
toroidal nodes (tested to round-off).

```python
for x in basis.families(eq.NFP):                    # x = 0 ... NFP // 2
    _, diffmat = basis.nodes_and_diffmat(eq.NFP, family=x)
    print(x, agnimhd.growth_rate(eq, diffmat))       # most unstable: the largest
```

- Family `NFP - x` has the eigenvalues of family `x`, so `basis.families(NFP)`
  lists only `x = 0 ... NFP // 2`.
- Families 0 and `NFP / 2` are real symmetric problems; the others are complex
  Hermitian: 16 bytes per matrix entry instead of 8, and four real
  multiply-adds per complex one. `"dense_mg"` solves only the real ones.
- The geometry does not depend on the family: evaluate once (`from_desc` returns
  family 0) and take each family's `DiffMat` from the basis.
  `from_desc(eq, basis, family=x)`, `AgniStability(eq, basis, family=x)` and
  `agnimhd solve --family x` take one family; `agnimhd solve` without it solves
  every family and names the most unstable.

Before families, a field-period run (`domain="field_period"`) solved family 0
only, `n = 0, +-NFP, +-2 NFP, ...`, and the other families needed the full
torus (`domain="full_torus"`), `NFP` times the unknowns. Measured on the QH test
case (`NFP = 4`) at 8x8x3: family 0 gives `gamma^2 = 2.28e-3`, family 1
`4.01e-3`, family 2 `5.78e-3`, the full torus's value.

## Grid resolution

`n_rho x n_theta x n_zeta` on the PEST grid; 24x12x8 is a reasonable start. The
eigenvalue converges fastest radially, and the placement of the radial nodes
changes it more than added resolution does: put the clustering point `x_0` of
the `automorphism` where the mode peaks (usually the resonant surface).
The innermost node sits at `rho = eps`, with `eps` between 1e-3 and 1e-2,
because several coefficients are singular on axis.

The toroidal nodes span one field period; every toroidal mode family is solved
on them ([Toroidal mode families](#toroidal-mode-families)).

## Radial basis

| basis | how to build | notes |
|---|---|---|
| Legendre-Lobatto with staircase map | `Basis(radial="lobatto")` | the test fixtures and the paper's convergence study |
| Gauss-Radau-Jacobi (GJ) | `Basis(radial="gauss_radau_jacobi")`, the default | no node on the axis; production choice for the QH runs with `alpha = -0.35`, `beta = -0.65` |
| Zernike, coupled (rho, theta) | `Basis(radial="zernike")` | regular at the axis by construction; dense operator, slower; the AGNI paper's tokamak runs |

The Zernike basis takes Gauss-Jacobi radial nodes inside `(0, 1)` and no
`automorphism`; its radial degree is `L = 2 (n_rho // 2 - 1)`, as in DESC (the
near-interpolating `2 (n_rho - 1)` gives a badly conditioned radial
pseudo-inverse). Its `D_rho` and `D_theta` act on `(rho, theta)` together, so the
assembly needs the per-direction counts:

```python
basis = agnimhd.Basis(96, 96, 1, radial="zernike", mpol=4 * n, zernike_penalty=0.01)
eq, diffmat = agnimhd.from_desc("dshape.h5", basis)
config = agnimhd.AssemblyConfig(axisym=True, n_mode_axisym=n, coupled_rt=True,
                                n_rho_coupled=96, n_theta_coupled=96)
```

These are the settings of the paper's DSHAPE tokamak ([Benchmarks](benchmarks.md)).

The derivative matrices annihilate the nodal content the Zernike basis does not
represent, and `zernike_penalty` is the only energy that content has against
the pressure drive: too small a penalty gives spurious unstable modes, and every
eigenvalue falls as the penalty grows. The penalty's strength after the
whitening grows with the node count, so a value that suffices at one resolution
can fail at a coarser one: the DSHAPE `n = 1` mode at 16x48 has 23 unstable
eigenvalues up to `gamma^2 = 7e-2` at 0.01, none from 0.3 on. Check that
`gamma^2` stays the same when the penalty is raised tenfold. A large penalty
makes the matrix stiff: solve with shift-invert (`"eigsh"`, `"jax_lanczos"`);
the dense GPU eigensolver returned wrong small eigenvalues at 96x96.

## MPOL and NTOR

`Basis(..., mpol=MPOL, ntor=NTOR)` keeps poloidal and toroidal modes up to MPOL
and NTOR (`basis.fourier_diffmat_truncated`). The grid must hold them,
`MPOL <= (n_theta - 1) // 2` and `NTOR <= (n_zeta - 1) // 2`, and a larger cap
(also `M` of `zernike_fourier_diffmat`) raises a `ValueError`.

- For a comparison with another code, cap them deliberately so one mode
  dominates. The paper used `m <= 8`, `n <= 8`; in its NIMSTELL benchmark the
  remaining mode is `m = n = 4`, an interchange mode near `iota = 1.02` where
  the shear vanishes.
- For a physics answer, raise them until `lambda` stops changing.
- NTOR counts field-period harmonics: every family keeps `|n| <= NTOR NFP`,
  the modes a full torus truncated at `NTOR NFP` keeps.
- With the two-level `"jd"` solver, coarse and fine levels need the same MPOL
  and NTOR; `basis.coarse()` keeps them and requires both to be set.

## Stellarator symmetry (`AssemblyConfig.parity`)

On a stellarator-symmetric equilibrium the reflection
`(rho, theta, zeta) -> (rho, -theta, -zeta)`, with the displacement components
`(xi^rho, xi^theta, xi^zeta)` taking the signs `(+, -, -)`, commutes with the
operator of the real families (0 and `NFP / 2`; measured 2.4e-15 on the
24x12x8 case). The operator then splits into an even and an odd block of about
half the size each. `AssemblyConfig(parity="even")` or `"odd"` solves one
block with `"eigsh"` or `"jax_lanczos"`; the eigenvector comes back on the
usual kept degrees of freedom. The most unstable mode is in one of the blocks,
so solve both and take the larger `gamma^2`: on the 24x12x8 case the even
block holds it (1.3376e-4, the full problem's lowest eigenvalue, 1.6e-9
apart) and the odd block holds the second mode (6.2454e-5). The factorization
of a block costs an eighth of the full one, so the pair costs a quarter, with
a quarter of the memory. The equilibrium is checked at every solve and a
non-symmetric one is refused. Not yet with `"jd"`.

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

`"jd"` is matrix-free Jacobi-Davidson with the ring block preconditioner,
deflated by the softest modes of its coarse level, which it requires (it raises
without one; without one it stalled on a near-zero mode of the 24x12x8 test
case). The coarse level is the same equilibrium on the same radial nodes and
with the same `mpol` and `ntor`, on fewer angular nodes:
`basis.coarse(n_theta, n_zeta)`, by default `2 mpol + 1` and `2 ntor + 1`.
Its modes reach the fine level by Fourier interpolation in theta and zeta (with
the family's phase in zeta), exact for every mode both levels keep. The coarse
dense matrix is assembled 256 rows at a time and the ring blocks of both levels
`ring_batch` (24) rings at a time. For a 24x12x8 coarse level (6,720 unknowns,
a 0.36 GB dense matrix) the assembly raised the peak resident memory by 1.3 GB
(1.9 GB assembled at once), the whole coarse build, with its generalized
eigensolve, by 2.2 GB either way.

```python
basis = agnimhd.Basis(48, 48, 16, mpol=8, ntor=2)
eq, diffmat, coarse = agnimhd.from_desc("eq.h5", basis, family=x,
                                        coarse=basis.coarse(20, 12))
jd = agnimhd.SolverConfig(eigensolver="jd", sigma=1e-3, jd_tol=1e-3,
                          jd_theta_tol=0.0, jd_inner=200, jd_outer=1000)
gamma2, v, residual = agnimhd.eigenpair(eq, diffmat, solver=jd, coarse=coarse)
```

`agnimhd solve eq.h5 --res R,T,Z --mpol M --ntor N --eigensolver jd [--coarse T,Z]`
does the same for a DESC file. Other codes evaluate the equilibrium on
`basis.coarse(...).nodes_and_diffmat(NFP)[0]` and pass
`coarse=basis.coarse_level(eq_coarse, family=x)`. `agnimhd.objective.coarse_space`
returns the coarse start vector and deflation space `(v0, Z)`, which can be
built in a separate run (e.g. on a CPU node) and passed as `coarse=(v0, Z)`.

The call above is the production setting, measured with the DESC
implementation this solver comes from on the Patil QH case (Gauss-Radau-Jacobi,
MPOL 8, NTOR 2): coarse 48x20x12 for 48x48x16, `k_defl` 50, `sigma = 1e-3`, 200
CG steps per correction, stop at residual 1e-3. At 48x48x16 it reached
`gamma^2 = 1.4401664e-4`, the value of a 1000-round run, after 82,200 operator
applications; at 32x48x16 (coarse 32x20x12) it matched the dense
`gamma^2 = 1.43848936e-4` to all printed digits after 52,200. Other production
coarse levels: 24x48x24 for 24x48x32 (MPOL 16, NTOR 4) and 24x27x19 for
24x48x48 (MPOL 13, NTOR 9). The defaults (`jd_tol = 0`, `jd_theta_tol = 1e-8`,
`jd_inner = 100`) are DESC's; for gradients stop on the residual.

Measured on the 24x12x8 test case, `ntor = 1` (`sigma = 1.3 gamma^2`, stop at
residual 1e-4):

| radial nodes | family | `mpol` | coarse | outer iterations | `gamma^2` vs dense |
|---|---|---|---|---|---|
| Lobatto | 0 | 4 | 24x12x4 | 121 | 3.4e-10 |
| Lobatto | 1 | 2 | 24x6x4 | 127 | 2.5e-9 |
| Lobatto | 1 | 5 | 24x12x4 | 103 | 5.2e-10 |
| Lobatto | 2 | 5 | 24x12x4 | 82 | 2.2e-10 |
| Gauss-Radau-Jacobi | 0 | 5 | 24x12x6 | 360 | 7.2e-9 |
| Gauss-Radau-Jacobi | 1 | 5 | 24x11x3 (default) | 237 | 3.5e-9 |
| Lobatto | 0 | 5 | 24x12x4 | 122 | second eigenvalue: 2.274e-4 for 3.963e-4 |
| Gauss-Radau-Jacobi | 0 | 5 | 24x11x3 (default) | 260 | second eigenvalue: 8.46e-5 for 3.54e-4 |

Check the residual `eigenpair` returns, and check the eigenvalue against a
second coarse level: in the last two rows JD converged, residual below 1e-4, to
the second eigenvalue. In the Lobatto case its start vector, the softest coarse
mode, overlapped the fine mode 1 by 1.7e-7 (mode 2: 0.50) and the iteration did
not leave that subspace; a random vector of norm 1e-6 added to the start gave
the right mode in 276 outer iterations. Three of the four production coarse
levels above have more angular nodes than the default.

## Shift (`sigma`)

`sigma` is in the convention of the returned value: it must lie above the
largest `gamma^2`, and not too far above it for the fixed-budget solvers
(`"jax_lanczos"`, `"jd"`). The solvers shift `A` by `-sigma`. On the 24x12x8
test case (`gamma^2 = 1.34e-4`) a 50-step Lanczos returned the wrong mode at
`sigma = 0.1` and at `1e-2`, and the right one at `sigma = 1e-3`. At
`sigma = 0.1`, 200 steps recover the eigenvalue at four times the cost, but
the vector is still not converged.

Procedure: estimate `gamma^2` from a cheap low-resolution run, then set
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
