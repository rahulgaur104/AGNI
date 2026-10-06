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
| `radial` | `"gauss_radau_jacobi"` | radial nodes, `"gauss_radau_jacobi"` or `"lobatto"` |
| `alpha`, `beta` | `-0.35`, `-0.65` | Jacobi exponents of `"gauss_radau_jacobi"` |
| `automorphism` | `dict(eps=1e-2, x_0=0.6, m_1=2.5, m_2=3.0)` | staircase map of the radial nodes onto `[eps, 1]`; `None` for none. Three ARIES-CS drivers used `eps=5e-2` |
| `mpol`, `ntor` | `None`: every mode the grid holds | highest poloidal mode, and highest toroidal mode in field-period harmonics (toroidal `n` up to `ntor NFP` in magnitude), that the derivative matrices keep |

The defaults are the choices of the Patil QH benchmark runs; the call above is
that benchmark's grid. The test fixtures use
`Basis(24, 12, 8, radial="lobatto", automorphism=dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0))`.
Codes without an adapter take `nodes, diffmat = basis.nodes_and_diffmat(NFP, family=x)`
and evaluate on the tensor product of `nodes`. `basis.coarse()` is the
Jacobi-Davidson coarse level: `round(2 n_rho / 3)` radial points, the rest
unchanged.

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
| Zernike, coupled (rho, theta) | `basis.zernike_fourier_diffmat`, `AssemblyConfig(coupled_rt=True)` | regular at the axis by construction; dense operator, slower |

For Zernike, build the `DiffMat` yourself on the same nodes the equilibrium is
evaluated at. The Zernike radial degree defaults to `L = 2 (n_rho // 2 - 1)`, as
in DESC; the near-interpolating `2 (n_rho - 1)` gives a badly conditioned radial
pseudo-inverse.

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
  and NTOR; `basis.coarse()` keeps them.

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
`growth_rate(eq, diffmat, solver=..., coarse=from_desc(eq_desc, basis.coarse(), family=x))`,
with the fine level's family `x`; on the 24x12x8 test case it did not converge
without one. The coarse-to-fine
transfer still assumes Lobatto radial nodes, so use `radial="lobatto"` with
`"jd"` for now. For gradients set a residual
stop (`jd_tol=1e-5`): with the default Ritz-value stop the gradient was off by up
to 3 %.

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
