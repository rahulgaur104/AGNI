# Choosing options

Three settings matter in every run, the resolution, the eigensolver and the
shift. Set them first and check the result as described below. The defaults of
everything else suit most cases. The later sections list those settings, first
the ones some cases need and then the ones for experts.

## Every run

### Resolution

```python
basis = agnimhd.Basis(24, 12, 8)          # n_rho, n_theta, n_zeta
```

The basis places `n_rho` radial, `n_theta` poloidal and `n_zeta` toroidal nodes,
the toroidal ones on one field period. Each term of the energy becomes a
quadrature over these nodes with spectral differentiation matrices (equations
(26) to (28) of the paper). The matrix has 3N rows, one for each component of
ξ at each of the N = n<sub>ρ</sub>n<sub>θ</sub>n<sub>ζ</sub> nodes.

Start at 24×12×8 and raise each number until γ² stops changing. The
eigenvalue usually converges fastest radially. A tokamak has `n_zeta = 1`
([Tokamaks](tokamaks.md)). The command line takes the same numbers as
`--res 24,12,8`.

The dense matrix holds (3N)² numbers of 8 bytes, or 16 for a complex family.
That is 0.4 GB at 24×12×8 and about 100 GB at 48×48×16, which decides the
eigensolver.

### Eigensolver

```python
gamma2, v, residual = agnimhd.solve("eq.h5", basis, "dense")
```

| `eigensolver` | hardware | memory | speed | largest case run |
|---|---|---|---|---|
| `"dense"` | one GPU, or a CPU | the dense matrix, at most 80 GB on an A100 | fast, about 6× faster on an A100 than `"eigsh"` on a 128-core CPU node (24×40×12, 26 s against 155 s) | 24×40×12 on one A100 |
| `"dense_mg"` | the GPUs of one or more nodes | the dense matrix split over the GPUs | fast for large matrices | 80×48×16, 182,784 unknowns, in 8.5 min on four A100s ([Several GPUs](multigpu.md)) |
| `"jd"` | one GPU, or a CPU | matrix-free, a few GB | moderate, 2× slower than `"dense"` where the matrix fits (96×96 tokamak, 60 s against 35 s), 4 to 5× faster than `"dense_mg"` (48×48×16 QH, 70 to 109 s against 406 to 536 s) and 9 to 12× faster than `"eigsh"` | 40×48×16 in 141 s on one A100 |
| `"eigsh"` (default) | CPU only (SciPy ARPACK), no GPU | the dense matrix in the memory of the node, 450 GB on a Perlmutter CPU node, more than one GPU holds | slow, 1130 s at 32×48×16 and 1670 s at 40×48×16 on 128 cores | 40×48×16 |

The CPU and GPU times are from table 4 of the paper, the others from
[Benchmarks](benchmarks.md#zernike-jd-against-the-dense-solve) and
[Several GPUs](multigpu.md). Use `"dense"` when the matrix fits on one GPU,
`"dense_mg"` when it fits on several, and `"jd"` beyond that. On a node without
GPUs, `"eigsh"` uses the larger memory of the CPU node to hold a larger matrix.

`"jd"` never forms the matrix. It needs `mpol` and `ntor` on the basis
([below](#poloidal-and-toroidal-truncation)) and a shift close to γ².

```python
basis = agnimhd.Basis(48, 48, 16, mpol=8, ntor=2)
gamma2, v, residual = agnimhd.solve("eq.h5", basis, "jd", sigma=1e-3)
```

The command line takes `--eigensolver jd --mpol 8 --ntor 2 --sigma 1e-3`.

### Shift

Every eigensolver finds the eigenvalue closest to a shift σ, by iterating
with (σI − A)⁻¹ (equation (47) of the paper, with A the matrix of
equation (46)). σ is in the units of γ² and must lie above the
largest γ². Below it the solve converges to the wrong mode.

`"dense"` and `"jd"` stop after a fixed number of iterations, so for them
σ must also be close to γ². Far above it the modes separate too
slowly and the solve can return the wrong mode. The default, 0.1, suits
`"eigsh"`, which iterates to a tolerance. For the other solvers estimate
γ² with a solve at low resolution and set `sigma` to 1.3 to 2.5 times
that value.

```python
gamma2, v, residual = agnimhd.solve("eq.h5", basis, "dense", sigma=2e-4)
```

### Checking a result

Every solve returns the relative residual
‖Av − γ²v‖ / (|γ²| ‖v‖) of its eigenvector v. A converged
mode has a small residual, and a wrong mode a large one, often above 1. Check the
residual, not only γ².

γ² is reliable only well above the roundoff of the matrix. Values with
|γ²| below about 1e-10 mean marginal stability (section 4.3 of the
paper). Raise the resolution until γ² stops changing before trusting its
sign.

## Settings some cases need

### Radial basis

```python
basis = agnimhd.Basis(24, 12, 8, radial="lobatto")
```

| `radial` | nodes | at the magnetic axis | suits |
|---|---|---|---|
| `"zernike"` (default) | Zernike polynomials, coupling ρ and θ | regular | any mode, including one that reaches the axis |
| `"gauss_radau_jacobi"` | Gauss-Radau-Jacobi, clustered around `x_0`, the paper's default | no condition imposed | a mode away from the axis |
| `"lobatto"` | Legendre-Lobatto, clustered around `x_0` | no condition imposed | a mode away from the axis |

The Zernike basis is regular at the axis by construction, so it suits any mode,
and it is the default. Gauss-Radau-Jacobi and Lobatto impose no condition at the
axis and work well for a mode that does not peak near it. Their nodes are mapped
onto `[eps, 1]` (equation (29) of the paper) and clustered around a radius
`x_0`, which should be where the mode peaks, usually a resonant surface. The
paper uses Gauss-Radau-Jacobi (section 3.2).

A saved `EquilibriumData` was evaluated on the radial nodes of the basis it was
exported with, and must be solved with that basis
([Interface](interface.md#saving-and-loading)).

### Poloidal and toroidal truncation

```python
basis = agnimhd.Basis(48, 48, 16, mpol=8, ntor=2)
```

`mpol` is the highest poloidal mode number the derivative matrices keep, and
`ntor` the highest toroidal one in units of NFP. They must satisfy
`mpol <= (n_theta - 1) // 2` and `ntor <= (n_zeta - 1) // 2`. By default every
mode the grid holds is kept. For a physics answer raise them until γ²
stops changing. For a comparison with another code, cap them as that code does.
The reference cases use the paper's truncation, `mpol = 8` and `ntor = 2` for
the QH stellarator and `mpol = 4 n` for the DSHAPE tokamak. `"jd"` needs both
set.

### Penalty

```python
basis = agnimhd.Basis(24, 12, 8, penalty=0.5)
```

The nodes can hold displacements that the basis does not represent, on the
Zernike basis those off the Zernike polynomials. On the Gauss-Radau-Jacobi and
Lobatto bases `"eigsh"` and `"dense"` solve only on the represented
displacements (the restriction, `restrict` below), and the penalty does not
enter. Elsewhere, on the Zernike basis and with `"jd"`, `"dense_mg"` or a
parity, the penalty gives the unrepresented displacements a positive energy, so
the most unstable mode is made of the represented ones. Above a penalty of about
1, γ² barely depends on it. From 1e2 to 1e4 it changes by 1.4 % on the
DSHAPE tokamak and by 0.1 % on the QH stellarator, and every eigensolver agrees
to 2.4e-7 or better up to 1e4
([Benchmarks](benchmarks.md#zernike-jd-against-the-dense-solve)). The default is
0.05, and 1.0 for a tokamak. How the penalty enters the energy is described in
[Theory](theory.md#penalty).

### Ratio of specific heats and incompressibility

```python
config = agnimhd.AssemblyConfig(gamma=1000.0)
gamma2, v, residual = agnimhd.solve("eq.h5", basis, assembly=config)
```

`gamma` is Γ in the compression term Γp|∇·ξ|² of the potential energy δW<sub>p</sub>
(equation (16) of the paper), and its default is 5/3. A larger Γ raises the
sound speed c<sub>s</sub> = √(Γp / (Mn₀)) and penalizes compression, so the most unstable mode
approaches the incompressible one, ∇·ξ → 0. It approaches it from the unstable
side, which keeps γ² well above the roundoff of the matrix. At Γ = 1000 the
eigenvalue and the eigenfunction of the QH case agree closely with those of the
exactly incompressible problem (section 6.2 and figure 7 of the paper). This is
how agnimhd imposes incompressibility, and it stays differentiable, so it serves
optimization as well. DESC's own stability objective defaults to Γ = 0, so state
the value with every result.

### Families, symmetry and toroidal mode number

A stellarator is solved one toroidal mode family at a time, and a
stellarator-symmetric one in an even and an odd block
([Stellarators](stellarators.md)). A tokamak is solved one toroidal mode number
at a time ([Tokamaks](tokamaks.md)).

## Settings for experts

### Jacobi-Davidson

`"jd"` follows Algorithm 2 of the paper. Its correction equation (equation (50))
is solved by preconditioned conjugate gradients. The preconditioner has one
block per poloidal ring on the Lobatto and Gauss-Radau-Jacobi bases (equation
(53)) and one block per Fourier mode on the Zernike basis
([Theory](theory.md#eigensolve)), and it is corrected by the softest modes of a
coarse level (equations (56) and (57)).

| `SolverConfig` field | paper (table 2) | default | meaning |
|---|---|---|---|
| `jd_outer` | n<sub>out</sub> | 200 | largest number of outer iterations |
| `jd_inner` | m<sub>in</sub> | 100 | conjugate gradient steps per correction |
| `jd_maxdim` | m<sub>max</sub> | 60 | search directions before a restart |
| `jd_keep` | k<sub>keep</sub> | 10 | vectors kept at a restart |
| `jd_theta_tol` | τ<sub>λ</sub> | 1e-8 | stop when γ² changes by less than this, relative |
| `jd_tol` | | 0, off | stop when the residual is below this, the better stop for gradients |
| `k_defl` | k<sub>defl</sub> | 50 | coarse modes in the preconditioner |
| `coarse_num_matvecs` | | 100 | Lanczos steps of the coarse eigensolve |
| `ring_batch` | | 24 | rings built at once, lower it if the build runs out of memory |

The coarse level is the same equilibrium on the same radial nodes with the same
`mpol` and `ntor`, on fewer angular nodes, `basis.coarse(n_theta, n_zeta)`. Its
default is 2`mpol` + 1 poloidal and 2`ntor` + 1 toroidal nodes, and the
command line sets it with `--coarse n_theta,n_zeta`. The production runs on the
QH case use

```python
basis = agnimhd.Basis(48, 48, 16, radial="gauss_radau_jacobi", mpol=8, ntor=2)
eq, diffmat, coarse = agnimhd.from_desc("eq.h5", basis, coarse=basis.coarse(20, 12))
jd = agnimhd.SolverConfig(eigensolver="jd", sigma=1e-3, jd_tol=1e-3,
                          jd_theta_tol=0.0, jd_inner=200, jd_outer=1000)
gamma2, v, residual = agnimhd.eigenpair(eq, diffmat, solver=jd, coarse=coarse)
```

and their measurements are in
[Benchmarks](benchmarks.md#jacobi-davidson-on-the-qh-case).
`agnimhd.objective.coarse_space` builds the coarse start vector and modes
`(v0, Z)`, which can be computed in a separate run and passed as
`coarse=(v0, Z)`.

### Dense shift-invert

| `SolverConfig` field | default | meaning |
|---|---|---|
| `num_matvecs` | 50 | Lanczos steps of `"dense"`, n<sub>mv</sub> of Algorithm 1 |
| `factor` | `"cholesky"` | factorization of the shifted matrix. `"lu"` factors any shift that is not an eigenvalue, at twice the cost of Cholesky |
| `sigma_mode` | `"fixed"` | `"adapt"` solves once, sets `sigma` to `sigma_factor` times that γ² and solves again |
| `sigma_factor` | 2.5 | the factor of `"adapt"` |
| `eigsh_tol` | 1e-8 | ARPACK tolerance of `"eigsh"` |
| `seed` | 0 | seed of the Lanczos start vector |

### Several GPUs

`mg_tile`, `mg_block`, `mg_iters`, `mg_tol` and `mg_theta_tol` set the solver
`"dense_mg"` and are described in [Several GPUs](multigpu.md).

### Radial nodes of the Lobatto and Gauss-Radau-Jacobi bases

| `Basis` keyword | default | meaning |
|---|---|---|
| `automorphism` | `dict(eps=1e-2, x_0=0.6, m_1=2.5, m_2=3.0)` | the map that clusters the radial nodes around `x_0`, starting at ρ = `eps` (appendix A.2 of the paper) |
| `alpha`, `beta` | -0.35, -0.65 | Jacobi exponents of `"gauss_radau_jacobi"` |
| `restrict` | chosen by the solver | `True` solves only on the displacements the derivative matrices represent, `False` uses the penalty instead |

`eps` lies between 1e-3 and 1e-2, since several coefficients are singular at
the axis (section 4.2 of the paper).
