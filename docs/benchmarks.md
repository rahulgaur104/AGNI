# Benchmarks

Published cases at their published resolution. Each has a driver in
`benchmarks/`: edit the settings at its top and run it on a GPU node. CI runs
the same discretization at a small resolution in `tests/`.

| case | driver | resolution | reference `gamma^2` | solver | wall time | hardware | agnimhd |
|---|---|---|---|---|---|---|---|
| DSHAPE tokamak | `benchmarks/dshape.py` | Zernike 96x96x1, `MPOL = 4 n`, penalty 0.08 (`n = 1`) and 0.01 (`n = 2 ... 5`), `Gamma = 5/3` | `n` = 2: 2.770639e-4, 3: 2.791126e-4, 4: 1.785811e-4, 5: 6.459412e-5 | shift-invert Lanczos, dense LU | about 265 s per mode (driver, `n = 1, 2`), mostly the penalty projector's SVD | one 80 GB A100 | `n = 2 ... 5` equal. `n = 1`: 5.31e-6, the paper's point is not reproduced |

## DSHAPE tokamak

AGNI paper, arXiv:2608.01750v3, section 5.2 and figure 5. The equilibrium is
`tests/data/dshape_imax0.98_1608.h5`: a DESC DSHAPE tokamak with major radius
4.0 m, minor radius 1.198 m, volume-averaged beta 2.5 %, and rotational
transform from 0.978 on the axis (low shear inside `rho = 0.5`) to 0.359 at the
edge. Each toroidal mode `n` is solved on its own (one plane, `d/dphi = i n`),
with no mass density profile. The paper does not state the Zernike penalty,
but its runs used 0.01 for every `n`. The benchmark uses 0.08 for `n = 1`, where
0.01 leaves a spurious unstable mode (below), and 0.01 for `n = 2 ... 5`.

The reference values are those runs. With `B_N = |psi_rr| / a^2`, ion density
2.2e20 m^-3 and the proton mass they give the plotted AGNI growth rates, 2.24e4,
2.25e4, 1.80e4 and 1.08e4 rad/s for `n = 2 ... 5` (`n = 1`: about 5e2 rad/s,
`gamma^2` about 1.5e-7, read off the plot). The NIMSTELL points of the same
figure are 2.48e4, 2.40e4, 1.91e4 and 1.09e4 rad/s, and `n = 1` stable.

Measured with `benchmarks/dshape.py`'s settings on one A100 (`n = 1` and `n = 2`
by the driver itself, `n = 3, 4, 5` by the same discretization and solver in a
separate script):

| `n` | penalty | `gamma^2` | rad/s | paper |
|---|---|---|---|---|
| 1 | 0.08 | 5.310873e-6 | 3.10e3 | about 1.5e-7 (5.1e2 rad/s): not reproduced |
| 2 | 0.01 | 2.770639e-4 | 2.24e4 | 2.770639e-4 |
| 3 | 0.01 | 2.791125e-4 | 2.25e4 | 2.791126e-4 |
| 4 | 0.01 | 1.785811e-4 | 1.80e4 | 1.785811e-4 |
| 5 | 0.01 | 6.459412e-5 | 1.08e4 | 6.459412e-5 |

The eigenpair residuals are large, 4e2 to 3e3 for `n = 2 ... 5` and 1.1e6 for
`n = 1`, because the whitened penalty makes the matrix stiff. The `n = 2 ... 5`
values still equal the paper's ARPACK values to seven digits, and the `n = 1`
value lies between those at penalty 0.05 and 0.1 (table below).

### The penalty

The Zernike derivative matrices annihilate the nodal content the basis does not
represent, so that content has no field-line bending energy. The penalty
(`penalty` times the projector onto it) is all that holds it against
the pressure drive. `gamma^2` therefore falls as the penalty grows, toward the
limit where that content is excluded, the displacement restricted to the
Zernike space. Only that limit is independent of the penalty.

The penalty is added to the energy matrix before the whitening, without the
quadrature weights, so the strength a mode sees varies over the nodes and grows
with the node count. Per unit penalty, the diagonal of the whitened penalty
ranges from 0.16 to 8.8e6 at 16x48 and from 0.42 to 2.3e9 at 32x64. At 16x48 the
paper's 0.01 is far too weak: `n = 1` has 23 unstable modes up to
`gamma^2 = 7.0e-2`, and the most unstable one lies 99.4 % outside the Zernike
space. They disappear as the penalty grows (`n = 1`, `MPOL = 4`, lowest
eigenvalue of the dense matrix):

| penalty | 0.01 | 0.03 | 0.05 | 0.1 | 0.3 | 1 | 10 | limit |
|---|---|---|---|---|---|---|---|---|
| unstable modes | 23 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `gamma^2` | 7.0e-2 | 1.2e-4 | 5.9e-5 | 1.6e-5 | -1.32e-6 | -1.351e-6 | -1.357e-6 | -1.361e-6 |

At 96x96, `MPOL = 4 n`, shift-invert Lanczos on an A100:

| `n` | paper (penalty 0.01) | agnimhd, 0.01 | agnimhd, 0.05 | limit (penalty-free) |
|---|---|---|---|---|
| 1 | about 1.5e-7 | 3.708e-5 | 8.488e-6 | -3.90e-7 |
| 2 | 2.770639e-4 | 2.770639e-4 | 2.282387e-4 | 1.948e-4 |
| 3 | 2.791126e-4 | 2.791125e-4 | 2.513169e-4 | 2.245e-4 |
| 4 | 1.785811e-4 | 1.785811e-4 | 1.665235e-4 | 1.503e-4 |
| 5 | 6.459412e-5 | 6.459412e-5 | 5.970116e-5 | 5.091e-5 |

`n = 1` at 96x96 with more penalty: 5.31e-6 (0.08), 4.18e-6 (0.1), 8.85e-7
(0.3), -2.0e-7 (1).
With all poloidal modes (`MPOL = 47`): 2.99e-5 (0.01), 1.44e-5 (0.05), 5.68e-6
(limit).

The penalty-free limit against resolution (`MPOL = 4 n`, with the Zernike radial
degree `L = 2 (n_rho // 2 - 1)` in brackets):

| `n` | 16x48 (14) | 24x48 (22) | 32x64 (30) | 48x96 (46) | 64x96 (62) | 96x96 (94) |
|---|---|---|---|---|---|---|
| 1 | -1.36e-6 | -9.5e-7 | -1.17e-6 | -5.2e-7 | -5.5e-7 | -3.9e-7 |
| 2 | -5.4e-6 | -4.2e-6 | 8.9e-6 | 9.23e-5 | 1.433e-4 | 1.948e-4 |
| 3 | -1.27e-5 | -6.1e-6 | -2.3e-6 | 1.130e-4 | | 2.245e-4 |
| 4 | -2.11e-5 | -1.71e-5 | -6.5e-6 | 7.29e-5 | | 1.503e-4 |
| 5 | -2.80e-5 | -2.24e-5 | -1.15e-5 | -3.7e-6 | | 5.09e-5 |

In this limit `n = 1` is near marginal at every resolution, `n = 2 ... 5` become
unstable as the radial degree grows and keep growing at 96x96, and the paper's
ordering (3 and 2, then 4, then 5) holds at 96x96. The limit was computed by
restricting the displacement to the Zernike space with a script outside the
package.

What CI checks (`tests/test_dshape.py`, 16x48, about 25 s): at this resolution
every unstable eigenvalue at the benchmark's penalties is a penalty artifact (the
limit is stable for every `n`, and `n = 1` at 0.08 gives 2.65e-5). So it checks

- code agreement, not physics: `n = 2 ... 5` at penalty 0.01 equal DESC's AGNI
  on the same nodes and settings (measured to 2e-11).
- `n = 1` with penalty 1 and 10: near marginal, the same to 1 %, within 1 % of
  the limit.

The paper's verdict, `n = 2 ... 5` unstable, holds in the limit only from 48x96
for `n = 2, 3, 4` and only at 96x96 for `n = 5` (-3.7e-6 at 48x96). A 48x96 solve
is already a 13,632-row complex dense matrix (3 GB). Its time on a CI runner was
not measured. The verdict is left to the benchmark.

## Zernike JD against the dense solve

`benchmarks/zernike_jd_vs_dense.py` solves two cases with `"jd"` and with the
dense solve, at penalty 1e-2 to 1e4 and `sigma = 1e-4`:

- the DSHAPE tokamak at 96x96, `n = 2`, `MPOL 8`, against `"dense"` on one
  80 GB A100;
- the QH stellarator of `tests/data` at 48x48x16, `MPOL 8`, `NTOR 2` (the AGNI
  paper's truncation), families 0 and 1, against `"dense_mg"` on the four 80 GB
  A100s of one node (jaxmg 1.2, tile 256).

Each solve appends one line to a JSON file, and `--report` compares the two
solvers. `gamma^2` from JD:

| case | 1e-2 | 1 | 1e2 | 1e3 | 1e4 |
|---|---|---|---|---|---|
| tokamak | 2.7706393e-4 | 2.0212496e-4 | 1.9801501e-4 | 1.9574976e-4 | 1.9518348e-4 |
| QH, family 0 | 7.1817015e-5 | 4.5760662e-5 | 4.4290857e-5 | 4.4240081e-5 | 4.4232874e-5 |
| QH, family 1 | 6.4876180e-5 | 4.6091993e-5 | 4.5218483e-5 | 4.5173817e-5 | 4.5165786e-5 |

Relative difference, JD against dense:

| case | 1e-2 | 1 | 1e2 | 1e3 | 1e4 |
|---|---|---|---|---|---|
| tokamak | -2.5e-10 | -9.2e-11 | -8.9e-12 | -1.8e-10 | +1.4e-9 |
| QH, family 0 | -3.9e-11 | -1.4e-10 | -2.4e-10 | -4.0e-10 | +5.2e-9 |
| QH, family 1 | -1.7e-11 | -1.3e-10 | +4.3e-11 | +6.1e-9 | +2.4e-7 |

The tokamak value at penalty 1e-2 is the paper's. As the penalty grows,
`gamma^2` falls by 30 % on the tokamak and by 38 % and 30 % on the two QH
families. From 1e3 to 1e4 it changes by 0.3 % on the tokamak and by 0.02 % on
the QH families.

The largest difference, +2.4e-7 at QH family 1 and penalty 1e4, has JD above a
dense solve whose last two iterations differ by 1e-12. At penalty 1e4 the
Rayleigh quotient of a fixed vector varies by 1e-7 to 3e-6 with the order of
its float64 evaluation (measured at 32x32x12).

JD runs at `sigma = 1e-4` in every case. One mode block reaches the shift ladder
([Theory](theory.md#eigensolve)): the `(0, 0)` block of QH family 0, which
factors at `10 sigma` at penalty 1e3 and at `100 sigma` at 1e4.

| case | JD per solve | JD peak memory | dense per solve | dense peak memory |
|---|---|---|---|---|
| tokamak | 57 to 64 s | 7.3 GB | 35 s | 26.2 GB |
| QH, family 0 | 72 to 160 s | 2.5 GB | 181 to 207 s | 28.8 GB per GPU |
| QH, family 1 | 70 to 109 s | 4.9 GB | 406 to 536 s | 58.4 GB per GPU |

`dense_mg` stops after 6 to 8 iterations, on a `gamma^2` change below 1e-8.

Before #183 the penalty projector was formed as `I - V V^H`, which is
indefinite in floating point ([Theory](theory.md#penalty)). JD and
dense then drifted apart as the penalty grew, to 2.4e-4 at QH family 1 and
penalty 1e4. At QH family 0 and penalty 1e3, `dense_mg` returned 3.67e-5 instead
of 4.42e-5. On the tokamak both solvers shared the error: they agreed to 6e-8,
while `gamma^2` was too large by 5.2e-7 at penalty 1e2, 4.2e-6 at 1e3 and 4.0e-5
at 1e4.

## Jacobi-Davidson on the QH case

The production setting of [Choosing options](options.md#jacobi-davidson) was measured with the DESC
implementation this solver comes from on the Patil QH case (Gauss-Radau-Jacobi,
MPOL 8, NTOR 2): coarse 48x20x12 for 48x48x16, `k_defl` 50, `sigma = 1e-3`, 200
CG steps per correction, stop at residual 1e-3. At 48x48x16 it reached
`gamma^2 = 1.4401664e-4`, the value of a 1000-round run, after 82,200 operator
applications. At 32x48x16 (coarse 32x20x12) it matched the dense
`gamma^2 = 1.43848936e-4` to all printed digits after 52,200. Other production
coarse levels: 24x48x24 for 24x48x32 (MPOL 16, NTOR 4) and 24x27x19 for
24x48x48 (MPOL 13, NTOR 9). The defaults (`jd_tol = 0`, `jd_theta_tol = 1e-8`,
`jd_inner = 100`) are DESC's. For gradients stop on the residual.

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
not leave that subspace. A random vector of norm 1e-6 added to the start gave
the right mode in 276 outer iterations. Three of the four production coarse
levels above have more angular nodes than the default.
