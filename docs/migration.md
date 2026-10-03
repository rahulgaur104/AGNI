# Migrating from AGNI in DESC

AGNI in DESC is [PR #1893](https://github.com/PlasmaControl/DESC/pull/1893),
built on the differentiation matrices of PR #1789.

Before, the eigenvalue came from DESC's compute machinery:

```python
data = eq.compute("finite-n lambda3", grid=grid, diffmat=diffmat, gamma=5/3)
```

Now:

```python
basis = agnimhd.Basis(n_rho, n_theta, n_zeta, domain="field_period")
eq_data, diffmat = agnimhd.from_desc(eq, basis)
gamma2 = agnimhd.growth_rate(eq_data, diffmat, agnimhd.AssemblyConfig(gamma=5/3))
```

The sign is flipped: agnimhd returns `gamma^2 = -lambda3`, positive when
unstable, and its `SolverConfig.sigma` is minus DESC's `sigma` (for example
`0.1` for DESC's `-0.1`).

Keyword options and `AGNI_*` environment variables became fields of
`AssemblyConfig` and `SolverConfig`. Environment variables are no longer read.

| DESC | agnimhd |
|---|---|
| `gamma`, `incompressible`, `axisym`, `coupled_rt` | `AssemblyConfig` |
| `sigma`, `num_matvecs`, `eigensolver` | `SolverConfig` |
| `eigensolver="eigsh_callback"` | `"eigsh"` |
| `eigensolver="pcg_deflated"` with `jd` options | `"jd"`, `coarse=(eq_c, diffmat_c)` |
| `FinitenStability` objective | `agnimhd.adapters.desc_objective.AgniStability` |

Gradients: in DESC, AGNI is a compute function of `R_lmn, Z_lmn, p_l, i_l, Psi`,
so `jax.grad` reaches the parameters directly and `ProximalProjection` keeps
force balance. Here the caller supplies that parameter-to-grid map:
`jax.grad(growth_rate_of)(params, equilibrium_map, diffmat)`.
`jax.grad(growth_rate)` on an `EquilibriumData` raises.

The agnimhd copy was taken from DESC commit `f625b0121` (2026-08-18) and has
since received DESC's later fixes up to `437ccf2ed`. It also fixes two bugs DESC
still has: `fourier_interp_matrix` ignored its `period` argument, and
`pcg_deflated` counted the start vector twice when a deflation space was given.
A third, on the complex Hermitian (`axisym=True`) operator, is fixed in both,
differently. matfree's Lanczos orthonormalized with `Q.T Q` instead of
`Q^H Q` and returned a wrong eigenvector with a plausible eigenvalue
(`+9.713e-02` against a dense `-2.660e-03` on one plane of the test case); it
was fixed upstream in matfree #288, hence `matfree>=0.6.2`, while DESC uses a
real `2n` embedding. And the `eigsh` callback declared a real output dtype; it
now comes from `assemble.operator_dtype`.
On the shipped test case agnimhd reproduces DESC's dense eigenvalue to 7e-10
relative.
