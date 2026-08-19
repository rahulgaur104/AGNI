# AGNI package extraction — status

Updated 2026-08-19. Entry point for the next session.

Repo: `/pscratch/sd/r/rgaur/AGNI` (git initialized, **nothing committed** — no
commits were requested).

DESC-free environment: `.venv-agni/` (jax 0.11.1, numpy 2.5.2, scipy 1.18.0,
matfree 0.5.5, h5py, pytest, and the lint tools). **DESC is absent from it**,
verified. Run tests as

    .venv-agni/bin/python -m pytest tests/ -q

DESC-present environment, for `tools/` only: `conda activate desc-env2`.

The paper is in the repository root (`Low_n_standard_formulation51_nu (3).pdf`)
and `docs/theory.md` is written against it, section by section.

---

## Measured results so far

| what | measured | reference |
|---|---|---|
| dense `lambda3`, 24x12x8, shipped QH case | **-1.337626871e-04** | -1.337622e-04 documented; 3.6e-6 relative, inside the 2.8e-5 noise floor |
| AGNI package vs its own DESC export | **2.3e-10 relative** | end-to-end milestone (Part 8 step 3) |
| Gauss-Lobatto nodes, AGNI vs DESC | **1.5e-16** | reimplementation is faithful |
| Zernike `R_l^m` vs DESC | **7.8e-16** | frozen reference |
| Zernike `dR/drho` vs DESC | **2.1e-14** | frozen reference |
| Zernike mode sets, both indexings, 8 cases | **exact** | frozen reference |
| coupled `D_rho`, `D_theta` vs DESC, 8 cases | **< 1e-12 relative** | frozen reference |
| drive: direct vs derived from the two vector fields | **2.8e-16** | both routes agree |
| matrix-free operator vs dense, per column | **4.8e-16 relative** | 2.9e-11 absolute = 0.16 ulp of `‖A‖_max = 8.2e5` |
| ring blocks vs the dense matrix's own sub-blocks | **< 1e-14 relative** | vmapped build |
| `growth_rate` vs the sidecar reference | **7.2e-10 relative** | milestone 6 |
| analytic `dλ/da` vs central difference at `h = 1e-7` | **agrees** | see the gradient test for why the step is not free |
| CLI `solve` vs the sidecar reference | **inside 2.8e-5** | end to end through the shell |

Test suite: **180 passing** (19 zernike, 41 diffmat, 27 equilibrium, 11
assembly, 45 solvers, 19 objective, 10 CLI, 9 plotting — 1 of those skips when
matplotlib is absent).

---

## Done

- `equilibrium.py` — `EquilibriumData`, the whole contract.
- `backend.py`, `quadrature.py`, `config.py` (`AssemblyConfig`, `SolverConfig`).
- `basis/zernike.py`, `basis/diffmat.py` — every basis, plus `standard_grid`,
  which returns the nodes **and** the matching `DiffMat` from one call so the
  two cannot drift apart.
- `assemble.py` — dense, ring, and matrix-free paths, all three checked against
  each other.
- `solvers.py` — ring preconditioner, PCG, deflated PCG, transfers, coarse
  generalized eigensolve. Two real bugs fixed (below).
- `objective.py` — `growth_rate`, `growth_rate_and_grad`, `eigenpair`, with the
  Hellmann-Feynman `custom_vjp`. Survives `jax.jit` and `jax.grad` applied from
  outside the package.
- `cli.py` — `info`, `validate`, `solve`. `plotting.py` — eigenvector back to a
  displacement, plus three plots behind a lazy matplotlib import.
- `tools/export_fixture.py`, `tools/export_zernike_reference.py`. Both fixtures
  generated and git-tracked.
- `pyproject.toml` — deps are exactly jax, numpy, scipy, matfree.
- Docs: `README.md`, `docs/theory.md`, `docs/interface.md`, `docs/adapters.md`,
  `docs/resolution.md`, `docs/api.md`, `docs/migration.md`, `mkdocs.yml`.
- Examples: `growth_rate.py`, `optimization_step.py`, `matrix_free_solve.py`,
  `basis_comparison.py`, and `desc_adapter.py` (the consumer-side reference,
  imported by nothing).
- CI: `.github/workflows/ci.yml` — lint, a 3.10/3.11/3.12 test matrix with a
  coverage gate, a **DESC-absent** job, and a **clean-install** job that asserts
  the built wheel's direct dependencies are exactly the four.

Coverage, measured `pytest tests -q --cov=agnimhd`: **91% overall**, above the
CI gate. By module: `solvers.py` 96%, `basis/diffmat.py` 96%,
`basis/zernike.py` 98%, `equilibrium.py` 97%, `cli.py` 94%, `quadrature.py`
91%, `objective.py` 90%, `assemble.py` 89%, `config.py` 74%, `plotting.py` 50%
(the uncovered half is the matplotlib branches, skipped where it is absent),
`backend.py` 56%.

## Next

1. `black`/`isort` have never been run on this tree — CI enforces both, and
   `black --check` currently reports 17 files. Format, then re-run the suite.
2. Coverage: the spec's target is 100% on `solvers.py` and `basis/`; they are at
   96%. `config.py` at 74% is the real gap — the env-var fallback paths in
   `resolve_option`/`resolve_flag` are untested.
3. Decide `psi_rr` (open question below).
4. Commit, when asked.

---

## Findings worth keeping

**A real bug was found and fixed in `zernike_penalty_projector_from_diffmat`.**
The original decided whether to re-add the constant mode with an ABSOLUTE test,
`||residual|| > 10 * eps`. When the constant already lies inside the derivative
row space — which happens as soon as the Zernike basis is over-resolved relative
to the nodes — that residual is pure roundoff, the test still passes, and
normalizing it appends a **unit-length noise vector** to the represented basis.
Two implementations agreeing on `D_rho`/`D_theta` to 4e-15 produced projectors
differing by **0.35** in the sup norm.

Measured on a 4x6 node set with L=8, M=3: residual norm 2e-12 (ansi) and 5e-12
(fringe) against `||ones|| = 4.9`. With the well-posed default (L = M = -1) the
residual is 4.9, so **production runs with the default were not affected**.
AGNI uses a relative floor, `1e-8 * sqrt(rt_size)`.

**Two more real bugs were found and fixed, both in ported DESC code.**

*`fourier_interp_matrix` ignored its own `period`.* The wavenumbers were never
scaled by `2*pi/period`, so the basis functions are not periodic on the grid and
the result is not an interpolation operator at all. Harmless at `period = 2*pi`
— the poloidal case, and so the case anyone would spot-check. The **toroidal**
transfer uses `2*pi/NFP`. Measured: at `n = 8`, `period = 2*pi/4`, the matrix
differed from the identity by **0.897** in the sup norm on the trivial
`n_src == n_dst` case; 4e-16 with the scaling in place.

*`pcg_deflated` double-counted the seed.* With **both** a deflation space `Z`
and an initial guess `x0`, the `span(Z)` component of the solution was added
twice. Measured on a synthetic SPD system: relative residual 9.6e-12, answer
**89% wrong**. The fix `H`-orthogonalizes `x0` against `Z` first. DESC never hit
this: its production path calls plain `pcg` and passes the coarse seed as the
*Lanczos* start vector.

**The default shift is wrong for a fixed-budget Lanczos, and the paper says so.**
`SolverConfig.sigma` defaults to `-1e-1`, inherited from DESC. Shift-invert maps
`lambda` to `1/(lambda - sigma)` and Lanczos separates modes at a rate set by the
*ratio* of those, so a shift far below the spectrum makes every small eigenvalue
indistinguishable. Measured on the shipped case (spectrum starts `-1.34e-4`,
`-6.25e-5`, then a null cluster at `1e-11`):

| `sigma` | `mu[0]/mu[1]` | `jax_lanczos`, 50 matvecs |
|---|---|---|
| `-1e-1` | 1.0007 | **wrong mode**, `+1.598e-04` |
| `-1e-2` | 1.0075 | `-1.337435e-04` |
| `-1e-3` | 1.0823 | `-1.337627e-04` (exact) |

The paper's own benchmarks used `|sigma| = 1e-3`, chosen from a low-resolution
full-spectrum pre-processing scan. `eigsh` is unaffected because it iterates to a
tolerance rather than stopping at a fixed matvec count, which is why the default
has never caused trouble on the default path. `sigma_mode="adapt"` does **not**
rescue it: the first pass returns a positive `lambda`, the `sigma2 < 0` guard
rejects it, and the second pass repeats the first. Re-shifting to
`-sigma_factor*|lambda|` instead is worse — it chases the null cluster down to
`1e-10`.

The failure is detectable and only detectable one way: the **Rayleigh residual**
is `4.6e+04` for the wrong mode against `1.6e-04` for the right one. Both tests
are in `tests/test_objective.py`; the default is left where it is, because a
shift *above* the spectrum has no recovery at all, and the behaviour is
documented in `SolverConfig.sigma` and `docs/resolution.md`.

**Sign convention differs from the paper, deliberately.** The paper writes
`dW_p = -lambda dK`, so its `lambda > 0` is unstable. `agnimhd.growth_rate`
returns the energy quotient `<xi|A|xi>/<xi|B|xi>`, so **`lambda < 0` is
unstable** and an optimizer must raise it. `docs/theory.md` states this in a
table; `tests/test_objective.py::test_a_descent_step_moves_lambda_the_right_way`
is the end-to-end assertion, because a globally flipped gradient passes every
magnitude and finiteness check.

## Open questions

- `psi_rr` is in the contract and is exported, but the solver never reads it —
  it recomputes the radial derivative spectrally. Kept required because the
  spec's contract table lists it. Worth revisiting.
- Serialization is `.npz`, not HDF5/NetCDF. Both of those need a dependency
  outside the allowed four. HDF5 is available via `save_hdf5`/`load_hdf5` behind
  an explicit `h5py` capability check.
- `standard_grid(automorphism=None)` places a node at `rho = 0`, on the singular
  axis. It is documented as being for derivative tests rather than for a
  stability solve; it may be better to require an `eps`.
