# agnimhd — handoff

Written 2026-08-19. **Start here.** This file is the complete inventory: what
exists, what every file is for, every number that has been measured, every
decision that was made and why, and everything that is still missing. `STATUS.md`
is the shorter running log; this is the exhaustive one.

Repo: `/pscratch/sd/r/rgaur/AGNI`. Git initialized, **nothing committed** — no
commit was ever requested.

---

## 1. What this is

`agnimhd` is a standalone extraction of the AGNI finite-*n* ideal MHD stability
solver and optimizer, previously living inside DESC
([PR #1789](https://github.com/PlasmaControl/DESC/pull/1789)). The physics is
described in the paper in this repository root,
`Low_n_standard_formulation51_nu (3).pdf` — Gaur, Patil, Gupta, Patch, Qian,
*AGNI: A differentiable MHD stability solver & optimizer for magnetic
confinement fusion devices* (2026).

**The single hard constraint:** `agnimhd` must never depend on DESC — not as a
dependency, not as an optional extra, not in a test, not behind a lazy import.
Dependencies are exactly **jax, numpy, scipy, matfree**. The dependency runs the
other way: DESC installs `agnimhd`.

Three mechanisms enforce it: `tests/test_objective.py::test_desc_is_not_a_dependency`
and `::test_only_the_allowed_dependencies_are_used`, the `desc-absent` and
`clean-install` CI jobs, and the `no-desc-import` pre-commit hook
(`tools/check_no_desc.sh`).

---

## 2. Environments and how to run things

| | |
|---|---|
| DESC-free test env | `.venv-agni/` — jax 0.11.1, numpy 2.5.2, scipy 1.18.0, matfree 0.5.5, h5py, pytest, pytest-cov, black, flake8, isort. **DESC is absent, verified.** |
| DESC-present env | `conda activate desc-env2` — only for `tools/`, never for the package or the tests |

```bash
cd /pscratch/sd/r/rgaur/AGNI
.venv-agni/bin/python -m pytest tests -q                              # ~15 min, CPU
.venv-agni/bin/python -m pytest tests -q --cov=agnimhd --cov-report=term-missing
.venv-agni/bin/python -m black --check src tests examples tools
.venv-agni/bin/python -m isort --check-only src tests examples tools
.venv-agni/bin/python -m flake8 src tests examples tools
bash tools/check_no_desc.sh
.venv-agni/bin/python examples/growth_rate.py
```

The suite is ~15 minutes because the objective tests each run a full dense
assembly plus eigensolve at the shipped 24x12x8 resolution. **Do not reduce the
resolution to make it cheaper** — that tests a different problem.

NERSC note: heavy runs go to compute nodes (`-A m4505 -C cpu`, or `-A m5194_g`
for GPU). The suite as it stands is small enough to run on a login node, and has
been.

---

## 3. File-by-file inventory

### Package — `src/agnimhd/` (~3900 lines)

| file | lines | what it holds |
|---|---|---|
| `__init__.py` | 37 | the public surface: `EquilibriumData`, `DiffMat`, `AssemblyConfig`, `SolverConfig`, `growth_rate`, `growth_rate_and_grad`, `eigenpair`, `FORMAT_VERSION`, `__version__` |
| `backend.py` | 106 | `jax`/`jnp` import, x64 enable, `errorif`, `check_posint` |
| `equilibrium.py` | 716 | `EquilibriumData` — the whole interface contract, validation, `.npz` save/load with a JSON sidecar, optional HDF5, JAX pytree registration, `instability_drive()` |
| `config.py` | 335 | `AssemblyConfig`, `SolverConfig` (frozen dataclasses), `resolve_option`, `resolve_flag` |
| `quadrature.py` | 321 | `leggauss_lob`, `gauss_radau_jacobi`, `bspline_nodes_weights`, `zernike_nodes_weights`, `automorphism_staircase1/2` |
| `basis/zernike.py` | 481 | DESC-free Zernike via Jacobi recurrence: `zernike_modes`, `zernike_radial`, `zernike_eval_matrix`, `zernike_fourier_diffmat`, `zernike_penalty_projector_from_diffmat` |
| `basis/diffmat.py` | 788 | `DiffMat`, `legendre_diffmat`, `fourier_diffmat`, `fourier_diffmat_truncated`, `jacobi_diffmat`, `bspline_diffmat`, `finite_difference_diffmat` (4th-order SBP), and `standard_grid` |
| `assemble.py` | 1235 | `assemble_dense`, `matfree_operator`, `ring_block`, `finish_ring_block`, `keep_indices`, `_normalized_fields` |
| `solvers.py` | 1114 | coordinates (`to_phys`/`from_phys`/`_h`, `level_meta`), transfers (`barycentric_matrix`, `fourier_interp_matrix`, `transfer_matrices`, `make_transfer`, `adjoint_defect`), ring preconditioner (`ring_nodes`, `ring_index_maps`, `group_index_matrix`, `build_ring_blocks`, `factor_ring_blocks[_traced]`, `make_block_precond`), iterations (`pcg`, `pcg_deflated`, `deflation_Y`, `coarse_gen_modes`, `coarse_seed_and_deflation`) |
| `objective.py` | 362 | `growth_rate` (the `custom_vjp`), `growth_rate_and_grad`, `eigenpair`, `_eigsh_host`, `_lanczos`, `_lanczos_at`, `_primal` |
| `cli.py` | 209 | `agnimhd info` / `validate` / `solve` |
| `plotting.py` | 298 | `mode_components`, `mode_displacement`, `mode_speed`, `plot_mode_cross_section`, `plot_radial_profile`, `plot_spectrum` |
| `py.typed` | — | PEP 561 marker |

### Tests — `tests/` (~2900 lines, 180 passing + 1 conditional skip)

| file | tests | covers |
|---|---|---|
| `conftest.py` | — | `eq_data`, `eq_meta`, `diffmat`, `config`, `dense` (session-scoped), `AUTO_KW`, `_require` |
| `test_zernike.py` | 19 | Zernike against the frozen DESC reference; the penalty-projector bug both ways |
| `test_diffmat.py` | 41 | every basis, SBP identities, exactness on polynomials, Fourier periodicity |
| `test_equilibrium.py` | 27 | the closed contract, rho-major ordering, both drive routes, NaN messages, npz/HDF5 round-trip, pytree behaviour, the `a`-sensitivity |
| `test_assemble.py` | 11 | `B` SPD, `A` symmetric, matrix-free vs dense per column, the reference eigenvalue |
| `test_solvers.py` | 45 | interpolation exactness, all five partitions, block preconditioner, PCG, deflation, coarse modes, ring blocks vs dense sub-blocks, adjointness |
| `test_objective.py` | 19 | sign first, value vs sidecar, both eigensolvers, the far-shift wrong-mode pin, `jax.grad`/`jax.jit` from outside, finite differences, the HF identity, the dependency scan |
| `test_cli.py` | 10 | `info` names every field and both traps, `validate` exit codes, `solve` vs the sidecar, a positive shift refused |
| `test_plotting.py` | 9 | exact round-trip, Dirichlet zeros, the Eq. (21) rescaling, scale invariance, the metric contraction |

**Fixtures** (`tests/data/`, all git-tracked, all read by the suite):
`qh_lowres_24x12x8.npz` (266 kB), its `.json` sidecar (provenance + the reference
eigenvalue), `zernike_reference.npz` (207 kB). `.gitignore` deliberately has no
blanket `*.npz`/`*.h5` rule, with a comment saying why. **A missing fixture fails,
it never skips** (`conftest._require` uses `pytest.fail`).

### Docs — `docs/` + `README.md`

`README.md` (overview, quickstart, sign-convention table, doc map),
`docs/theory.md` (the paper, section by section, mapped onto the code),
`docs/interface.md` (the `EquilibriumData` contract, field by field, with units),
`docs/adapters.md` (checklist + the DESC compute-key table),
`docs/resolution.md` (accuracy floor, resolution, eigensolver choice, the shift,
the matrix-free path, the FD step), `docs/api.md` (API map),
`docs/migration.md` (from AGNI-inside-DESC), `mkdocs.yml`.

### Examples — `examples/`

`growth_rate.py` (load, solve, read the sign), `optimization_step.py` (grad from
outside + one ascent step), `matrix_free_solve.py` (matrix-free vs dense, ring
preconditioning, CG), `basis_comparison.py` (the paper's Fig. 7 convergence
table), `desc_adapter.py` (**the consumer-side reference; imports DESC; imported
by nothing**).

### Tools — `tools/` (dev-only, may import DESC)

`export_fixture.py` (produced the fixture; its `KEY_MAP` *is* the DESC adapter),
`export_zernike_reference.py`, `job_export_fixture.sl`, `check_no_desc.sh`,
`export_fixture_57266626.log`.

### Infrastructure

`pyproject.toml` (deps exactly the four; extras `hdf5`, `plot`, `test`, `dev`,
`docs`; console script `agnimhd`; pytest/coverage/black/isort config),
`.flake8` (**separate file because flake8 cannot read pyproject.toml** — without
it flake8 falls back to 79 columns and fails on code black itself produced),
`.pre-commit-config.yaml` (pinned black/isort/flake8 + hygiene hooks + the
no-DESC hook; the test suite is deliberately *not* a hook at ~15 min),
`.github/workflows/ci.yml` (lint; 3.10/3.11/3.12 matrix with a fixture-presence
check and a 90% coverage gate; `desc-absent`; `clean-install` asserting the built
wheel's direct dependencies are exactly the four).

---

## 4. Every measured number

Physics/numerics, all reproduced by the suite on CPU:

| what | measured |
|---|---|
| dense `lambda3`, 24x12x8 shipped QH case | **-1.337626871e-04** (documented -1.337622e-04; 3.6e-6 relative, inside the noise floor) |
| package vs its own DESC export | **2.3e-10 relative** |
| `growth_rate` vs the sidecar reference | **7.2e-10 relative** |
| CLI `solve` vs the sidecar | inside **2.8e-5** |
| Gauss-Lobatto nodes vs DESC | **1.5e-16** |
| Zernike `R_l^m` vs DESC | **7.8e-16**; `dR/drho` **2.1e-14**; mode sets **exact** |
| coupled `D_rho`, `D_theta` vs DESC, 8 cases | **< 1e-12 relative** |
| drive: direct vs derived from the two vector fields | **2.8e-16** |
| matrix-free operator vs dense, per column | **4.8e-16 relative** = 2.9e-11 absolute = 0.16 ulp of `‖A‖_max = 8.2e5` |
| ring blocks vs the dense matrix's sub-blocks | **< 1e-14 relative** |
| analytic `dλ/da` vs central difference at `h = 1e-7` | agrees; recorded 0.45% |
| `a` definition: QuadratureGrid vs LinearGrid | **3.76%** apart |
| eigenvalue absolute noise floor | **1e-10** (`eps·‖Â‖₂`, `‖Â‖₂ ~ 1e6`) |
| eigenvalue relative noise floor | **2.8e-5** |
| ARPACK `eigsh` vs our JAX Lanczos, CPU | **1.53x faster** |
| coarse radial floor for the two-level solve | **16** (below it: sign flip) |
| Rayleigh residual, right mode vs wrong mode | **1.6e-04** vs **4.6e+04** |

Test/coverage state at the last full run **before** the black reformat: 180
passed, 1 skipped, **91% coverage** (`solvers.py` 96%, `basis/diffmat.py` 96%,
`basis/zernike.py` 98%, `equilibrium.py` 97%, `cli.py` 94%, `quadrature.py` 91%,
`objective.py` 90%, `assemble.py` 89%, `config.py` 74%, `backend.py` 56%,
`plotting.py` 50% — the last is the matplotlib branches, skipped when it is
absent).

---

## 5. Decisions, with the reasoning

**Sign convention.** `growth_rate` returns `λ = <ξ|A|ξ>/<ξ|B|ξ>`, the energy
quotient, so **`λ < 0` is unstable** and an optimizer must *raise* it. The paper
writes `δW_p = -λ δK`, so its `λ > 0` is unstable. The two differ by an overall
sign; `docs/theory.md` has the table and
`test_a_descent_step_moves_lambda_the_right_way` is the end-to-end assertion,
because a globally flipped gradient passes every magnitude and finiteness check
there is.

**`s -> rho`.** Every `s = rho²` in the TERPSICHORE-derived expressions is
replaced by `rho`. Not a consistent change of variable — a redefinition of the
radial coordinate the functional is written in. Getting it wrong changes the
drive by a rho-dependent factor of order two.

**The gradient.** Hellmann-Feynman via a `custom_vjp` whose backward rule returns
**zero** cotangents; the returned Rayleigh quotient then differentiates to
`vᵀ(dA/dq)v / vᵀv` exactly. Consequences: the eigensolve need not be
differentiable (so host ARPACK behind a `pure_callback` can be the default), and
the eigenvector-selection `argmax` leaves the graph. `v` is still recomputed at
every call — fixed-vector, not stale-vector.

**Configs are frozen dataclasses, not dicts.** They drive Python branches and
array shapes. A dict would retrace on every `jit` call, so it is refused with a
`TypeError`. Resolution order is keyword → environment → default, which inverts
the DESC pattern where an exported variable silently discarded an explicit
argument.

**`a` is an input, not recomputed.** So that an adapter must choose a definition
consciously. AGNI's is the QuadratureGrid cross-section area integral.

**No adapters ship.** An adapter imports an equilibrium code; the package must
not. `examples/desc_adapter.py` is the reference and belongs, conceptually, in
the consumer.

**`standard_grid` exists** because building the Legendre+Fourier `DiffMat` with
the automorphism was being duplicated in the CLI, conftest, and every example —
and a `DiffMat` built with different automorphism kwargs than the geometry is not
an error, it is a wrong eigenvalue. It returns nodes and matrices from one call.
Verified bit-identical to the hand construction.

**The default shift is left at `-1e-1`.** See below.

---

## 6. Bugs found and fixed during the extraction

All three were found by tests, not by reading. If numbers are compared against
older DESC runs, these are the places they can legitimately differ.

**1. `zernike_penalty_projector_from_diffmat` used an absolute test.**
It decided whether to re-add the constant mode with `||residual|| > 10*eps`. When
the constant already lies in the derivative row space (Zernike over-resolved
relative to the nodes) that residual is pure roundoff, the test still passes, and
normalizing it appends a **unit-length noise vector** to the represented basis.
Two implementations agreeing on `D_rho`/`D_theta` to 4e-15 produced projectors
differing by **0.35** in the sup norm. Measured on 4x6 nodes with L=8, M=3:
residual 2e-12 (ansi), 5e-12 (fringe) against `||ones|| = 4.9`. With the default
(L = M = -1) the residual is 4.9, so **production runs with the default were not
affected**. Fix: a relative floor, `1e-8*sqrt(rt_size)`.

**2. `fourier_interp_matrix` ignored its own `period`.** The wavenumbers were
never scaled by `2π/period`, so the basis functions are not periodic on the grid
and the result is not an interpolation operator. Harmless at `period = 2π` — the
poloidal case, and therefore the one anyone spot-checks. The **toroidal** transfer
uses `2π/NFP`. Measured at `n = 8`, `period = 2π/4`: the matrix differed from the
identity by **0.897** in the sup norm on the trivial `n_src == n_dst` case; 4e-16
after the fix. Every coarse-to-fine prolongation in a two-level solve was
affected. **The bug is still upstream in DESC.**

**3. `pcg_deflated` double-counted the seed.** With both a deflation space `Z`
and an initial guess `x0`, the `span(Z)` component was added twice — once from
the coarse solve, once from the guess — and the projected CG, running on the
complement, could not remove it. Measured on a synthetic SPD system: relative
residual 9.6e-12, answer **89% wrong**. `Z` alone (2.6e-11) and `x0` alone
(6.7e-12) are both fine, so only a comparison against a direct solve catches it.
Fix: a new `deflate_x` projector that `H`-orthogonalizes `x0` against `Z` first.
DESC's production path never hit this — it calls plain `pcg` with an additive
`M_ring + Y Yᵀ` preconditioner and passes the coarse seed as the *Lanczos* start
vector.

---

## 7. The shift, in full

`SolverConfig.sigma` defaults to `-1e-1`, inherited from DESC. Shift-invert maps
`λ` to `μ = 1/(λ - σ)`, and Lanczos separates modes at a rate set by the *ratio*
of their `μ`; as `σ` recedes every `μ` collapses onto `-1/σ`. Measured on the
shipped case (spectrum starts `-1.34e-4`, `-6.25e-5`, then a numerically null
cluster at `1e-11`):

| `sigma` | `mu[0]/mu[1]` | `jax_lanczos`, 50 matvecs |
|---|---|---|
| `-1e-1` (default) | 1.0007 | **wrong mode**, `+1.598e-04` |
| `-1e-2` | 1.0075 | `-1.337435e-04` (1.4e-5 off) |
| `-1e-3` | 1.0823 | `-1.337627e-04` (exact) |

`num_matvecs = 200` also recovers the right mode at the far shift, at four times
the cost of moving `σ`. **The paper's own benchmarks used `|σ| = 1e-3`**, chosen
from a low-resolution full-spectrum pre-processing scan — that is the recommended
procedure. `eigsh` is unaffected because ARPACK iterates to `eigsh_tol` instead
of stopping at a fixed matvec count, which is why the default never caused
trouble on the default path.

`sigma_mode="adapt"` does **not** rescue it: the first pass returns a positive
`λ`, the `sigma2 < 0` guard rejects it, and the second pass repeats the first.
Re-shifting to `-sigma_factor*|λ|` instead was measured and is *worse* — from a
wrong first pass it chases the null cluster down to `σ = -1e-10` and converges
there.

The default was left in place because a shift *above* the spectrum has no
recovery at all (wrong mode, and `H` indefinite so CG is not legal), and because
the failure is detectable: the **Rayleigh residual** is 4.6e+04 for the wrong
mode against 1.6e-04 for the right one. Two tests pin this; the behaviour is
documented in `SolverConfig.sigma` and `docs/resolution.md`.

`sigma_mode="track"` is deliberately **not implemented** and should not be added.

---

## 8. What is still needed

Ordered. Nothing here is blocking a first review.

1. **Commit.** Nothing has been committed. Nothing was asked to be.
2. **Coverage gaps.** 91% overall against a >90% gate, but the spec's target was
   100% on `solvers.py` and `basis/` (at 96%). The real gap is `config.py` at
   **74%**: the environment-variable fallback paths in `resolve_option` and
   `resolve_flag` have no test at all, and they are the mechanism by which a job
   script configures a run. `backend.py` at 56% is mostly the `errorif` message
   branches.
3. **`pcg_deflated` is not wired into `growth_rate`.** It raises
   `NotImplementedError` naming itself. The pieces are all in `solvers.py` and
   tested individually; what is missing is the driver that builds a coarse level.
   That needs an equilibrium re-evaluated at coarse nodes, which only an
   equilibrium code can produce — so it needs either a second fixture or a
   consumer-side test.
4. **The two-level example is partial** (`matrix_free_solve.py`) for the same
   reason, and says so.
5. **`psi_rr` is required by the contract but never read** — the solver
   recomputes the radial derivative spectrally. Kept required so adapters do not
   have to change later. Decide.
6. **`standard_grid(automorphism=None)` puts a node at `rho = 0`**, on the
   singular axis. Documented as being for derivative tests; it may be better to
   require an `eps`.
7. **No VMEC/GVEC adapter guidance beyond the general note.** The `s = rho²`
   conversion (`d/drho = 2 rho d/ds`) is the thing to get right.
8. **CI has never run.** The workflow is written but there is no remote. The
   `clean-install` job's dependency assertion is the one most likely to need a
   tweak on first contact.
9. **`pre-commit` has never run** either — `pre-commit install` needs network on
   first use to fetch the pinned hook repos. black/isort/flake8 have been run by
   hand and the tree is clean.
10. **Docs are unbuilt.** `mkdocs.yml` exists; `mkdocs build` has not been run,
    and the `../README.md` nav entry is the kind of thing that works in the
    repository and not in a built site.
11. **Not benchmarked at production resolution** in this package. Every number
    above is 24x12x8 on CPU. The paper's table 3 timings are DESC-side.

## 9. The order of work from here (the user's plan)

1. ~~Handoff + an initial version that runs with tests and coverage~~ — done.
2. Review, changes, test and documentation updates.
3. Test against DESC.
4. Package and release.
