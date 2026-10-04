# Roadmap

Written 2026-10-03. Developer notes, not user documentation. Replaces
`desc_sync_plan.md` (its sections 1-5 are done on `sync-desc-10-02` and `jdclean`).

## Goals

1. **Stability of any equilibrium.** Load an equilibrium (DESC first; loading is
   an interface other codes implement), choose the basis by keyword arguments with
   defaults, and solve with one of three solvers:
   - `dense`: dense matrix on one GPU, Cholesky shift-invert, no JAXMg;
   - `dense_mg`: dense matrix split over the GPUs of one node with JAXMg;
   - `jd`: matrix-free Jacobi-Davidson with the ring preconditioner and the
     coarse-level deflation, both always on.
2. **DESC optimization with the agnimhd objective.** A DESC script imports
   `AgniStability` from agnimhd, passes the same basis and solver keywords, and
   optimizes the boundary through `ProximalProjection`. Order: one GPU (`dense`,
   then `jd`), then several GPUs (`dense_mg`).

Target usage:

```python
import agnimhd as ag

src = ag.load("eq.h5")                        # DESC file or Equilibrium
basis = ag.Basis(40, 48, 16)                  # other knobs: keywords with defaults
lam, v = ag.solve(src, basis, solver="jd")    # "dense" | "dense_mg" | "jd"
```

```python
# DESC script
from agnimhd.adapters.desc_objective import AgniStability
obj = ObjectiveFunction(AgniStability(eq, basis=ag.Basis(24, 12, 8), solver="jd"))
```

## How every change lands

1. An issue: what changes, the acceptance test.
2. A branch named after the change, from `origin/master` or, while earlier PRs
   are open, from the branch it builds on.
3. One-line commit messages, no trailers.
4. In the same PR: tests for the change (and their entries in
   `.test_durations`, which CI checks), coverage not below `master`, docs for
   every user-visible change, src/tests/docs line deltas in the PR text.
5. CI green, merge into `master` with "Create a merge commit" (not squash: the
   PRs are stacked, and a squash commit is not an ancestor of the next branch, so
   the next merge conflicts; measured 2026-10-03), delete the branch.

One topic per PR. Claude's shell has no GitHub credentials: Claude prepares the
branch, the issue text and the PR text; the user pushes and opens them.

## Conventions

- **Sign.** Everything agnimhd returns is the squared growth rate
  `gamma^2 = -lambda_min`: **positive means unstable**. The minus sign is applied
  once, where the eigenvalue leaves the package; solvers inside work on `A`
  unchanged. The shift `sigma` is given in the same convention: a value above
  the largest `gamma^2` (for example `1.05 * gamma2_estimate`).
- **Names.** A function's name says what it does in words (`conjugate_transpose`,
  not `_cT`). No abbreviations beyond standard ones (`rho`, `eq`, `jd`). A leading
  underscore marks an internal helper; it does not excuse a cryptic name.
- **Docstrings.** Every function has one: a one-line summary; public functions
  also list parameters and returns (numpydoc).
- **Tests.** A test reads like a short user script: load an equilibrium, build
  the basis, solve, compare with a reference. Someone new learns how to write a
  solve script by reading them, as with DESC's tests. One behavior per test,
  parametrized instead of copied.
- **Size.** No code that the change does not need; src/tests/docs line deltas in
  every PR text.

## Where things stand

| branch | content |
|---|---|
| `master` afeccb5 | refactor of 09-03/04; nothing below is merged |
| `sync-desc-10-02` e1944a9 (+9) | DESC fixes since the extraction, component-major whitening, `v_fixed`/`v_guess`, `from_desc`, `agnimhd solve eq.h5` |
| `jdclean` 0cc8dbc (+15; 9a76004 local) | JD, `AgniStability`, `assemble_rows`, `dense_mg`, docs |

Gaps, each closed by a PR below:

- **Basis.** `standard_grid` gives Lobatto radial nodes (optionally through the
  staircase map) and untruncated Fourier only. Gauss-Radau-Jacobi nodes
  (`quadrature.gauss_radau_jacobi`) and truncated Fourier
  (`fourier_diffmat_truncated`) exist but need a hand-built
  `grid=(nodes, diffmat)`; the Patil drivers built that grid with DESC functions.
- **Loading.** Only `from_desc`; other codes fill `EquilibriumData` by hand.
- **Density.** Mass weighting by `ni` reaches `assemble_dense` and `dense_mg`
  only; `growth_rate`, `eigenpair`, `growth_rate_of` and `AgniStability` cannot
  take it.
- **Single-GPU dense.** Exists as `eigensolver="jax_lanczos", factor="cholesky"`
  (the default factor is LU). No GPU measurement in this repository.
- **JD.** Coarse deflation only if the caller passes `coarse=(eq_c, diffmat_c)`;
  without it JD stalled on the 24x12x8 fixture (10-02). The coarse-to-fine
  transfer assumes Lobatto radial nodes (`objective._coarse_space`), so it is
  wrong for a Gauss-Radau-Jacobi basis.
- **`dense_mg`.** Measured on Patil QH up to 182,784 unknowns
  (`docs/multigpu.md`); gradients through it never run.
- **`AgniStability`.** Value and gradient checked at 24x12x8; takes only `res`
  and `automorphism`; no optimization has run (the `ProximalProjection` build
  was killed at 30 GB on a login node).
- **CI.** Newest jax, no DESC, no GPU: `test_adapters` is skipped, so the DESC
  adapter and objective have no coverage; GPU paths are covered only through CPU
  stand-ins.
- **Versions.** DESC needs jax < 0.10, jaxmg >= 1.0 needs jax 0.11. DESC and
  `dense_mg` in one process means jaxmg 0.0.9 on one node.

## Phase 0: land the existing branches

Cherry-picks of existing commits, no new code. Each PR: full suite on its branch,
pass count and coverage in the PR text. A commit that does not split cleanly
goes in whole and the PR says so.

| PR | commits | content |
|---|---|---|
| 0.1 | aebd63c, cb5bb42, 49355eb | DESC fixes since the extraction |
| 0.2 | 854d00e | component-major whitening |
| 0.3 | 225edfb | `v_fixed`, `v_guess` |
| 0.4 | beb8d10 | shorter comments, docstrings and tests |
| 0.5 | 79b1daf, 1c11fc4, e1944a9 + one new CI commit | `from_desc`, `agnimhd solve eq.h5`, dependency tests in a fresh interpreter; CI job with a pinned DESC release (jax < 0.10) that runs the DESC tests and adds their coverage |
| 0.6 | 58d2936, ac9251c, 1851b48 | Jacobi-Davidson |
| 0.7 | 14c35c5 | docs rewritten compactly |
| 0.8 | 71d144d | `AgniStability` |
| 0.9 | ab4b579, e56f5d5, 6cb85d1, b55cc81, 0cc8dbc, 9a76004 | `assemble_rows`, `dense_mg`, docs |
| 0.10 | this file | roadmap; removes `desc_sync_plan.md` |

## Phase 1: goal 1 on one GPU

**1.1 `Basis`.** One frozen object builds the nodes and the `DiffMat`: `radial`
("lobatto", "gauss_radau_jacobi"), `alpha`, `beta`, `automorphism` (staircase
parameters), `mpol`, `ntor` (Fourier truncation), `domain` (one field period or
full torus). Replaces `standard_grid` and the `grid=` escape hatch.
Tests: reproduces `standard_grid` exactly; reproduces the Patil 40x48x16 grid
saved from the DESC driver. Docs: `options.md`, choosing the basis.
Cap: src +60 net.

**1.2 Equilibrium sources.** `ag.load(path_or_object)` returns a source with
`evaluate(basis, params=None) -> EquilibriumData`. DESC source (`from_desc`
folded in) and an `.npz` source for other codes. Tests: DESC source equals
today's `from_desc`; `.npz` round trip. Docs: `interface.md`. Cap: src +40 net.

**1.3 Density.** `ni` becomes an optional `EquilibriumData` field (default:
none, unweighted), so every solver and the objective get it without new
arguments. Tests: each solver with density equals `assemble_dense` with density.
Cap: src +20 net.

**1.4 Solver names.** `dense` (today's `jax_lanczos` with `factor="cholesky"`),
`dense_mg`, `jd`; `eigsh` stays as the CPU reference. Tests: `dense` equals
`eigsh` on both fixtures. Docs: `options.md` solver table.

**1.5 JD with its coarse level, always.** `ag.solve(..., solver="jd")` builds the
coarse level from the same source with `basis.coarse()` (same MPOL/NTOR on both
levels); the transfer uses the basis's own radial nodes; `jd` without a coarse
level raises. Tests: JD equals `dense` on 24x12x8 for a Lobatto and a
Gauss-Radau-Jacobi basis; error without a coarse level.

**1.6 `ag.solve` and the CLI.** `ag.solve(src, basis, solver, **knobs)` returns
`(lam, v)`; `agnimhd solve eq.h5 --res 40,48,16 --solver jd`. Docs: quickstart
in `index.md` is the three lines above. Tests: CLI on the fixture.

**1.7 GPU validation (jobs, no code).** Patil QH with `dense` (largest grid one
80 GB GPU holds), `jd`, and `dense_mg` on the same basis; reference
-1.43963328604e-4 at 40x48x16. Table into the docs. Setup and go from the user.

## Phase 2: goal 2

**2.1 `AgniStability` on the new pieces.** Takes `basis`, `solver` and solver
knobs; the source is evaluated at the current params on every call, the JD coarse
level too; density included. Tests: value equals `ag.solve`; AD against finite
differences on Psi at 24x12x8. Docs: new `desc.md` (loading and objective).

**2.2 One-GPU optimization.** Small QH case, `ProximalProjection` over boundary
modes with `ForceBalance`; `dense` first, then `jd`. Acceptance: lambda
decreases, force error stays at the solve level, gradient matches finite
differences at the start; time and memory per iteration reported. Example
script in `examples/`. 80 GB GPU job; setup and go from the user.

**2.3 `dense_mg` in the objective.** Four GPUs, one node, jaxmg 0.0.9. Work
known so far: the Python iteration loop and its early stop under DESC's `jit`;
DESC's arrays on one GPU while the solve spans four. Acceptance: same value and
gradient as `dense` on a grid that fits one GPU.

**2.4 Several nodes.** Needs jax 0.11 for jaxmg >= 1.0: either DESC raises its
pin or the eigensolve runs in its own process. Decide after 2.3.

## Benchmarks

Fixed cases, each with a driver in `benchmarks/` (standalone, settings on top)
and a row in `docs/benchmarks.md`: reference value, resolution, solver, wall
time, memory, hardware. Run on GPU nodes with the user's go, not in CI (CI keeps
the small fixtures). Every branch that changes numerics (families, parity,
solvers) reports these against the branch it replaces.

| case | type | reference (old sign: lambda) | source |
|---|---|---|---|
| Patil QH, beta 1.5 %, iota_min 1.02 | stellarator, NFP 4 | 40x48x16 GJ, MPOL 8 NTOR 2, one field period: -1.43963328604e-4 (dense CPU); `dense_mg` sizes up to 80x48x16 | `AGNI_var/Patil_benchmark/qh_beta1.5_imin1.02/`, docs/multigpu.md |
| DSHAPE, imax 0.98 | tokamak, axisymmetric | Zernike 96x96, ZPEN 0.01, MPOL = 4n: n=2 -2.770639e-4, n=3 -2.791126e-4, n=4 -1.785811e-4, n=5 -6.459412e-5 | `AGNI_var/DSHAPE_results_09-20-26/README.md`, eq `AGNI_var/Patil_benchmark/DSHAPE/dshape_imax0.98_1608.h5` |
| Solov'ev internal kink, q0 1.035 / 1.045 | tokamak, n = 1 | DCON marginal q0 1.03956-1.03959 | phase 3 |
| ARIES-CS | stellarator, full torus | dense and JD results in `AGNI_var/ARIES-CS/` (JD bug open there) | to be chosen |
| QAS3 | stellarator, NFP 3 | TERPSICHORE: -7.03701e-7, 5 unstable | phase 3 |

DSHAPE needs a Zernike radial option in `Basis` (`zernike_fourier_diffmat`
exists, with `coupled_rt`); added with the benchmark PR. Where the equilibrium
files live (repo, release asset, or DESC's examples) is decided with that PR
(repository size).

## Acceleration ideas

Toroidal mode families solved on one field period, and the stellarator-symmetry
split: `docs/CODE_ACCELERATION.md`. Not scheduled yet.

## Phase 3 (final goal): published benchmarks

After phases 1 and 2: run published cases from the literature (same
equilibrium, mode numbers and resolution as the published result), report
accuracy, wall time and memory on CPU and GPU, and show optimization through
DESC on them.

Outside this repository: the JAXMg factor-reuse issue
(`AGNI_var/ARIES-CS/jaxmg_test_10-02-26/JAXMG_ISSUE_factor_reuse.md`).

## Decisions taken (2026-10-03; change any)

1. Phase 0 as ten topic PRs.
2. Basis defaults follow the AGNI_var drivers (Patil QH, ARIES-CS):
   Gauss-Radau-Jacobi radial nodes, `alpha=-0.35, beta=-0.65`, through the
   staircase map `eps=1e-2, x_0=0.6, m_1=2.5, m_2=3.0` (the ARIES-CS JD runs
   named `*_eps05` use `eps=5e-2`); `mpol`, `ntor` default to the most the grid resolves.
   Lobatto by keyword (the test fixtures were built with it).
3. `domain` ("field_period" or "full_torus") was a required keyword; toroidal
   families removed it (the grid is always one field period).
4. `jax_lanczos` renamed `dense`, no alias.
5. JD coarse level default: radial 2/3, angles kept (the 24x12x8 / 16x12x8
   fixture pair), revisited after 1.7.
6. `sigma` is given in the returned convention (above the largest `gamma^2`).
7. The user opens issues and PRs from Claude's text and merges them with merge
   commits, in stack order (squash conflicts on stacked branches).

## Order of work

Phase 0 is stacked in its table order. After it: the sign convention, then
`Basis` (1.1), then at once the JD coarse transfer from the basis's own radial
nodes (from 1.5: with Gauss-Radau-Jacobi as the default, JD's Lobatto-only
transfer is wrong until then), then toroidal mode families if the prototype
holds (`CODE_ACCELERATION.md` section 5: the grid is always one field period,
`domain` and the `axisym` branches go), then names and docstrings, then the
rest of phase 1. Alongside: dead code goes in two PRs (the deflated-CG path that JD
replaced; helpers nothing calls), from the audit of 2026-10-03
(`AGNI_wt/audit/NAMES_DOCSTRINGS_TESTS.md`: about 675 src lines dead). Tests are rewritten as
user scripts once `ag.solve` exists (1.6).
