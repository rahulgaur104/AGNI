# Developer guide

## How a change lands

1. An issue: what changes, and the test that shows it is done.
2. A branch named after the change, from `master`. If the change needs one
   that is still in review, branch from that branch and open the PR against it
   (GitHub retargets it to `master` when the base merges). At most three
   branches deep.
3. One-line commit messages, no trailers.
4. One PR per change, with: tests for the change and their entries in
   `.test_durations` (CI fails on a missing entry), coverage not below
   `master` (gate: 90 %), docs for every user-visible change, and the
   src / tests / docs line deltas in the PR text. Measured numbers only.
5. CI green; merged with a merge commit (not squash: a squash commit is not an
   ancestor of the next stacked branch, so that branch then conflicts); the
   branch is deleted.

The plan these changes follow is `docs/roadmap.md`.

## Checks CI runs

| job | what |
|---|---|
| lint | `black --check`, `isort --check-only`, `flake8` with the pinned versions in `.pre-commit-config.yaml`; `tools/check_no_desc.sh` |
| test | the suite in four groups (`pytest-split` by `.test_durations`), with coverage, figures compared against `tests/baseline/` (`--mpl`) |
| desc | `tests/test_adapters.py` with `desc-opt==0.17.3` |
| desc-absent | the suite with DESC absent: the package must not need it |
| coverage | the groups combined, `--fail-under=90`, uploaded to codecov |
| docs | `mkdocs build --strict`, published to GitHub Pages from `master` |

`pre-commit install` runs the lint job's checks on every commit.

## Conventions

- **Sign.** Everything returned is `gamma^2 = -lambda`, positive when unstable;
  `sigma` is in the same convention.
- **Names.** A function's name says what it does; a leading underscore marks a
  helper and does not excuse a cryptic name.
- **Docstrings.** Every function: one line; public ones with parameters and
  returns (numpydoc).
- **Tests.** A test reads like a short user script: load, build the basis,
  solve, compare with a reference. One behavior per test, parametrized rather
  than copied. Slow tests are marked `slow` and still run in CI.
- **Size.** No code the change does not need.
- **No DESC import** in `src/` outside `adapters/` (lazy, inside functions) or
  in `tests/`.
- **Figures.** A test that draws returns its figure under
  `@pytest.mark.mpl_image_compare`; regenerate a baseline with
  `pytest tests/test_plotting.py --mpl-generate-path=tests/baseline` on
  matplotlib 3.10.6 and commit it with the change that moved it.

## Measuring

Test durations: `pytest tests --store-durations --durations-path .test_durations`
(or edit the entry). Coverage: `pytest tests --cov=agnimhd --cov-report=term`.
Peak memory of a solve: `/usr/bin/time -f "%M KB" python script.py`.
