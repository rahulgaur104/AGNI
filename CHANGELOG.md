# Changelog

Every pull request adds a line under "Unreleased". Numbers in parentheses are
pull requests on GitHub.

## Unreleased

- Zenodo DOI in the README and `CITATION.cff` (#97).

## 0.2.0 (2026-10-15)

Breaking changes

- `growth_rate_of` is removed; `growth_rate` is the one function for `gamma^2`.
  Optimize mode is `growth_rate(params, diffmat, ..., equilibrium_map=f)`;
  `on_vector` or `sigma` without `equilibrium_map` raise a `TypeError` (#95).

New features

- `eigensolver="dense_mg"` solves the complex toroidal families (#90).

Other

- Codecov badge in the README again (#93).

## 0.1.0 (2026-10-09)

The first release on PyPI. It contains the initial development (below) and
everything merged from 4 to 8 October 2026.

Breaking changes

- The eigenvalue returned everywhere is `gamma^2 = -lambda`, positive when
  unstable, and `sigma` is given in the same convention (#25).
- `Basis` has no `domain` any more: the grid always spans one field period and
  every toroidal mode family is solved on it (#29).
- `from_desc(..., density=True)` stores the density in `eq_data.density` and
  returns `(eq, diffmat)` instead of `(eq, diffmat, density)` (#40).
- The eigensolvers `"pcg"` and `"pcg_deflated"` are removed (#52).
- The eigensolver `"jax_lanczos"` is renamed `"dense"`, with a Cholesky
  factorization by default (#64).

New features

- `from_desc` loads a DESC `.h5` file or `Equilibrium` into an `EquilibriumData`;
  `agnimhd solve eq.h5` on the command line (#13).
- `growth_rate` and `eigenpair`: `v_fixed` skips the eigensolve, `v_guess`
  warm-starts it (#9).
- `eigensolver="jd"`: matrix-free Jacobi-Davidson with the ring preconditioner
  and coarse-level deflation (#15); a production coarse level with fewer
  angular nodes, always on, for any radial basis and family (#38); the start
  vector from the softest coarse modes (#42); the coarse level of a coupled
  Zernike basis (#44); the coarse level without forming the transformed
  matrix (#48).
- `eigensolver="dense_mg"`: dense shift-invert split over several GPUs with
  JAXMg, and `assemble_rows` for any block of rows of the matrix (#21).
- `AgniStability`, the growth rate as a DESC objective, in
  `agnimhd.adapters.desc_objective` (#19); differentiable under DESC's `jit`
  (#37); with the JD solver (#50), optional density weighting (#56), a warm
  start from the previous eigenvector (#80) and an adaptive JD shift (#82).
- `Basis`: nodes and derivative matrices chosen by keyword (#27);
  `radial="zernike"`, the DSHAPE tokamak of the AGNI paper (#31).
- Toroidal mode families: every family `n = x + k NFP` is solved on one field
  period (#29).
- Stellarator symmetry: `AssemblyConfig(parity="even" | "odd")` solves one
  parity block of the operator (#70).
- `agnimhd.load` and `agnimhd.solve` (#60).
- `eigenpair` and `growth_rate` are jitted by default (#46).

Bug fixes

- Conjugate transposes in the Hermitian solver pieces, and 1-D `DiffMat`
  weights so that 2-D quadrature weights work: the fixes made in DESC after the
  extraction (#3).
- Mass-density weighting reached only `dense_mg`. `EquilibriumData` has an
  optional field `density`, and every solver weights the kinetic energy with
  it (#40).

Performance

- `assemble_dense` whitens in component-major order, without the permutation
  round trip (#5).

Documentation

- A compact rewrite for `from_desc` and `jd` (#17); the roadmap and
  `CODE_ACCELERATION.md` (#23); citation of the preprint (#32); local paths
  and machine-specific wording removed (#33); Installation and Developer guide
  pages (#68); an Examples page with measured output and figures, and
  `desc_optimization.py` (#62); a tutorial notebook that runs on a CPU (#72);
  the DESC coupling page (#58).
- Shorter comments and docstrings, one synthetic axisymmetric fixture and
  consolidated tests (#11); unused helpers removed (#54).

Tests, CI and packaging

- A CI job with DESC 0.17.3 for the adapter tests (#13); the `AgniStability`
  objective in a DESC optimization (#74).
- CI on every pull request, with figure tests against baseline images (#66);
  runtime benchmarks of every pull request against its base (#78); an
  aggregate job `all` for the branch ruleset (#84); the DESC tests split over
  three jobs, and a workflow that times the suite on a runner (#86).
- The version comes from the git tag; PyPI release workflow and Zenodo
  metadata (#76).

## Initial development (2026-08-19 to 2026-09-04)

AGNI was extracted from its implementation inside DESC
([PR #1893](https://github.com/PlasmaControl/DESC/pull/1893)) into the
standalone package `agnimhd`, committed directly to `master`.

- The package, `pyproject.toml`, MIT license, pre-commit and flake8
  configuration, MkDocs documentation and examples (2026-08-19 to 08-20).
- GitHub Actions CI and coverage uploads to Codecov (2026-08-19 to 09-03); the
  test suite split across CI jobs (2026-09-03).
- Variable names made consistent with AGNI in DESC (2026-08-20).
- A bug in the matrix-free operator that made it non-Hermitian fixed
  (2026-09-03).
- The highest Fourier mode a derivative matrix keeps is limited by the number
  of quadrature points (2026-09-04).
