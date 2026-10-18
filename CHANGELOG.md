Changelog
=========

New Features

- ``Basis`` uses the coupled Zernike-Fourier radial basis by default (``radial="zernike"``), which is regular at the magnetic axis. The default ``automorphism`` now applies only to ``"lobatto"`` and ``"gauss_radau_jacobi"``, and ``agnimhd solve --radial`` defaults to ``zernike`` too [#105](https://github.com/rahulgaur104/AGNI/pull/105).
- Adds ``DiffMat.coupled_rt``: the assembly reads coupled (rho, theta) mode from the derivative matrices themselves, so a Zernike basis works with the default ``AssemblyConfig()`` [#105](https://github.com/rahulgaur104/AGNI/pull/105).
- Adds ``CHANGELOG.md`` [#103](https://github.com/rahulgaur104/AGNI/pull/103).
- Rewrites the documentation of the basis in ``docs/options.md`` as one section with a single table of the radial bases, and renames ``examples/cross_sections.py`` to ``examples/eigenfunction.py`` [#101](https://github.com/rahulgaur104/AGNI/pull/101).
- Adds the Zenodo DOI to the README and ``CITATION.cff`` [#97](https://github.com/rahulgaur104/AGNI/pull/97).

Breaking Changes and Deprecations

- Removes ``coupled_rt``, ``n_rho_coupled`` and ``n_theta_coupled`` from ``AssemblyConfig``; nothing replaces them [#105](https://github.com/rahulgaur104/AGNI/pull/105).
- A ``Basis`` built without ``radial`` is now a Zernike basis, so scripts that relied on the old Gauss-Radau-Jacobi default must pass ``radial="gauss_radau_jacobi"`` to keep their results [#105](https://github.com/rahulgaur104/AGNI/pull/105).

Bug Fixes

- Fixes the ``jacobi_diffmat`` docstring, which called Gauss-Radau-Jacobi the converged reference for the Zernike path [#105](https://github.com/rahulgaur104/AGNI/pull/105).
- Fixes ``docs/theory.md`` and ``docs/options.md``, which named two different default radial bases [#101](https://github.com/rahulgaur104/AGNI/pull/101).


v0.2.0
------

New Features

- ``eigensolver="dense_mg"`` solves the complex toroidal families [#90](https://github.com/rahulgaur104/AGNI/pull/90).
- Adds the Codecov badge to the README again [#93](https://github.com/rahulgaur104/AGNI/pull/93).

Breaking Changes and Deprecations

- Removes ``growth_rate_of``; ``growth_rate`` is the one function for ``gamma^2``. Optimize mode is ``growth_rate(params, diffmat, ..., equilibrium_map=f)``, and ``on_vector`` or ``sigma`` without ``equilibrium_map`` raise a ``TypeError`` [#95](https://github.com/rahulgaur104/AGNI/pull/95).


v0.1.0
------

The first release on PyPI.

New Features

- Adds ``from_desc``, which loads a DESC ``.h5`` file or ``Equilibrium`` into an ``EquilibriumData``, and ``agnimhd solve eq.h5`` on the command line [#13](https://github.com/rahulgaur104/AGNI/pull/13).
- Adds ``v_fixed`` (skips the eigensolve) and ``v_guess`` (warm start) to ``growth_rate`` and ``eigenpair`` [#9](https://github.com/rahulgaur104/AGNI/pull/9).
- Adds ``eigensolver="jd"``, matrix-free Jacobi-Davidson with the ring preconditioner and coarse-level deflation [#15](https://github.com/rahulgaur104/AGNI/pull/15); with the production coarse level (fewer angular nodes), always on, for any radial basis and family [#38](https://github.com/rahulgaur104/AGNI/pull/38); started from the softest coarse modes [#42](https://github.com/rahulgaur104/AGNI/pull/42); on a coupled Zernike basis [#44](https://github.com/rahulgaur104/AGNI/pull/44); without forming the transformed coarse matrix [#48](https://github.com/rahulgaur104/AGNI/pull/48).
- Adds ``eigensolver="dense_mg"``, dense shift-invert split over several GPUs with JAXMg, and ``assemble_rows`` for any block of rows of the matrix [#21](https://github.com/rahulgaur104/AGNI/pull/21).
- Adds ``AgniStability``, the growth rate as a DESC objective, in ``agnimhd.adapters.desc_objective`` [#19](https://github.com/rahulgaur104/AGNI/pull/19); differentiable under DESC's ``jit`` [#37](https://github.com/rahulgaur104/AGNI/pull/37); with the JD solver [#50](https://github.com/rahulgaur104/AGNI/pull/50), optional density weighting [#56](https://github.com/rahulgaur104/AGNI/pull/56), a warm start from the previous eigenvector [#80](https://github.com/rahulgaur104/AGNI/pull/80) and an adaptive JD shift [#82](https://github.com/rahulgaur104/AGNI/pull/82).
- Adds ``Basis``, which chooses the nodes and derivative matrices by keyword [#27](https://github.com/rahulgaur104/AGNI/pull/27), and ``radial="zernike"``, the DSHAPE tokamak of the AGNI paper [#31](https://github.com/rahulgaur104/AGNI/pull/31).
- Adds toroidal mode families: every family ``n = x + k NFP`` is solved on one field period [#29](https://github.com/rahulgaur104/AGNI/pull/29).
- Adds ``AssemblyConfig(parity="even" | "odd")``, which solves one stellarator-symmetry block of the operator [#70](https://github.com/rahulgaur104/AGNI/pull/70).
- Adds the optional field ``EquilibriumData.density``; every solver weights the kinetic energy with it [#40](https://github.com/rahulgaur104/AGNI/pull/40).
- Adds ``agnimhd.load`` and ``agnimhd.solve`` [#60](https://github.com/rahulgaur104/AGNI/pull/60).
- ``eigenpair`` and ``growth_rate`` are jitted by default [#46](https://github.com/rahulgaur104/AGNI/pull/46).
- Adds the documentation pages: a compact rewrite for ``from_desc`` and ``jd`` [#17](https://github.com/rahulgaur104/AGNI/pull/17), the roadmap and ``CODE_ACCELERATION.md`` [#23](https://github.com/rahulgaur104/AGNI/pull/23), Installation and Developer guide [#68](https://github.com/rahulgaur104/AGNI/pull/68), Examples with measured output and figures and ``desc_optimization.py`` [#62](https://github.com/rahulgaur104/AGNI/pull/62), a tutorial notebook that runs on a CPU [#72](https://github.com/rahulgaur104/AGNI/pull/72), DESC coupling [#58](https://github.com/rahulgaur104/AGNI/pull/58), and the citation of the preprint [#32](https://github.com/rahulgaur104/AGNI/pull/32).
- Adds CI on every pull request with figure tests against baseline images [#66](https://github.com/rahulgaur104/AGNI/pull/66), runtime benchmarks against the base branch [#78](https://github.com/rahulgaur104/AGNI/pull/78), an aggregate job ``all`` for the branch ruleset [#84](https://github.com/rahulgaur104/AGNI/pull/84), the DESC tests in three jobs and a workflow that times the suite [#86](https://github.com/rahulgaur104/AGNI/pull/86), a CI job with DESC 0.17.3 for the adapter tests [#13](https://github.com/rahulgaur104/AGNI/pull/13) and a test of ``AgniStability`` in a DESC optimization [#74](https://github.com/rahulgaur104/AGNI/pull/74).
- Adds packaging: the version from the git tag, a PyPI release workflow and Zenodo metadata [#76](https://github.com/rahulgaur104/AGNI/pull/76).

Performance Improvements

- ``assemble_dense`` whitens in component-major order, without the permutation round trip [#5](https://github.com/rahulgaur104/AGNI/pull/5).

Breaking Changes and Deprecations

- The eigenvalue returned everywhere is ``gamma^2 = -lambda``, positive when unstable, and ``sigma`` is given in the same convention [#25](https://github.com/rahulgaur104/AGNI/pull/25).
- Removes ``domain`` from ``Basis``: the grid always spans one field period and every toroidal mode family is solved on it [#29](https://github.com/rahulgaur104/AGNI/pull/29).
- ``from_desc(..., density=True)`` stores the density in ``eq_data.density`` and returns ``(eq, diffmat)`` instead of ``(eq, diffmat, density)`` [#40](https://github.com/rahulgaur104/AGNI/pull/40).
- Removes the eigensolvers ``"pcg"`` and ``"pcg_deflated"`` [#52](https://github.com/rahulgaur104/AGNI/pull/52).
- Renames the eigensolver ``"jax_lanczos"`` to ``"dense"``, with a Cholesky factorization by default [#64](https://github.com/rahulgaur104/AGNI/pull/64).
- Removes unused helpers [#54](https://github.com/rahulgaur104/AGNI/pull/54); shorter comments and docstrings, one synthetic axisymmetric fixture and consolidated tests [#11](https://github.com/rahulgaur104/AGNI/pull/11); local paths and machine-specific wording removed from the documentation [#33](https://github.com/rahulgaur104/AGNI/pull/33).

Bug Fixes

- Fixes the conjugate transposes in the Hermitian solver pieces, and makes the ``DiffMat`` weights 1-D so that 2-D quadrature weights work: the fixes made in DESC after the extraction [#3](https://github.com/rahulgaur104/AGNI/pull/3).
- Fixes the mass-density weighting, which reached only ``dense_mg`` [#40](https://github.com/rahulgaur104/AGNI/pull/40).


Initial development
-------------------

AGNI was extracted from its implementation inside DESC ([PR #1893](https://github.com/PlasmaControl/DESC/pull/1893)) into the standalone package ``agnimhd``, between 19 August and 4 September 2026, with commits directly to ``master``.

New Features

- Adds the package, ``pyproject.toml``, the MIT license, pre-commit and flake8 configuration, MkDocs documentation and examples.
- Adds GitHub Actions CI with coverage uploads to Codecov, and splits the test suite across CI jobs.
- Makes the variable names consistent with AGNI in DESC.
- Limits the highest Fourier mode a derivative matrix keeps to what the number of quadrature points can represent.

Bug Fixes

- Fixes a bug in the matrix-free operator that made it non-Hermitian.
