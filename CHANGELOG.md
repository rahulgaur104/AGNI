Changelog
=========

v0.3.1
------

New Features

- ``make_mode_precond`` requires ``op_blocks``. The path that built the blocks from the true operator, unused since #173, is removed [#181](https://github.com/rahulgaur104/AGNI/pull/181).
- Documents the penalty and how it is applied, the Fourier-mode preconditioner of JD with its shift ladder, the dense matrix built from matrix-free rows, and JD against the dense solve on the 96x96 tokamak and the 48x48x16 QH case at ``MPOL 8``, penalty 1e-2 to 1e4 [#180](https://github.com/rahulgaur104/AGNI/pull/180).
- ``dense_mg`` moves its shift halfway to the current ``gamma^2`` estimate after every successful factorization, and back halfway to the last shift that worked after a failed one, instead of iterating at a fixed shift. It stops once ``gamma^2`` changes by less than ``SolverConfig.mg_theta_tol`` (new, default 1e-8 relative, 0 turns it off), and all processes share one stop decision, which roundoff used to split and hang the job. On the 16x16x8 stellarator started at 1e-3, ``gamma^2`` agrees with the one-GPU dense solve to 1.6e-8 or better instead of 2 to 9 % [#177](https://github.com/rahulgaur104/AGNI/pull/177).
- ``eigensolver="dense_mg"`` runs on ``jaxmg`` 1.2.0 with jax 0.11.0 as well as on 0.0.9. With 1.x there is one process per GPU, on one node or several, and the matrix is overwritten instead of copied, so the 48x48x16 stellarator, family 1, at ``penalty`` 1e-2 runs at 59.4 GB per GPU on one node instead of running out of memory at 98 GB. Adds ``benchmarks/zernike_jd_vs_dense.py``, which compares JD with the dense solve on the 96x96 tokamak and the 48x48x16 stellarator for ``penalty`` from 1e-2 to 1e4 [#176](https://github.com/rahulgaur104/AGNI/pull/176).
- ``eigensolver="dense"`` and ``"eigsh"`` build ``A`` from the matrix-free operator, 64 rows at a time, instead of with ``assemble_dense``, whose Kronecker-product temporaries are several times the size of ``A``. On the 96x96 DSHAPE tokamak with ``n = 2`` the dense solve peaks at 29.1 GB instead of 68.1 GB and fits on one 80 GB GPU, and its eigenvalue agrees with the earlier CPU result to 2e-8 [#174](https://github.com/rahulgaur104/AGNI/pull/174).
- The Zernike mode blocks of the JD preconditioner come from a second operator whose equilibrium coefficients are averaged over theta and zeta on each radius. It keeps every ``(m, n)`` mode separate, so one product gives the same column of every block. Building the blocks takes ``3 n_rho`` matrix-free products instead of ``3 n_total`` (192 instead of 196,608 at 64x64x16) and 6 to 17 s at every grid up to 64x64x16. JD still solves the true operator, and on the stellarator family 1 at 16x24x8 it takes as many corrections as with the exact blocks [#173](https://github.com/rahulgaur104/AGNI/pull/173).
- JD on the Zernike basis uses a new preconditioner with one block per Fourier mode ``(m, n)``, over all radii and the three displacement components, applied with an FFT in theta and zeta. The ring blocks it replaces cannot hold the Zernike penalty, which couples radii within a mode, and JD on the stellarator took 22 and 55 corrections at ``penalty`` 0.1 and 1 and failed at 1e2 and 1e4. JD now takes 6 to 10 corrections from 0.1 to 1e4 and matches the dense solve. A toroidal family ``x`` is multiplied by ``exp(-i x zeta)`` before the FFT, so the families other than 0 separate into modes as well [#172](https://github.com/rahulgaur104/AGNI/pull/172).
- The ring blocks of the JD preconditioner take each ring's entries of the Zernike penalty from its (rho, theta) and zeta factors instead of forming the ``n_total x n_total`` penalty matrix as a Kronecker product (34 GB at 64x64x16). JD on the Patil QH stellarator at 64x64x16 peaks at 6.0 GB instead of 38.7 GB, with the same ring blocks to rounding [#171](https://github.com/rahulgaur104/AGNI/pull/171).
- The matrix-free operator on the Zernike basis applies each real derivative matrix to the real and imaginary parts of a complex vector in one real product, and applies ``D^T`` by swapping the indices of that product. JAX no longer promotes the dense derivative matrices to complex or forms their transposes, and JD on the 96x96 DSHAPE tokamak with ``n = 2`` peaks at 5.6 GB instead of 53.8 GB. The operator and the ring preconditioner also scatter the two parts separately, which is about 100 times faster on the GPU than scattering the complex vector [#170](https://github.com/rahulgaur104/AGNI/pull/170).

Breaking Changes and Deprecations

- The default eigensolver is ``"dense"`` instead of ``"eigsh"``, and the default shift ``sigma`` is 1e-3, as in the paper, instead of 1e-1, in ``SolverConfig``, ``solve`` and ``agnimhd solve``. A case with ``gamma^2`` above 1e-3 needs a larger ``sigma`` [#187](https://github.com/rahulgaur104/AGNI/pull/187).

Bug Fixes

- The documentation follows the paper's convention, λ the largest eigenvalue, and cites its equations correctly. The penalty, radial-basis and Γ sections follow the measurements after #183, the eigensolver table gives hardware, memory and speed, and the math is in Unicode [#185](https://github.com/rahulgaur104/AGNI/pull/185).
- The Zernike penalty is applied as ``alpha U (U^H u)``, with ``U`` an orthonormal basis of what the derivative matrices do not represent (``unrepresented_space``). ``I - V V^H`` formed as a matrix has eigenvalues of about -1e-15, which the whitening turns into negative energies of order ``alpha``: spurious unstable modes, failed dense Cholesky factorizations, and ``gamma^2`` too large by up to 4e-5 (relative) at ``penalty`` 1e4. ``DiffMat`` stores ``U`` as ``penalty_complement``; ``penalty_projector`` is ``U U^H`` [#183](https://github.com/rahulgaur104/AGNI/pull/183).
- ``dense_mg`` returns NaN when the jaxmg Cholesky solve fails because the shift is above the softest eigenvalue, as the one-GPU dense solve does. A failed ``jaxmg`` 1.x solve returns a finite but wrong solution, and only its status code reports the failure [#176](https://github.com/rahulgaur104/AGNI/pull/176).
- A Zernike mode block of the JD preconditioner that is not positive definite at the shift is factored at the first of 10, 100 and 1000 times the shift where it is (``SHIFT_LADDER``), and JD no longer returns NaN because of it. JD keeps its target and every other block is unchanged. Adds tests of JD against the dense solve on the tokamak and the stellarator (24x24x8, MPOL 8) at ``penalty`` 1e-2 and 1e4 [#175](https://github.com/rahulgaur104/AGNI/pull/175).

v0.3.0
------

New Features

- Separable bases solve with ``eigsh`` and ``dense`` on the displacements the angular derivative matrices represent (``Basis(restrict=...)``, by Rogerio Jorge), leaving out the Nyquist and truncated modes they map to zero, which on an even-``NFP`` torus can pose as a spurious instability; ``jd``, ``dense_mg``, a parity and the Zernike basis use the penalty. Both are now the default: ``Basis(restrict=None)`` restricts where it can, ``True`` insists, ``False`` always uses the penalty. ``Basis(penalty=None)`` takes its strength from ``AssemblyConfig.axisym``: ``DEFAULT_PENALTY_ALPHA_AXISYM = 1.0`` for a tokamak, ``DEFAULT_PENALTY_ALPHA = 0.05`` in 3D [#156](https://github.com/rahulgaur104/AGNI/pull/156).
- Applies ``penalty`` to the Lobatto and Gauss-Radau-Jacobi bases, on the nodal modes ``D_theta`` maps to zero, and to the null modes of ``D_zeta`` on every basis (``DiffMat.penalty_projector_zeta``): the Nyquist modes and modes above ``mpol`` or ``ntor`` no longer pose as instabilities. The CLI takes it as ``--penalty`` (0 for the plain nodal operator) [#148](https://github.com/rahulgaur104/AGNI/pull/148).
- Adds ``agnimhd.plotting.mode_delta_b``, the perturbed magnetic field ``delta B = curl(xi x B)`` of an eigenvector in contravariant PEST components; without pressure and current, ``int |delta B|^2 dV`` equals the operator's ``v^H A v`` [#107](https://github.com/rahulgaur104/AGNI/pull/107).
- Adds Bernstein's double-adiabatic (Chew-Goldberger-Low) energy principle for anisotropic pressure. With the optional ``EquilibriumData`` fields ``p_perp``, ``p_par``, ``grad_p_perp``, ``grad_p_par``, ``grad_lnB``, ``T_b``, ``J_sup_rho`` and ``J_sup_theta``, supplied together, every solver assembles it instead of the isotropic energy principle; ``matfree_operator`` also returns ``Ax_full``, the operator on the full ``3 * n_total`` vector [#88](https://github.com/rahulgaur104/AGNI/pull/88).
- Adds ``agnimhd.desc_from_eqdsk``, which turns an EFIT g-file into a fixed-boundary DESC equilibrium with the g-file's rotational transform and solves it, ready for ``from_desc``. It is a port of ``convert_EFIT_to_DESC`` from efit2desc and needs no OMFIT [#99](https://github.com/rahulgaur104/AGNI/pull/99).
- ``Basis`` uses the coupled Zernike-Fourier radial basis by default (``radial="zernike"``), which is regular at the magnetic axis. The default ``automorphism`` now applies only to ``"lobatto"`` and ``"gauss_radau_jacobi"``, and ``agnimhd solve --radial`` defaults to ``zernike`` too [#105](https://github.com/rahulgaur104/AGNI/pull/105).
- Adds ``DiffMat.coupled_rt``: the assembly reads coupled (rho, theta) mode from the derivative matrices themselves, so a Zernike basis works with the default ``AssemblyConfig()`` [#105](https://github.com/rahulgaur104/AGNI/pull/105).
- Adds ``CHANGELOG.md`` [#103](https://github.com/rahulgaur104/AGNI/pull/103).
- Rewrites the documentation of the basis in ``docs/options.md`` as one section with a single table of the radial bases, and renames ``examples/cross_sections.py`` to ``examples/eigenfunction.py`` [#101](https://github.com/rahulgaur104/AGNI/pull/101).
- Adds the Zenodo DOI to the README and ``CITATION.cff`` [#97](https://github.com/rahulgaur104/AGNI/pull/97).

Breaking Changes and Deprecations

- ``EquilibriumData`` no longer takes ``Psi``: the solver uses only ``psi = Psi / (2 pi)`` through ``psi_r`` and ``psi_rr``, and normalizes the field by ``B_N = |psi_rr| / a^2``. ``Psi`` is no longer a pytree leaf or saved, and the file format is version 2 [#151](https://github.com/rahulgaur104/AGNI/pull/151).
- The penalty is named for what it does, not for the Zernike basis, since every basis uses it now: ``Basis(penalty=...)``, ``DiffMat.penalty_alpha``, ``penalty_projector``, ``penalty_projector_zeta``, ``penalty_rank``, ``penalty_svd_tol``, ``agnimhd.basis.penalty_projector_from_diffmat`` and ``DEFAULT_PENALTY_ALPHA`` replace the ``zernike_penalty`` names, with no aliases [#148](https://github.com/rahulgaur104/AGNI/pull/148).
- Removes ``coupled_rt``, ``n_rho_coupled`` and ``n_theta_coupled`` from ``AssemblyConfig``; nothing replaces them [#105](https://github.com/rahulgaur104/AGNI/pull/105).
- A ``Basis`` built without ``radial`` is now a Zernike basis, so scripts that relied on the old Gauss-Radau-Jacobi default must pass ``radial="gauss_radau_jacobi"`` to keep their results [#105](https://github.com/rahulgaur104/AGNI/pull/105).

Bug Fixes

- ``fourier_diffmat`` takes the sign ``(-1)^(i-j)`` from integer parity: on TPU the float power ``(-1.0) ** (i - j)`` returned +1 for negative odd ``i - j``, which gave wrong ``D_theta``/``D_zeta`` and a wrong ``gamma^2`` [#155](https://github.com/rahulgaur104/AGNI/pull/155).
- Fixes the interface contract in the ``EquilibriumData`` docstrings and ``docs/interface.md``: the adapters that ship, the units of ``g_rr`` (m^2) and ``J_cross_grad_rho`` (A m^-3), ``psi_r`` and ``psi_rr`` as derivatives of ``psi = Psi / (2 pi)``, the dimensionless ``density`` and its reference density, the claim that a kinetic energy density in ``p`` gives NaN (it gives a wrong eigenvalue), the gradient example, the partial and total derivative of an ``equilibrium_map``, and that gradients are first-order reverse mode only [#151](https://github.com/rahulgaur104/AGNI/pull/151).
- Caps pytest below 9.2, which removed the ``Function._nodeid`` that pytest-benchmark 5.3.0 reads, so every benchmark errored, and passes a ``float`` to ``pytest.approx`` in the complex-family gradient test, which failed under pytest 9.2.0 on a JAX scalar [#147](https://github.com/rahulgaur104/AGNI/pull/147).
- ``EquilibriumData.save(path, rho=...)`` records the radial nodes of an ``.npz`` file, and ``agnimhd.load`` then refuses a ``Basis`` whose nodes differ, which used to load silently when the resolution matched, e.g. a Lobatto file on the default Zernike basis. Files without ``rho`` load as before and ``FORMAT_VERSION`` is unchanged [#127](https://github.com/rahulgaur104/AGNI/pull/127).
- ``eigensolver="jd"`` returns NaN when the ring factorization is not finite or when ``jd_outer`` corrections end without a stop test, as the dense path does, instead of a vector that looks converged [#126](https://github.com/rahulgaur104/AGNI/pull/126).
- ``EquilibriumData`` rejects ``NFP`` and ``n_rho``, ``n_theta``, ``n_zeta`` that are zero or not integers, which ``int()`` used to truncate (``n_rho = 3.9`` became 3, ``NFP = 0`` was kept) [#125](https://github.com/rahulgaur104/AGNI/pull/125).
- ``EquilibriumData.validate`` rejects a mirror (``iota`` zero everywhere), which used to pass and give an all-NaN ``A`` because the compressibility terms multiply by ``1 / iota``; mirrors wait for anisotropic pressure [#88](https://github.com/rahulgaur104/AGNI/pull/88) [#124](https://github.com/rahulgaur104/AGNI/pull/124).
- Fixes ``parity``, which was checked only by solve-mode ``growth_rate``: ``eigenpair``, ``solve`` and optimize-mode ``growth_rate`` silently projected a non-symmetric equilibrium onto one parity block. They now run the same checks, and optimize mode refuses ``parity`` [#123](https://github.com/rahulgaur104/AGNI/pull/123).
- Adds a test that ``eigsh`` returns the eigenvalue nearest its shift, not the most unstable one, and a ``SolverConfig.sigma`` note that its residual cannot flag this [#116](https://github.com/rahulgaur104/AGNI/pull/116).
- ``AssemblyConfig(incompressible=True)`` raises ``NotImplementedError``; nothing read the flag, so it silently gave the compressible operator, and ``docs/theory.md`` described a projection that does not exist [#115](https://github.com/rahulgaur104/AGNI/pull/115).
- ``EquilibriumData.validate`` rejects a negative Jacobian ``sqrtg``, which used to pass and give an all-NaN ``A`` [#114](https://github.com/rahulgaur104/AGNI/pull/114).
- ``EquilibriumData.validate`` rejects an optional ``density`` that is zero, negative or not finite, which used to pass and give NaN entries in ``A`` [#113](https://github.com/rahulgaur104/AGNI/pull/113).
- Fixes the assembly's grid check, which compared only the node count, so a DiffMat built for 4x16x3 was accepted for an 8x8x3 equilibrium and the matrices were reshaped wrongly. It now compares ``(n_rho, n_theta, n_zeta)`` [#112](https://github.com/rahulgaur104/AGNI/pull/112).
- Fixes ``leggauss_lob(3)``, which returned no interior node and failed its size assertion, so ``Basis(3, ..., radial="lobatto")`` could not be built. The rule is now ``[-1, 0, 1]`` with weights ``[1/3, 4/3, 1/3]`` [#111](https://github.com/rahulgaur104/AGNI/pull/111).
- Fixes ``mode_displacement``, which multiplied the solver's radial unknown by the SI ``psi'`` instead of the normalized one, so ``xi^rho`` was too large by ``a^2 B_N = |Psi| / pi`` (13.6 on the shipped QH case) and ``|delta V|`` over-weighted it; eigenvalues are unchanged, the eigenfunction figures are redone [#109](https://github.com/rahulgaur104/AGNI/pull/109).
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
