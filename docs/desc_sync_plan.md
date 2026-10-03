# Syncing agnimhd with AGNI-in-DESC, and the two integration directions

Written 2026-10-02. Line numbers refer to the files at that date. Developer
notes, not part of the user documentation.

Status 2026-10-02: section 1 items 1-7 done on branch `sync-desc-10-02`;
section 2 (JD) done on `jdclean`; section 4 tier 1 done as `agnimhd.from_desc`;
section 5 sketched in `examples/desc_objective.py`. Open: items 8-9, the native
`.h5` reader (tier 2), regenerating the fixtures with the current DESC.

`agnimhd` was extracted from DESC commit `f625b0121` (2026-08-18; recorded in
`tests/data/qh_lowres_24x12x8.json` as `desc_version`). DESC's AGNI code on
branch `rg/AGNI_var` moved on through 437ccf2ed (2026-10-02), and branch
`rg/AGNI_var_jdclean` replaced the shift-invert matrix-free solver with a
Jacobi-Davidson (JD) solver (47253fb45, 2026-09-17). This page records what
has to move into `agnimhd`, how the tests change, and how `agnimhd` should talk
to DESC in both directions.

Branch layout in this repository:

| branch | mirrors | content |
|---|---|---|
| `master` | DESC `rg/AGNI_var` up to 437ccf2ed | bug fixes since the extraction + whitening + `v_fixed` + warm start |
| `jdclean` | DESC `rg/AGNI_var_jdclean` (ba8847d86) | `master` + JD as the matrix-free solver, coarse deflation wired in |

A note on JAX versions: `agnimhd` is **not** tied to DESC's pin
(`jax < 0.10`). DESC-free environments can run the newest jax, which is what
the multi-GPU dense solver `jaxmg >= 1.0` needs (it pins jax 0.10/0.11 and
cuSOLVERMp). That is why the multi-GPU dense work happens here, not in DESC.

## 1. `master`: changes to port from DESC (in order)

Each item names the DESC commit, the DESC location and the `agnimhd` location.
Items 1-5 are correctness fixes and go first; each has a test that fails
before and passes after.

1. **Coupled-rt iota placement in the matrix-free operator** (DESC 36bfa01af).
   `src/agnimhd/assemble.py:1148` has `iota * d_dv(D_theta0^T, jq*xr)`; with a
   coupled Zernike `D_theta` iota does not commute with the operator. DESC:
   `d_dv(D_theta0^T, iota*jq*xr)` (`DESC2/DESC/desc/compute/_stability.py:2261`).
   Test: matrix-free vs dense with `coupled_rt=True` (today only the separable
   case is compared, `tests/test_assemble.py:124`).
2. **Hermitian transposes** (DESC cacf77de7). Plain `swapaxes` ->
   `conj(swapaxes)` in `solvers.py:605` (block preconditioner), `:689-702`
   (deflation), `:849-850` (coarse modes), `:949-959` (`deflation_Y`),
   `:1111` (ring blocks). Only matters for complex (axisymmetric n != 0) runs.
   Test: an axisymmetric complex operator, preconditioner must be Hermitian.
3. **Complex Lanczos** (DESC 12ef6a674). `objective.py:95-137` runs real
   symmetric Lanczos on a complex matrix; DESC embeds into a real 2n problem
   (`_stability.py:2906-2937`). Test: `jax_lanczos` vs `eigsh` on an
   axisymmetric case.
4. **Zernike default radial resolution** (DESC f022f4979).
   `basis/zernike.py:381` uses `L = 2*(n_rho-1)`; DESC uses `2*(n_rho//2-1)`
   after measuring `||D_rho||` 4.5e8 -> 1.5e2 at 16x24. The frozen reference
   `tests/data/zernike_reference.npz` was produced with the old formula, so the
   tests that use it must pass `L` explicitly (`tests/conftest.py:136`,
   `tools/export_zernike_reference.py:77`) and a new test pins the new default.
5. **Quadrature weight shape** (DESC f0ad1ed3e). `DiffMat` accepts 2-D
   `W_*` (`basis/diffmat.py:588-597`) but `assemble.py:267` and `:943` `kron`
   them as 1-D. Add `w_rho/w_theta/w_zeta` properties that always return the
   1-D diagonal, use them everywhere, test with a 2-D weight input.
6. **Component-major whitening** (DESC 437ccf2ed; this is the memory saving
   that motivated the JAXMg work). `assemble.py:748-770` still permutes
   `A[p][:, p]`, whitens, permutes back. Replace with the two one-sided
   einsums in component-major order and delete the derivative operators before
   the tail. Bit-identical on DESC's ARIES 8x12x10 check; here the test is
   `dense_eigenvalue_reproduces_reference` plus an exact-equality test against
   the old tail kept as a helper in the test.
7. **`v_fixed` and warm start** (DESC 117526c97, 1ec584a00): `growth_rate(...,
   v_fixed=)` skips the eigensolve and returns the Rayleigh quotient of the
   given vector; `v_guess` seeds eigsh / Lanczos. This is what a DESC objective
   needs for cheap re-evaluations (section 4).
8. **Mode data** (DESC eb18859b7): eigenfunction, xi, delta-B, delta-V on the
   grid as plain arrays, from one function. `plotting.mode_*` already does xi
   and V; add delta-B.
9. **Kron-slice ring assembly** (cacf77de7, `_stability.py:763-790, 828-870`)
   and `lax.map(batch_size=64)` ring batching: only needed once ring
   preconditioning is used from `growth_rate` (section 2).

Two fixes exist only in `agnimhd` and should go the other way (into DESC):
`fourier_interp_matrix` period (`solvers.py:229`) and the double-counted seed
in `pcg_deflated` (`solvers.py:711`). Both are already documented in
`docs/migration.md`.

Dead configuration to remove or wire: `resolve_option`/`resolve_flag`
(`config.py:29,62`) are never called, so no `AGNI_*` environment variable is
read, contrary to `docs/migration.md:55`. `SolverConfig` fields
`coarse_num_matvecs, cg_tol, cg_maxiter, cg_maxiter_cold, k_defl, rr_refine`
are unread because `eigensolver="pcg_deflated"` raises
`NotImplementedError` (`objective.py:199-203`). `AssemblyConfig.gamma`
defaults to 5/3 here, DESC's objective defaults to 0.0: pick one and say so.

## 2. `jdclean`: the Jacobi-Davidson matrix-free solver

Source: `DESC2/DESC_jdclean/desc/compute/_stability.py:3064-3220` (`_jd_solve`),
options at `:2528-2539`, preconditioner build `:3019-3060`, coarse space
`:3226-3300`, objective wiring in `desc/objectives/_stability.py` (`adapt=`,
`jd_*` kwargs). What JD needs that `agnimhd` already has: `ring_block`,
`build_ring_blocks`, `factor_ring_blocks`, `make_block_precond`,
`coarse_seed_and_deflation`, `pcg`/projected CG. What is missing is the
orchestration:

- `solvers.jacobi_davidson(Ax, precond, v0, Z, *, outer, inner, maxdim, keep,
  tol, theta_tol)`: Rayleigh-Ritz on a growing basis, correction equation
  solved by projected PCG with the ring preconditioner, deflated by the coarse
  modes `Z`, restart to `keep` vectors at `maxdim`. Pure JAX, jit-able.
- `objective.growth_rate(..., solver=SolverConfig(eigensolver="jd"))`: build
  the ring preconditioner at `sigma`, build `Z, v0` from a coarse
  `EquilibriumData` (an optional `coarse=(eq_c, diffmat_c)` argument), run
  JD, return the Rayleigh quotient with the same `custom_vjp` as today.
- Keep `pcg_deflated` name as an alias that raises with a message pointing
  to `"jd"`, so old scripts fail loudly.

Tests (fixture 24x12x8, CPU, each under a minute): JD eigenvalue equals the
dense eigenvalue to 1e-6 relative; JD eigenvector residual below 1e-6; JD with
a coarse level (12x6x4) converges in fewer outer iterations than without;
gradient through `growth_rate_of` with JD equals the dense-eigenvector gradient
(DESC measured 2.2e-5 at this resolution); a jitted call compiles once.

## 3. Test revisions

Keep the fixture-driven, DESC-free design. Changes:

- Regenerate `tests/data/qh_lowres_24x12x8.{npz,json}` with the current DESC
  (all exporter keys still exist in DESC 437ccf2ed, checked 10-02) and record
  the DESC commit hash in the sidecar, not only the version string. The
  pinned `dense_lambda3 = -1.3376268714e-4` is expected to stay within the
  present 2.8e-5 tolerance; if it moves, say why in the sidecar.
- Add a second fixture at a resolution with `coupled_rt=True`, because the
  iota-placement bug (item 1) is invisible on the separable fixture.
- Add an axisymmetric fixture (DSHAPE-like, n != 0 so the operator is
  complex) for items 2 and 3.
- `test_solvers.py`: Hermitian checks on every factor; JD convergence tests.
- `test_objective.py`: `v_fixed` round trip; warm-start speedup measured by
  iteration count, not wall time.
- Drop or implement the unread `SolverConfig` fields; test that every field is
  consumed (a static check over `config.py` and the solver sources).
- Markers: keep `slow` for anything above 60 s; CI splits by file.

## 4. Reading and analysing a DESC file inside `agnimhd`

Today nothing in `agnimhd` reads a DESC `.h5`; `cli._load` sends `.h5` to
`EquilibriumData.load_hdf5`, which expects agnimhd's own layout. Two tiers:

**Tier 1 (works now): export from DESC.** `tools/export_fixture.py` is a
complete DESC adapter (key table at `:37-64`). Make it a maintained command in
the DESC-side package (section 5), not a dev tool: `agnimhd-desc export
eq.h5 --res 24,12,8 --out eq_24x12x8.npz`. The output is the interface
contract and carries `desc_commit`, resolution, automorphism and `a`.

**Tier 2 (development): a native reader, no DESC import.** A DESC `.h5` is an
`EquilibriaFamily`: for each equilibrium the Fourier-Zernike coefficients
`R_lmn, Z_lmn, L_lmn` with their bases (`_modes (n,3)` = (l, m, n), `_sym`,
`_spectral_indexing`, `NFP`), `Psi`, and profiles as `PowerSeriesProfile`
params for pressure, current (or iota). Reading that into `EquilibriumData`
needs:

1. `io/desc_h5.py`: parse the family, pick the last equilibrium, load the
   coefficient arrays and mode tables with `h5py` (already an optional extra).
2. `basis/fourier_zernike.py`: evaluate `sum c_lmn Z_l^m(rho) cos/sin(m theta
   - n NFP zeta)` and its first and second derivatives. `basis/zernike.py`
   already has the radial Zernike recurrence (`zernike_radial`,
   `zernike_eval_matrix`); add the toroidal Fourier factor and the
   derivative orders DESC uses (up to second order in each coordinate).
3. PEST mapping: the file stores `lambda` (`L_lmn`), so `theta_PEST = theta +
   lambda(rho, theta, zeta)`. For the tensor grid in `(rho, theta_PEST, phi)`
   solve `theta + lambda(rho, theta, phi) = theta_PEST` per node by Newton
   iteration (vectorised, jit-able; DESC's `map_coordinates` does exactly
   this). The reader must hit DESC's `tol=1e-12` because the geometry inherits
   the error.
4. Metrics and currents: with `R, Z, phi` and their derivatives, the covariant
   basis vectors, the PEST metric `g_ab|PEST`, `sqrt(g)` and its derivatives,
   `B`, `J = curl B / mu0`, `J^zeta`, `|J|`, `J x grad rho` and
   `(B . grad) grad rho` follow DESC's compute formulas (the drive formula is
   in `equilibrium.instability_drive`). Use `jax.jacfwd` for the derivatives
   of the series rather than re-deriving them by hand; the series is cheap.
5. Profiles: power-series evaluation of `p(rho)`, `p_r`; iota from the
   current profile needs DESC's `iota_num/iota_den` surface averages, which
   require flux-surface integrals of the metric. Start with iota-constrained
   files (iota profile stored) and fall back to the exporter for
   current-constrained ones.
6. `a` and `Psi`: `Psi` is stored; `a` must use the QuadratureGrid definition
   (`docs/interface.md`), i.e. a surface integral at rho=1; implement that
   integral in the reader so the 3.76 % trap is closed.
7. Validation: a test that reads `tests/inputs/AGNI_QH_lowres.h5` (to be
   vendored, 1 MB) and reproduces every array of
   `tests/data/qh_lowres_24x12x8.npz` to 1e-10, and the eigenvalue to the
   pinned value. That test is the acceptance criterion for the whole reader.

Effort: steps 1-3 are a day, step 4 is the bulk (two or three days with the
DESC formulas open), 5-7 a day. Until then the exporter is the supported path.

**Analysis** on top of either tier: `agnimhd solve` (eigenvalue, residual,
mode plots), `agnimhd spectrum` (full dense spectrum via `scipy.linalg.eigh`
on CPU today, `jaxmg.syevd` on multi-GPU later), `agnimhd scan --res ...`
(resolution convergence table, the `docs/options.md` procedure as a
command). The dense spectrum is also where `jaxmg.potrs/syevd` plug in: the
reduced matrix `A` from `assemble_dense` is exactly what the JAXMg test driver
(`AGNI_var/ARIES-CS/jaxmg_test_10-02-26/jaxmg_dense_test.py`) shards.

## 5. Loading `agnimhd` into DESC as a stability objective

The dependency direction stays: DESC imports `agnimhd`. The DESC-side code is
one objective class plus the adapter, living in DESC (or a tiny
`agnimhd-desc` package that depends on both). Shape:

```python
class AgniStability(_Objective):
    _coordinates = "rtz"; _units = "(dimensionless)"; _print_value_fmt = "finite-n lambda: "
    _static_attrs = _Objective._static_attrs + ["_res", "_automorphism", "_assembly", "_solver"]

    def __init__(self, eq, *, res, automorphism, assembly=None, solver=None,
                 coarse_res=None, target=0, bounds=None, weight=1, name="agni"):
        ...

    def build(self, use_jit=True, verbose=1):
        # 1. nodes + DiffMat from agnimhd.basis.standard_grid(*res, NFP=eq.NFP)
        # 2. DESC transforms for the KEY_MAP keys on a Grid whose theta is
        #    solved from theta_PEST ONCE here (map_coordinates) -- the same
        #    approach FinitenStability takes in DESC (_mapped_grid); the grid
        #    is re-mapped in compute when L_lmn is a free parameter
        # 3. constants = {"transforms", "profiles", "diffmat", "a_quad_grid"}

    def compute(self, params, constants=None):
        return agnimhd.growth_rate_of(params, self._eq_data, constants["diffmat"],
                                      self._assembly, self._solver)

    def _eq_data(self, params):  # the params -> EquilibriumData map, no solve
        data = compute_fun(eq, KEYS, params=params, transforms=..., profiles=...)
        return agnimhd.EquilibriumData(**{v: data[k] for k, v in KEY_MAP.items()},
                                       J_cross_grad_rho=data["J x grad(rho)"],
                                       B_dot_grad_grad_rho=data["(B*grad) grad(rho)"],
                                       Psi=params["Psi"], a=data["a"], n_rho=..., NFP=eq.NFP)
```

Why this works: `growth_rate_of` (optimize mode) takes that chain as its
`params -> EquilibriumData` map and keeps its eigensolve behind a `custom_vjp`
with zero cotangent, so `jax.grad` through DESC's compute chain gives the
Hellmann-Feynman derivative with respect to `R_lmn, Z_lmn, L_lmn, p_l, c_l,
Psi` for free. Points that need care, all learned in DESC's own
`FinitenStability` (`DESC2/DESC_jdclean/desc/objectives/_stability.py:602`):

- **The PEST grid moves with `L_lmn`.** If lambda is free, `theta` must be
  re-solved inside `compute` (DESC's `map_coordinates` is jit-able with a
  fixed iteration count), otherwise the geometry is evaluated at the wrong
  theta and the eigenvalue drifts. Fix the grid only for fixed-boundary
  solves where ProximalProjection re-solves the equilibrium anyway.
- **Never cache the eigenvector across equilibrium changes.** DESC measured
  that a 7e-5 relative mesh shift sends the Rayleigh residual to about 4800
  and flips the sign. `v_fixed` is only for a second evaluation at the same
  parameters (value + gradient without a second eigensolve).
- **`a` from the QuadratureGrid**, computed in `build` and passed as a
  constant, never from the PEST grid (3.76 % difference, item in
  `docs/interface.md`).
- **Static attributes.** Everything `growth_rate` reads as a Python value
  (resolution, `coupled_rt`, solver names, `sigma`) has to be in
  `_static_attrs`, or the jitted `ObjectiveFunction` tries to trace it.
- **Eigensolver on GPU.** `eigsh` runs on the host through `pure_callback`
  and copies the dense matrix; for optimisation at 3N > 20,000 use the JD
  path (`jdclean` branch) which stays on device.
- **Metric.** Return `lambda` raw and let `target`/`bounds` do the work, or
  implement DESC's `shifted_relu` (`w0 * sum + w1 * max` above `lambda0`) as
  `metric=`.

Deliverable for this section: `examples/desc_objective.py` (a runnable
sketch against `tests/inputs/AGNI_QH_lowres.h5`, requires DESC) and, once
`v_fixed` and JD are on `master`/`jdclean`, a DESC pull request that adds the
class next to `FinitenStability` and makes `FinitenStability` delegate to it.
