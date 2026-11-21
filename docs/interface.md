# Interface: `EquilibriumData`

`EquilibriumData` holds everything the solver needs: flat arrays on the PEST
grid and two scalars. `agnimhd.from_desc` fills it from DESC, and other codes
fill it directly. Check a result with `agnimhd validate eq.npz -v`.

## Grid and ordering

- Coordinates: PEST (ρ, θ<sub>PEST</sub>, φ), written `(r, v, p)` in field
  names. ρ = √(ψ/ψ<sub>edge</sub>), never s = ρ². φ is the geometric
  toroidal angle over [0, 2π/NFP), one field period. Every toroidal mode
  number is solved on it, one family at a time (n = x + k·NFP, see
  [Stellarators](stellarators.md#toroidal-mode-families)).
- Ordering: rho-major. Node `(i, j, k)` has flat index
  `(i * n_theta + j) * n_zeta + k`, so `arr.reshape(n_rho, n_theta, n_zeta)`
  recovers the grid. A wrong ordering does not raise. It solves a different
  problem.
- Units: SI, not normalized. The solver normalizes by a and
  B<sub>N</sub> = |ψ″|/a².

## Fields

Scalar: `a` (minor radius, m). `NFP` and the resolution are static integers.

Arrays, each of length `n_rho * n_theta * n_zeta`:

| field | meaning |
|---|---|
| `g_rr, g_rv, g_rp, g_vv, g_vp, g_pp` | covariant PEST metric e<sub>a</sub>·e<sub>b</sub>, all in m² (e<sub>ρ</sub> = de/dρ is a length) |
| `g_sup_rr` | ∇ρ·∇ρ |
| `sqrtg`, `sqrtg_r`, `sqrtg_v`, `sqrtg_p` | PEST Jacobian, positive, and its partial derivatives |
| `J_sup_zeta`, `abs_J` | J<sup>ζ</sup> and \|J\| |
| `iota`, `psi_r`, `psi_rr`, `p`, `p_r` | profiles at every node (`p` is pressure in Pa) |

`psi_r` and `psi_rr` are the first and second ρ derivatives, in Wb, of
ψ = Ψ/(2π), the toroidal flux Ψ divided by 2π (the paper's and
DESC's ψ). With ρ = √(ψ/ψ<sub>b</sub>), `psi_rr` = 2ψ<sub>b</sub> at every node.
It sets B<sub>N</sub> = |ψ″|/a². AGNI never needs Ψ itself.

Instability drive, one of:

- `finite_n_instability_drive`, or
- `J_cross_grad_rho` (A m⁻³) and `B_dot_grad_grad_rho` (T m⁻²), shape `(n, 3)`, from which
  AGNI forms 2(J×∇ρ)·((B·∇)∇ρ)/(g<sup>ρρ</sup>)².

The second route avoids the s → ρ conversion of the published formula
(TERPSICHORE, Eq. 5), which changes the drive by a ρ-dependent factor.

Anisotropic pressure, all eight together or none (see
[theory](theory.md#anisotropic-pressure)). With them `p`, `p_r` and the drive
are ignored:

| field | shape | meaning |
|---|---|---|
| `p_perp`, `p_par` | `(n,)` | perpendicular and parallel pressure, Pa |
| `grad_p_perp`, `grad_p_par` | `(n, 3)` | partials (d/dρ, d/dθ<sub>PEST</sub>, d/dφ) at fixed PEST coordinates |
| `grad_lnB` | `(n, 3)` | the same partials of ln\|B\| |
| `T_b` | `(n, 3, 3)` | T[i, k] = e<sub>i</sub>·∂<sub>k</sub>b, covariant basis vector e<sub>i</sub> dotted with the partial derivative of the unit vector b = B/\|B\|, and B<sup>i</sup>T[i, k] = 0 is checked |
| `J_sup_rho`, `J_sup_theta` | `(n,)` | J·∇ρ, J·∇θ<sub>PEST</sub>, A m⁻³ |

## Two inputs that are easy to get wrong

- `a`: the eigenvalue is very sensitive to it. Use the cross-section area
  definition (DESC: `a` computed on a `QuadratureGrid`). DESC's `LinearGrid`
  value differs by 3.76 % on the test case.
- `p`: pressure in pascals. A kinetic energy density or n T in eV is not
  caught: `p` enters linearly, so the matrix stays finite and the eigenvalue is
  wrong. Do not rely on a `NaN` as the symptom.

The optional `density` is a dimensionless weight such as `ni / max(ni)`.
γ² is in units of B<sub>N</sub>²/(μ₀ρ₀a²), with ρ₀ the mass
density that a weight of 1 stands for. Declare ρ₀ to express the growth
rate in s⁻¹.

The optional `phi_matrix`, shape `(n_theta * n_zeta, n_theta * n_zeta)` in
meters, is the vacuum response of the plasma boundary: on the outer shell the
periodic scalar potential is `Phi = -phi_matrix @ B_n` (exterior Neumann, `B_n`
along the outward normal), so the vacuum energy is
`+(1/2mu_0) int B_n (phi_matrix @ B_n) dS >= 0`; rows and columns in the shell's theta-outer,
zeta-fastest order. Its presence makes every solver **free-boundary**: `xi^rho`
on the outer shell is kept (`keep_indices(..., free_boundary=True)`) and the
vacuum energy is added. The outer radial node must then be the plasma boundary,
`rho = 1`. `EquilibriumData.free_boundary` reports it.

## DESC

`from_desc(eq_or_path, basis)` fills `EquilibriumData` from DESC. The keys,
the grids and the differentiable route (`AgniStability`) are in
[DESC coupling](desc.md).

DESC's single key `finite-n instability drive` may replace `J x grad(rho)` and
`(B*grad) grad(rho)`, and the export scripts in `tools` use it.

`from_desc` converts through NumPy, which breaks the JAX graph, so its output
serves solve mode only. Optimize mode needs the conversion written in JAX inside
the `equilibrium_map`, as `agnimhd.adapters.desc_objective.AgniStability` does
through DESC's compute functions. If a code's conversion cannot be made differentiable, only
solve mode is available. The remaining option is to finite-difference the whole
objective, one equilibrium solve and one eigensolve per parameter.

## VMEC, GVEC and others

Evaluate the fields above on the PEST grid. VMEC's radial label is s = ρ²,
so every radial derivative needs d/dρ = 2ρ d/ds. Take the nodes and the
`DiffMat` from one `Basis`, `basis.nodes_and_diffmat(NFP, family=x)`, and
evaluate on those nodes. The nodes are the same for every family.

## Saving and loading

```python
eq.save("eq.npz")
eq = agnimhd.EquilibriumData.load("eq.npz")
eq.save_hdf5("eq.h5")                 # needs h5py
gamma2, v, residual = agnimhd.solve("eq.npz", basis)   # basis of the file's nodes
```

`agnimhd.load("eq.npz")` is a source that evaluates only on a basis of the
file's resolution, with the file's own density. Another resolution needs the
equilibrium code.

To solve on a machine without DESC, export the equilibrium once and solve the
file there.

```bash
python tools/export_fixture.py --eq equilibrium.h5 --res 24,12,8 \
    --out case.npz --meta case.json                       # needs DESC
agnimhd solve case.npz --radial lobatto \
    --automorphism '{"eps": 0.01, "x_0": 0.65, "m_1": 2.0, "m_2": 3.0}'
```

The radial basis and `automorphism` must be the ones used at the export, since
they place the radial nodes. A different choice gives a wrong eigenvalue without
an error.

`save` writes the arrays, `a`, the resolution, `NFP`, the format version and,
with `save(path, rho=...)`, the radial nodes, which `agnimhd.load` then checks
against the basis. `load` refuses a newer format version. The `.json` sidecars
in `tests/data` and `examples/data` come from the export scripts in `tools` and
record the source equilibrium, the clustering parameters and, for the QH cases,
a reference eigenvalue that the tests read.

`EquilibriumData` is a JAX pytree: arrays and scalars are leaves, resolution and
`NFP` are static. That does not make it a set of design variables:
`jax.grad(growth_rate)` raises (see [Solve mode and optimize mode](desc.md#solve-mode-and-optimize-mode)). Construct
it from traced arrays inside an `equilibrium_map` with `validate=False`. A map
at fixed state gives the partial derivative of γ², while a map through a
converged equilibrium solve gives the total. Gradients are first-order, reverse
mode only (`jax.grad`, `jax.vjp`): `jax.jvp` and second derivatives are not
supported.
