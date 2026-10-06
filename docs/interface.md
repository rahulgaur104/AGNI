# Interface: `EquilibriumData`

`EquilibriumData` holds everything the solver needs: flat arrays on the PEST
grid and two scalars. `agnimhd.from_desc` fills it from DESC; other codes fill
it directly. Check a result with `agnimhd validate eq.npz -v`.

## Grid and ordering

- Coordinates: PEST `(rho, theta_PEST, phi)`, written `(r, v, p)` in field
  names. `rho = sqrt(psi / psi_edge)`, never `s = rho^2`. `phi` is the geometric
  toroidal angle over `[0, 2 pi / NFP)`, one field period. Every toroidal mode
  number is solved on it, one family at a time (`n = x + k NFP`; see
  [Choosing options](options.md#toroidal-mode-families)).
- Ordering: rho-major. Node `(i, j, k)` has flat index
  `(i * n_theta + j) * n_zeta + k`, so `arr.reshape(n_rho, n_theta, n_zeta)`
  recovers the grid. A wrong ordering does not raise; it solves a different
  problem.
- Units: SI, not normalized. The solver normalizes by `a` and
  `B_N = |Psi| / (pi a^2)`.

## Fields

Scalars: `Psi` (total toroidal flux, Wb) and `a` (minor radius, m). `NFP` and
the resolution are static integers.

Arrays, each of length `n_rho * n_theta * n_zeta`:

| field | meaning |
|---|---|
| `g_rr, g_rv, g_rp, g_vv, g_vp, g_pp` | covariant PEST metric `e_a . e_b` |
| `g_sup_rr` | `grad rho . grad rho` |
| `sqrtg`, `sqrtg_r`, `sqrtg_v`, `sqrtg_p` | PEST Jacobian and its partial derivatives |
| `J_sup_zeta`, `abs_J` | `J^zeta` and `|J|` |
| `iota`, `psi_r`, `psi_rr`, `p`, `p_r` | profiles at every node; `p` is pressure in Pa |

Instability drive, one of:

- `finite_n_instability_drive`, or
- `J_cross_grad_rho` and `B_dot_grad_grad_rho` (shape `(n, 3)`), from which
  AGNI forms `2 (J x grad rho) . ((B . grad) grad rho) / (g^rr)^2`.

The second route avoids the `s -> rho` conversion of the published formula
(TERPSICHORE, Eq. 5), which changes the drive by a rho-dependent factor.

## Two inputs that are easy to get wrong

- `a`: the eigenvalue is very sensitive to it. Use the cross-section area
  definition (DESC: `a` computed on a `QuadratureGrid`). DESC's `LinearGrid`
  value differs by 3.76 % on the test case.
- `p`: pressure in pascals. A kinetic energy density or `n T` in eV gives `NaN`.

## DESC

`from_desc(eq_or_path, basis)` fills `EquilibriumData` from DESC; the keys, the
grids and the differentiable route (`AgniStability`) are in
[DESC coupling](desc.md).

DESC's single key `finite-n instability drive` may replace `J x grad(rho)` and
`(B*grad) grad(rho)`; the export scripts in `tools` use it.

`from_desc` converts through NumPy, which breaks the JAX graph, so its output
serves solve mode only. Optimize mode needs the conversion written in JAX inside
the `equilibrium_map`, as `agnimhd.adapters.desc_objective.AgniStability` does
through DESC's compute functions. If a code's conversion cannot be made differentiable, only
solve mode is available; the remaining option is to finite-difference the whole
objective, one equilibrium solve and one eigensolve per parameter.

## VMEC, GVEC and others

Evaluate the fields above on the PEST grid. VMEC's radial label is `s = rho^2`,
so every radial derivative needs `d/drho = 2 rho d/ds`. Take the nodes and the
`DiffMat` from one `Basis`, `basis.nodes_and_diffmat(NFP, family=x)`, and
evaluate on those nodes; the nodes are the same for every family.

## Saving and loading

```python
eq.save("eq.npz")
eq = agnimhd.EquilibriumData.load("eq.npz")
eq.save_hdf5("eq.h5")                 # needs h5py
```

`save` writes the arrays, the scalars, the resolution, `NFP` and a format
version, nothing else; `load` refuses a newer format version. The `.json`
sidecars in `tests/data` and `examples/data` come from the export scripts in
`tools` and record the source equilibrium, the clustering parameters and a
reference eigenvalue, which the tests read.

`EquilibriumData` is a JAX pytree: arrays and scalars are leaves, resolution and
`NFP` are static. That does not make it a set of design variables:
`jax.grad(growth_rate)` raises (see [Two modes](index.md#two-modes)). Construct
it from traced arrays inside an `equilibrium_map` with `validate=False`.
