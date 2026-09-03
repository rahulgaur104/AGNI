# Writing an adapter

An **adapter** turns your equilibrium into an
[`EquilibriumData`](interface.md). It is about fifty lines, it lives in *your*
repository, and `agnimhd` deliberately ships none — shipping a DESC adapter
would mean the package knows about DESC, and the whole point of the extraction
is that it does not.

`examples/desc_adapter.py` is a complete worked reference for DESC. It is not
imported by the package or by any test.

---

## The four steps

### 1. Choose the grid

Tensor product, `n_rho x n_theta x n_zeta`, in PEST straight-field-line
coordinates.

* **Radial**: Legendre-Lobatto nodes pushed through the clustering map
  `agnimhd.quadrature.automorphism_staircase1`. Nodes must start at
  `rho = eps ~ 1e-3..1e-2`, never at 0.
* **Poloidal**: uniform on `[0, 2*pi)`.
* **Toroidal**: uniform on `[0, 2*pi/NFP)`.

Write down the automorphism kwargs. You will need the *same* ones to build the
`DiffMat`, and nothing checks that you did.

### 2. Evaluate the equilibrium at those nodes

In PEST coordinates. If your code is native to a different angle (DESC's `theta`
is not `theta_PEST`), map the coordinates first — that is a root-find, and it is
usually the slowest part of an export.

### 3. Flatten rho-major and pack

```
n = (i * n_theta + j) * n_zeta + k
```

If you built the nodes as a `(n_rho, n_theta, n_zeta)` meshgrid in that order,
`ravel()` is already right.

### 4. Validate

```bash
agnimhd validate my_equilibrium.npz -v
```

Then solve once and compare against anything you already trust.

---

## Checklist

Copy this into your adapter's test.

- [ ] Nodes are **PEST**, and the innermost `rho` is `eps`, not 0.
- [ ] Flattening is **rho-major**. Reshape to `(n_rho, n_theta, n_zeta)` and
      spot-check that `arr[i, j, k]` is the value at node `(i, j, k)`.
- [ ] Toroidal nodes span **one field period** and `NFP` is passed.
- [ ] Everything is **SI and unnormalized**.
- [ ] `p` is **pressure in pascals** — not `n T` in eV, not a kinetic energy
      density. (Symptom of getting this wrong: `NaN`, not a wrong number.)
- [ ] `a` uses the **cross-section area integral** definition, and you recorded
      which one you used. (Symptom of getting this wrong: a plausible eigenvalue
      that is several percent off. Two DESC definitions differ by 3.76%.)
- [ ] The drive uses `rho`, **not** `s = rho^2`. If you supply the two vector
      fields instead of `finite_n_instability_drive`, this is handled for you —
      prefer that.
- [ ] `g_sup_rr` is `grad(rho).grad(rho)`, not `1 / g_rr`.
- [ ] `sqrtg` is nonzero everywhere; `iota` is nonzero everywhere.
- [ ] The `DiffMat` is built on **the same nodes**, with the same automorphism
      kwargs, and `D_zeta`/`W_zeta` carry the `NFP` scaling.
- [ ] Round-trip: `save`, `load`, solve, and compare against the direct solve.

---

## DESC

The mapping is a table. This *is* the adapter:

| DESC compute key | `EquilibriumData` field |
|---|---|
| `g_rr\|PEST`, `g_rv\|PEST`, `g_rp\|PEST`, `g_vv\|PEST`, `g_vp\|PEST`, `g_pp\|PEST` | `g_rr`, `g_rv`, `g_rp`, `g_vv`, `g_vp`, `g_pp` |
| `g^rr` | `g_sup_rr` |
| `sqrt(g)_PEST` | `sqrtg` |
| `(sqrt(g)_PEST_r)\|PEST`, `(sqrt(g)_PEST_v)\|PEST`, `(sqrt(g)_PEST_p)\|PEST` | `sqrtg_r`, `sqrtg_v`, `sqrtg_p` |
| `J^zeta`, `\|J\|` | `J_sup_zeta`, `abs_J` |
| `iota`, `psi_r`, `psi_rr`, `p`, `p_r` | same names |
| `J x grad(rho)`, `(B*grad) grad(rho)` | `J_cross_grad_rho`, `B_dot_grad_grad_rho` |

plus `Psi = eq.params_dict["Psi"]` and `a` from `eq.compute("a")` on a
`QuadratureGrid`.

Two DESC-specific notes:

* **`theta` is not `theta_PEST`.** Build the PEST nodes, then
  `eq.map_coordinates(..., inbasis=("rho", "theta_PEST", "zeta"),
  outbasis=("rho", "theta", "zeta"))` and compute on the result. Use a tight
  `tol`; the geometry inherits the root-find's error.
* **Compute `a` on a `QuadratureGrid`.** The `LinearGrid` value differs by 3.76%
  and the eigenvalue notices.

`tools/export_fixture.py` in this repository does exactly this to produce the
test fixture. It is dev-only, imports DESC, and is never imported by the package
or by a test.

## VMEC, GVEC, and others

No adapter is written yet. The same table applies once quantities are on a PEST
grid; the work is the coordinate map from your code's native angle, and the
`s -> rho` substitution if you take the drive from published expressions rather
than building it from the two vector fields.

If your code's radial label is `s = rho^2` — VMEC's is — then every radial
derivative in the contract must be converted: `d/drho = 2 rho d/ds`. This is the
single most likely source of a wrong answer that still looks like a physical
mode.

---

## Consuming the gradient

An adapter that only converts an equilibrium supports solve mode: one
equilibrium in, one `lambda` out, no derivative. A differentiable adapter is
required for optimize mode. `jax.grad(agnimhd.growth_rate)(eq, diffmat)`
raises; [Two modes](index.md#two-modes) gives the reason. The required chain
rule is

```
d lambda / d params  =  d lambda / d(EquilibriumData)  x  d(EquilibriumData) / d params
    ^ the objective          ^ private, AGNI's             ^ the adapter and solve
```

so the entry point takes the parameters and the map rather than an equilibrium:

```python
def equilibrium_map(params):
    eq_desc = solve_equilibrium(params)          # differentiable
    return to_equilibrium_data(eq_desc)          # the adapter

lam = agnimhd.growth_rate_of(params, equilibrium_map, diffmat)
g = jax.grad(agnimhd.growth_rate_of)(params, equilibrium_map, diffmat)
```

`g` has the structure of `params`. Instability is `lambda < 0`, so an optimizer
must raise it and a minimizer requires the negated objective; see
[docs/theory.md](theory.md#sign-convention). Under `jit`, `equilibrium_map` is
a Python callable and therefore static, alongside the two configs:
`jax.jit(jax.grad(growth_rate_of), static_argnums=(1, 3, 4))`.

[DESC](https://github.com/PlasmaControl/DESC) supplies `solve_equilibrium`, a
gradient through force balance, geometry on the PEST grid, and the outer
optimizer. Note that `examples/desc_adapter.py` is numpy-based and therefore
solve-mode only: a `float(np.asarray(…))` anywhere breaks the graph and the
gradient terminates there.

Two conditions must hold for the composed gradient, neither of which AGNI can
enforce. The equilibrium must be **converged at both finite-difference points**
if the gradient is validated against a finite difference; see
[resolution.md](resolution.md) for the width of the usable step window. And the
**mode must not swap** between those points, since a mode swap is
indistinguishable from an incorrect gradient.

If the equilibrium code is not differentiable — a root-find without a custom
rule, or a call into Fortran — the chain does not close and only solve mode is
available. The inner factor is not exposed for completing the chain by hand,
because on its own it is not a derivative with respect to any design variable.
The remaining option is to finite-difference the whole objective, at one
equilibrium solve and one eigensolve per parameter per step.
