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
      fields instead of `drive`, this is handled for you — prefer that.
- [ ] `g_sup_rr` is `grad(rho).grad(rho)`, not `1 / g_rr`.
- [ ] `sqrt_g` is nonzero everywhere; `iota` is nonzero everywhere.
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
| `sqrt(g)_PEST` | `sqrt_g` |
| `(sqrt(g)_PEST_r)\|PEST`, `(sqrt(g)_PEST_v)\|PEST`, `(sqrt(g)_PEST_p)\|PEST` | `sqrt_g_r`, `sqrt_g_v`, `sqrt_g_p` |
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

Once the adapter is a JAX function of your own parameters, the composition is
just autodiff:

```python
def objective(params):
    eq_desc = solve_equilibrium(params)          # your code
    eq = to_equilibrium_data(eq_desc)            # your adapter
    return -agnimhd.growth_rate(eq, diffmat)     # minimize instability

g = jax.grad(objective)(params)
```

Note the sign: `growth_rate` returns the energy quotient, so **instability is
`lambda < 0`** and an optimizer must raise it. See
[docs/theory.md](theory.md#sign-convention).

If your adapter is not differentiable — a root-find without a custom rule, a
call into Fortran — you still get `dlambda/d(EquilibriumData)` and can carry it
the rest of the way yourself.
