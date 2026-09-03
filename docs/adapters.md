# Writing an adapter

An **adapter** turns your equilibrium into an
[`EquilibriumData`](interface.md). It is about fifty lines, it lives in *your*
repository, and `agnimhd` ships none, because shipping a DESC adapter
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
is not `theta_PEST`), map the coordinates first. That is a root-find, and it is
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
- [ ] `p` is **pressure in pascals**, not `n T` in eV, not a kinetic energy
      density. (Symptom of getting this wrong: `NaN`, not a wrong number.)
- [ ] `a` uses the **cross-section area integral** definition, and you recorded
      which one you used. (Symptom of getting this wrong: a plausible eigenvalue
      that is several percent off. Two DESC definitions differ by 3.76%.)
- [ ] The drive uses `rho`, **not** `s = rho^2`. If you supply the two vector
      fields instead of `finite_n_instability_drive`, this is handled here:
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
  `tol`, and the geometry inherits the root-find's error.
* **Compute `a` on a `QuadratureGrid`.** The `LinearGrid` value differs by 3.76%
  and the eigenvalue notices.

`tools/export_fixture.py` in this repository does exactly this to produce the
test fixture. It is dev-only, imports DESC, and is never imported by the package
or by a test.

## VMEC, GVEC, and others

No adapter is written yet. The same table applies once quantities are on a PEST
grid. The work is the coordinate map from the code's native angle, and the
`s -> rho` substitution if you take the drive from published expressions rather
than building it from the two vector fields.

If the code's radial label is `s = rho^2`, as VMEC's is, then every radial
derivative in the contract must be converted: `d/drho = 2 rho d/ds`. This is the
single most likely source of a wrong answer that still looks like a physical
mode.

---

## Consuming the gradient

An adapter that only converts an equilibrium supports solve mode: one
equilibrium in, one `lambda` out, no derivative.
`jax.grad(agnimhd.growth_rate)(eq, diffmat)` raises, for the reason given in
[Two modes](index.md#two-modes).

Optimize mode requires the adapter to be differentiable, and takes the
equilibrium's parameters together with the map from them to an
`EquilibriumData`:

```python
def equilibrium_map(params):
    data = evaluate_on_pest_grid(params)         # geometry and profiles
    return to_equilibrium_data(data)             # the adapter

g = jax.grad(agnimhd.growth_rate_of)(params, equilibrium_map, diffmat)
```

**`equilibrium_map` contains no equilibrium solve.** It evaluates geometry and
profiles from the spectral coefficients and packs them. Differentiating through
a Newton iteration is not required and is not how the derivative is obtained in
practice.

`g` is therefore a partial derivative at a fixed force balance residual. Force
balance is a constraint on the optimization, enforced by the optimizer. DESC
does this with `ProximalProjection`: after each step the equilibrium is
perturbed and re-solved back onto the constraint surface, and the reduced
derivative

```
d lambda / dc  =  @lambda/@c  -  (@lambda/@x) (@F/@x)^-1 (@F/@c)
```

is assembled from the force balance residual `F`, the equilibrium state
`x = (R_lmn, Z_lmn, L_lmn)` and the free parameters `c`. AGNI and the adapter
supply the `@lambda` factors. DESC supplies `F`, its Jacobians and the
projection. An adapter that breaks the JAX graph, for instance with
`float(np.asarray(...))`, removes the `@lambda` factors and leaves only solve
mode. `examples/desc_adapter.py` is numpy-based and is solve-mode only for that
reason.

`g` has the structure of `params`. Instability is `lambda < 0`, so an optimizer
must raise it and a minimizer requires the negated objective. See
[docs/theory.md](theory.md#sign-convention). Under `jit`, `equilibrium_map` is
a Python callable and therefore static, alongside the two configs:
`jax.jit(jax.grad(growth_rate_of), static_argnums=(1, 3, 4))`.

Two conditions must hold when the composed gradient is checked against a finite
difference, neither of which AGNI can enforce. The equilibrium must be
converged at **both** points, since otherwise the difference measures a solver
residual. And the **mode must not swap** between them, because a mode swap is
indistinguishable from an incorrect gradient. See
[resolution.md](resolution.md) for the width of the usable step window.

If the adapter cannot be made differentiable, only solve mode is available. The
inner factor is not exposed for completing the derivative by hand, because on
its own it is not a derivative with respect to any design variable. The
remaining option is to finite-difference the whole objective, at one
equilibrium solve and one eigensolve per parameter per step.
