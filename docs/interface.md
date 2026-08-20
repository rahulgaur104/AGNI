# The interface contract: `EquilibriumData`

`EquilibriumData` is the **only** way the solver learns about an equilibrium. It
holds flat arrays and two scalars — nothing else, and in particular no reference
to an equilibrium object, a file handle, or a code-specific type. It can be
constructed with no equilibrium code installed, saved to disk, and reloaded.

`agnimhd` ships **no adapters**. Producing an `EquilibriumData` from a DESC
`Equilibrium`, a VMEC `wout`, or a GVEC state is the consumer's job and lives in
the consumer's repository. What this page owes them is a contract precise enough
to implement against without reading the solver.

Check your adapter's output with

```bash
agnimhd validate my_equilibrium.npz -v
```

which loads the file, runs every structural and finiteness check, and prints the
resolution, the scalars, and the range of every array.

---

## Coordinates, ordering, units

**Coordinates.** PEST straight-field-line `(rho, theta_PEST, phi)`, abbreviated
`(r, v, p)` in the field names.

* `rho = sqrt(psi / psi_edge)` on `(0, 1]` — never `s = rho^2`, anywhere.
* `theta_PEST` on `[0, 2*pi)`.
* `phi` is the **geometric** toroidal angle. With `NFP > 1` the nodes span one
  field period, `[0, 2*pi/NFP)`.

**Ordering.** rho-major. The flat index of node `(i, j, k)` is

```
n = (i * n_theta + j) * n_zeta + k
```

so `numpy.reshape(arr, (n_rho, n_theta, n_zeta))` recovers the tensor structure,
and `numpy.ravel` of a `(n_rho, n_theta, n_zeta)` array is already correct. An
adapter that emits a different ordering **will not raise** — it will solve a
different problem. This is the single most likely way to be wrong.

**Units.** SI, unnormalized. The solver normalizes internally with `a` and
`B_N = |Psi| / (pi a^2)`. Do not pre-normalize.

**Radial nodes.** The innermost surface must sit at `rho = eps` with `eps` in
roughly `[1e-3, 1e-2]`, not at `rho = 0`: several coefficients are singular on
axis. `n_rho >= 3` is required (two of the shells are Dirichlet-constrained).

---

## Required scalars

| name | units | meaning |
|---|---|---|
| `Psi` | Wb | total toroidal flux through the boundary. Signed, following your code's convention. Differentiable. |
| `a` | m | minor radius. **Read the warning below.** Differentiable. |

`NFP` is an ordinary integer keyword, not differentiable; it sets the toroidal
period of the node set.

## Required arrays

Each of shape `(n_nodes,)`, `n_nodes = n_rho * n_theta * n_zeta`, in rho-major
order. Profile quantities are evaluated *at every node*, constant on each `rho`
surface.

### Geometry

| name | units | meaning |
|---|---|---|
| `g_rr`, `g_rv`, `g_rp`, `g_vv`, `g_vp`, `g_pp` | m² | covariant PEST metric, `g_ab = e_a . e_b`, with `e_r = de/drho`, `e_v = de/dtheta_PEST`, `e_p = de/dphi` |
| `g_sup_rr` | m⁻² | `grad(rho) . grad(rho)` — the *contravariant* radial component, not `1/g_rr` |
| `sqrtg` | m³ | PEST Jacobian `e_rho . (e_theta x e_phi)`. Nonzero everywhere; the solver divides by it |
| `sqrtg_r`, `sqrtg_v`, `sqrtg_p` | m³ | partials of `sqrtg` with respect to `rho`, `theta_PEST`, `phi`, at fixed PEST coordinates |

### Current

| name | units | meaning |
|---|---|---|
| `J_sup_zeta` | A m⁻³ | contravariant toroidal current density `J^zeta` |
| `abs_J` | A m⁻² | current density magnitude `|J|` |

The poloidal current is *not* an input: force balance supplies it internally as
`j^theta = iota j^zeta + p' / psi'`.

### Profiles

| name | units | meaning |
|---|---|---|
| `iota` | — | rotational transform. Must be nonzero — a variable change divides by it. A true mirror (`iota == 0`) is detected and routed to a separate mass-matrix branch |
| `psi_r`, `psi_rr` | Wb | `dpsi/drho`, `d²psi/drho²` |
| `p` | Pa | **plasma pressure.** Read the second warning below |
| `p_r` | Pa (per unit rho) | `dp/drho` |

`psi_rr` is part of the contract but is currently recomputed spectrally by the
solver. Supply it anyway, so that adapters do not have to change when that stops
being true.

## The instability drive — one of two ways

Supply **either** `finite_n_instability_drive`, **or** both vector fields and
let AGNI form it. The field name matches DESC's own compute key,
`"finite-n instability drive"`.

| name | shape | units | meaning |
|---|---|---|---|
| `finite_n_instability_drive` | `(n_nodes,)` | T A m⁻¹ | the drive term `F`, precomputed |
| `J_cross_grad_rho` | `(n_nodes, 3)` | A m⁻² | `J x grad(rho)`, Cartesian components |
| `B_dot_grad_grad_rho` | `(n_nodes, 3)` | T m⁻² | `(B . grad) grad(rho)`, Cartesian components |

From the two vector fields,

```
finite_n_instability_drive = 2 * dot(J_cross_grad_rho, B_dot_grad_grad_rho) / g_sup_rr**2
```

(TERPSICHORE doi:10.1007/978-1-4613-0659-7_8 Eq. 5 p. 162, **with `s -> rho`**).
The two routes agree to 2.8e-16 — that is asserted in
`tests/test_equilibrium.py`. If you compute the drive yourself from the
literature, check the `s -> rho` substitution before anything else: without it
the drive is wrong by a rho-dependent factor of order two, which moves the
eigenvalue's magnitude and can flip its sign near marginality.

Providing neither route is an error. The contract is **closed** in the other
direction too: an unrecognized field name raises rather than being ignored, so a
typo cannot silently do nothing.

---

## Two traps

Both produce results that look fine.

### 1. `a` is not a free choice

The eigenvalue is hypersensitive to the minor radius: it enters through
`B_N = |Psi| / (pi a^2)`, and the operator's terms carry `a^2`, `a^3` and `a^4`.
Two defensible definitions in DESC — the `QuadratureGrid` area integral and the
`LinearGrid` boundary line integral — differ by **3.76%** on the shipped QH case,
which moves `lambda` far more than any discretization error you will be chasing.

AGNI's definition is the **cross-section area integral**:

```
A = (1 / n_zeta) * sum_zeta  INT_{S(zeta)} |e_rho x e_theta| dtheta drho
a = sqrt(A / pi)
```

the zeta-average of the enclosed constant-`phi` cross-sectional area, by direct
area quadrature — *not* a boundary line integral, and *not* extrapolated from
the outermost surface. `a` is an explicit field rather than something recomputed
internally exactly so that an adapter has to make this choice consciously.
`EquilibriumData.save` records the definition you used in the sidecar.

### 2. `p` is pressure, not kinetic energy

`p` is the plasma pressure in pascals. A raw kinetic-energy density (`(3/2) n T`)
or a `n*T` in eV produces **`NaN`** out of the assembly rather than a wrong
number, because the compressibility term takes square roots of quantities built
from it. Convert to pressure first. `validate()` asserts finiteness and names
this cause in the message.

---

## Constructing, saving, loading

```python
from agnimhd import EquilibriumData

eq = EquilibriumData(
    n_rho=24, n_theta=12, n_zeta=8, NFP=4,
    Psi=-0.5, a=1.7,
    g_rr=g_rr, ..., p=p, p_r=p_r,
    finite_n_instability_drive=finite_n_instability_drive,
)
eq.save("qh.npz")               # .npz + a .json sidecar with provenance
eq2 = EquilibriumData.load("qh.npz")
eq.save_hdf5("qh.h5")           # optional; requires h5py
```

* **`.npz` is the native format.** It needs nothing beyond numpy. HDF5 is
  available through `save_hdf5`/`load_hdf5` behind an explicit `h5py`
  capability check that raises a named `ImportError` if it is absent.
* `save` writes a **JSON sidecar** alongside the array file recording the
  resolution, `NFP`, `Psi`, `a`, the string you used for `a_definition`, the
  source equilibrium, and — when the exporter computed one — a reference
  eigenvalue. That sidecar is what regression tests read, so that no number in
  this repository is ever typed in from a document.
* `load` refuses a file whose format major version it does not recognize
  (`agnimhd.FORMAT_VERSION`).

## As a JAX pytree

`EquilibriumData` is a registered pytree. **Every array and both scalars are
dynamic leaves**; the resolution and `NFP` are static aux data. So

```python
g = jax.grad(growth_rate)(eq, diffmat)   # g is an EquilibriumData of gradients
float(g.a), float(g.Psi)                 # scalars get gradients too
```

and `eq` may be passed through `jit`, `vmap`, and `scan` boundaries. When
constructing one from traced arrays inside a transformation, pass
`validate=False` — the finiteness checks cannot be evaluated on a tracer.

`eq.replace(a=new_a)` returns a copy with fields replaced, leaving the original
untouched.

## See also

* [docs/adapters.md](adapters.md) — the per-code checklist and a worked DESC
  adapter.
* [docs/theory.md](theory.md) — where each of these quantities enters the energy
  functional.
