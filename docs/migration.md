# Migrating from AGNI-inside-DESC

AGNI began as a set of modules inside DESC ([PR #1789]). If you have scripts
written against that version, this page is the translation table and the list of
things that changed on purpose.

[PR #1789]: https://github.com/PlasmaControl/DESC/pull/1789

---

## What changed, in one sentence

The solver no longer takes a DESC `Equilibrium`; it takes an
[`EquilibriumData`](interface.md) of plain arrays, and you write the ten-line
conversion.

## Call sites

**Before** — the growth rate came out of `eq.compute`, alongside everything
else DESC computes:

```python
data = eq.compute(
    "finite-n lambda3", grid=grid, diffmat=diffmat,
    incompressible=False, gamma=5/3, v_guess=v0,
)
lam = data["finite-n lambda3"]
```

**After** — two steps, and the first one is yours:

```python
from agnimhd import AssemblyConfig, growth_rate
from agnimhd.basis import standard_grid

nodes, diffmat = standard_grid(n_rho, n_theta, n_zeta, NFP=eq.NFP,
                               automorphism=dict(eps=1e-2, x_0=0.65,
                                                 m_1=2.0, m_2=3.0))
eq_data = desc_to_agnimhd(eq, nodes)          # your adapter; see examples/
lam = growth_rate(eq_data, diffmat, AssemblyConfig(gamma=5/3))
```

`examples/desc_adapter.py` is a complete `desc_to_agnimhd`, and
[docs/adapters.md](adapters.md) has the DESC compute-key table it is built from.

## Configuration

Loose `kwargs` and `AGNI_*` environment variables are now fields of two frozen
dataclasses, `AssemblyConfig` and `SolverConfig`.

| before | after |
|---|---|
| `gamma=...`, `incompressible=...`, `axisym=...`, `coupled_rt=...` | `AssemblyConfig(...)` |
| `sigma=...`, `num_matvecs=...`, `cg_tol=...`, `k_defl=...`, `rr_refine=...` | `SolverConfig(...)` |
| `AGNI_EIGENSOLVER` | `SolverConfig(eigensolver=...)`, env still read as a fallback |
| `AGNI_SIGMA_MODE`, `AGNI_NUM_MATVECS`, `AGNI_RR_REFINE` | corresponding `SolverConfig` fields |

The resolution order is now **keyword first, environment second, default last**.
That is a behaviour change: the old pattern

```python
os.environ.get("AGNI_NUM_MATVECS", str(kwargs.get("num_matvecs", 50)))
```

used the keyword only as the environment's default, so an exported variable
silently discarded an explicit argument. A value you pass now wins.

`AGNI_SIGMA_MODE=track` is **not implemented** and never will be. `adapt` is the
measured best strategy and `fixed` is the other option.

## Gradients

In DESC, AGNI is a compute function of `params` — `R_lmn`, `Z_lmn`, `p_l`,
`i_l`, `Psi` — with the geometry computed from them in the same graph, so
`jax.grad` lands on the parameters. The standalone package contains no
equilibrium solve, so the caller supplies that structure, and the signature
reflects it:

```python
def equilibrium_map(params):                 # the DESC solve and the adapter
    return to_equilibrium_data(solve_equilibrium(params))

g = jax.grad(growth_rate_of)(params, equilibrium_map, diffmat, assembly)
```

`g` has the structure of `params`. **`jax.grad(growth_rate)` on an
`EquilibriumData` raises**: `dlambda/d(EquilibriumData)` is a sensitivity to
grid samples that are in force balance only because a solve put them there, so
it is private and `growth_rate_of` is the only public route to it.

The Hellmann-Feynman machinery is unchanged in substance — the eigensolve
wrapped in a `custom_vjp` with a zero backward rule — but it is now part of the
public contract and tested as such, under `jax.jit` applied by a caller with
`equilibrium_map` static alongside the configs.

## Things that changed because they were wrong

All three were found by the extraction's test suite. The first two are fixed
here and not in the DESC branch; the third is fixed in both, differently. If you
compare numbers against an older run, these are the places they can legitimately
differ.

**`fourier_interp_matrix` ignored its `period`.** The wavenumbers were never
scaled by `2*pi/period`, so at any period other than `2*pi` the result was not
an interpolation operator at all. Measured: at `n = 8`, `period = 2*pi/4`, the
matrix differed from the identity by **0.897** on the trivial `n_src == n_dst`
case; 4e-16 after the fix. The poloidal transfer (`period = 2*pi`) is
bit-identical before and after — only the **toroidal** one was affected, which
means every coarse-to-fine prolongation in a two-level solve was.

**`pcg_deflated` double-counted the seed.** With both a deflation space `Z` and
an initial guess `x0`, the `span(Z)` component of the solution was added twice.
Measured on a synthetic SPD system: relative residual 9.6e-12, answer **89%
wrong**. DESC's production path never hit it — it calls plain `pcg` with an
additive preconditioner and passes the coarse seed as the *Lanczos* start
vector — so this affects you only if you called `pcg_deflated` with both
arguments.

**The complex Hermitian operator could not be solved at all.**
`AssemblyConfig(axisym=True)` analyzes a single toroidal Fourier mode, so
`d/dphi` becomes multiplication by `i n` and the operator is complex Hermitian
rather than real symmetric. Two things were wrong on that path, and both were
silent.

`matfree`'s Lanczos orthonormalized with `Q.T @ Q` instead of
`Q.conj().T @ Q` — the same thing on a real symmetric operator, a different
thing here. The returned Ritz *values* stayed plausible while the Ritz
*vectors* were wrong, so nothing raised. Measured on the shipped QH case at one
zeta plane: the Rayleigh quotient came back `+9.713e-02` against a dense
`-2.660e-03` — an unstable equilibrium reported as stable — with a residual of
`1.14e+03`. Fixed upstream in
[matfree #288](https://github.com/pnkraemer/matfree/pull/288), which is why the
dependency floor is now `matfree>=0.6.2` rather than a preference.

`objective._primal` declared ARPACK's callback output with the default *float*
dtype. `jax.pure_callback` casts the host result to what it is told rather than
checking it, so the `eigsh` path failed inside the callback on every
axisymmetric case. The dtype now comes from
[`assemble.operator_dtype`](api.md).

DESC's branch works around the first of these with a real `2n` embedding,
`[[Re A, -Im A], [Im A, Re A]]`, which doubles the Krylov space. That is not
needed here — against `matfree>=0.6.2` the complex operator is solved directly,
and both eigensolvers reproduce the dense eigenvalue to nine digits.

## Things that are now explicit

* **`a` is an input**, not something recomputed internally, and the definition
  is the `QuadratureGrid` cross-section area integral. The `LinearGrid` value
  differs by 3.76% and the eigenvalue notices.
* **The instability drive** can be supplied as `finite_n_instability_drive` or
  built by AGNI from `J x grad(rho)` and `(B.grad)grad(rho)`. Prefer the
  second: it keeps the `s -> rho` substitution out of your code.
* **`psi_rr`** is in the contract though the solver currently recomputes the
  radial derivative spectrally. Supply it; adapters will not have to change when
  that stops being true.

## What did not change

The physics, the discretization, the node ordering (rho-major), the sign
convention (`lambda < 0` is unstable), the normalization (`a`, and
`B_N = |Psi|/(pi a^2)`), and the numbers. The package reproduces its own DESC
export to **2.3e-10 relative**, which is well inside the eigenvalue's 2.8e-5
relative noise floor.
