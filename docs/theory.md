# Theory

Notation follows Gaur et al. (2026), except that every `s = rho^2` of the
TERPSICHORE expressions is replaced by `rho`.

## Energy principle

Linearized ideal MHD about a static equilibrium gives the potential energy

```
dW = INT dV [ |C|^2 + Gamma p |div xi|^2 - F |xi . grad rho|^2 ]
C  = curl(xi x B) + (J x grad rho) / |grad rho|^2 (xi . grad rho)
F  = 2 (J x grad rho) . ((B . grad) grad rho) / |grad rho|^4
```

(paper Eqs. 16-17, where `|grad rho|^4` is `(g^rr)^2` in the code): field-line
bending, compression, and the instability drive `F`, the only term that can be
negative. With the kinetic energy `dK = INT dV |xi|^2`, the discrete problem is
`A xi = lambda B xi` (paper Eq. 45), `A` from `dW` and `B` from `dK`, real
symmetric for the toroidal mode families 0 and `NFP / 2` and complex Hermitian
for the others and with `axisym=True`, where a single toroidal harmonic is kept
and `d/dphi = i n`. With `lambda = <xi|A|xi> /
<xi|B|xi>` the lowest eigenvalue, agnimhd returns the squared growth rate
`gamma^2 = -lambda`, so `gamma^2 > 0` is unstable (the paper's `lambda`,
Eq. 19, `dW_p = -lambda dK`).

## Coordinates and normalization

PEST coordinates `(rho, theta_PEST, phi)`, with `phi` the geometric toroidal
angle over one field period. Inputs are SI, and the solver normalizes lengths
by `a` and fields by `B_N = |psi_rr| / a^2` in `assemble._normalized_fields`.

## Discretization

Each term of `dW` is a quadrature `xi^T D^T diag(W sqrt(g) E) D xi` with `E` an
equilibrium coefficient at the nodes and `D` a spectral differentiation matrix
(paper Eqs. 26-27). The 3D operators are Kronecker products of 1D ones, except
that the Zernike basis couples `rho` and `theta` into one operator each, and
assembling `dW` rather than the force operator makes `A` symmetric by
construction. `A` is `3N x 3N`, `N = n_rho n_theta n_zeta`.

- Bases: Fourier in both angles. Radially, a coupled Zernike-Fourier
  (rho, theta) basis, regular at the axis (the default of `Basis`),
  Gauss-Radau-Jacobi (the paper's default, chosen for modularity),
  Legendre-Lobatto (the most accurate in the paper's scan, Fig. 7c),
  B-spline or SBP finite differences ([Radial basis](options.md#radial-basis)).
- Radial clustering: the equilibrium is evaluated at `rho_s = f(x)`, with `f`
  the staircase map of paper Eq. 65 (`automorphism_staircase1`). The
  transformation is Eq. 64. The derivative
  and weights transform with `f'`, taken by `jax.grad`.
- Axis: the displacement is rescaled (`xi^rho / psi'`, `iota xi^zeta`,
  `upsilon = xi^theta - iota xi^zeta`, paper Eq. 24) and no node sits on the axis.
- Wall: `xi^rho = 0` on the innermost and outermost surfaces, so
  `n_keep = 3N - 2 n_theta n_zeta` unknowns remain (`assemble.keep_indices`).

## Toroidal mode families

The equilibrium repeats every field period `L = 2 pi / NFP`, so `A` and `B`
commute with the shift `T` that takes `zeta` to `zeta + L`. The eigenvectors can
therefore be chosen as eigenvectors of `T` as well (Bloch's theorem),

```
xi(zeta + L) = exp(2 pi i x / NFP) xi(zeta),        x = 0, 1, ..., NFP - 1,
```

and family `x` holds the toroidal mode numbers `n = x + k NFP`. A mode of family
`x` is known on the whole torus once it is known on one period, so each family
is solved on the one-period grid. The projectors
`P_x = (1/NFP) sum_p exp(-2 pi i x p / NFP) T^p` commute with `A`, so the
full-torus matrix splits into `NFP` independent blocks and its spectrum is the
union of the family spectra, which the tests check to round-off. The
equilibrium coefficients have period `L` and the quadrature weights of one
period are unchanged, so only the toroidal derivative depends on the family.

That derivative is the family block of the Fourier differentiation matrix
`D_full` on the `NFP n_zeta` nodes of the full torus,

```
D_x[i, j] = sum_{p=0}^{NFP-1} D_full[i, j + p n_zeta] exp(2 pi i x p / NFP),     i, j = 0 ... n_zeta - 1.
```

The simpler `NFP D_period + i x I`, acting on `exp(-i x zeta) xi`, keeps a
different window of harmonics. For `x = NFP / 2`, and for every `x != 0` at even
`n_zeta`, it differs from `D_x` by one harmonic, which moved the most unstable
eigenvalue of the QH test case by 14 to 26 %. `ntor` truncates `D_full` before
its family blocks are taken, so every family keeps its modes with
`|n| <= ntor NFP`.

`D_{NFP-x}` is the complex conjugate of `D_x`, so families `x` and `NFP - x`
have the same eigenvalues. Families 0 and `NFP / 2` are real symmetric and the
others complex Hermitian. A tokamak is the case of a single toroidal node, where
`D_zeta = i n` (section 3.1 of the paper).

## Stellarator symmetry

A stellarator-symmetric equilibrium is unchanged under the reflection
`(rho, theta, zeta) -> (rho, -theta, -zeta)`. `B` and `J` change sign under it,
but `delta W_p` and `delta K` are quadratic in them. The reflection acts on a
displacement as

```
(S xi)^rho  (rho, theta, zeta) = + xi^rho  (rho, -theta, -zeta)
(S xi)^theta(rho, theta, zeta) = - xi^theta(rho, -theta, -zeta)
(S xi)^zeta (rho, theta, zeta) = - xi^zeta (rho, -theta, -zeta)
```

and the rescaled unknowns `xi^rho / psi'`, `upsilon` and `iota xi^zeta` (paper
Eq. 24) take the same signs. `S` commutes with `A` and `B` and `S^2 = I`, so
every eigenvector is even (`S xi = xi`) or odd (`S xi = -xi`) and the problem
splits in two. `S` maps family `x` to `-x`, so only families 0 and `NFP / 2`
split this way. For `NFP / 2` the reflection carries an extra factor of -1 on
the nodes that wrap around the period.

On the grid, node `theta_j = 2 pi j / n_theta` mirrors to
`theta_((n_theta - j) mod n_theta)`, and likewise in `zeta`. An even or odd
vector is fixed by its values on half the nodes. With `C` the matrix that builds
it from them (`1/sqrt(2)` on each mirror pair with the component's sign, 1 or 0
on a node that is its own mirror), each block solves
`(C^T A C) u = lambda (C^T B C) u` with about `N / 2` unknowns, and its dense
factorization costs an eighth of the full one (`agnimhd.symmetry`).

## Penalty

The derivative matrices annihilate the nodal content they do not represent: on
the Zernike basis the content off the Zernike polynomials, and on every basis
the Nyquist modes and the modes above `mpol` and `ntor`. That content has no
field-line bending energy, so only the penalty holds it against the pressure
drive. The penalty adds `alpha |U^H u|^2` to the energy of each
displacement component `u`, before the whitening. `U` is an orthonormal basis
of the unrepresented content (`agnimhd.basis.unrepresented_space`). It is the
complement of `V`, the orthonormal basis of what the derivative matrices
represent: the row space of the stacked `D_rho` and `D_theta` from an SVD, plus
the constant. `U` is the last columns of a complete QR of `V`, and
`Q = U U^H = I - V V^H` is the projector onto that content.

The matrix-free operator applies the penalty as `alpha U (U^H u)`, so its energy
cannot be negative. Formed as `I - V V^H`, `Q` has eigenvalues of about -1e-15
on the represented content. The whitening multiplies the penalty by up to 1e13
at 48x48x16, so at a large `alpha` those directions gained a negative energy of
order `alpha`, and a spurious unstable mode appeared (#184). `assemble_dense`
still adds `U U^H` as a matrix. The solvers use it only for the ring blocks of
the JD preconditioner on the separable bases, where its roundoff changes how
many corrections JD needs, not `gamma^2`.

`gamma^2` falls as `alpha` grows, towards the limit in which the displacement
has no unrepresented content. Choosing `alpha` is described in
[Radial basis](options.md#radial-basis) and the effect on the DSHAPE tokamak in
[The penalty](benchmarks.md#the-penalty).

## Reduction to a standard problem

`B` is block diagonal with one 3x3 block per node, so its Cholesky factor
`B = L L^T` costs O(N). With `v = L^T xi` (paper Eq. 46) the problem becomes
`L^-1 A L^-T v = lambda v`. No generalized eigensolve is done anywhere. The whitening is done per node pair directly in the
component-major layout, with no permutation copies of `A`.
`assemble_dense` returns this matrix, and `matfree_operator` applies it
without forming it.

## Eigensolve

The lowest eigenvalue is found by shift-invert (paper Eq. 47), iterating on
`(A + sigma I)^-1` so that eigenvalues near `-sigma` dominate. `sigma` is given
in the convention of `gamma^2`, above the largest one. `eigsh` and
`dense` factor the dense shifted matrix. They build `A` 64 rows at a time from
the matrix-free operator, since row `i` of the Hermitian `A` is the conjugate of
`A e_i`, and never form the Kronecker derivative matrices of `assemble_dense`.
`dense_mg` splits these rows over several GPUs and runs block inverse iteration,
with a shift that moves towards `gamma^2` after every factorization
([Dense solves on several GPUs](multigpu.md)).
`jd` (Jacobi-Davidson) never forms `A`. It grows a search space by solving a
projected correction equation with preconditioned CG. The condition number of
`A + sigma I` is about 1e10, and the softest modes of a coarse level are added
to the preconditioner, `M^-1 + Z (Z^H H Z)^-1 Z^H` (Eq. 55). Projecting them
out of the operator instead returns a wrong-sign eigenvalue.

The preconditioner `M` depends on the radial basis.

- **Separable bases** (Lobatto, Gauss-Radau-Jacobi). One block of
  `A + sigma I` per ring, the `n_theta` nodes at one radius and one toroidal
  angle (paper Eqs. 51-54). It brings the condition number to about 1e8.
- **Zernike basis.** The penalty couples the radii within one Fourier mode
  `(m, n)`, which a ring block cannot hold. `M` therefore has one block per mode,
  over all radii and the three components, applied with an FFT in theta and
  zeta. A family `x` is multiplied by `exp(-i x zeta)` before the FFT, which
  makes `D_zeta` and the zeta penalty diagonal. The blocks come from a second
  operator whose equilibrium coefficients are averaged over theta and zeta on
  each radius. It couples no modes, so one product gives the same column of
  every block, `3 n_rho` products in all. A block that is not positive definite
  at `sigma` is factored at the first of `10 sigma`, `100 sigma` and
  `1000 sigma` where it is. `M` changes only how many corrections JD needs, so
  neither the averaging nor the larger shift changes `gamma^2`.

## Gradient

At an eigenvector, `d lambda / dx = <v| dA/dx |v> / <v|v>` (Hellmann-Feynman,
paper Eq. 59). The eigensolve sits in a `jax.custom_vjp` with a zero backward
rule and minus the Rayleigh quotient is returned, so autodiff of it with `v`
fixed is exactly `d gamma^2 / dx = -d lambda / dx`. Here `x` are the
equilibrium's boundary or profile parameters at fixed force balance (paper
Sec. 5.2), reached through
`growth_rate` with an `equilibrium_map`. The equilibrium is not re-solved
inside the derivative. The eigensolver itself need
not be differentiable, and `dA/dx v` comes from reverse-mode differentiation of
the matrix-free operator without forming a matrix.

## Incompressibility

The compression term is stabilizing. A large `gamma` drives `div xi` toward
zero and stays cheap and differentiable (paper Sec. 6.2). The exact projection
of the paper (Eqs. 61-62) is not implemented: `incompressible=True` raises
`NotImplementedError`. The compressible branch approaches the incompressible
one from the unstable side.

## Anisotropic pressure

With the anisotropy fields of `EquilibriumData` present (`p_perp`, `p_par`,
their partials, `grad_lnB`, `T_b`, `J_sup_rho`, `J_sup_theta`) the solver
assembles Bernstein's double-adiabatic energy principle (Bernstein et al. 1958,
anisotropic-pressure section) with the Chew-Goldberger-Low stress tensor
`P = p_perp (I - bb) + p_par bb`. With `D = div xi`, the field-line stretching
`s = b . (grad xi) . b`, `sigma = p_par - p_perp`, the perturbed unit vector
`e~ = Q_perp / |B|`, `v = (grad xi) . b` and `w = b . grad xi`,

```
dW = INT dV [ |Q|^2 - xi . (j x Q) + D (xi . grad p_perp)
              + (5/3) p_perp D^2 + (1/3) p_perp (D - 3 s)^2
              + s (xi . grad sigma) + sigma (s D + 2 s^2 - e~ . v - e~ . w) ]
```

Checks built into `tests/test_anisotropy.py`: on a uniform cylinder the shear
Alfven form is `(B^2 - sigma) |d xi/dz|^2` (firehose for `sigma > B^2`), the
fast and slow forms give `B^2 + 2 p_perp` and `3 p_par`, and for
`p_par = p_perp = p` the functional exceeds the isotropic one by exactly
`(1/3) INT p (D - 3s)^2`, which also verifies the direct
`- xi.(j x Q) + D xi.grad p` against the rearranged
`|C|^2 - F |xi.grad rho|^2` of the isotropic code.

The unknowns are unchanged: the same component-major vector of `xi^rho =
xi~^rho / psi'`, `upsilon = xi^theta - xi^zeta` and `xi^zeta = iota xi~^zeta`
(paper Eq. 24), the same Dirichlet mask and the same mass matrix. The physical
contravariant components `psi' xi^rho`, `upsilon + xi^zeta`, `xi^zeta / iota`
are formed inside the terms, as the compressibility term (paper Eq. 38) does,
and `Q` is paper Eq. 22. `xi` above is Bernstein's symbol for the displacement.

No force-balance rearrangement is used on this branch: the isotropic
`|C|^2`, `F` and `j^theta = iota j^zeta + p'/psi'` all assume `j x B = grad p`,
`p = p(rho)` and `j . grad rho = 0`, none of which holds once `p_perp` varies on
a surface. The terms are written as a linear map from `xi` to a few fields
(`Q`, `D`, `s`, `e~`, `v + w`, ...) and a node-local pairing. The operator is
the adjoint of that map applied to the pairing (`agnimhd.anisotropy`), which
serves the dense, ring and matrix-free routes alike. `T_ik = e_i . d_k b` is
the only new geometric input. It replaces Christoffel symbols in
`v_i = d_i(xi . b) - xi^k T_ki`, `u_i = xi^k T_ik`, and
`w = (Q + (xi . grad ln|B| + D) B)/|B| + u`.

The CGL closure does not reduce to ideal MHD for the perturbation even when
the equilibrium is isotropic (the `(1/3) p (D - 3 s)^2` term is physical), so
`p_perp = p_par = p` is not a regression of the isotropic code. For a mirror
(`iota = 0`) the third unknown is the unscaled `xi^zeta`, as in the mass matrix.

## References

- R. Gaur et al., AGNI: a differentiable MHD stability solver and optimizer for
  magnetic confinement fusion devices (2026).
- I. B. Bernstein et al., Proc. R. Soc. A 244 (1958).
- D. V. Anderson et al., TERPSICHORE, doi:10.1007/978-1-4613-0659-7_8, Eq. 5.
