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

(paper Eqs. 16-17; `|grad rho|^4` is `(g^rr)^2` in the code): field-line
bending, compression, and the instability drive `F`, the only term that can be
negative. With the kinetic energy `dK = INT dV |xi|^2`, the discrete problem is
`A xi = lambda B xi` (paper Eq. 45), `A` from `dW` and `B` from `dK`, real
symmetric on a 3D grid and complex Hermitian with `axisym=True`, where a single
toroidal harmonic is kept and `d/dphi = i n`. agnimhd returns
`lambda = <xi|A|xi> / <xi|B|xi>`, so `lambda < 0` is unstable (the paper's
Eq. 19, `dW_p = -lambda dK`, uses the opposite sign).

## Coordinates and normalization

PEST coordinates `(rho, theta_PEST, phi)`, with `phi` the geometric toroidal
angle over one field period. Inputs are SI; the solver normalizes lengths by
`a` and fields by `B_N = |Psi| / (pi a^2)` in `assemble._normalized_fields`.

## Discretization

Each term of `dW` is a quadrature `xi^T D^T diag(W sqrt(g) E) D xi` with `E` an
equilibrium coefficient at the nodes and `D` a spectral differentiation matrix
(paper Eqs. 26-27). The 3D operators are Kronecker products of 1D ones, and
assembling `dW` rather than the force operator makes `A` symmetric by
construction. `A` is `3N x 3N`, `N = n_rho n_theta n_zeta`.

- Bases: Fourier in both angles; radially Legendre-Lobatto (default here, the
  most accurate in the paper's scan, Fig. 7c), Gauss-Radau-Jacobi (the paper's
  default, chosen for modularity), Chebyshev, B-spline, SBP finite differences,
  or a coupled Zernike-Fourier (rho, theta) basis.
- Radial clustering: the equilibrium is evaluated at `rho_s = f(x)`, with `f`
  the staircase map of paper Eq. 65 (`automorphism_staircase1`; the
  transformation is Eq. 64). The derivative
  and weights transform with `f'`, taken by `jax.grad`.
- Axis: the displacement is rescaled (`xi^rho / psi'`, `iota xi^zeta`,
  `upsilon = xi^theta - xi^zeta`, paper Eq. 24) and no node sits on the axis.
- Wall: `xi^rho = 0` on the innermost and outermost surfaces, so
  `n_keep = 3N - 2 n_theta n_zeta` unknowns remain (`assemble.keep_indices`).

## Reduction to a standard problem

`B` is block diagonal with one 3x3 block per node, so its Cholesky factor
`B = L L^T` costs O(N). With `v = L^T xi` (paper Eq. 46) the problem becomes
`L^-1 A L^-T v = lambda v`; no generalized eigensolve is done anywhere. The whitening is done per node pair directly in the
component-major layout, with no permutation copies of `A`.
`assemble_dense` returns this matrix; `matfree_operator` applies it without
forming it.

## Eigensolve

The lowest eigenvalue is found by shift-invert (paper Eq. 47, written there
with `sigma I - A` for its sign convention), iterating on `(A - sigma I)^-1` so
that eigenvalues near `sigma` dominate. `eigsh` and `jax_lanczos` factor the
dense shifted matrix. `jd` (Jacobi-Davidson) never forms it: it grows a search
space by solving a projected correction equation with preconditioned CG. The
condition number of `A - sigma I` is about 1e10; its ring block preconditioner
(paper Eqs. 51-54) brings that to about 1e8, and the softest modes of a coarse
level are added to the preconditioner, `M^-1 + Z (Z^H H Z)^-1 Z^H` (Eq. 55).
Projecting them out of the operator instead returns a wrong-sign eigenvalue.

## Gradient

At an eigenvector, `d lambda / dx = <v| dA/dx |v> / <v|v>` (Hellmann-Feynman,
paper Eq. 59). The eigensolve sits in a `jax.custom_vjp` with a zero backward
rule and the Rayleigh quotient is returned, so autodiff of the quotient with
`v` fixed is exactly this derivative. Here `x` are the equilibrium's boundary or
profile parameters at fixed force balance (paper Sec. 5.2), reached through
`growth_rate_of` and an `equilibrium_map`; the equilibrium is not re-solved
inside the derivative. The eigensolver itself need
not be differentiable, and `dA/dx v` comes from reverse-mode differentiation of
the matrix-free operator without forming a matrix.

## Incompressibility

The compression term is stabilizing. A large `gamma` drives `div xi` toward
zero and stays cheap and differentiable (paper Sec. 6.2); `incompressible=True`
projects the compressible part out (paper Eqs. 61-62, `div xi ~ 1e-8`) but needs
a dense Cholesky of the Gram matrix inside the derivative.
The compressible branch approaches the incompressible one from the unstable
side.

## Anisotropic pressure

With the anisotropy fields of `EquilibriumData` present (`p_perp`, `p_par`,
their partials, `grad_lnB`, `T_b`, `J_sup_rho`, `J_sup_theta`) the solver
assembles Bernstein's double-adiabatic energy principle (Bernstein et al. 1958,
anisotropic-pressure section; Chew-Goldberger-Low stress tensor
`P = p_perp (I - bb) + p_par bb`). With `D = div xi`, the field-line stretching
`s = b . (grad xi) . b`, `sigma = p_par - p_perp`, the perturbed unit vector
`e~ = Q_perp / |B|`, `v = (grad xi) . b` and `w = b . grad xi`,

```
dW = INT dV [ |Q|^2 - xi . (j x Q) + D (xi . grad p_perp)
              + (5/3) p_perp D^2 + (1/3) p_perp (D - 3 s)^2
              + s (xi . grad sigma) + sigma (s D + 2 s^2 - e~ . v - e~ . w) ]
```

Checks built into `tests/test_anisotropy.py`: on a uniform cylinder the shear
Alfven form is `(B^2 - sigma) |d xi/dz|^2` (firehose for `sigma > B^2`), the
fast and slow forms give `B^2 + 2 p_perp` and `3 p_par`; for `p_par = p_perp =
p` the functional exceeds the isotropic one by exactly `(1/3) INT p (D - 3s)^2`,
which also verifies the direct `- xi.(j x Q) + D xi.grad p` against the
rearranged `|C|^2 - F |xi.grad rho|^2` of the isotropic code.

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
(`Q`, `D`, `s`, `e~`, `v + w`, ...) and a node-local pairing; the operator is
the adjoint of that map applied to the pairing (`agnimhd.anisotropy`), which
serves the dense, ring and matrix-free routes alike. `T_ik = e_i . d_k b` is
the only new geometric input; it replaces Christoffel symbols in
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
