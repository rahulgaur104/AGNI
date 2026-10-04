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
toroidal harmonic is kept and `d/dphi = i n`. With `lambda = <xi|A|xi> /
<xi|B|xi>` the lowest eigenvalue, agnimhd returns the squared growth rate
`gamma^2 = -lambda`, so `gamma^2 > 0` is unstable (the paper's `lambda`,
Eq. 19, `dW_p = -lambda dK`).

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

The lowest eigenvalue is found by shift-invert (paper Eq. 47), iterating on
`(A + sigma I)^-1` so that eigenvalues near `-sigma` dominate; `sigma` is given
in the convention of `gamma^2`, above the largest one. `eigsh` and
`jax_lanczos` factor the dense shifted matrix. `jd` (Jacobi-Davidson) never
forms it: it grows a search space by solving a projected correction equation
with preconditioned CG. The
condition number of `A + sigma I` is about 1e10; its ring block preconditioner
(paper Eqs. 51-54) brings that to about 1e8, and the softest modes of a coarse
level are added to the preconditioner, `M^-1 + Z (Z^H H Z)^-1 Z^H` (Eq. 55).
Projecting them out of the operator instead returns a wrong-sign eigenvalue.

## Gradient

At an eigenvector, `d lambda / dx = <v| dA/dx |v> / <v|v>` (Hellmann-Feynman,
paper Eq. 59). The eigensolve sits in a `jax.custom_vjp` with a zero backward
rule and minus the Rayleigh quotient is returned, so autodiff of it with `v`
fixed is exactly `d gamma^2 / dx = -d lambda / dx`. Here `x` are the
equilibrium's boundary or profile parameters at fixed force balance (paper
Sec. 5.2), reached through
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

## References

- R. Gaur et al., AGNI: a differentiable MHD stability solver and optimizer for
  magnetic confinement fusion devices (2026).
- I. B. Bernstein et al., Proc. R. Soc. A 244 (1958).
- D. V. Anderson et al., TERPSICHORE, doi:10.1007/978-1-4613-0659-7_8, Eq. 5.
