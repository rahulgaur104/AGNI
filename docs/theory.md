# Theory

Notation follows Gaur et al. (2026), except that every s = ρ² of the
TERPSICHORE expressions is replaced by ρ.

## Energy principle

Linearized ideal MHD about a static equilibrium gives the potential energy

```
δW = ∫ dV [ |C|² + Γp |∇·ξ|² − F |ξ·∇ρ|² ]
C  = ∇×(ξ×B) + (J×∇ρ) / |∇ρ|² (ξ·∇ρ)
F  = 2 (J×∇ρ)·((B·∇)∇ρ) / |∇ρ|⁴
```

(paper Eqs. 16-17, where |∇ρ|⁴ is (g<sup>ρρ</sup>)² in the code): field-line
bending, compression, and the instability drive F, the only term that can be
negative. With the kinetic energy δK = ∫ dV |ξ|², the discrete problem is
Aξ = λBξ (paper Eq. 45), A from δW and B from δK, real
symmetric for the toroidal mode families 0 and NFP/2 and complex Hermitian
for the others and with `axisym=True`, where a single toroidal harmonic is kept
and d/dφ = i n. As in the paper, A is the matrix of −δW<sub>p</sub>, so λ is
the largest eigenvalue of Aξ = λBξ and δW<sub>p</sub> = −λ δK (paper Eq. 19).
agnimhd returns λ = γ²(a<sub>N</sub>/v<sub>A</sub>)² as `gamma2`, and γ² > 0 is
unstable.

## Coordinates and normalization

PEST coordinates (ρ, θ<sub>PEST</sub>, φ), with φ the geometric toroidal
angle over one field period. Inputs are SI, and the solver normalizes lengths
by a and fields by B<sub>N</sub> = |ψ″|/a² in `assemble._normalized_fields`.

## Discretization

Each term of δW is a quadrature ξᵀ Dᵀ diag(W √g E) D ξ with E an
equilibrium coefficient at the nodes and D a spectral differentiation matrix
(paper Eqs. 26-27). The 3D operators are Kronecker products of 1D ones, except
that the Zernike basis couples ρ and θ into one operator each, and
assembling δW rather than the force operator makes A symmetric by
construction. A is 3N × 3N, N = n<sub>ρ</sub>n<sub>θ</sub>n<sub>ζ</sub>.

- Bases: Fourier in both angles. Radially, a coupled Zernike-Fourier
  (ρ, θ) basis, regular at the axis (the default of `Basis`),
  Gauss-Radau-Jacobi (the paper's default, chosen for modularity),
  Legendre-Lobatto (the most accurate in the paper's scan, Fig. 7c),
  B-spline or SBP finite differences ([Radial basis](options.md#radial-basis)).
- Radial clustering: the equilibrium is evaluated at ρ<sub>s</sub> = f(x), with f
  the staircase map of paper Eq. 65 (`automorphism_staircase1`). The
  transformation is Eq. 64. The derivative
  and weights transform with f′, taken by `jax.grad`.
- Axis: the displacement is rescaled (ξ<sup>ρ</sup>/ψ′, ιξ<sup>ζ</sup>,
  υ = ξ<sup>θ</sup> − ιξ<sup>ζ</sup>, paper Eq. 24) and no node sits on the axis.
- Wall: ξ<sup>ρ</sup> = 0 on the innermost and outermost surfaces, so
  n<sub>keep</sub> = 3N − 2n<sub>θ</sub>n<sub>ζ</sub> unknowns remain (`assemble.keep_indices`).

## Toroidal mode families

The equilibrium repeats every field period L = 2π/NFP, so A and B
commute with the shift T that takes ζ to ζ + L. The eigenvectors can
therefore be chosen as eigenvectors of T as well (Bloch's theorem),

```
ξ(ζ + L) = exp(2πix/NFP) ξ(ζ),        x = 0, 1, …, NFP − 1,
```

and family x holds the toroidal mode numbers n = x + k·NFP. A mode of family
x is known on the whole torus once it is known on one period, so each family
is solved on the one-period grid. The projectors
P<sub>x</sub> = (1/NFP) Σ<sub>p</sub> exp(−2πixp/NFP) Tᵖ commute with A, so the
full-torus matrix splits into NFP independent blocks and its spectrum is the
union of the family spectra, which the tests check to round-off. The
equilibrium coefficients have period L and the quadrature weights of one
period are unchanged, so only the toroidal derivative depends on the family.

That derivative is the family block of the Fourier differentiation matrix
D<sub>full</sub> on the NFP·n<sub>ζ</sub> nodes of the full torus,

```
Dₓ[i, j] = Σₚ₌₀ᴺᶠᴾ⁻¹ D_full[i, j + p n_ζ] exp(2πixp/NFP),     i, j = 0 … n_ζ − 1.
```

The simpler NFP·D<sub>period</sub> + ixI, acting on exp(−ixζ)ξ, keeps a
different window of harmonics. For x = NFP/2, and for every x ≠ 0 at even
n<sub>ζ</sub>, it differs from D<sub>x</sub> by one harmonic, which moved the most unstable
eigenvalue of the QH test case by 14 to 26 %. `ntor` truncates D<sub>full</sub> before
its family blocks are taken, so every family keeps its modes with
|n| ≤ `ntor`·NFP.

D<sub>NFP−x</sub> is the complex conjugate of D<sub>x</sub>, so families x and NFP − x
have the same eigenvalues. Families 0 and NFP/2 are real symmetric and the
others complex Hermitian. A tokamak is the case of a single toroidal node, where
D<sub>ζ</sub> = i n (section 3.1 of the paper).

## Stellarator symmetry

A stellarator-symmetric equilibrium is unchanged under the reflection
(ρ, θ, ζ) → (ρ, −θ, −ζ). B and J change sign under it,
but δW<sub>p</sub> and δK are quadratic in them. The reflection acts on a
displacement as

```
(Sξ)^ρ (ρ, θ, ζ) = + ξ^ρ (ρ, −θ, −ζ)
(Sξ)^θ (ρ, θ, ζ) = − ξ^θ (ρ, −θ, −ζ)
(Sξ)^ζ (ρ, θ, ζ) = − ξ^ζ (ρ, −θ, −ζ)
```

and the rescaled unknowns ξ<sup>ρ</sup>/ψ′, υ and ιξ<sup>ζ</sup> (paper
Eq. 24) take the same signs. S commutes with A and B and S² = I, so
every eigenvector is even (Sξ = ξ) or odd (Sξ = −ξ) and the problem
splits in two. S maps family x to −x, so only families 0 and NFP/2
split this way. For NFP/2 the reflection carries an extra factor of -1 on
the nodes that wrap around the period.

On the grid, node θ<sub>j</sub> = 2πj/n<sub>θ</sub> mirrors to
θ<sub>(n<sub>θ</sub> − j) mod n<sub>θ</sub></sub>, and likewise in ζ. An even or odd
vector is fixed by its values on half the nodes. With C the matrix that builds
it from them (1/√2 on each mirror pair with the component's sign, 1 or 0
on a node that is its own mirror), each block solves
(CᵀAC) u = λ (CᵀBC) u with about N/2 unknowns, and its dense
factorization costs an eighth of the full one (`agnimhd.symmetry`).

## Penalty

The derivative matrices annihilate the nodal content they do not represent: on
the Zernike basis the content off the Zernike polynomials, and on every basis
the Nyquist modes and the modes above `mpol` and `ntor`. That content has no
field-line bending energy, so only the penalty holds it against the pressure
drive. The penalty adds α|Uᴴu|² to the energy of each
displacement component u, before the whitening. U is an orthonormal basis
of the unrepresented content (`agnimhd.basis.unrepresented_space`). It is the
complement of V, the orthonormal basis of what the derivative matrices
represent: the row space of the stacked D<sub>ρ</sub> and D<sub>θ</sub> from an SVD, plus
the constant. U is the last columns of a complete QR of V, and
Q = UUᴴ = I − VVᴴ is the projector onto that content.

The matrix-free operator applies the penalty as αU(Uᴴu), so its energy
cannot be negative. Formed as I − VVᴴ, Q has eigenvalues of about -1e-15
on the represented content. The whitening multiplies the penalty by up to 1e13
at 48×48×16, so at a large α those directions gained a negative energy of
order α, and a spurious unstable mode appeared (#184). `assemble_dense`
still adds UUᴴ as a matrix. The solvers use it only for the ring blocks of
the JD preconditioner on the separable bases, where its roundoff changes how
many corrections JD needs, not γ².

γ² falls as α grows, towards the limit in which the displacement
has no unrepresented content. Choosing α is described in
[Radial basis](options.md#radial-basis) and the effect on the DSHAPE tokamak in
[The penalty](benchmarks.md#the-penalty).

## Reduction to a standard problem

B is block diagonal with one 3×3 block per node, so its Cholesky factor
B = LLᵀ costs O(N). With v = Lᵀξ (paper Eq. 46) the problem becomes
L⁻¹AL⁻ᵀv = λv. No generalized eigensolve is done anywhere. The whitening is done per node pair directly in the
component-major layout, with no permutation copies of A.
`assemble_dense` returns this matrix, and `matfree_operator` applies it
without forming it.

## Eigensolve

The largest eigenvalue is found by shift-invert (paper Eq. 47), iterating on
(σI − A)⁻¹ so that eigenvalues just below σ dominate. σ lies above the largest
eigenvalue. `eigsh` and
`dense` factor the dense shifted matrix. They build A 64 rows at a time from
the matrix-free operator, since row i of the Hermitian A is the conjugate of
Ae<sub>i</sub>, and never form the Kronecker derivative matrices of `assemble_dense`.
`dense_mg` splits these rows over several GPUs and runs block inverse iteration,
with a shift that moves towards γ² after every factorization
([Dense solves on several GPUs](multigpu.md)).
`jd` (Jacobi-Davidson) never forms A. It grows a search space by solving a
projected correction equation with preconditioned CG. The condition number of
σI − A is about 1e10, and the softest modes of a coarse level are added
to the preconditioner, M⁻¹ + Z(ZᴴHZ)⁻¹Zᴴ (Eq. 55). Projecting them
out of the operator instead returns a wrong-sign eigenvalue.

The preconditioner M depends on the radial basis.

- **Separable bases** (Lobatto, Gauss-Radau-Jacobi). One block of
  σI − A per ring, the n<sub>θ</sub> nodes at one radius and one toroidal
  angle (paper Eqs. 51-54). It brings the condition number to about 1e8.
- **Zernike basis.** The penalty couples the radii within one Fourier mode
  (m, n), which a ring block cannot hold. M therefore has one block per mode,
  over all radii and the three components, applied with an FFT in θ and
  ζ. A family x is multiplied by exp(−ixζ) before the FFT, which
  makes D<sub>ζ</sub> and the ζ penalty diagonal. The blocks come from a second
  operator whose equilibrium coefficients are averaged over θ and ζ on
  each radius. It couples no modes, so one product gives the same column of
  every block, 3n<sub>ρ</sub> products in all. A block that is not positive definite
  at σ is factored at the first of 10σ, 100σ and
  1000σ where it is. M changes only how many corrections JD needs, so
  neither the averaging nor the larger shift changes γ².

## Gradient

At an eigenvector, ∂λ/∂x = ⟨v|∂A/∂x|v⟩ / ⟨v|v⟩ (Hellmann-Feynman,
paper Eq. 60). The eigensolve sits in a `jax.custom_vjp` with a zero backward
rule and the Rayleigh quotient at v is returned, so autodiff of it with v
fixed is exactly ∂λ/∂x. Here x are the
equilibrium's boundary or profile parameters at fixed force balance (paper
Sec. 5.2), reached through
`growth_rate` with an `equilibrium_map`. The equilibrium is not re-solved
inside the derivative. The eigensolver itself need
not be differentiable, and (dA/dx)v comes from reverse-mode differentiation of
the matrix-free operator without forming a matrix.

## Incompressibility

The compression term is stabilizing. A large Γ drives ∇·ξ toward
zero and stays cheap and differentiable (paper Sec. 6.2). The exact projection
of the paper (Eqs. 61-62) is not implemented: `incompressible=True` raises
`NotImplementedError`. The compressible branch approaches the incompressible
one from the unstable side.

## Anisotropic pressure

With the anisotropy fields of `EquilibriumData` present (`p_perp`, `p_par`,
their partials, `grad_lnB`, `T_b`, `J_sup_rho`, `J_sup_theta`) the solver
assembles Bernstein's double-adiabatic energy principle (Bernstein et al. 1958,
anisotropic-pressure section) with the Chew-Goldberger-Low stress tensor
P = p<sub>⊥</sub>(I − bb) + p<sub>∥</sub>bb. With D = ∇·ξ, the field-line stretching
s = b·(∇ξ)·b, σ = p<sub>∥</sub> − p<sub>⊥</sub>, the perturbed unit vector
ẽ = Q<sub>⊥</sub>/|B|, v = (∇ξ)·b and w = b·∇ξ,

```
δW = ∫ dV [ |Q|² − ξ·(j×Q) + D (ξ·∇p_⊥)
            + (5/3) p_⊥ D² + (1/3) p_⊥ (D − 3s)²
            + s (ξ·∇σ) + σ (sD + 2s² − ẽ·v − ẽ·w) ]
```

Checks built into `tests/test_anisotropy.py`: on a uniform cylinder the shear
Alfven form is (B² − σ) |dξ/dz|² (firehose for σ > B²), the
fast and slow forms give B² + 2p<sub>⊥</sub> and 3p<sub>∥</sub>, and for
p<sub>∥</sub> = p<sub>⊥</sub> = p the functional exceeds the isotropic one by exactly
(1/3) ∫ p (D − 3s)², which also verifies the direct
−ξ·(j×Q) + Dξ·∇p against the rearranged
|C|² − F|ξ·∇ρ|² of the isotropic code.

The unknowns are unchanged: the same component-major vector of ξ<sup>ρ</sup> =
ξ̃<sup>ρ</sup>/ψ′, υ = ξ<sup>θ</sup> − ξ<sup>ζ</sup> and ξ<sup>ζ</sup> = ιξ̃<sup>ζ</sup>
(paper Eq. 24), the same Dirichlet mask and the same mass matrix. The physical
contravariant components ψ′ξ<sup>ρ</sup>, υ + ξ<sup>ζ</sup>, ξ<sup>ζ</sup>/ι
are formed inside the terms, as the compressibility term (paper Eq. 38) does,
and Q is paper Eq. 22. ξ above is Bernstein's symbol for the displacement.

No force-balance rearrangement is used on this branch: the isotropic
|C|², F and j<sup>θ</sup> = ιj<sup>ζ</sup> + p′/ψ′ all assume j×B = ∇p,
p = p(ρ) and j·∇ρ = 0, none of which holds once p<sub>⊥</sub> varies on
a surface. The terms are written as a linear map from ξ to a few fields
(Q, D, s, ẽ, v + w, …) and a node-local pairing. The operator is
the adjoint of that map applied to the pairing (`agnimhd.anisotropy`), which
serves the dense, ring and matrix-free routes alike. T<sub>ik</sub> = e<sub>i</sub>·∂<sub>k</sub>b is
the only new geometric input. It replaces Christoffel symbols in
v<sub>i</sub> = ∂<sub>i</sub>(ξ·b) − ξ<sup>k</sup>T<sub>ki</sub>, u<sub>i</sub> = ξ<sup>k</sup>T<sub>ik</sub>, and
w = (Q + (ξ·∇ln|B| + D)B)/|B| + u.

The CGL closure does not reduce to ideal MHD for the perturbation even when
the equilibrium is isotropic (the (1/3) p (D − 3s)² term is physical), so
p<sub>⊥</sub> = p<sub>∥</sub> = p is not a regression of the isotropic code. For a mirror
(ι = 0) the third unknown is the unscaled ξ<sup>ζ</sup>, as in the mass matrix.

## References

- R. Gaur et al., AGNI: a differentiable MHD stability solver and optimizer for
  magnetic confinement fusion devices (2026).
- I. B. Bernstein et al., Proc. R. Soc. A 244 (1958).
- D. V. Anderson et al., TERPSICHORE, doi:10.1007/978-1-4613-0659-7_8, Eq. 5.
