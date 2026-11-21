# General flux coordinates

The paper (Gaur et al. 2026, arXiv:2608.01750) writes the equilibrium field in
PEST coordinates, its Eq. 20, and derives every term of the energy integral
from that form. This page derives the same terms for a field written in **any**
flux coordinates, shows that the eigenvalues do not depend on the angles, and
records what was measured. With the angles straight, every equation below is
the paper's.

Terms used on this page:

- *Flux coordinates* (ρ, θ, ζ): ρ labels the nested surfaces the field lies in,
  θ is any poloidal angle and ζ any toroidal angle, both of period 2π (ζ of
  period 2π/NFP on one field period, NFP being the number of field periods).
- *Straight-field-line angles*: angles in which the field lines are straight
  lines in the (θ, ζ) plane of every surface. PEST coordinates are the
  straight-field-line angles whose ζ is the geometric toroidal angle φ.
- *Covariant basis* e<sub>i</sub> = ∂r/∂x<sup>i</sup> and *contravariant basis*
  ∇x<sup>i</sup>, with e<sub>i</sub>·∇x<sup>j</sup> = δ<sub>i</sub><sup>j</sup>;
  a vector has contravariant components A<sup>i</sup> = A·∇x<sup>i</sup> and
  covariant components A<sub>i</sub> = A·e<sub>i</sub>.
- *Jacobian* √g = e<sub>ρ</sub>·(e<sub>θ</sub>×e<sub>ζ</sub>) =
  1/(∇ρ·∇θ×∇ζ), taken positive.
- ψ = Ψ/2π with Ψ the toroidal flux, ψ′ = dψ/dρ, ι the rotational transform,
  p the pressure, j the current density (all equilibrium quantities, the
  paper's subscript 0 dropped), ξ̃ the physical displacement.

## 1. The field

A field that lies in the surfaces has no ρ component,

```
B = B^θ e_θ + B^ζ e_ζ .                                                   (1)
```

Define the two dimensionless functions

```
b^θ = √g B^θ / ψ′ ,        b^ζ = √g B^ζ / ψ′ ,                             (2)

√g B = ψ′ ( b^θ e_θ + b^ζ e_ζ ) .                                         (3)
```

∇·B = (1/√g) ∂<sub>i</sub>(√g B<sup>i</sup>), so the field is divergence-free
exactly when

```
∂_θ b^θ + ∂_ζ b^ζ = 0 .                                                   (4)
```

The toroidal flux between two neighbouring surfaces, through a surface of
constant ζ, is dρ ∮ √g B<sup>ζ</sup> dθ = 2π ψ′ dρ, and the poloidal flux
through a surface of constant θ is dρ ∮ √g B<sup>θ</sup> dζ = 2π ι ψ′ dρ. So

```
(1/2π) ∮ b^ζ dθ = 1 ,        (1/2π) ∮ b^θ dζ = ι                           (5)
```

in every flux coordinate system. Eqs. (4) and (5) together give

```
b^θ = ι − ∂_ζ λ ,        b^ζ = 1 + ∂_θ λ                                   (6)
```

for a periodic function λ(ρ, θ, ζ): b<sup>ζ</sup> − 1 has zero θ-mean, so it is
∂<sub>θ</sub>λ; then (4) makes b<sup>θ</sup> + ∂<sub>ζ</sub>λ independent of θ,
and the part of λ that depends on (ρ, ζ) alone can be chosen so that it is
independent of ζ too, where (5) says it is ι.

**The angles are straight exactly when λ = 0**, that is when
(b<sup>θ</sup>, b<sup>ζ</sup>) = (ι, 1), and (3) is then the paper's Eq. 20.
θ + λ is the straight poloidal angle that goes with the same ζ. In DESC, λ of
Eq. (6) is the code's own λ: θ<sub>PEST</sub> = θ + λ.

Eq. (3) is the only way the direction of the field enters what follows. The
paper's formulation is the special case in which b<sup>θ</sup> and
b<sup>ζ</sup> are constant on a surface.

## 2. The unknowns

Write ξ̃ = ξ̃<sup>ρ</sup> e<sub>ρ</sub> + ξ̃<sup>θ</sup> e<sub>θ</sub> +
ξ̃<sup>ζ</sup> e<sub>ζ</sub>. With e<sub>i</sub>×e<sub>j</sub> = √g
ε<sub>ijk</sub> ∇x<sup>k</sup> and (3),

```
ξ̃ × B = ψ′ ( b^ζ ξ̃^θ − b^θ ξ̃^ζ ) ∇ρ  −  ψ′ b^ζ ξ̃^ρ ∇θ  +  ψ′ b^θ ξ̃^ρ ∇ζ .     (7)
```

The unknowns are

```
X = ξ̃^ρ / ψ′ ,        υ = b^ζ ξ̃^θ − b^θ ξ̃^ζ ,        Z = ι ξ̃^ζ ,              (8)
```

so that (7) reads

```
ξ̃ × B = ψ′ υ ∇ρ  −  ψ′² b^ζ X ∇θ  +  ψ′² b^θ X ∇ζ .                         (9)
```

X and υ are, up to factors of ψ′ and b<sup>ζ</sup>, the covariant components of
ξ̃×B. With straight angles (8) is the paper's Eq. 24 with its υ =
ξ<sup>θ</sup> − ξ<sup>ζ</sup>. The inverse of (8) is

```
ξ̃^ρ = ψ′ X ,        ξ̃^θ = ( υ + b^θ Z/ι ) / b^ζ ,        ξ̃^ζ = Z / ι ,       (10)
```

that is ξ̃ = M u with u = (X, υ, Z) and

```
        ⎛ ψ′    0        0          ⎞
    M = ⎜ 0     1/b^ζ    b^θ/(b^ζ ι) ⎟ .                                   (11)
        ⎝ 0     0        1/ι        ⎠
```

The third column of M is (b<sup>θ</sup>, b<sup>ζ</sup>)/(b<sup>ζ</sup> ι) in the
angles, which is along B: **at fixed X and υ the third unknown is a
displacement along the field**, and it drops out of ξ̃×B, as (9) shows.

(10) divides by b<sup>ζ</sup>, so the formulation needs b<sup>ζ</sup> ≠ 0, that
is B·∇ζ ≠ 0 everywhere. By (6) the straight poloidal angle θ + λ increases
with θ at the rate b<sup>ζ</sup>, so b<sup>ζ</sup> > 0 is the condition for the
straight angle to exist on the same ζ. The general formulation therefore
applies wherever the paper's does, and needs nothing more. The division by ι in
(8) is the paper's; a mirror (ι = 0) takes Z = ξ̃<sup>ζ</sup>, as in the PEST
code.

## 3. Field-line bending

Q = ∇×(ξ̃×B), and the curl of a vector with covariant components
A<sub>i</sub> has contravariant components
(1/√g) ε<sup>ijk</sup> ∂<sub>j</sub>A<sub>k</sub>. From (9),

```
√g Q^ρ = ∂_θ(ψ′² b^θ X) + ∂_ζ(ψ′² b^ζ X) = ψ′² ( b^θ ∂_θ + b^ζ ∂_ζ ) X ,      (12)
√g Q^θ = ψ′ ∂_ζ υ − ∂_ρ( b^θ ψ′² X ) ,                                      (13)
√g Q^ζ = − ψ′ ∂_θ υ − ∂_ρ( b^ζ ψ′² X ) .                                    (14)
```

The second form of (12) uses (4). It is B·∇ acting on ξ̃<sup>ρ</sup>:
√g B·∇ = ψ′(b<sup>θ</sup>∂<sub>θ</sub> + b<sup>ζ</sup>∂<sub>ζ</sub>).
(12) to (14) are the paper's Eq. 25 with ι → b<sup>θ</sup> and 1 →
b<sup>ζ</sup>, both **inside** the radial derivatives. The field-line bending
energy is

```
∫ dV |Q|² = ∫ dρ dθ dζ (1/√g) g_ij q^i q^j ,        q^i = √g Q^i .          (15)
```

## 4. The rest of |C|²

C = Q + c (ξ̃·∇ρ) with c = (j×∇ρ)/|∇ρ|² (paper Eq. 17), so

```
|C|² = |Q|² + 2 ξ̃^ρ c·Q + |c|² (ξ̃^ρ)² .                                   (16)
```

For a current in the surfaces, j = j<sup>θ</sup>e<sub>θ</sub> +
j<sup>ζ</sup>e<sub>ζ</sub>, the covariant components of c are (paper Eq. 39,
which holds in any coordinates)

```
c_ρ = √g ( j^θ g^{ζρ} − j^ζ g^{θρ} ) / g^{ρρ} ,     c_θ = √g j^ζ ,     c_ζ = − √g j^θ ,   (17)
```

and |c|² = |j|²/g<sup>ρρ</sup>. The mixed term is therefore

```
∫ dV ξ̃^ρ c·Q = ∫ dρ dθ dζ √g ψ′ X [ (j^θ g^{ζρ} − j^ζ g^{θρ})/g^{ρρ} q^ρ + j^ζ q^θ − j^θ q^ζ ] ,   (18)
```

the paper's Eq. 40 with (12) to (14) for q. The two contravariant metric
components come from the covariant ones,
g<sup>ρθ</sup> = (g<sub>ρζ</sub>g<sub>θζ</sub> −
g<sub>ρθ</sub>g<sub>ζζ</sub>)/g and g<sup>ρζ</sup> =
(g<sub>ρθ</sub>g<sub>θζ</sub> − g<sub>ρζ</sub>g<sub>θθ</sub>)/g, as before.

Radial force balance supplies j<sup>θ</sup>. j×B =
(j<sup>θ</sup>B<sup>ζ</sup> − j<sup>ζ</sup>B<sup>θ</sup>) √g ∇ρ = ∇p, so with
(2)

```
ψ′ ( j^θ b^ζ − j^ζ b^θ ) = p′ ,        j^θ = ( b^θ j^ζ + p′/ψ′ ) / b^ζ .     (19)
```

With straight angles this is the paper's j<sup>θ</sup> = ι j<sup>ζ</sup> +
p′/ψ′. j<sup>ζ</sup> = j·∇ζ does not depend on the poloidal angle;
j<sup>θ</sup> does, and (19) gives the one that belongs to the angle in use.

## 5. Compression, drive and kinetic energy

```
∇·ξ̃ = (1/√g) [ ∂_ρ(√g ψ′ X) + ∂_θ(√g ξ̃^θ) + ∂_ζ(√g ξ̃^ζ) ]                 (20)
```

with ξ̃<sup>θ</sup> and ξ̃<sup>ζ</sup> from (10): the paper's Eq. 41 with υ +
ξ<sup>ζ</sup> replaced by (υ + b<sup>θ</sup>Z/ι)/b<sup>ζ</sup>.

The instability drive F = 2 (j×∇ρ)·(B·∇∇ρ)/|∇ρ|⁴ is a scalar built from
vectors and ρ, and it multiplies (ξ̃·∇ρ)² = ψ′²X². Neither depends on the
angles: **this term is unchanged** (paper Eq. 42).

The kinetic energy is

```
∫ dV |ξ̃|² = ∫ dρ dθ dζ √g uᵀ ( Mᵀ g M ) u ,                               (21)
```

a 3×3 block per node, with entries

```
XX: ψ′² g_ρρ        Xυ: ψ′ g_ρθ / b^ζ        υυ: g_θθ / (b^ζ)²
XZ: ψ′ ( g_ρθ r + g_ρζ ) / ι
υZ: ( g_θθ r + g_θζ ) / ( b^ζ ι )                     r = b^θ / b^ζ
ZZ: ( g_θθ r² + 2 g_θζ r + g_ζζ ) / ι²                                     (22)
```

With (b<sup>θ</sup>, b<sup>ζ</sup>) = (ι, 1), r = ι and (22) is the paper's
Eq. 43. The block is still diagonal over the nodes, so the O(N) Cholesky
factorization of the paper's section 4.1 is unchanged.

## 6. The eigenvalues do not depend on the angles

Take two flux coordinate systems with the same ρ and ζ and poloidal angles
related by θ′ = θ + Λ(ρ, θ, ζ). Then, by the chain rule on the position vector,

```
e_ρ = e′_ρ + Λ_ρ e′_θ ,                                                    (23)
```

while ∇ρ and ∇ζ are the same vectors in both. From (8) and (9): X = ξ̃·∇ρ/ψ′
and Z = ι ξ̃·∇ζ are scalars of the two systems, and ψ′υ = (ξ̃×B)·e<sub>ρ</sub>,
so by (23) and (9) in the primed system

```
X = X′ ,        Z = Z′ ,        υ = υ′ − ψ′ b′^ζ Λ_ρ X′ .                    (24)
```

For the primed system straight (b′<sup>ζ</sup> = 1, Λ = λ) this is υ =
υ<sub>PEST</sub> − ψ′ λ<sub>ρ</sub> X. The two sets of unknowns are related by
a 3×3 unit lower-triangular matrix at each point,

```
                  ⎛ 1               0   0 ⎞
u = T u′ ,    T = ⎜ −ψ′ b′^ζ Λ_ρ    1   0 ⎟ .                              (25)
                  ⎝ 0               0   1 ⎠
```

δW and δK are integrals of scalars (|C|², (∇·ξ̃)², F (ξ̃·∇ρ)², |ξ̃|²), and the
current of (19) is the same vector in both systems (substitute (6) and
e<sub>θ</sub> = b<sup>ζ</sup>e′<sub>θ</sub>, e<sub>ζ</sub> = e′<sub>ζ</sub> +
λ<sub>ζ</sub>e′<sub>θ</sub> for a straight primed system). So δW[u] =
δW′[u′] and δK[u] = δK′[u′] whenever u = T u′: the two eigenvalue problems
are congruent, A′ = TᵀAT and B′ = TᵀBT, and **have the same eigenvalues**.
T leaves X alone, so the wall condition X = 0 is the same condition in both.

What does depend on the angles is the discretization: uniform nodes in θ are
not uniform in θ′, and a mode that is one Fourier harmonic in one angle is
several in the other. The two discrete eigenvalues agree to their
discretization errors, not to round-off. Section 10 measures this.

## 7. Discretization

Each derivative in (12) to (14) and (20) is a differentiation matrix acting on
nodal values, as in the paper's section 3.1, and b<sup>θ</sup>, b<sup>ζ</sup>
are node values that multiply before or after it:

```
q^ρ = ψ′² ( b^θ D_θ X + b^ζ D_ζ X )
q^θ = ψ′ D_ζ υ − D_ρ( b^θ ψ′² X )
q^ζ = − ψ′ D_θ υ − D_ρ( b^ζ ψ′² X )
D   = ψ′ (∂_ρ ln√g) X + D_ρ(ψ′X) + (∂_θ ln√g) ξ̃^θ + D_θ ξ̃^θ + (∂_ζ ln√g) ξ̃^ζ + D_ζ ξ̃^ζ
```

q<sup>ρ</sup> uses the second form of (12), with the pitch outside the
derivative. That is the choice that reduces to the published discretization on
every basis: on the coupled Zernike basis D<sub>θ</sub> does not commute with
multiplication by a general function of ρ (measured on 8×12 nodes with
`mpol=4`: D<sub>θ</sub>(f x) and f D<sub>θ</sub>x differ by 2.0 × 10⁻³ of
their size for f = 1/(0.4 + 0.3ρ²), and by round-off for f = 0.4 + 0.3ρ²), so
D<sub>θ</sub>(ι X) and ι D<sub>θ</sub>X are different matrices, and the PEST
code has the second. (The first form, the curl of the
nodal products, has the property that
D<sub>ρ</sub>q<sup>ρ</sup> + D<sub>θ</sub>q<sup>θ</sup> +
D<sub>ζ</sub>q<sup>ζ</sup> vanishes identically when the matrices commute. The
energy integral does not use that property.)

The operator is assembled as A = Φᴴ G Φ, where Φ maps the unknowns to the
fields (q, D, ξ̃<sup>ρ</sup>) and G is the node-local Hermitian pairing

```
E = Σ_nodes W [ (1/√g) q̄ᵀ g q + √g ( ξ̃^ρ̄ c̃·q + c.c. ) + √g (|j|²/g^{ρρ}) |ξ̃^ρ|² + √g Γ p |D|² ] ,
c̃ = ( (j^θ g^{ζρ} − j^ζ g^{θρ})/g^{ρρ} , j^ζ , −j^θ ) ,
```

with Φᴴ taken by reverse-mode differentiation of Φ, so A is Hermitian by
construction (`agnimhd.general`). The drive, the penalty, the mass whitening
and the wall condition are the PEST code's, untouched.

## 8. What the solver needs

| | PEST | general |
|---|---|---|
| nodes uniform in | θ<sub>PEST</sub>, φ | any θ, ζ |
| metric, √g and its three partial derivatives | of PEST coordinates | of (ρ, θ, ζ) |
| direction of the field | ι (profile) | `B_sup_theta`, `B_sup_zeta` (two node arrays), and ι |
| j<sup>ζ</sup>, \|j\|, drive, ψ′, ψ″, p, p′, a | the same | the same |
| coordinate map before the solve | Newton solve for θ at every node | none |

`EquilibriumData` takes the two arrays as optional fields. Without them the
data is PEST and the PEST code path runs, unchanged.

Not covered: the anisotropic-pressure fields, or a free boundary (`phi_matrix`),
together with the general ones (`NotImplementedError`). The vacuum term is
written for the PEST form of B·∇ on the boundary.

## 9. Nodes that do not move in the equilibrium code's coordinates

In PEST the nodes are fixed in θ<sub>PEST</sub>, so in the equilibrium code's
coordinates they move whenever λ changes: every evaluation solves θ + λ(ρ, θ,
ζ) = θ<sub>PEST</sub> for θ at every node, and a gradient differentiates
through that solve and through the code's basis functions at the moved nodes.
A formulation that does not need straight angles can keep the nodes where they
are. Two choices are implemented for DESC.

**`coordinates="native"`** (`from_desc`, `AgniStability`, `agnimhd.load`, the
command line): nodes uniform in DESC's own θ. The fields are DESC's own metric,
Jacobian and `B^theta`, `B^zeta`.

**`coordinates="frozen"`** (`AgniStability`): the nodes are the PEST nodes of
the equilibrium at build time, found once. They are uniform in the angle

```
θ′ = θ + λ₀(ρ, θ, ζ) ,        λ₀ = λ at build time,                        (26)
```

which is a fixed function of DESC's coordinates, straight at build time and
close to straight while λ stays close to λ₀: by (6) the pitch in the primed
angles is b′<sup>ζ</sup> = (1 + λ<sub>θ</sub>)/(1 + λ₀<sub>θ</sub>) and
b′<sup>θ</sup> = ι − ∂′<sub>ζ</sub>(λ − λ₀). DESC's fields are moved to the
primed angles algebraically, with s = 1 + λ₀<sub>θ</sub>, a<sub>ρ</sub> =
λ₀<sub>ρ</sub>/s and a<sub>ζ</sub> = λ₀<sub>ζ</sub>/s stored at the nodes
(`agnimhd.adapters.desc.shift_theta`):

```
e′_θ = e_θ / s ,     e′_ρ = e_ρ − a_ρ e_θ ,     e′_ζ = e_ζ − a_ζ e_θ ,     √g′ = √g / s ,
∂′_θ = ∂_θ / s ,     ∂′_ρ = ∂_ρ − a_ρ ∂_θ ,     ∂′_ζ = ∂_ζ − a_ζ ∂_θ ,
B·∇θ′ = s B^θ + λ₀_ζ B^ζ ,     B·∇ζ unchanged.                             (27)
```

The metric follows from the first line, and the derivatives of √g′ from the
second with the second derivatives λ₀<sub>ρθ</sub>, λ₀<sub>θθ</sub>,
λ₀<sub>θζ</sub>.

## 10. Measurements

All on two CPU cores, double precision. "Relative difference" is general/PEST −
1.

**PEST limit.** With (b<sup>θ</sup>, b<sup>ζ</sup>) = (ι, 1) written out, the
general operator against the PEST one, as a fraction of max|A|
(`tests/test_general.py`):

| case | size | dense | matrix-free |
|---|---|---|---|
| QH, 8×8×3, real family | 528 | 1.9 × 10⁻¹⁶ | 2.7 × 10⁻¹⁶ |
| QH, 8×8×3, family 1 (complex) | 528 | 1.9 × 10⁻¹⁶ | 2.9 × 10⁻¹⁶ |
| DSHAPE, 16×48, n = 2, Zernike, penalty 0.01 | 2208 | 3.3 × 10⁻¹⁶ | 9.0 × 10⁻¹⁶ |

**Energies of one displacement**, PEST operator on straight nodes against the
general operator on nodes of angles with λ depending on all three coordinates
(b<sup>ζ</sup> from 0.50 to 1.50, b<sup>θ</sup> from −0.16 to 1.04 with ι from
0.41 to 0.44; a synthetic analytic configuration):

| resolution | δW | δK |
|---|---|---|
| 8×8×4 | 5.3 × 10⁻⁴ | 9.7 × 10⁻³ |
| 12×12×6 | 6.0 × 10⁻⁵ | 2.2 × 10⁻⁴ |
| 16×16×8 | 5.5 × 10⁻⁷ | 4.8 × 10⁻⁷ |
| 20×24×12 | 6.1 × 10⁻¹⁰ | 7.5 × 10⁻¹³ |
| 24×32×16 | 3.5 × 10⁻¹² | below 10⁻¹⁵ |

**Most unstable γ² of one toroidal mode** (n = 1) of the same configuration
with an axisymmetric λ (b<sup>ζ</sup> from 0.70 to 1.50):

| resolution | PEST | general | relative difference |
|---|---|---|---|
| 16×17 | 0.2656454 | 0.2469994 | −7.0 × 10⁻² |
| 20×25 | 0.2670181 | 0.2658280 | −4.5 × 10⁻³ |
| 24×33 | 0.2670203 | 0.2670041 | −6.1 × 10⁻⁵ |

**DESC's own angles are far from straight.** On the nodes, with the two flux
integrals (5) holding to 2 × 10⁻¹⁶ and 4 × 10⁻¹⁶:

| equilibrium | b<sup>ζ</sup> | b<sup>θ</sup> − ι |
|---|---|---|
| QH (`AGNI_QH_lowres.h5`, max ι = 1.10), 12×16×12 nodes | 0.69 to 1.28 | −1.31 to 2.25 (b<sup>θ</sup> changes sign) |
| DSHAPE (`dshape_imax0.98_1608.h5`), 8×8 nodes | 0.33 to 1.67 | 0 (axisymmetric) |

**DSHAPE tokamak, n = 3**, Lobatto radial nodes (default map), no penalty, odd
n<sub>θ</sub>, all eigenvalues by LAPACK; most unstable γ² in PEST and in DESC's
own angles:

| n<sub>ρ</sub> × n<sub>θ</sub> | PEST | DESC's angles | relative difference |
|---|---|---|---|
| 16 × 33 | 8.670 × 10⁻⁴ | 5.829 × 10⁻⁴ | −3.3 × 10⁻¹ |
| 24 × 33 | 5.574 × 10⁻⁴ | 4.278 × 10⁻⁴ | −2.3 × 10⁻¹ |
| 32 × 33 | 2.892 × 10⁻⁴ | 2.840 × 10⁻⁴ | −1.8 × 10⁻² |
| 48 × 25 | 3.261 × 10⁻⁴ | 2.872 × 10⁻⁴ | −1.2 × 10⁻¹ |
| 48 × 33 | 2.898 × 10⁻⁴ | 2.867 × 10⁻⁴ | −1.1 × 10⁻² |
| 48 × 49 | 2.880 × 10⁻⁴ | 2.868 × 10⁻⁴ | −4.3 × 10⁻³ |
| 64 × 49 | 2.879 × 10⁻⁴ | 2.873 × 10⁻⁴ | −2.1 × 10⁻³ |

Below 32 radial nodes neither value is converged. From there the two approach
each other as either resolution grows. In n<sub>θ</sub> at 48 radial nodes the
PEST value moves by 12 % from 25 to 49 nodes and the one in DESC's angles by
0.1 %; in n<sub>ρ</sub> at 33 poloidal nodes, from 32 to 48, the PEST value
moves by 0.2 % and the other by 1.0 %, and at 49 poloidal nodes, from 48 to 64,
by 0.05 % and 0.17 %. The second and third eigenvalues are not radially converged in
either system at these resolutions (the second: 7.7 × 10⁻⁵ and 4.3 × 10⁻⁵ at
48 × 49, 3.5 × 10⁻⁵ and 1.9 × 10⁻⁵ at 64 × 49) and say nothing either way.

**QH stellarator, family 0**, same method, Lobatto nodes through the staircase
map of the test fixtures. Not converged in either
system at what two cores and 7 GB hold, so this is not evidence for or against
agreement:

| n<sub>ρ</sub> × n<sub>θ</sub> × n<sub>ζ</sub> | PEST | DESC's angles |
|---|---|---|
| 12 × 13 × 9 | 3.090 × 10⁻⁴ | 1.372 × 10⁻⁴ |
| 12 × 17 × 13 | 9.656 × 10⁻⁵ | 1.282 × 10⁻⁴ |
| 12 × 25 × 13 | 9.616 × 10⁻⁵ | 9.326 × 10⁻⁵ |
| 10 × 25 × 13 | 6.559 × 10⁻⁵ | 9.020 × 10⁻⁵ |

The last two rows differ only in n<sub>ρ</sub>, and the PEST value changes by
32 % between them.

**How many nodes.** A single PEST harmonic exp i(mθ<sub>PEST</sub> + nζ), seen
in angles with θ<sub>PEST</sub> = θ + λ, has by (6) the local wavenumbers

```
∂_θ :  m b^ζ ,        ∂_ζ :  n + m ( ι − b^θ ) .                            (28)
```

For the QH case above a harmonic with m = 8 therefore reaches toroidal
wavenumbers n ± 18, which is 4.5 field-period harmonics more than in PEST, and
poloidal wavenumbers 1.28 m. An eigenmode is a sum of harmonics and can be
narrower or wider in either angle (the DSHAPE mode above converges faster in
DESC's θ), so (28) is a guide to the node count, not a bound on the error.

**Cost of the geometry and of the objective** (QH, `AgniStability`, 7444
parameters, compiled, best of 2 or 3 calls). "Geometry" is the map from DESC's
parameters to `EquilibriumData`, "its gradient" the reverse-mode derivative of
a scalar of it; "objective" and "Jacobian" include the eigensolve
(`eigensolver="eigsh"`, σ = 0.1):

| nodes | coordinates | geometry | its gradient | objective | Jacobian |
|---|---|---|---|---|---|
| 8×8×3 | `pest` | 444 ms | 951 ms | 1.43 s | 2.47 s |
| 8×8×3 | `native` | 7.5 ms | 14.4 ms | 0.88 s | 1.03 s |
| 8×8×3 | `frozen` | 7.4 ms | 14.4 ms | 0.75 s | 0.96 s |
| 16×12×8 | `pest` | 4160 ms | 11490 ms | 13.95 s | 22.80 s |
| 16×12×8 | `native` | 26.9 ms | 63.4 ms | 8.02 s | 8.18 s |
| 16×12×8 | `frozen` | 27.1 ms | 62.1 ms | 6.38 s | 6.68 s |

**One matrix-free product** (QH, 24×12×8, 6720 unknowns, compiled, best of 7
× 50): the PEST expansion 0.502 ms, the general operator fed the same PEST
fields 0.374 ms, the two results equal to 4.2 × 10⁻¹⁶ of their size.

**Jacobi-Davidson** on the general operator fed PEST fields (QH, 24×12×8,
`mpol=5`, `ntor=1`, σ = 1.3 γ²): γ² is 6.2 × 10⁻¹² from the dense PEST value
3.962982733 × 10⁻⁴, eigen-residual 9.3 × 10⁻⁵ (2.1 × 10⁻¹¹ and 6.7 × 10⁻⁵ for
the PEST operator). Its Fourier-mode preconditioner is built from coefficients
averaged over each surface; how many corrections it needs when b<sup>θ</sup>
and b<sup>ζ</sup> vary on the surface was not measured.

**`frozen` against `pest` at the build-time equilibrium** (QH, 16×12×8 unless
stated). The pitch on the frozen nodes is (ι, 1) to 7 × 10⁻¹⁶. γ² agrees to
1.4 × 10⁻¹³ at 8×8×3 and to the eleven digits printed, 1.3366135186 × 10⁻⁴,
at 16×12×8. The Jacobians agree to 1.6 × 10⁻¹⁰ of their norm in every block of
DESC's parameters except `L_lmn`, the coefficients of λ, where they differ by
1.1 × 10⁻¹ at 8×8×3 and 3.9 × 10⁻² at 16×12×8: those are the parameters that
move the PEST nodes and leave the frozen ones where they are. In that block
each is the derivative of its own discrete γ² (against central differences in
the three largest `L_lmn` entries at 8×8×3: 9 × 10⁻⁹ for `pest`, 1.1 × 10⁻⁷ for
`frozen`), and the two discrete functions share a value at the build point,
not a slope. The difference was not measured at higher resolution.
