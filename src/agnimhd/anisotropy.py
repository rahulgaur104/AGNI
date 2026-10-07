"""Anisotropic pressure: Bernstein's double-adiabatic energy principle.

With ``P = p_perp (I - bb) + p_par bb`` and the double-adiabatic equations of
state, Bernstein et al. (Proc. R. Soc. A 244, 1958, anisotropic-pressure
section) give, for a fixed boundary and with ``D = div xi``,
``s = b . (grad xi) . b`` (field-line stretching), ``sigma = p_par - p_perp``,

.. math::

    dW = \\int dV \\big[ |Q|^2 - \\xi\\cdot(j\\times Q) + D\\,\\xi\\cdot\\nabla p_\\perp
         + \\tfrac53 p_\\perp D^2 + \\tfrac13 p_\\perp (D - 3s)^2
         + s\\,\\xi\\cdot\\nabla\\sigma
         + \\sigma\\,(sD + 2s^2 - \\tilde e\\cdot v - \\tilde e\\cdot w) \\big]

with ``e~ = Q_perp / |B|`` the perturbed unit vector, ``v = (grad xi) . b`` and
``w = b . grad xi``. ``docs/theory.md`` has the derivation and the checks
(firehose threshold, the fast and slow speeds, the isotropic limit).

Everything here except ``|Q|^2`` -- which the assembler already has -- is a sum
of bilinear terms ``sum_k w_k <a_k(xi), b_k(xi)>`` in a handful of fields that
are **linear** in ``xi``. So the operator is written once, as

    A xi = Phi^H ( G ( Phi xi ) )

with ``Phi`` the linear map ``xi -> fields`` and ``G`` a node-local pairing of
the fields with the equilibrium weights. ``Phi^H`` comes from ``jax.vjp``; the
dense assembler materializes columns of the same map. There is therefore one
definition of these terms for the dense, ring and matrix-free paths.

No force-balance rearrangement is used (the isotropic code's ``|C|^2`` and
instability drive assume ``j x B = grad p``, ``p = p(rho)`` and ``j . grad rho
= 0``, none of which holds here), and no Christoffel symbols appear: the only
new geometric input is ``T_ik = e_i . d_k b``.
"""

from scipy.constants import mu_0

from .backend import jax, jnp

__all__ = ["normalized_fields", "operator", "materialize"]


def normalized_fields(eq, f):
    """Add the normalized anisotropy fields to the assembler's dict ``f``.

    Pressures scale like ``p0``, currents like ``j_sup_zeta``, ``T_b`` by the
    minor radius, ``grad_lnB`` not at all. Also forms the covariant ``B_i``,
    ``|B|^2`` and the contravariant metric, all from fields already in ``f``.
    """
    a_N, B_N = f["a_N"], f["B_N"]
    pscale = mu_0 / B_N**2
    jscale = mu_0 * a_N**2 / B_N
    f["p_perp"] = pscale * jnp.asarray(eq.p_perp)[:, None]
    f["sigma"] = pscale * jnp.asarray(eq.p_par - eq.p_perp)[:, None]
    f["grad_p_perp"] = pscale * jnp.asarray(eq.grad_p_perp)
    f["grad_sigma"] = pscale * jnp.asarray(eq.grad_p_par - eq.grad_p_perp)
    f["grad_lnB"] = jnp.asarray(eq.grad_lnB)
    f["T_b"] = jnp.asarray(eq.T_b) / a_N
    f["j_sup_rho"] = jscale * jnp.asarray(eq.J_sup_rho)[:, None]
    f["j_sup_theta_in"] = jscale * jnp.asarray(eq.J_sup_theta)[:, None]
    g = jnp.stack(
        [
            jnp.concatenate([f["g_rr"], f["g_rv"], f["g_rp"]], axis=1),
            jnp.concatenate([f["g_rv"], f["g_vv"], f["g_vp"]], axis=1),
            jnp.concatenate([f["g_rp"], f["g_vp"], f["g_pp"]], axis=1),
        ],
        axis=1,
    )  # (n, 3, 3) covariant metric
    f["g_inv"] = jnp.linalg.inv(g)
    # B^rho = 0, B^theta = iota psi'/sqrt(g), B^zeta = psi'/sqrt(g).
    b_sup = jnp.concatenate(
        [0 * f["iota"], f["iota"] * f["psi_r_over_sqrtg"], f["psi_r_over_sqrtg"]],
        axis=1,
    )
    f["B_sup"] = b_sup
    f["B_cov"] = jnp.einsum("nij,nj->ni", g, b_sup)
    f["B2"] = jnp.sum(f["B_sup"] * f["B_cov"], axis=1, keepdims=True)
    return f


def operator(g, W, d_r, d_t, d_z, ismirror):
    """Return ``apply(xi) -> A_an xi`` on grid arrays of shape ``(..., 3)``.

    Parameters
    ----------
    g : dict
        Normalized fields reshaped to the 3D grid (vectors ``(.., 3)``,
        ``T_b`` and ``g_inv`` ``(.., 3, 3)``).
    W : ndarray
        Quadrature weights on the grid (already including the radial map).
    d_r, d_t, d_z : callable
        Derivatives on grid scalars, as in :func:`agnimhd.assemble.matfree_operator`.
    ismirror : bool-like
        ``iota == 0`` everywhere (traced); selects the unscaled ``xi^zeta``.
    """
    iota, psi_r, sqrtg = g["iota"], g["psi_r"], g["sqrtg"]
    psi_r2, psi_r_over_sqrtg = g["psi_r2"], g["psi_r_over_sqrtg"]
    B_cov, B_sup, B2 = g["B_cov"], g["B_sup"], g["B2"]
    B = jnp.sqrt(B2)
    T, g_inv = g["T_b"], g["g_inv"]
    gmet = jnp.stack(
        [
            jnp.stack([g["g_rr"], g["g_rv"], g["g_rp"]], -1),
            jnp.stack([g["g_rv"], g["g_vv"], g["g_vp"]], -1),
            jnp.stack([g["g_rp"], g["g_vp"], g["g_pp"]], -1),
        ],
        -2,
    )
    j = jnp.stack([g["j_sup_rho"], g["j_sup_theta_in"], g["j_sup_zeta"]], -1)
    # xi . (j x Q) = sqrt(g) eps_ijk X^i j^j Q^k = X . (M Q),
    # with M_ik = sqrt(g) eps_ijk j^j.
    eps = jnp.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        ],
        dtype=float,
    )
    M = sqrtg[..., None, None] * jnp.einsum("ijk,...j->...ik", eps, j)
    omega = W * sqrtg
    dlog_v, dlog_p = g["sqrtg_v"] / sqrtg, g["sqrtg_p"] / sqrtg
    p_perp, sigma = g["p_perp"], g["sigma"]
    grad_pp, grad_sig, grad_lnB = g["grad_p_perp"], g["grad_sigma"], g["grad_lnB"]

    def phi(xi):
        """Linear map: displacement -> the fields the energy is bilinear in."""
        xr, xu, xz = xi[..., 0], xi[..., 1], xi[..., 2]
        # physical contravariant components (paper Eq. 21; unscaled for a mirror)
        Xr = psi_r * xr
        Xt = jnp.where(ismirror, xu, xu + xz)
        # The unselected branch must stay finite, or the vjp of `where` turns
        # its zero cotangent into 0 * inf.
        Xz = jnp.where(ismirror, xz, xz / jnp.where(ismirror, 1.0, iota))
        X = jnp.stack([Xr, Xt, Xz], -1)
        # Q = curl(xi x B), paper Eq. 22
        Qr = psi_r2 / sqrtg * (iota * d_t(xr) + d_z(xr))
        Qt = psi_r_over_sqrtg * (d_z(xu) - d_r(iota * psi_r2 * xr) / psi_r)
        Qz = -psi_r_over_sqrtg * (d_t(xu) + d_r(psi_r2 * xr) / psi_r)
        Q = jnp.stack([Qr, Qt, Qz], -1)
        Q_cov = jnp.einsum("...ij,...j->...i", gmet, Q)
        # div xi. Radially d(sqrt(g) X^rho)/sqrt(g): sqrt(g) psi' xi^rho ~
        # rho^2 xi^rho is even in rho and in the Zernike span, psi' xi^rho ~
        # rho xi^rho is not. In the angles the rho factor of sqrt(g) commutes
        # with the derivative, so d X + X d ln sqrt(g) keeps smooth X smooth.
        D = d_r(sqrtg * Xr) / sqrtg + d_t(Xt) + Xt * dlog_v + d_z(Xz) + Xz * dlog_p
        BQ = jnp.sum(B_sup * Q_cov, -1)  # B . Q = B^i Q_i
        xi_lnB = jnp.sum(X * grad_lnB, -1)
        s = BQ / B2 + xi_lnB + D
        xi_b = jnp.sum(X * B_cov, -1) / B  # xi . b
        u = jnp.einsum("...ik,...k->...i", T, X)  # (xi . grad) b
        dxb = jnp.stack([d_r(xi_b), d_t(xi_b), d_z(xi_b)], -1)
        v = dxb - jnp.einsum("...ki,...k->...i", T, X)  # (grad xi) . b
        w = (Q_cov + (xi_lnB + D)[..., None] * B_cov) / B[..., None] + u  # b.grad xi
        et = (Q_cov - (BQ / B2)[..., None] * B_cov) / B[..., None]  # e~ = Q_perp/B
        gp = jnp.sum(X * grad_pp, -1)
        gs = jnp.sum(X * grad_sig, -1)
        return dict(X=X, Q=Q, D=D, s=s, gp=gp, gs=gs, et=et, vw=v + w)

    def pair(y):
        """The node-local weights G: fields -> conjugate fields, E = <phi, G phi>."""
        X, Q, D, s, gp, gs, et, vw = (
            y[k] for k in ("X", "Q", "D", "s", "gp", "gs", "et", "vw")
        )
        om = omega
        return dict(
            X=-0.5 * om[..., None] * jnp.einsum("...ik,...k->...i", M, Q),
            Q=-0.5 * om[..., None] * jnp.einsum("...ik,...i->...k", M, X),
            D=om
            * (
                0.5 * gp
                + (5.0 / 3.0) * p_perp * D
                + p_perp / 3.0 * (D - 3 * s)
                + 0.5 * sigma * s
            ),
            gp=0.5 * om * D,
            s=om * (-p_perp * (D - 3 * s) + 0.5 * gs + 0.5 * sigma * D + 2 * sigma * s),
            gs=0.5 * om * s,
            et=-0.5
            * (om * sigma)[..., None]
            * jnp.einsum("...ij,...j->...i", g_inv, vw),
            vw=-0.5
            * (om * sigma)[..., None]
            * jnp.einsum("...ij,...i->...j", g_inv, et),
        )

    def apply(xi):
        fields, vjp = jax.vjp(phi, xi)
        y = jax.tree_util.tree_map(jnp.conj, pair(fields))
        return jnp.conj(vjp(y)[0])

    apply.grid_shape = sqrtg.shape
    return apply


def materialize(apply, n_total, dtype, rows=None, batch=256):
    """Rows ``rows`` of the matrix of ``apply`` in component-major ordering.

    ``apply`` acts on ``(n_rho, n_theta, n_zeta, 3)`` arrays; the returned
    matrix acts on the flat vector ``[xi^rho, ups, xi^zeta]`` of length
    ``3 * n_total``. The operator is Hermitian, so ``A[r, :]`` is the conjugate
    of ``A e_r``. With ``rows=None`` all ``3 * n_total`` rows are built.
    """
    rows = jnp.arange(3 * n_total) if rows is None else jnp.asarray(rows)
    shape = apply.grid_shape

    def row(r):
        e = jnp.zeros(3 * n_total, dtype).at[r].set(1.0)
        y = apply(e.reshape(3, n_total).T.reshape(shape + (3,)))
        return jnp.conj(y.reshape(n_total, 3).T.reshape(-1))

    return jax.lax.map(row, rows, batch_size=batch)
