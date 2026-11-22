"""General flux coordinates: the energy principle without straight field lines.

The paper (arXiv:2608.01750, Eq. 20) writes the equilibrium field in PEST
coordinates, ``sqrt(g) B = psi' (iota e_theta + e_zeta)``. Nothing in the energy
principle needs that. In **any** flux coordinates ``(rho, theta, zeta)`` with
``B . grad(rho) = 0`` the field is

.. math::

    \\sqrt{g}\\, \\mathbf{B} = \\psi' \\left( b^\\theta \\mathbf{e}_\\theta
        + b^\\zeta \\mathbf{e}_\\zeta \\right), \\qquad
    b^\\theta = \\sqrt{g} B^\\theta / \\psi', \\quad
    b^\\zeta = \\sqrt{g} B^\\zeta / \\psi',

with ``d_theta b^theta + d_zeta b^zeta = 0`` and ``(b^theta, b^zeta) = (iota,
1)`` exactly when the angles are straight. With the unknowns

.. math::

    X = \\tilde\\xi^\\rho / \\psi', \\qquad
    \\upsilon = b^\\zeta \\tilde\\xi^\\theta - b^\\theta \\tilde\\xi^\\zeta, \\qquad
    Z = \\iota\\, \\tilde\\xi^\\zeta

(the paper's Eq. 24 and its ``upsilon`` when the angles are straight),
``xi x B = psi' upsilon grad(rho) - psi'^2 b^zeta X grad(theta) + psi'^2
b^theta X grad(zeta)`` and the field-line bending term is

.. math::

    \\sqrt{g}\\, Q^\\rho &= \\psi'^2 (b^\\theta \\partial_\\theta
        + b^\\zeta \\partial_\\zeta) X, \\\\
    \\sqrt{g}\\, Q^\\theta &= \\psi' \\partial_\\zeta \\upsilon
        - \\partial_\\rho (b^\\theta \\psi'^2 X), \\\\
    \\sqrt{g}\\, Q^\\zeta &= -\\psi' \\partial_\\theta \\upsilon
        - \\partial_\\rho (b^\\zeta \\psi'^2 X),

which is the paper's Eq. 25 with ``iota -> b^theta`` and ``1 -> b^zeta`` inside
the derivatives. ``docs/general_coordinates.md`` has the derivation of every
term and the proof that the eigenvalues do not depend on the angles.

As in :mod:`agnimhd.anisotropy`, the operator is written once, as

    A u = Phi^H ( G ( Phi u ) )

with ``Phi`` the linear map from the unknowns to ``sqrt(g) Q``, ``div xi`` and
``xi^rho``, and ``G`` the node-local pairing of those fields with the metric,
the current and the pressure. ``Phi^H`` comes from ``jax.vjp``, so the operator
is Hermitian by construction, and the dense, ring and matrix-free paths share
this one definition. The instability drive, the de-aliasing penalty and the
mass whitening are node-local or coordinate-independent and stay in
:mod:`agnimhd.assemble`.

With ``(b^theta, b^zeta) = (iota, 1)`` the operator is the PEST one of
:mod:`agnimhd.assemble` to round-off, term by term (``tests/test_general.py``).
"""

from .backend import jax, jnp

__all__ = ["mass_blocks", "normalized_fields", "operator"]


def normalized_fields(eq, f):
    """Add ``b_t`` and ``b_z`` to the assembler's dict ``f``, and replace the
    force-balance ``j_sup_theta`` by its general-coordinate form.

    ``b_t = sqrt(g) B^theta / psi'`` and ``b_z = sqrt(g) B^zeta / psi'`` are
    dimensionless, so they are formed from the SI inputs directly. Radial force
    balance, ``(j x B) . e_rho = p'``, reads ``psi' (j^theta b^zeta - j^zeta
    b^theta) = p'`` in any flux coordinates; the paper's ``j^theta = iota j^zeta
    + p'/psi'`` is its straight-angle case.
    """
    b_t, b_z = eq.field_pitch()
    f["b_t"] = jnp.asarray(b_t)[:, None]
    f["b_z"] = jnp.asarray(b_z)[:, None]
    f["j_sup_theta"] = (f["b_t"] * f["j_sup_zeta"] + f["p_r"] / f["psi_r"]) / f["b_z"]
    return f


def _third_scale(iota, ismirror):
    """``1 / s`` with ``Z = s xi^zeta``: ``s = iota``, or 1 for a mirror.

    The unselected branch stays finite, so that neither ``where`` nor its
    derivative sees ``1 / 0``.
    """
    return jnp.where(ismirror, 1.0, 1.0 / jnp.where(ismirror, 1.0, iota))


def mass_blocks(g, nW, ismirror):
    """The six independent entries of the per-node mass block, ``M^T g M``.

    The physical contravariant components are ``xi = M u`` with ``u = (X,
    upsilon, Z)`` and

    .. math::

        M = \\begin{pmatrix} \\psi' & 0 & 0 \\\\
            0 & 1/b^\\zeta & b^\\theta / (b^\\zeta s) \\\\
            0 & 0 & 1/s \\end{pmatrix},

    whose third column is along ``B``: at fixed ``X`` and ``upsilon`` the third
    unknown is a displacement along the field.

    Parameters
    ----------
    g : dict
        Normalized fields, any common shape.
    nW : ndarray
        Density weight times quadrature weight, same shape.
    ismirror : bool-like
        ``iota == 0`` everywhere (traced): the third unknown is then unscaled.

    Returns
    -------
    dict
        ``rr, ru, uu, rz, uz, zz``: the entries of ``nW sqrt(g) M^T g M``.
    """
    psi_r, sqrtg = g["psi_r"], g["sqrtg"]
    b_z = g["b_z"]
    pitch = g["b_t"] / b_z
    sinv = _third_scale(g["iota"], ismirror)
    w = nW * sqrtg
    return dict(
        rr=w * psi_r**2 * g["g_rr"],
        ru=w * psi_r * g["g_rv"] / b_z,
        uu=w * g["g_vv"] / b_z**2,
        rz=w * psi_r * (g["g_rv"] * pitch + g["g_rp"]) * sinv,
        uz=w * (g["g_vv"] * pitch + g["g_vp"]) * sinv / b_z,
        zz=w * (g["g_vv"] * pitch**2 + 2.0 * g["g_vp"] * pitch + g["g_pp"]) * sinv**2,
    )


def operator(g, W, d_r, d_t, d_z, gamma, ismirror, dtype):
    """Return ``apply(u) -> A u`` on grid arrays of shape ``(..., 3)``.

    ``A`` holds the field-line bending term ``|Q|^2``, the mixed ``Q``-current
    term and ``|j|^2`` term of ``|C|^2`` (paper Eq. 31), and the compression
    term. The instability drive and the penalty are added by the caller.

    Parameters
    ----------
    g : dict
        Normalized fields reshaped to the 3D grid, with ``b_t`` and ``b_z``.
    W : ndarray
        Quadrature weights on the grid (already including the radial map).
    d_r, d_t, d_z : callable
        Derivatives on grid scalars, as in :func:`agnimhd.assemble.matfree_operator`.
    gamma : float
        Ratio of specific heats in the compression term.
    ismirror : bool-like
        ``iota == 0`` everywhere (traced); selects the unscaled ``xi^zeta``.
    dtype : dtype
        The operator's dtype (:func:`agnimhd.assemble.operator_dtype`); a real
        input to a complex operator is promoted to it.
    """
    psi_r, psi_r2, sqrtg = g["psi_r"], g["psi_r2"], g["sqrtg"]
    b_t, b_z = g["b_t"], g["b_z"]
    sinv = _third_scale(g["iota"], ismirror)
    dlog_r = g["sqrtg_r"] / sqrtg
    dlog_v = g["sqrtg_v"] / sqrtg
    dlog_p = g["sqrtg_p"] / sqrtg
    gmet = jnp.stack(
        [
            jnp.stack([g["g_rr"], g["g_rv"], g["g_rp"]], -1),
            jnp.stack([g["g_rv"], g["g_vv"], g["g_vp"]], -1),
            jnp.stack([g["g_rp"], g["g_vp"], g["g_pp"]], -1),
        ],
        -2,
    )
    j_t, j_z = g["j_sup_theta"], g["j_sup_zeta"]
    # (j x grad rho)/|grad rho|^2 . Q = c_i (sqrt(g) Q^i), paper Eq. 39, with
    # c = ((j^t g^{zr} - j^z g^{tr}) / g^{rr}, j^z, -j^t). The `_term` factors
    # are psi' sqrt(g) g^{r.} from the covariant metric, hence the division.
    c_r = (j_t * g["g_sup_rp_term"] - j_z * g["g_sup_rv_term"]) / (
        psi_r * sqrtg * g["g_sup_rr"]
    )
    cvec = jnp.stack([c_r, j_z, -j_t], -1)
    w_q = W / sqrtg  # |Q|^2 sqrt(g) = q . g . q / sqrt(g), with q = sqrt(g) Q
    w_v = W * sqrtg
    j2 = g["J2"] / g["g_sup_rr"]
    gp = gamma * g["p0"]

    def phi(u):
        """Linear map: unknowns -> the fields the energy is bilinear in."""
        xr, xu, xz = u[..., 0], u[..., 1], u[..., 2]
        # physical contravariant components of the displacement
        Xr = psi_r * xr
        Xz = xz * sinv
        Xt = (xu + b_t * Xz) / b_z
        # q = sqrt(g) Q, Q = curl(xi x B)
        qr = psi_r2 * (b_t * d_t(xr) + b_z * d_z(xr))
        qt = psi_r * d_z(xu) - d_r(b_t * psi_r2 * xr)
        qz = -psi_r * d_t(xu) - d_r(b_z * psi_r2 * xr)
        # div xi = sum_i (d_i xi^i + xi^i d_i ln sqrt(g))
        D = (
            psi_r * dlog_r * xr
            + d_r(psi_r * xr)
            + dlog_v * Xt
            + d_t(Xt)
            + dlog_p * Xz
            + d_z(Xz)
        )
        return dict(q=jnp.stack([qr, qt, qz], -1), D=D, Xr=Xr)

    def pair(y):
        """The node-local weights G: fields -> conjugate fields, E = <phi, G phi>."""
        q, D, Xr = y["q"], y["D"], y["Xr"]
        cq = jnp.sum(cvec * q, -1)
        return dict(
            q=w_q[..., None] * jnp.einsum("...ij,...j->...i", gmet, q)
            + (w_v * Xr)[..., None] * cvec,
            D=w_v * gp * D,
            Xr=w_v * (cq + j2 * Xr),
        )

    def apply(u):
        u = jnp.asarray(u, dtype=jnp.result_type(u.dtype, dtype))
        fields, vjp = jax.vjp(phi, u)
        y = jax.tree_util.tree_map(jnp.conj, pair(fields))
        return jnp.conj(vjp(y)[0])

    apply.grid_shape = sqrtg.shape
    return apply
