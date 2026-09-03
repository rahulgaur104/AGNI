"""Assembly of the AGNI operator: dense, ring-restricted, and matrix-free.

AGNI discretizes the ideal-MHD **energy functional**, not the force operator.
That gives a generalized symmetric eigenvalue problem

.. math::  A x = \\lambda B x

with ``B`` -- the kinetic/mass matrix -- symmetric positive definite. ``B`` is
Cholesky-factored per node into ``3x3`` blocks, and the congruence
``L^{-1} A L^{-T}`` reduces the problem to a standard symmetric one. The most
negative eigenvalue is the squared growth rate, and its **sign** decides
stability.

The displacement is discretized on the 3D tensor-product PEST grid with three
vector components per node, so the unknown has length ``3 * n_rho * n_theta *
n_zeta`` before boundary and constraint degrees of freedom are removed. A
Dirichlet condition on ``xi^rho`` removes that component on the innermost and
outermost radial shells, leaving ``n_keep = 3 * n_total - 2 * n_theta *
n_zeta``.

Three entry points, one definition
----------------------------------
:func:`assemble_dense` builds the reduced whitened matrix. :func:`ring_block`
builds one poloidal ring's sub-block of the same matrix -- exactly, not
approximately, because every step after the ring restriction is node-diagonal
or a permutation. :func:`matfree_operator` applies the same operator without
ever forming it. All three are checked against each other in
``tests/test_assemble.py``; the matrix-free operator reproduces the dense matrix
to 2e-11 and the ring blocks to ~1e-16.

Node ordering is rho-major throughout: node ``(i, j, k)`` is at flat index
``(i * n_theta + j) * n_zeta + k``, and component ``c`` lives at
``c * n_total + ...``.
"""

import numpy as np
from scipy.constants import mu_0

from .backend import errorif, jax, jnp
from .config import AssemblyConfig

__all__ = [
    "assemble_dense",
    "finish_ring_block",
    "keep_indices",
    "matfree_operator",
    "operator_dtype",
    "ring_block",
]


def operator_dtype(config):
    """Return the dtype the assembled operator carries under ``config``.

    ``axisym=True`` analyzes a single toroidal Fourier mode ``exp(i n phi)``, so
    ``d/dphi`` becomes multiplication by ``i n`` and every matrix built from it
    is complex **Hermitian** rather than real symmetric.

    This is a function rather than a literal at each construction site because
    callers outside the assembler need the answer *before* the matrix exists:
    ``objective._primal`` must declare ARPACK's output shape and dtype to
    ``jax.pure_callback``, which cannot infer either. Hardcoding a real dtype
    there made the ``eigsh`` path fail on every axisymmetric case.

    Parameters
    ----------
    config : AssemblyConfig

    Returns
    -------
    numpy.dtype
    """
    return jnp.complex128 if config.axisym else jnp.float64


def _cT(x):
    """Conjugate transpose."""
    return jnp.conjugate(jnp.transpose(x))


def keep_indices(n_rho, n_theta, n_zeta):
    """Indices of the degrees of freedom retained after the Dirichlet mask.

    ``xi^rho`` is fixed to zero on the innermost and outermost radial shells;
    the two tangential components are kept everywhere.

    Parameters
    ----------
    n_rho, n_theta, n_zeta : int

    Returns
    -------
    ndarray of int, shape (3 * n_total - 2 * n_theta * n_zeta,)
        Indices into the length-``3 * n_total`` component-major vector.

    Notes
    -----
    Derived from the resolution alone and returned as **concrete NumPy**. It
    carries no dependence on the equilibrium, and downstream consumers use it to
    size arrays -- shapes cannot come from traced values, so making this a
    device array only moves the failure one frame deeper.
    """
    n_total = n_rho * n_theta * n_zeta
    n_shell = n_theta * n_zeta
    return np.concatenate(
        [np.arange(n_shell, n_total - n_shell), np.arange(n_total, 3 * n_total)]
    )


def _zernike_penalty(diffmat, rt_size, coupled_rt):
    """Return ``(alpha, Q, rank)`` for a ``DiffMat``-owned de-aliasing penalty."""
    if not coupled_rt:
        return 0.0, None, None
    alpha = float(getattr(diffmat, "zernike_penalty_alpha", 0.0) or 0.0)
    if alpha <= 0.0:
        return alpha, None, None
    Q = getattr(diffmat, "zernike_penalty_projector", None)
    errorif(
        Q is None,
        ValueError,
        "DiffMat has zernike_penalty_alpha > 0 but no "
        "zernike_penalty_projector. Rebuild the DiffMat with coupled "
        "D_rho/D_theta, or pass a precomputed projector.",
    )
    errorif(
        tuple(Q.shape) != (rt_size, rt_size),
        ValueError,
        "DiffMat zernike_penalty_projector shape does not match the coupled_rt "
        f"grid: got {tuple(Q.shape)}, expected {(rt_size, rt_size)}.",
    )
    return alpha, Q, getattr(diffmat, "zernike_penalty_rank", None)


def _normalized_fields(eq, config):
    """Non-dimensionalize the equilibrium onto AGNI's internal scaling.

    Returns a dict of column vectors (shape ``(n_nodes, 1)``) plus the scalars.
    The normalization is ``a`` for lengths and ``B_N = |Psi| / (pi a^2)`` for
    the field, so the operator's terms carry ``a^2``, ``a^3`` and ``a^4``. This
    is why ``a`` is hypersensitive -- see :class:`agnimhd.EquilibriumData`.
    """
    a_N = eq.a
    B_N = jnp.abs(eq.Psi / (jnp.pi * a_N**2))

    def col(x):
        return jnp.asarray(x)[:, None]

    iota = col(eq.iota)
    psi_r = col(eq.psi_r) / (a_N**2 * B_N)
    sqrtg = col(eq.sqrtg) / a_N**3

    out = dict(
        a_N=a_N,
        B_N=B_N,
        iota=iota,
        iotainv=1.0 / iota,
        psi_r=psi_r,
        psi_r2=psi_r**2,
        psi_r3=psi_r**3,
        iota_psi_r2=iota * psi_r**2,
        # A tiny shift because the pressure can go slightly negative at the edge
        # of a fitted profile; the compressibility term takes products of it.
        p0=mu_0 * col(eq.p) / B_N**2 + 1e-12,
        p_r=mu_0 * col(eq.p_r) / B_N**2,
        sqrtg=sqrtg,
        sqrtg_r=col(eq.sqrtg_r) / a_N**3,
        sqrtg_v=col(eq.sqrtg_v) / a_N**3,
        sqrtg_p=col(eq.sqrtg_p) / a_N**3,
        g_rr=col(eq.g_rr) / a_N**2,
        g_vv=col(eq.g_vv) / a_N**2,
        g_pp=col(eq.g_pp) / a_N**2,
        g_rv=col(eq.g_rv) / a_N**2,
        g_rp=col(eq.g_rp) / a_N**2,
        g_vp=col(eq.g_vp) / a_N**2,
        g_sup_rr=col(eq.g_sup_rr) * a_N**2,
        J2=col((mu_0 * eq.abs_J) ** 2) * (a_N / B_N) ** 2,
        j_sup_zeta=mu_0 * col(eq.J_sup_zeta) * a_N**2 / B_N,
        F=-mu_0 * col(eq.instability_drive()) / B_N**2,
    )
    out["psi_r_over_sqrtg"] = out["psi_r"] / out["sqrtg"]
    # From ideal-MHD force balance.
    out["j_sup_theta"] = out["iota"] * out["j_sup_zeta"] + out["p_r"] / out["psi_r"]
    # g^{12} = (g_13 g_23 - g_12 g_33) / (sqrt(g))^2 for the 3x3 metric. These
    # absorb a psi_r * sqrt(g): g_sup_rv_term == psi_r * sqrt(g) * g^{rv}.
    out["g_sup_rv_term"] = out["psi_r_over_sqrtg"] * (
        out["g_rp"] * out["g_vp"] - out["g_rv"] * out["g_pp"]
    )
    out["g_sup_rp_term"] = out["psi_r_over_sqrtg"] * (
        out["g_rv"] * out["g_vp"] - out["g_rp"] * out["g_vv"]
    )
    return out


def _resolution(eq, diffmat, config):
    """Per-direction node counts, honouring coupled mode."""
    if config.coupled_rt:
        n_rho = int(config.n_rho_coupled)
        n_theta = int(config.n_theta_coupled)
    else:
        n_rho = int(diffmat.D_rho.shape[0])
        n_theta = int(diffmat.D_theta.shape[0])
    if config.axisym:
        n_zeta = 1
    else:
        n_zeta = int(diffmat.D_zeta.shape[0])
    errorif(
        n_rho * n_theta * n_zeta != eq.n_nodes,
        ValueError,
        f"the DiffMat implies a {n_rho}x{n_theta}x{n_zeta} grid "
        f"({n_rho * n_theta * n_zeta} nodes) but the EquilibriumData has "
        f"{eq.n_nodes} nodes ({eq.resolution}). The matrices must be built for "
        "the nodes the equilibrium was evaluated on.",
    )
    return n_rho, n_theta, n_zeta


def assemble_dense(eq, diffmat, config=None, density=None, ring_nodes=None):
    """Assemble the reduced, whitened dense AGNI matrix ``A``.

    Parameters
    ----------
    eq : EquilibriumData
        Equilibrium quantities on the PEST grid.
    diffmat : DiffMat
        Differentiation and quadrature matrices built on the same nodes.
    config : AssemblyConfig, optional
        Static settings. Defaults to ``AssemblyConfig()``.
    density : ndarray, shape (n_nodes,), optional
        Mass-density weight in the kinetic matrix. Arbitrary up to the
        eigenvalue scaling -- it mostly sets the spread of ``B``'s spectrum, and
        preconditioning removes that factor. Defaults to ones.
    ring_nodes : ndarray of int, optional
        Restrict the assembly to these nodes, producing one ring's block rather
        than the full matrix. When set, the return value is the reduced
        ``{"A", "Linv", "au_diag"}`` triple and the caller finishes the block
        with :func:`finish_ring_block`; everything past the ring restriction is
        global to the full matrix and meaningless for a single block.

    Returns
    -------
    dict
        With ``ring_nodes=None``: ``"A"`` is the reduced whitened matrix of
        shape ``(n_keep, n_keep)``, plus the intermediates callers need
        (``"Linv"``, ``"d"``, ``"keep"``, the resolution, and the normalized
        metric fields). With ``ring_nodes`` set: ``{"A", "Linv", "au_diag"}``.

    Notes
    -----
    With ``ring_nodes=None`` every restriction helper below is the identity, so
    the full-matrix path is bit-for-bit what it would be without the ring
    machinery present.
    """
    config = AssemblyConfig() if config is None else config
    n_rho_max, n_theta_max, n_zeta_max = _resolution(eq, diffmat, config)
    n_total = n_rho_max * n_theta_max * n_zeta_max

    f = _normalized_fields(eq, config)
    gamma = config.gamma

    if config.axisym:
        # Each component of xi is a single toroidal Fourier mode.
        D_zeta0 = 1j * config.n_mode_axisym * jnp.array([[1]])
    else:
        D_zeta0 = diffmat.D_zeta
    D_rho0 = diffmat.D_rho
    D_theta0 = diffmat.D_theta
    W_rho, W_theta, W_zeta = diffmat.W_rho, diffmat.W_theta, diffmat.W_zeta

    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))
    if config.coupled_rt:
        # D_rho0/D_theta0 already couple (rho, theta); only tensor with zeta.
        I_rt0 = jax.lax.stop_gradient(jnp.eye(n_rho_max * n_theta_max))
        D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, I_zeta0))
        D_theta = jax.lax.stop_gradient(jnp.kron(D_theta0, I_zeta0))
        D_zeta = jax.lax.stop_gradient(jnp.kron(I_rt0, D_zeta0))
        D_thetaT = jax.lax.stop_gradient(jnp.kron(_cT(D_theta0), I_zeta0))
        D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rt0, _cT(D_zeta0)))
    else:
        I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
        I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
        D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
        D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
        D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))
        D_thetaT = jax.lax.stop_gradient(
            jnp.kron(I_rho0, jnp.kron(_cT(D_theta0), I_zeta0))
        )
        D_zetaT = jax.lax.stop_gradient(
            jnp.kron(I_rho0, jnp.kron(I_theta0, _cT(D_zeta0)))
        )

    # Quadrature weights factorize (tensor product) in both modes.
    W = jnp.kron(W_rho, jnp.kron(W_theta, W_zeta))[:, None]

    # -- ring restriction ------------------------------------------------
    # With ring_nodes=None every helper is the identity. With a ring supplied,
    # the derivative matrices are sliced to the ring's columns (rows for the
    # transposes), the accumulators shrink to 3*|R|, and node-diagonal
    # quantities are restricted to the ring's nodes.
    if ring_nodes is None:
        _nR = n_total

        def _selc(M):
            return M

        def _selr(M):
            return M

        def _nodesel(v):
            return v

        def _diag_r(v):
            return jnp.diag(v)

        def _diag_col(v):
            return jnp.diag(jnp.asarray(v).reshape(-1))

        def _fit(M):
            return M

    else:
        _Rnode = jnp.asarray(ring_nodes)
        _nR = int(_Rnode.size)
        _ar = jnp.arange(_nR)

        def _selc(M):
            return M[:, _Rnode]

        def _selr(M):
            return M[_Rnode, :]

        def _nodesel(v):
            v = jnp.asarray(v)
            return v[_Rnode] if v.shape[0] == n_total else v

        def _diag_r(v):
            v = jnp.asarray(v)
            if v.ndim == 2:
                return jnp.diag(v)
            v = v.reshape(-1)
            return jnp.diag(v[_Rnode]) if v.size == n_total else jnp.diag(v)

        def _diag_col(v):
            # A node-diagonal OPERATOR added to a D matrix must be
            # column-sliced, not reduced to a ring block: diag(v)[:, R].
            v = jnp.asarray(v).reshape(-1)
            return (
                jnp.zeros((n_total, _nR), dtype=v.dtype).at[_Rnode, _ar].set(v[_Rnode])
            )

        def _fit(M):
            # A term with a derivative on only one side keeps the full node
            # dimension on the underived side, where the implicit identity was
            # never sliced. Restrict any surviving full-length axis.
            M = jnp.asarray(M)
            if M.ndim == 2 and M.shape[0] == n_total:
                M = M[_Rnode, :]
            if M.ndim == 2 and M.shape[1] == n_total:
                M = M[:, _Rnode]
            return M

    D_rho = _selc(D_rho)
    D_theta = _selc(D_theta)
    D_zeta = _selc(D_zeta)
    D_thetaT = _selr(D_thetaT)
    D_zetaT = _selr(D_zetaT)

    n0 = jnp.asarray(jnp.ones(n_total) if density is None else density).reshape(
        n_total, 1
    )

    rho_idx = slice(0, _nR)
    ups_idx = slice(_nR, 2 * _nR)
    zeta_idx = slice(2 * _nR, 3 * _nR)

    dtype = operator_dtype(config)
    A = jnp.zeros((3 * _nR, 3 * _nR), dtype=dtype)
    B = jnp.zeros((3 * _nR, 3 * _nR), dtype=dtype)

    # Unpack the normalized fields for readability below.
    iota, iotainv = f["iota"], f["iotainv"]
    psi_r, psi_r2, psi_r3 = f["psi_r"], f["psi_r2"], f["psi_r3"]
    iota_psi_r2 = f["iota_psi_r2"]
    p0, sqrtg = f["p0"], f["sqrtg"]
    psi_r_over_sqrtg = f["psi_r_over_sqrtg"]
    g_rr, g_vv, g_pp = f["g_rr"], f["g_vv"], f["g_pp"]
    g_rv, g_rp, g_vp = f["g_rv"], f["g_rp"], f["g_vp"]
    g_sup_rr = f["g_sup_rr"]
    g_sup_rv_term, g_sup_rp_term = f["g_sup_rv_term"], f["g_sup_rp_term"]
    J2, j_sup_zeta, j_sup_theta = f["J2"], f["j_sup_zeta"], f["j_sup_theta"]
    F = f["F"]

    partial_z_log_sqrtg = (f["sqrtg_p"] / sqrtg).flatten()
    partial_r_log_sqrtg = (f["sqrtg_r"] / sqrtg).flatten()
    partial_v_log_sqrtg = (f["sqrtg_v"] / sqrtg).flatten()

    C_zeta = _diag_col(partial_z_log_sqrtg) + D_zeta
    C_rho = _diag_col(partial_r_log_sqrtg) + D_rho
    C_theta = _diag_col(partial_v_log_sqrtg) + D_theta

    # ---- Q^2_rr ----
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
            + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
            + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
            + _cT((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta) @ D_theta
        )
    )

    # ---- Q^2_vv ---- (symmetry enforced exactly)
    A = A.at[ups_idx, ups_idx].add(
        _fit(
            0.5
            * (
                D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
                + _cT((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta) @ D_zeta
            )
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            _cT(D_rho * _nodesel(iota_psi_r2).T)
            @ (
                (psi_r_over_sqrtg * W * g_vv / psi_r)
                * (D_rho * _nodesel(iota_psi_r2).T)
            )
        )
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(
            -1
            * _cT(D_rho * _nodesel(iota_psi_r2).T)
            @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
        )
    )

    # ---- Q^2_zz ----
    A = A.at[ups_idx, ups_idx].add(
        _fit(
            0.5
            * (
                _cT(D_theta) @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
                + _cT((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta) @ D_theta
            )
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            _cT(D_rho * _nodesel(psi_r2).T)
            @ ((psi_r_over_sqrtg * W * g_pp / psi_r) * (D_rho * _nodesel(psi_r2).T))
        )
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(
            _cT(D_rho * _nodesel(psi_r2).T) @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
        )
    )

    # ---- Q^2_rv, and its transpose along the rho-rho block diagonal ----
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(D_theta)
                @ (
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
                + _cT(D_zeta)
                @ (
                    (psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
            )
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
                @ D_theta
                + _cT(
                    (psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
                @ D_zeta
            )
        )
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(
            _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    # ---- Q^2_rz ----
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(D_theta)
                @ (
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rp)
                    * (D_rho * _nodesel(psi_r2).T)
                )
                + _cT(D_zeta)
                @ ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * _nodesel(psi_r2).T))
            )
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rp)
                    * (D_rho * _nodesel(psi_r2).T)
                )
                @ D_theta
                + _cT(
                    (psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * _nodesel(psi_r2).T)
                )
                @ D_zeta
            )
        )
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(
            -1
            * (
                _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
                + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            )
        )
    )

    # ---- Q^2_vz ----
    A = A.at[ups_idx, ups_idx].add(
        _fit(
            -1
            * (
                _cT(D_zeta) @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
                + _cT((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta) @ D_zeta
            )
        )
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(
            -1
            * (
                _cT(D_rho * _nodesel(psi_r2).T)
                @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
                - _cT(D_rho * _nodesel(iota_psi_r2).T)
                @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
            )
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            _cT(D_rho * _nodesel(iota_psi_r2).T)
            @ ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * _nodesel(psi_r2).T))
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            _cT((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * _nodesel(psi_r2).T))
            @ (D_rho * _nodesel(iota_psi_r2).T)
        )
    )

    # ---- mixed Q-J term: xi^rho (J x grad rho)/|grad rho|^2 . Q ----
    # g^rv and g^rz are replaced using the metric identity, so this route
    # matches the matrix-free operator's by construction.
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            (
                W
                * psi_r2
                * (j_sup_theta * g_sup_rp_term - j_sup_zeta * g_sup_rv_term)
                / g_sup_rr
            )
            * (iota * D_theta + D_zeta)
            - (W * sqrtg * psi_r * j_sup_zeta) * (D_rho * _nodesel(iota_psi_r2).T)
            + (W * sqrtg * psi_r * j_sup_theta) * (D_rho * _nodesel(psi_r2).T)
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            _cT(
                (
                    W
                    * psi_r2
                    * (j_sup_theta * g_sup_rp_term - j_sup_zeta * g_sup_rv_term)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
            )
            - _cT((W * sqrtg * psi_r * j_sup_zeta) * (D_rho * _nodesel(iota_psi_r2).T))
            + _cT((W * sqrtg * psi_r * j_sup_theta) * (D_rho * _nodesel(psi_r2).T))
        )
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(
            (W * psi_r2 * sqrtg * j_sup_theta) * D_theta
            + (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
        )
    )

    # ---- diagonal |J|^2 term ----
    A = A.at[rho_idx, rho_idx].add(
        _fit(_diag_r((psi_r2 * W * sqrtg * J2 / g_sup_rr).flatten()))
    )

    # ---- mass matrix (symmetric positive definite) ----
    B = B.at[rho_idx, rho_idx].add(
        _fit(_diag_r((n0 * W * psi_r2 * sqrtg * g_rr).flatten()))
    )
    B = B.at[ups_idx, ups_idx].add(_fit(_diag_r((n0 * W * sqrtg * g_vv).flatten())))
    B = B.at[rho_idx, ups_idx].add(
        _fit(_diag_r((n0 * W * psi_r * sqrtg * g_rv).flatten()))
    )

    # A magnetic mirror (iota identically zero) takes a different set of
    # couplings. `ismirror` depends on the equilibrium and is therefore TRACED,
    # so a Python `if` on it raises under jit. `jnp.where` selects the same
    # values without a host-side branch; for iota != 0 both arms are finite.
    ismirror = jnp.all(jnp.abs(iota) < 1e-12)
    zz = jnp.where(
        ismirror,
        n0 * W * sqrtg * g_pp,
        n0 * W * sqrtg * (g_vv + 2 * iotainv * g_vp + iotainv**2 * g_pp),
    )
    rz = jnp.where(
        ismirror,
        n0 * W * psi_r * sqrtg * g_rp,
        n0 * W * psi_r * sqrtg * (g_rv + iotainv * g_rp),
    )
    uz = jnp.where(
        ismirror,
        n0 * W * psi_r * sqrtg * g_vp,
        n0 * W * sqrtg * (g_vv + iotainv * g_vp),
    )
    B = B.at[zeta_idx, zeta_idx].add(_fit(_diag_r(zz.flatten())))
    B = B.at[rho_idx, zeta_idx].add(_fit(_diag_r(rz.flatten())))
    B = B.at[ups_idx, zeta_idx].add(_fit(_diag_r(uz.flatten())))

    # ---- compressibility: purely stabilizing, does not move marginality ----
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            _cT(C_rho * _nodesel(psi_r).T)
            @ ((gamma * sqrtg * W * p0) * (C_rho * _nodesel(psi_r).T))
        )
    )
    A = A.at[ups_idx, ups_idx].add(
        _fit(_cT(C_theta) @ ((gamma * sqrtg * W * p0) * C_theta))
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(_cT(C_rho * _nodesel(psi_r).T) @ ((gamma * sqrtg * W * p0) * C_theta))
    )
    A = A.at[zeta_idx, zeta_idx].add(
        _fit(
            _cT(C_theta + C_zeta * _nodesel(iotainv).T)
            @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * _nodesel(iotainv).T))
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        _fit(
            _cT(C_rho * _nodesel(psi_r).T)
            @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * _nodesel(iotainv).T))
        )
    )
    A = A.at[ups_idx, zeta_idx].add(
        _fit(
            _cT(C_theta)
            @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * _nodesel(iotainv).T))
        )
    )

    # ---- instability drive, kept as a diagonal rather than materialized ----
    au_diag = _nodesel((W * psi_r2 * sqrtg * F).flatten())

    rt_size = n_rho_max * n_theta_max
    alpha, Q_rt, _rank = _zernike_penalty(diffmat, rt_size, config.coupled_rt)
    if alpha > 0.0:
        # Device only when the input is actually traced. `jnp.kron` here
        # unconditionally moves the (rt_size * n_zeta)^2 intermediate onto the
        # accelerator, where NumPy kept it on the host for free -- a real
        # memory regression at production resolutions, where this assembly
        # already runs near the device limit.
        if isinstance(Q_rt, jax.core.Tracer) or isinstance(A, jax.core.Tracer):
            Q = Q_rt if n_zeta_max == 1 else jnp.kron(Q_rt, jnp.eye(n_zeta_max))
        else:
            Q = (
                np.asarray(Q_rt)
                if n_zeta_max == 1
                else np.kron(np.asarray(Q_rt), np.eye(n_zeta_max))
            )
        penalty = jnp.asarray(alpha * Q, dtype=A.dtype)
        A = A.at[rho_idx, rho_idx].add(_fit(penalty))
        A = A.at[ups_idx, ups_idx].add(_fit(penalty))
        A = A.at[zeta_idx, zeta_idx].add(_fit(penalty))

    # Mirror the upper blocks into the lower ones.
    A = A.at[ups_idx, rho_idx].set(_cT(A[rho_idx, ups_idx]))
    A = A.at[zeta_idx, rho_idx].set(_cT(A[rho_idx, zeta_idx]))
    A = A.at[zeta_idx, ups_idx].set(_cT(A[ups_idx, zeta_idx]))
    B = B.at[ups_idx, rho_idx].set(_cT(B[rho_idx, ups_idx]))
    B = B.at[zeta_idx, rho_idx].set(_cT(B[rho_idx, zeta_idx]))
    B = B.at[zeta_idx, ups_idx].set(_cT(B[ups_idx, zeta_idx]))

    d = 1 / jnp.sqrt(_diag_r(B))

    # The A whitening is DEFERRED until after B_blocks is extracted and B is
    # released. Doing it here holds A_old + A_new + a broadcast transient while
    # B_old + B_new are also live -- about five full (3*n_total)^2 copies.
    au_diag = d[rho_idx] ** 2 * au_diag
    B = d[:, None] * B * d[None, :]

    if config.axisym:
        B_blocks = jnp.zeros((_nR, 3, 3), dtype=jnp.complex128)
        I3 = jnp.tile(jnp.eye(3, dtype=jnp.complex128), (_nR, 1, 1))
    else:
        B_blocks = jnp.zeros((_nR, 3, 3))
        I3 = jnp.tile(jnp.eye(3), (_nR, 1, 1))

    B_blocks = B_blocks.at[:, 0, 0].set(_diag_r(B[rho_idx, rho_idx]))
    B_blocks = B_blocks.at[:, 1, 1].set(_diag_r(B[ups_idx, ups_idx]))
    B_blocks = B_blocks.at[:, 2, 2].set(_diag_r(B[zeta_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 0, 1].set(_diag_r(B[rho_idx, ups_idx]))
    B_blocks = B_blocks.at[:, 1, 0].set(_diag_r(B[ups_idx, rho_idx]))
    B_blocks = B_blocks.at[:, 2, 0].set(_diag_r(B[rho_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 0, 2].set(_diag_r(B[zeta_idx, rho_idx]))
    B_blocks = B_blocks.at[:, 1, 2].set(_diag_r(B[ups_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 2, 1].set(_diag_r(B[zeta_idx, ups_idx]))

    # B is dead here -- only the small (N, 3, 3) B_blocks is used downstream.
    # The barrier is LOAD-BEARING: XLA schedules by dataflow, not source order,
    # so merely moving these lines does not stop it interleaving the two
    # whitenings and holding both matrices at once.
    A, B_blocks, d = jax.lax.optimization_barrier((A, B_blocks, d))
    B = None
    A = d[:, None] * A * d[None, :]

    # Enforce the physical xi^rho boundary condition in the per-node blocks
    # BEFORE taking the Cholesky.
    n_per_shell = n_theta_max * n_zeta_max
    node_ids = np.arange(n_total)
    rho_shell = node_ids // n_per_shell
    boundary = _nodesel(jnp.asarray((rho_shell == 0) | (rho_shell == (n_rho_max - 1))))
    # Written as jnp.where over the full leading axis rather than a boolean
    # index: a boolean-mask index needs a concrete mask, and under the ring
    # vmap `boundary` is traced. The two forms are identical elementwise.
    for _i, _j in ((0, 1), (1, 0), (0, 2), (2, 0)):
        B_blocks = B_blocks.at[:, _i, _j].set(
            jnp.where(boundary, 0.0, B_blocks[:, _i, _j])
        )

    L = jnp.linalg.cholesky(B_blocks)
    Linv = jax.lax.linalg.triangular_solve(L, I3, left_side=True, lower=True)

    if ring_nodes is not None:
        # The RING PATH ends here. Everything past this point -- the
        # component/node permutation, the congruence, the keep-mask reduction
        # -- is global to the full matrix and meaningless for one block. The
        # caller finishes it with `finish_ring_block`.
        return {"A": A, "Linv": Linv, "au_diag": au_diag}

    p = _component_to_node_permutation(n_total)
    A = A[p][:, p]

    # L^-1 A L^-T
    A = A.reshape(n_total, 3, n_total, 3)
    A = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, A, Linv)

    node_idx = jnp.arange(n_total)
    # A constant diagonal shift for positive-definiteness, applied in the
    # whitened basis BEFORE the instability drive. It shifts every eigenvalue
    # uniformly, and the matrix-free operator must add the same one.
    A = A.at[node_idx, :, node_idx, :].add(1e-14 * jnp.eye(3))

    # The transformed drive, without materializing the full Au matrix.
    L0 = Linv[:, :, 0]
    au_node = au_diag[:, None, None] * L0[:, :, None] * L0[:, None, :]
    A = A.at[node_idx, :, node_idx, :].add(au_node)

    A = A.reshape(3 * n_total, 3 * n_total)
    pinv = jnp.empty_like(p).at[p].set(jnp.arange(3 * n_total))
    A = A[pinv][:, pinv]

    keep = jnp.asarray(keep_indices(n_rho_max, n_theta_max, n_zeta_max))
    A = A[jnp.ix_(keep, keep)]

    out = {
        "A": A,
        "Linv": Linv,
        "d": d,
        "keep": keep,
        "n_rho": n_rho_max,
        "n_theta": n_theta_max,
        "n_zeta": n_zeta_max,
        "n_total": n_total,
        "n_keep": int(keep.size),
        "D_rho0": D_rho0,
        "D_theta0": D_theta0,
        "D_zeta0": D_zeta0,
        "coupled_rt": config.coupled_rt,
        "rho_idx": rho_idx,
        "ups_idx": ups_idx,
        "zeta_idx": zeta_idx,
    }
    out.update({k: f[k] for k in ("g_rr", "g_rv", "g_rp", "g_vv", "g_vp", "g_pp")})
    out.update({k: f[k] for k in ("iota", "psi_r", "psi_r_over_sqrtg")})
    return out


def _component_to_node_permutation(N):
    """Permutation from component-major to node-major ordering.

    Component-major is ``[rho_1..N | theta_1..N | zeta_1..N]``; node-major is
    ``[rho_1, theta_1, zeta_1 | ... ]``. The returned ``p`` satisfies
    ``x_node = x_comp[p]`` and ``M_node = M_comp[p][:, p]``.

    Parameters
    ----------
    N : int
        Number of spatial nodes per component.

    Returns
    -------
    jax.Array of int, shape (3 * N,)
    """
    k = jnp.arange(N, dtype=jnp.int64)
    perm = jnp.empty(3 * N, dtype=jnp.int64)
    perm = perm.at[3 * k + 0].set(k)
    perm = perm.at[3 * k + 1].set(N + k)
    perm = perm.at[3 * k + 2].set(2 * N + k)
    return perm


def finish_ring_block(A_blk, Linv, au_diag_blk, n_nodes):
    """Reproduce the assembler's tail on one ring: permute, whiten, shift, drive.

    The assembler has already applied the symmetric ``d`` scaling and built
    ``Linv``, so only the node-major permutation, the ``Linv A Linv^T``
    congruence, the ``1e-14`` shift and the drive diagonal remain. All of those
    are node-diagonal or a permutation, so they restrict to a ring **exactly**
    -- which is why a ring block equals the corresponding sub-block of the full
    matrix rather than merely approximating it.

    Parameters
    ----------
    A_blk : ndarray, shape (3 * n_nodes, 3 * n_nodes)
    Linv : ndarray, shape (n_nodes, 3, 3)
    au_diag_blk : ndarray, shape (n_nodes,)
    n_nodes : int

    Returns
    -------
    jax.Array, shape (3 * n_nodes, 3 * n_nodes)
    """
    N = n_nodes
    k = jnp.arange(N)
    p = jnp.zeros(3 * N, dtype=jnp.int64)
    p = p.at[3 * k + 0].set(k)
    p = p.at[3 * k + 1].set(N + k)
    p = p.at[3 * k + 2].set(2 * N + k)

    Ap = A_blk[p][:, p].reshape(N, 3, N, 3)
    Ap = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, Ap, Linv)
    node = jnp.arange(N)
    Ap = Ap.at[node, :, node, :].add(1e-14 * jnp.eye(3))
    L0 = Linv[:, :, 0]
    Ap = Ap.at[node, :, node, :].add(
        au_diag_blk[:, None, None] * L0[:, :, None] * L0[:, None, :]
    )
    Ap = Ap.reshape(3 * N, 3 * N)
    pinv = jnp.zeros_like(p).at[p].set(jnp.arange(3 * N))
    return Ap[pinv][:, pinv]


def ring_block(eq, diffmat, config, nodes, density=None):
    """Assemble one poloidal ring's block of the whitened operator.

    Convenience wrapper: restricted assembly followed by
    :func:`finish_ring_block`.

    Parameters
    ----------
    eq : EquilibriumData
    diffmat : DiffMat
    config : AssemblyConfig
    nodes : ndarray of int
        Node indices of the ring.
    density : ndarray, optional

    Returns
    -------
    jax.Array, shape (3 * len(nodes), 3 * len(nodes))
    """
    out = assemble_dense(eq, diffmat, config, density=density, ring_nodes=nodes)
    return finish_ring_block(
        out["A"], out["Linv"], out["au_diag"], jnp.asarray(nodes).size
    )


def matfree_operator(eq, diffmat, config=None, density=None):
    """Build the reduced, whitened AGNI operator as a matrix-free callable.

    This is the **single definition** of the operator used by every matrix-free
    path: the Lanczos solve, the deflated PCG, and the ring preconditioner's
    blocks are all sub-blocks of the same matrix. They therefore agree by
    construction rather than by maintenance.

    Parameters
    ----------
    eq : EquilibriumData
    diffmat : DiffMat
    config : AssemblyConfig, optional
    density : ndarray, shape (n_nodes,), optional

    Returns
    -------
    dict
        ``"Ax"`` applies the operator to a reduced vector of length
        ``n_keep``; ``"Ax_full"`` applies it to the full ``3 * n_total``
        vector. Also returns ``"Linv_DT"``, ``"diagBsqinv"``, ``"keep"``,
        ``"n_keep"`` and the resolution, which the transfer and preconditioner
        machinery needs.
    """
    config = AssemblyConfig() if config is None else config
    n_rho, n_theta, n_zeta = _resolution(eq, diffmat, config)
    n_total = n_rho * n_theta * n_zeta

    if config.axisym:
        D_zeta0 = 1j * config.n_mode_axisym * jnp.array([[1]])
    else:
        D_zeta0 = diffmat.D_zeta
    D_rho0, D_theta0 = diffmat.D_rho, diffmat.D_theta
    gamma = config.gamma

    def rs(u):
        return jnp.asarray(u).reshape(n_rho, n_theta, n_zeta)

    f = _normalized_fields(eq, config)
    # Same fields, reshaped to the 3D grid instead of column vectors.
    g = {k: rs(v) for k, v in f.items() if getattr(v, "ndim", 0) == 2}
    iota, iotainv = g["iota"], g["iotainv"]
    psi_r, psi_r2, psi_r3 = g["psi_r"], g["psi_r2"], g["psi_r3"]
    iota_psi_r2 = g["iota_psi_r2"]
    p0, sqrtg = g["p0"], g["sqrtg"]
    psi_r_over_sqrtg = g["psi_r_over_sqrtg"]
    g_rr, g_vv, g_pp = g["g_rr"], g["g_vv"], g["g_pp"]
    g_rv, g_rp, g_vp = g["g_rv"], g["g_rp"], g["g_vp"]
    g_sup_rr = g["g_sup_rr"]
    g_sup_rv_term, g_sup_rp_term = g["g_sup_rv_term"], g["g_sup_rp_term"]
    J2, j_sup_zeta, j_sup_theta = g["J2"], g["j_sup_zeta"], g["j_sup_theta"]
    F = g["F"]
    partial_r_log_sqrtg = g["sqrtg_r"] / sqrtg
    partial_v_log_sqrtg = g["sqrtg_v"] / sqrtg
    partial_p_log_sqrtg = g["sqrtg_p"] / sqrtg

    n0 = rs(jnp.ones(n_total) if density is None else density)
    W = rs(jnp.kron(diffmat.W_rho, jnp.kron(diffmat.W_theta, diffmat.W_zeta)))

    if config.coupled_rt:

        def d_dr(D, u):
            """Radial derivative via the coupled (rho, theta) operator."""
            return (D @ u.reshape(n_rho * n_theta, n_zeta)).reshape(
                n_rho, n_theta, n_zeta
            )

        def d_dv(D, u):
            """Poloidal derivative via the coupled (rho, theta) operator."""
            return (D @ u.reshape(n_rho * n_theta, n_zeta)).reshape(
                n_rho, n_theta, n_zeta
            )

    else:

        def d_dr(D, u):
            """Radial derivative, separable."""
            return jnp.einsum("ij,jkl->ikl", D, u)

        def d_dv(D, u):
            """Poloidal derivative, separable."""
            return jnp.einsum("ij,kjl->kil", D, u)

    def d_dz(D, u):
        """Toroidal derivative, always separable."""
        return jnp.einsum("ij,klj->kli", D, u)

    dtype = operator_dtype(config)
    B_blocks = jnp.zeros((n_total, 3, 3), dtype=dtype)
    B_blocks = B_blocks.at[:, 0, 0].set((n0 * W * psi_r2 * sqrtg * g_rr).flatten())
    B_blocks = B_blocks.at[:, 1, 1].set((n0 * W * sqrtg * g_vv).flatten())
    B_blocks = B_blocks.at[:, 0, 1].set((n0 * W * psi_r * sqrtg * g_rv).flatten())
    B_blocks = B_blocks.at[:, 1, 0].set((n0 * W * psi_r * sqrtg * g_rv).flatten())

    # The mirror branch is selected from the equilibrium, exactly as the dense
    # assembler does, rather than from a separate flag the caller has to
    # remember to set. The two paths disagreed on a real mirror when this was a
    # manual keyword defaulting to False.
    ismirror = jnp.all(jnp.abs(iota) < 1e-12)
    zz = jnp.where(
        ismirror,
        n0 * W * sqrtg * g_pp,
        n0 * W * sqrtg * (g_vv + 2.0 * iotainv * g_vp + iotainv**2 * g_pp),
    )
    rz = jnp.where(
        ismirror,
        n0 * W * psi_r * sqrtg * g_rp,
        n0 * W * psi_r * sqrtg * (g_rv + iotainv * g_rp),
    )
    uz = jnp.where(
        ismirror,
        n0 * W * psi_r * sqrtg * g_vp,
        n0 * W * sqrtg * (g_vv + iotainv * g_vp),
    )
    B_blocks = B_blocks.at[:, 2, 2].set(zz.flatten())
    B_blocks = B_blocks.at[:, 0, 2].set(rz.flatten())
    B_blocks = B_blocks.at[:, 2, 0].set(rz.flatten())
    B_blocks = B_blocks.at[:, 1, 2].set(uz.flatten())
    B_blocks = B_blocks.at[:, 2, 1].set(uz.flatten())

    diagBsqinv = 1.0 / jnp.sqrt(
        jnp.stack((B_blocks[:, 0, 0], B_blocks[:, 1, 1], B_blocks[:, 2, 2]), axis=1)
    )
    B_scaled = jnp.einsum("...ij,...i,...j->...ij", B_blocks, diagBsqinv, diagBsqinv)

    n_per_shell = n_theta * n_zeta
    boundary_idx = jnp.concatenate(
        [jnp.arange(n_per_shell), jnp.arange(n_total - n_per_shell, n_total)]
    )
    keep = jnp.asarray(keep_indices(n_rho, n_theta, n_zeta))
    n_keep = int(keep.size)

    B_scaled = B_scaled.at[boundary_idx, 0, 1].set(0)
    B_scaled = B_scaled.at[boundary_idx, 1, 0].set(0)
    B_scaled = B_scaled.at[boundary_idx, 0, 2].set(0)
    B_scaled = B_scaled.at[boundary_idx, 2, 0].set(0)

    L_D = jnp.linalg.cholesky(B_scaled)
    I3 = jnp.broadcast_to(jnp.eye(3, dtype=L_D.dtype), L_D.shape)
    Linv_D = jax.lax.linalg.triangular_solve(L_D, I3, left_side=True, lower=True)
    Linv_DT = jnp.swapaxes(Linv_D, -1, -2)

    rt_size = n_rho * n_theta
    alpha, Q_rt, _ = _zernike_penalty(diffmat, rt_size, config.coupled_rt)
    apply_penalty = alpha > 0.0
    if apply_penalty:
        alphaQ_rt = jnp.asarray(alpha * Q_rt)

        def _apply_penalty(u):
            """Q = kron(Q_rt, I_zeta) acting on rho-major data."""
            return (alphaQ_rt @ u.reshape(rt_size, n_zeta)).reshape(
                n_rho, n_theta, n_zeta
            )

    def Ax_full(x_flat):
        """Apply the whitened operator to a full ``3 * n_total`` vector."""
        x = jnp.transpose(x_flat.reshape(3, n_total), axes=(1, 0))
        x = diagBsqinv * jnp.einsum("lij,lj->li", Linv_DT, x)
        x = x.reshape((n_rho, n_theta, n_zeta, 3))
        xr, xu, xz = x[..., 0], x[..., 1], x[..., 2]

        xr_v = d_dv(D_theta0, xr)
        xr_z = d_dz(D_zeta0, xr)
        xu_v = d_dv(D_theta0, xu)
        xu_z = d_dz(D_zeta0, xu)
        xz_v = d_dv(D_theta0, xz)
        xz_z = d_dz(D_zeta0, xz)

        xr1_r = d_dr(D_rho0, iota_psi_r2 * xr)
        xr2_r = d_dr(D_rho0, psi_r2 * xr)
        xr3_r = d_dr(D_rho0, psi_r * xr)

        Ar = jnp.zeros((n_rho, n_theta, n_zeta), dtype=xr.dtype)
        Au = jnp.zeros_like(Ar)
        Az = jnp.zeros_like(Ar)

        # Q^2_rr
        Ar += (
            d_dv(_cT(D_theta0), (psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * xr_v)
            + d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * psi_r3 * W * g_rr) * xr_z)
            + d_dv(_cT(D_theta0), (psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * xr_z)
            + d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * xr_v)
        )

        # Q^2_vv
        Au += d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * psi_r * W * g_vv) * xu_z)
        Ar += iota_psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_vv / psi_r) * xr1_r
        )
        Ar += -iota_psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vv) * xu_z)
        Au += -d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * W * g_vv) * xr1_r)

        # Q^2_zz
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * psi_r * W * g_pp) * xu_v)
        Ar += psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_pp / psi_r) * xr2_r)
        Ar += psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_pp) * xu_v)
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * W * g_pp) * xr2_r)

        # Q^2_rv and transpose
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r * psi_r_over_sqrtg * W * g_rv) * xr1_r)
            + d_dz(_cT(D_zeta0), (psi_r * psi_r_over_sqrtg * W * g_rv) * xr1_r)
        )
        Ar += -(
            iota_psi_r2
            * d_dr(_cT(D_rho0), (iota * psi_r * psi_r_over_sqrtg * W * g_rv) * xr_v)
            + iota_psi_r2
            * d_dr(_cT(D_rho0), (psi_r * psi_r_over_sqrtg * W * g_rv) * xr_z)
        )
        Ar += d_dv(
            _cT(D_theta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * xu_z
        ) + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rv) * xu_z)
        Au += d_dz(
            _cT(D_zeta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * xr_v
        ) + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rv) * xr_z)

        # Q^2_rz and transpose
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r * psi_r_over_sqrtg * W * g_rp) * xr2_r)
            + d_dz(_cT(D_zeta0), (psi_r * psi_r_over_sqrtg * W * g_rp) * xr2_r)
        )
        Ar += -(
            psi_r2
            * d_dr(_cT(D_rho0), (iota * psi_r * psi_r_over_sqrtg * W * g_rp) * xr_v)
            + psi_r2 * d_dr(_cT(D_rho0), (psi_r * psi_r_over_sqrtg * W * g_rp) * xr_z)
        )
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * xu_v)
            + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rp) * xu_v)
        )
        Au += -(
            d_dv(_cT(D_theta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * xr_v)
            + d_dv(_cT(D_theta0), (psi_r2 * psi_r_over_sqrtg * W * g_rp) * xr_z)
        )

        # Q^2_vz and transpose
        Au += -(
            d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * psi_r * W * g_vp) * xu_v)
            + d_dv(_cT(D_theta0), (psi_r_over_sqrtg * psi_r * W * g_vp) * xu_z)
        )
        Ar += -psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vp) * xu_z)
        Ar += iota_psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vp) * xu_v)
        Au += -d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * W * g_vp) * xr2_r)
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * W * g_vp) * xr1_r)
        Ar += iota_psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_vp / psi_r) * xr2_r
        )
        Ar += psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vp / psi_r) * xr1_r)

        # J x Q terms, by the same route as the dense assembler: g^rv and g^rz
        # come from the PEST lower metric via the identity, not from a supplied
        # contravariant metric. The `_term` factors already carry psi_r*sqrt(g),
        # hence psi_r2 here.
        jq = (
            psi_r2
            * W
            * (j_sup_theta * g_sup_rp_term - j_sup_zeta * g_sup_rv_term)
            / g_sup_rr
        )
        Ar += jq * (iota * xr_v + xr_z)
        Ar += -(psi_r * sqrtg * W * j_sup_zeta) * xr1_r
        Ar += (psi_r * sqrtg * W * j_sup_theta) * xr2_r
        Ar += iota * d_dv(_cT(D_theta0), jq * xr) + d_dz(_cT(D_zeta0), jq * xr)
        Ar += -iota_psi_r2 * d_dr(_cT(D_rho0), psi_r * sqrtg * W * j_sup_zeta * xr)
        Ar += psi_r2 * d_dr(_cT(D_rho0), psi_r * sqrtg * W * j_sup_theta * xr)

        Ar += (W * psi_r2 * sqrtg * j_sup_theta) * xu_v + (
            W * psi_r2 * sqrtg * j_sup_zeta
        ) * xu_z
        Au += d_dv(_cT(D_theta0), W * psi_r2 * sqrtg * j_sup_theta * xr)
        Au += d_dz(_cT(D_zeta0), W * psi_r2 * sqrtg * j_sup_zeta * xr)

        # |J|^2 and the instability drive
        Ar += (psi_r2 * W * sqrtg * J2) / g_sup_rr * xr
        Aur = (W * psi_r2 * sqrtg * F) * xr

        # Compressibility
        gp = gamma * sqrtg * W * p0
        cr = psi_r * partial_r_log_sqrtg * xr + xr3_r
        cu = partial_v_log_sqrtg * xu + xu_v
        cz = (
            partial_v_log_sqrtg * xz
            + xz_v
            + iotainv * (partial_p_log_sqrtg * xz + xz_z)
        )

        Ar += psi_r * (partial_r_log_sqrtg * gp * cr + d_dr(_cT(D_rho0), gp * cr))
        Au += partial_v_log_sqrtg * gp * cu + d_dv(_cT(D_theta0), gp * cu)
        Az += (
            partial_v_log_sqrtg * gp * cz
            + d_dv(_cT(D_theta0), gp * cz)
            + iotainv * (partial_p_log_sqrtg * gp * cz + d_dz(_cT(D_zeta0), gp * cz))
        )
        Ar += psi_r * (partial_r_log_sqrtg * gp * cu + d_dr(_cT(D_rho0), gp * cu))
        Au += partial_v_log_sqrtg * gp * cr + d_dv(_cT(D_theta0), gp * cr)
        Ar += psi_r * (partial_r_log_sqrtg * gp * cz + d_dr(_cT(D_rho0), gp * cz))
        Az += (
            partial_v_log_sqrtg * gp * cr
            + d_dv(_cT(D_theta0), gp * cr)
            + iotainv * (partial_p_log_sqrtg * gp * cr + d_dz(_cT(D_zeta0), gp * cr))
        )
        Au += partial_v_log_sqrtg * gp * cz + d_dv(_cT(D_theta0), gp * cz)
        Az += (
            partial_v_log_sqrtg * gp * cu
            + d_dv(_cT(D_theta0), gp * cu)
            + iotainv * (partial_p_log_sqrtg * gp * cu + d_dz(_cT(D_zeta0), gp * cu))
        )

        if apply_penalty:
            Ar = Ar + _apply_penalty(xr)
            Au = Au + _apply_penalty(xu)
            Az = Az + _apply_penalty(xz)

        As = jnp.stack([Ar, Au, Az], axis=-1).reshape((n_total, 3))
        Aus = jnp.stack(
            [Aur, jnp.zeros_like(Aur), jnp.zeros_like(Aur)], axis=-1
        ).reshape((n_total, 3))
        ys = jnp.einsum("lij,lj->li", Linv_D, diagBsqinv * As)
        yus = jnp.einsum("lij,lj->li", Linv_D, diagBsqinv * Aus)
        y = ys + yus
        # The same constant diagonal shift the dense assembler adds in the
        # whitened basis. These must stay in sync or the two paths differ by a
        # uniform shift.
        return y.T.reshape(-1) + 1e-14 * x_flat

    def Ax(x_reduced):
        """Apply the operator to a reduced vector of length ``n_keep``."""
        x_full = jnp.zeros(3 * n_total, dtype=x_reduced.dtype)
        # unique_indices=True: `keep` is a concatenation of disjoint aranges, so
        # declaring it lets JAX form the scatter's transpose, which the CG
        # shift-invert path needs for its symmetric transpose-solve.
        x_full = x_full.at[keep].set(x_reduced, unique_indices=True)
        return Ax_full(x_full)[keep]

    return {
        "Ax": Ax,
        "Ax_full": Ax_full,
        "Linv_DT": Linv_DT,
        "diagBsqinv": diagBsqinv,
        "keep": keep,
        "n_keep": n_keep,
        "n_rho": n_rho,
        "n_theta": n_theta,
        "n_zeta": n_zeta,
        "n_total": n_total,
        "NFP": eq.NFP,
        "d_dr": d_dr,
        "d_dv": d_dv,
        "d_dz": d_dz,
    }
