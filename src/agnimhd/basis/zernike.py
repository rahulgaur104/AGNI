"""Zernike polynomials and the coupled Zernike-Fourier differentiation matrices.

Why this module exists
----------------------
Zernike polynomials do **not** factor into independent ``rho`` and ``theta``
operators the way a tensor-product basis does. A single pseudo-inverse fits
nodal values to the Zernike-Fourier spectral basis, and the derivative
evaluation matrices are composed with that fit to give two *coupled* real-space
operators of shape ``(n_rho * n_theta, n_rho * n_theta)``. That coupling is the
entire reason the machinery exists: it is what makes the disc center well
behaved, where a separable radial basis has a coordinate singularity.

What is implemented here
------------------------
The radial polynomials come from the **Jacobi-polynomial recurrence**

.. math::

    R_l^{m}(\\rho) = (-1)^{n}\\, \\rho^{|m|}\\,
                     P_n^{(|m|,\\,0)}(1 - 2\\rho^2), \\qquad n = (l - |m|)/2

which is stable at moderate and large ``L``. The explicit factorial sum is
*not* used: it loses accuracy badly by ``L ~ 20`` through cancellation between
large alternating terms, and AGNI runs well past that.

Cross-checking without a DESC dependency
----------------------------------------
Mode ORDERING (the ``"ansi"`` and ``"fringe"`` conventions) and the conditioning
of the nodal-to-spectral fit are *conventions*, not theorems -- an independent
implementation can be internally consistent and still disagree, and a consumer
mixing the two would then get silently wrong derivatives. So the cross-check is
preserved **by value**: ``tools/export_zernike_reference.py`` recorded what DESC
produces, those values are committed as ``tests/data/zernike_reference.npz``,
and ``tests/test_zernike.py`` compares against that file. DESC is nowhere in the
dependency graph.

References
----------
Born & Wolf, *Principles of Optics*, for the Zernike definition.
ANSI Z80.28 for the ANSI indexing convention; the "fringe"/University of Arizona
convention for the other.
"""

import numpy as np

from ..backend import errorif, jnp

__all__ = [
    "fourier",
    "zernike_eval_matrix",
    "zernike_fourier_diffmat",
    "zernike_modes",
    "zernike_penalty_projector_from_diffmat",
    "zernike_radial",
]


def _binom(n, k):
    """Binomial coefficient ``C(n, k)``, elementwise, for non-negative integers.

    Evaluated by the multiplicative recurrence ``b *= (n + 1 - i) / i``, which
    avoids forming factorials that overflow well before the polynomial degrees
    AGNI uses do.
    """
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    kmax = int(np.max(k)) if k.size else 0
    b = np.ones(np.broadcast(n, k).shape, dtype=float)
    for i in range(1, kmax + 1):
        active = i <= k
        b = np.where(active, b * (n + 1 - i) / i, b)
    return b


def _jacobi(n, alpha, beta, x, dx=0):
    """Jacobi polynomial ``P_n^{(alpha, beta)}(x)`` and its ``dx``-th derivative.

    Uses the standard forward recurrence (the same one SciPy's
    ``eval_jacobi`` uses). Derivatives use the identity

    .. math::

        \\frac{d^k}{dx^k} P_n^{(a,b)}
            = \\frac{\\Gamma(a + b + n + 1 + k)}{2^k\\,\\Gamma(a + b + n + 1)}
              P_{n-k}^{(a+k,\\,b+k)}

    so a derivative costs no more than the value.

    Parameters
    ----------
    n : ndarray of int, shape (K,)
        Degree. Negative degrees return 0.
    alpha, beta : ndarray, shape (K,)
        Jacobi parameters.
    x : ndarray, shape (N, 1)
        Evaluation points, shaped to broadcast against the mode axis.
    dx : int
        Derivative order.

    Returns
    -------
    ndarray, shape (N, K)
    """
    n = np.asarray(n, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    x = np.asarray(x, dtype=float)

    coeff = np.ones_like(n)
    for j in range(1, dx + 1):
        coeff = coeff * (n + alpha + beta + j) / 2.0
    n = n - dx
    alpha = alpha + dx
    beta = beta + dx

    d = (alpha + beta + 2) * (x - 1) / (2 * (alpha + 1))
    p = d + 1.0

    steps = np.maximum(n - 1, 0)
    kmax = int(np.max(steps)) if steps.size else 0
    for kk in range(kmax):
        k = kk + 1.0
        t = 2 * k + alpha + beta
        d_new = (
            (t * (t + 1) * (t + 2)) * (x - 1) * p + 2 * k * (k + beta) * (t + 2) * d
        ) / (2 * (k + alpha + 1) * (k + alpha + beta + 1) * t)
        active = kk < steps
        p = np.where(active, d_new + p, p)
        d = np.where(active, d_new, d)

    out = _binom(n + alpha, n) * p
    out = np.where(n < 0, 0.0, out)
    out = np.where(n == 0, 1.0, out)
    out = np.where(n == 1, 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (x - 1)), out)
    return coeff * out


def zernike_radial(r, l, m, dr=0):  # noqa: E741 -- `l` is the radial degree
    """Radial part ``R_l^m(r)`` of the Zernike polynomials, or its derivative.

    Parameters
    ----------
    r : ndarray, shape (N,) or (N, 1)
        Radial coordinates in ``[0, 1]``.
    l : ndarray of int, shape (K,)
        Radial mode numbers.
    m : ndarray of int, shape (K,)
        Azimuthal mode numbers. Only ``|m|`` matters here; the sign selects
        cosine or sine in :func:`fourier`.
    dr : {0, 1}
        Derivative order with respect to ``r``.

    Returns
    -------
    ndarray, shape (N, K)
        ``R_l^m`` evaluated at each ``r``. Modes with ``(l - |m|)`` odd are
        identically zero, as the definition requires.

    Raises
    ------
    NotImplementedError
        For ``dr > 1``. AGNI's operators are first order; higher derivatives
        would be dead code, so they are refused rather than shipped untested.
    """
    errorif(
        dr not in (0, 1),
        NotImplementedError,
        f"zernike_radial supports dr in (0, 1), got {dr}. AGNI's differentiation "
        "matrices are first order, so higher derivatives have no caller and are "
        "deliberately not implemented.",
    )
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    l = np.asarray(l)  # noqa: E741 -- matches the standard R_l^m notation
    m_abs = np.abs(np.asarray(m)).astype(float)
    n = (l - m_abs) // 2
    s = (-1.0) ** n
    beta = np.zeros_like(m_abs)
    arg = 1 - 2 * r**2

    if dr == 0:
        out = r**m_abs * _jacobi(n, m_abs, beta, arg, 0)
    else:
        f = _jacobi(n, m_abs, beta, arg, 0)
        df = _jacobi(n, m_abs, beta, arg, 1)
        out = m_abs * r ** np.maximum(m_abs - 1, 0) * f - 4 * r ** (m_abs + 1) * df

    return s * np.where((l - m_abs) % 2 == 0, out, 0.0)


def fourier(theta, m, dt=0):
    """Real Fourier basis, or its derivative, in the AGNI/DESC sign convention.

    ``m >= 0`` gives ``cos(|m| theta)`` and ``m < 0`` gives ``sin(|m| theta)``,
    written as a single shifted sine so that the derivative is one expression.

    Parameters
    ----------
    theta : ndarray, shape (N,) or (N, 1)
        Poloidal angle.
    m : ndarray of int, shape (K,)
        Mode numbers; the sign selects cosine (``>= 0``) or sine (``< 0``).
    dt : int
        Derivative order.

    Returns
    -------
    ndarray, shape (N, K)
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[:, None]
    m = np.asarray(m)
    m_pos = (m >= 0).astype(int)
    m_abs = np.abs(m).astype(float)
    shift = m_pos * np.pi / 2 + dt * np.pi / 2
    return m_abs**dt * np.sin(m_abs * theta + shift)


def zernike_modes(L, M, spectral_indexing="ansi"):
    """Mode numbers ``(l, m)`` of a Zernike basis, in AGNI's canonical order.

    Parameters
    ----------
    L : int
        Maximum radial resolution.
    M : int
        Maximum poloidal resolution.
    spectral_indexing : {"ansi", "fringe"}
        Which region of the Zernike pyramid to fill.

        ``"ansi"`` fills the pyramid with triangles of decreasing size, ending
        in a triangle. For ``L == M`` this is the traditional ANSI pyramid; for
        ``L > M`` it adds rows to the bottom, giving a "house" shape.

        ``"fringe"`` fills with chevrons of decreasing size, ending in a diamond
        at ``L == 2*M`` -- the traditional fringe / University of Arizona
        indexing. For ``L > 2*M`` it adds chevrons, giving a hexagonal diamond.

    Returns
    -------
    ndarray of int, shape (n_modes, 2)
        Rows ``(l, m)``, **lexicographically sorted by (l, m)**. The indexing
        convention chooses which modes are present; the ordering afterwards is
        always this sort, so two bases with the same mode set are always
        ordered identically regardless of how they were specified.

    Raises
    ------
    ValueError
        For an unknown ``spectral_indexing``.
    """
    errorif(
        spectral_indexing not in ("ansi", "fringe"),
        ValueError,
        f"spectral_indexing must be 'ansi' or 'fringe', got " f"{spectral_indexing!r}.",
    )
    L, M = int(L), int(M)
    errorif(L < 0 or M < 0, ValueError, f"L and M must be >= 0, got {L}, {M}.")

    pol_posm = []
    if spectral_indexing == "ansi":
        for d in range(0, L + 1, 2):
            pol_posm += [(m + d, m) for m in range(0, M + 1) if m + d < M + 1]
        if L > M:
            pol_posm += [
                (ll, m)
                for ll in range(M + 1, L + 1)
                for m in range(0, M + 1)
                if (ll - m) % 2 == 0
            ]
    else:
        for d in range(0, L + 1, 2):
            pol_posm += [
                (m + d // 2, m - d // 2) for m in range(0, M + 1) if m - d // 2 >= 0
            ]
        if L > 2 * M:
            for ll in range(2 * M, L + 1, 2):
                pol_posm += [(ll - m, m) for m in range(0, M + 1)]

    pol = []
    for ll, m in pol_posm:
        pol.append((ll, m))
        if m != 0:
            pol.append((ll, -m))
    pol = np.asarray(pol, dtype=int).reshape(-1, 2)

    order = np.lexsort((pol[:, 1], pol[:, 0]))
    return pol[order]


def zernike_eval_matrix(rho, theta, modes, dr=0, dt=0):
    """Evaluate a Zernike-Fourier basis on the tensor product of the nodes.

    Parameters
    ----------
    rho : ndarray, shape (n_rho,)
        Radial nodes.
    theta : ndarray, shape (n_theta,)
        Poloidal nodes.
    modes : ndarray of int, shape (n_modes, 2)
        Rows ``(l, m)``, from :func:`zernike_modes`.
    dr, dt : int
        Derivative orders in ``rho`` and ``theta``.

    Returns
    -------
    ndarray, shape (n_rho * n_theta, n_modes)
        Rows are nodes in **rho-major** order -- node ``(i, j)`` is row
        ``i * n_theta + j`` -- matching the ordering the rest of the package
        uses. Columns follow ``modes``.
    """
    rho = np.asarray(rho, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    modes = np.asarray(modes, dtype=int).reshape(-1, 2)
    ell, m = modes[:, 0], modes[:, 1]

    radial = zernike_radial(rho, ell, m, dr=dr)  # (n_rho, K)
    poloidal = fourier(theta, m, dt=dt)  # (n_theta, K)
    # rho-major: repeat each rho row across all theta, tile theta.
    return radial[:, None, :].repeat(theta.size, axis=1).reshape(
        -1, modes.shape[0]
    ) * np.tile(poloidal, (rho.size, 1))


def zernike_fourier_diffmat(
    rho, theta, L=-1, M=-1, spectral_indexing="ansi", rcond=None
):
    """Coupled radial and poloidal Zernike-Fourier derivative operators.

    A single pseudo-inverse fits nodal values to the Zernike basis; the radial
    and poloidal derivative evaluation matrices are then composed with that fit
    to give two coupled real-space operators::

        D_rho   = (dA/drho)   @ pinv(A)
        D_theta = (dA/dtheta) @ pinv(A)

    Parameters
    ----------
    rho, theta : array-like
        One-dimensional radial and poloidal collocation nodes.
    L, M : int
        Zernike radial and poloidal resolutions. ``-1`` picks a resolution from
        the node counts: ``L = 2 * (n_rho - 1)``, ``M = (n_theta - 1) // 2``.
    spectral_indexing : {"ansi", "fringe"}
        Zernike indexing convention.
    rcond : float, optional
        Relative singular-value cutoff for the pseudo-inverse. ``None`` uses
        NumPy's default, ``max(A.shape) * eps``, which is what DESC's
        ``Transform(build_pinv=True)`` uses.

        **The fit is rank-deficient whenever the nodes under-determine the
        basis** -- e.g. a basis chosen with ``L`` or ``M`` larger than the node
        counts support. The pseudo-inverse then silently returns the
        minimum-norm solution rather than erroring, and the resulting operators
        annihilate the unresolved content instead of differentiating it. That is
        the intended behavior (the de-aliasing penalty below is built to
        penalize exactly that content), but it means a badly chosen ``L``/``M``
        degrades accuracy without any diagnostic. Prefer the ``-1`` defaults.

    Returns
    -------
    D_rho, D_theta : tuple of jax.Array
        Coupled first-derivative matrices, each of shape
        ``(n_rho * n_theta, n_rho * n_theta)``, in rho-major node ordering.

    Raises
    ------
    ValueError
        If ``rho`` or ``theta`` is not one-dimensional, or is empty.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    errorif(
        rho.ndim != 1 or theta.ndim != 1,
        ValueError,
        "rho and theta must be one-dimensional.",
    )
    errorif(
        rho.size < 1 or theta.size < 1, ValueError, "rho and theta cannot be empty."
    )

    M = max((theta.size - 1) // 2, 0) if M == -1 else int(M)
    L = 2 * (rho.size - 1) if L == -1 else int(L)

    modes = zernike_modes(L, M, spectral_indexing)
    A = zernike_eval_matrix(rho, theta, modes, dr=0, dt=0)
    dA_drho = zernike_eval_matrix(rho, theta, modes, dr=1, dt=0)
    dA_dtheta = zernike_eval_matrix(rho, theta, modes, dr=0, dt=1)

    pinv = np.linalg.pinv(A, rcond=rcond) if rcond is not None else np.linalg.pinv(A)
    return jnp.asarray(dA_drho @ pinv), jnp.asarray(dA_dtheta @ pinv)


def zernike_penalty_projector_from_diffmat(D_rho, D_theta, svd_tol=1e-10):
    """Build the coupled Zernike-Fourier de-aliasing penalty projector.

    The stacked derivative row space spans the represented basis modulo
    constants (a constant is annihilated by both derivatives, so it never
    appears in the row space). Adding the constant mode back leaves exactly the
    unresolved nodal complement to be penalized, and the returned ``Q = I - P``
    projects onto it.

    Parameters
    ----------
    D_rho, D_theta : array-like
        Coupled radial and poloidal derivative matrices of matching square
        shape ``(n_rho * n_theta, n_rho * n_theta)``.
    svd_tol : float
        Relative SVD cutoff used to decide the rank of the represented space.

    Returns
    -------
    projector : jax.Array
        Hermitian projector onto unresolved nodal content.
    rank : int
        Dimension of the represented range, after adding the constant mode.

    Raises
    ------
    ValueError
        If either matrix is missing or not square, or if their shapes differ.
    """
    errorif(
        D_rho is None or D_theta is None,
        ValueError,
        "D_rho and D_theta are required to build a Zernike penalty projector.",
    )
    D_rho = np.asarray(D_rho)
    D_theta = np.asarray(D_theta)
    errorif(
        D_rho.ndim != 2 or D_rho.shape[0] != D_rho.shape[1],
        ValueError,
        "D_rho must be a square matrix.",
    )
    errorif(
        D_theta.shape != D_rho.shape,
        ValueError,
        "D_theta must have the same shape as D_rho.",
    )

    rt_size = D_rho.shape[0]
    D_stack = np.vstack((D_rho, D_theta))
    _, svals, vh = np.linalg.svd(D_stack, full_matrices=False)
    cutoff = float(svd_tol) * max(float(svals[0]), 1.0)
    rank = int(np.count_nonzero(svals > cutoff))

    range_basis = vh[:rank].conj().T
    dtype = range_basis.dtype if range_basis.size else D_stack.dtype
    const = np.ones((rt_size, 1), dtype=dtype)
    if rank:
        const = const - range_basis @ (range_basis.conj().T @ const)
    const_norm = np.linalg.norm(const)
    # RELATIVE threshold, against the norm of the vector being projected
    # (||ones|| = sqrt(rt_size)). An ABSOLUTE `> 10 * eps` test, which is what
    # this code originally used, is wrong whenever the constant already lies
    # inside the derivative row space: the residual is then pure roundoff, the
    # absolute test still passes, and normalizing that residual appends a
    # UNIT-LENGTH NOISE VECTOR to the represented basis. The penalty then fails
    # to penalize an essentially arbitrary direction, differently on every
    # machine and every build.
    #
    # Measured on a 4x6 node set with an over-resolved L=8, M=3 basis: the
    # residual norm is 2e-12 (ansi) and 5e-12 (fringe) against ||ones|| = 4.9,
    # and two implementations agreeing on D_rho/D_theta to 4e-15 produced
    # projectors differing by 0.35 in the sup norm. With the well-posed default
    # (L = M = -1) the residual is 4.9 -- the constant is genuinely outside the
    # row space, as it must be, since both derivatives annihilate it -- so the
    # default path is unaffected and still matches to 1e-15.
    #
    # 1e-8 sits ~1e4 above the observed noise and ~1e8 below a genuine
    # residual, so the gap is not close on either side.
    const_floor = 1e-8 * np.sqrt(rt_size)
    if const_norm > const_floor:
        range_basis = np.concatenate((range_basis, const / const_norm), axis=1)

    P_rt = range_basis @ range_basis.conj().T
    Q_rt = np.eye(rt_size, dtype=P_rt.dtype) - P_rt
    Q_rt = 0.5 * (Q_rt + Q_rt.conj().T)
    return jnp.asarray(Q_rt), int(range_basis.shape[1])
