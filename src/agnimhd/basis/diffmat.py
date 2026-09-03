"""Differentiation and quadrature matrix pairs, and the ``DiffMat`` that holds them.

Every basis here returns a pair ``(D, W)``: a first-derivative matrix and a
quadrature matrix on the *same* nodes. They must be used together and on the
nodes they were built for.

Summation by parts
------------------
Most of these pairs satisfy a **summation-by-parts (SBP)** identity,

.. math::  D^T W + W D = B

with ``B = diag(-1, 0, ..., 0, 1)`` on a non-periodic interval and ``B = 0`` on a
periodic one. This is the discrete analogue of integration by parts, and it is
what makes the discrete energy functional match the continuous one. A basis that
quietly loses SBP does not give a visibly broken answer -- it gives a plausible,
*wrong* eigenvalue. ``tests/test_diffmat.py`` therefore checks SBP explicitly for
every basis that is supposed to have it.

References
----------
Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM.
Canuto et al. (2006). *Spectral Methods -- Fundamentals in Single Domains*.
Fornberg, B. (1998). *A Practical Guide to Pseudospectral Methods*, §3.2.
Carpenter, Gottlieb & Abarbanel (1994), for the fourth-order SBP boundary
closures used by :func:`finite_difference_diffmat`.
"""

import numpy as np

from ..backend import check_posint, errorif, jax, jnp
from ..quadrature import (
    bspline_clamped_uniform_knots,
    bspline_nodes_weights,
    gauss_radau_jacobi,
    leggauss_lob,
)
from .zernike import zernike_penalty_projector_from_diffmat

__all__ = [
    "DiffMat",
    "bspline_diffmat",
    "finite_difference_diffmat",
    "fourier_diffmat",
    "fourier_diffmat_truncated",
    "fourier_pts",
    "jacobi_diffmat",
    "legendre_diffmat",
    "standard_grid",
]

#: Default de-aliasing penalty strength for the coupled Zernike-Fourier path.
#: This is the value the converged production runs used; it is a documented
#: default, not a magic number. Zero disables the penalty entirely.
DEFAULT_ZERNIKE_PENALTY_ALPHA = 0.05


def _barycentric_weights(x):
    """Barycentric weights ``lambda_i = 1 / prod_{j != i} (x_i - x_j)``."""
    x = jnp.asarray(x)
    diff = x[:, None] - x[None, :]
    diff_eye = diff + jnp.eye(x.size)
    return 1.0 / jnp.prod(diff_eye, axis=1)


def _barycentric_diffmat(nodes):
    """First-derivative matrix on arbitrary nodes, by barycentric interpolation.

    The diagonal is set by the negative-sum trick rather than by its closed
    form: enforcing ``sum_j D_ij = 0`` exactly makes the operator annihilate
    constants to machine precision, which matters because a constant is in the
    null space of every derivative and any leak there shows up directly in the
    eigenvalue.
    """
    nodes = jnp.asarray(nodes)
    n = nodes.size
    lam = _barycentric_weights(nodes)
    difference = nodes[:, None] - nodes[None, :]
    safe = difference + jnp.eye(n)
    D = (lam[None, :] / lam[:, None]) / safe
    D = D.at[jnp.diag_indices(n)].set(0.0)
    D = D.at[jnp.diag_indices(n)].set(-jnp.sum(D, axis=1))
    return D


def legendre_diffmat(N):
    """Legendre-Gauss-Lobatto differentiation and quadrature matrices.

    Parameters
    ----------
    N : int
        Number of nodes, at least 2.

    Returns
    -------
    D, W : tuple of jax.Array
        ``(N, N)`` first-derivative matrix and diagonal quadrature matrix on
        the LGL nodes of ``[-1, 1]``.

    Notes
    -----
    Satisfies SBP with ``B = diag(-1, 0, ..., 0, 1)`` to machine precision.
    """
    N = check_posint(N, "N", False)
    errorif(N < 2, ValueError, f"N must be at least 2, got {N}.")
    x, w = leggauss_lob(N)
    D = _barycentric_diffmat(x)
    W = jnp.zeros((N, N)).at[jnp.diag_indices(N)].set(w)
    return D, W


def standard_grid(n_rho, n_theta, n_zeta, NFP=1, automorphism=None):
    """Build AGNI's default node set and its matching :class:`DiffMat`.

    Legendre-Lobatto radially -- pushed through the clustering automorphism if
    one is given -- and Fourier in both angles, with the toroidal pair scaled
    for a single field period.

    This exists as library code rather than as a snippet in each driver for one
    reason: **the nodes and the matrices must come from the same construction**.
    Nothing downstream can check that they do. A ``DiffMat`` built with
    different automorphism parameters than the geometry was evaluated on is not
    an error, it is a wrong eigenvalue. Returning both from one call makes the
    mismatch hard to write.

    Parameters
    ----------
    n_rho, n_theta, n_zeta : int
        Grid resolution.
    NFP : int, optional
        Field periods. The toroidal nodes span ``[0, 2*pi/NFP)`` and the
        toroidal pair is scaled accordingly. Default 1.
    automorphism : dict, optional
        Keyword arguments for
        :func:`~agnimhd.quadrature.automorphism_staircase1`, e.g.
        ``dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)``. ``None`` leaves the
        Lobatto nodes unclustered, mapped affinely onto ``[0, 1]`` -- which
        **puts a node on the magnetic axis**, where ``1/sqrt(g)`` and ``g_rv``
        are singular. That is fine for derivative and quadrature tests and wrong
        for a stability solve; a physical run wants an automorphism with
        ``eps`` in roughly ``[1e-3, 1e-2]``.

    Returns
    -------
    nodes : dict
        ``{"rho": (n_rho,), "theta": (n_theta,), "zeta": (n_zeta,)}``. Evaluate
        the equilibrium at the tensor product of these, flattened rho-major.
    diffmat : DiffMat

    Notes
    -----
    The radial mapping enters twice, and both are handled here: the derivative
    matrix is divided by ``f'`` and the quadrature weights are multiplied by it,
    which is the discrete form of the change of variable
    ``INT drho_s X = INT drho f'(rho) X(f(rho))``.

    Examples
    --------
    >>> from agnimhd.basis import standard_grid
    >>> nodes, dm = standard_grid(24, 12, 8, NFP=4,
    ...                           automorphism=dict(eps=1e-2, x_0=0.65,
    ...                                             m_1=2.0, m_2=3.0))
    >>> nodes["rho"].shape, nodes["theta"].shape, nodes["zeta"].shape
    ((24,), (12,), (8,))
    """
    from ..quadrature import automorphism_staircase1

    n_rho = check_posint(n_rho, "n_rho", False)
    n_theta = check_posint(n_theta, "n_theta", False)
    n_zeta = check_posint(n_zeta, "n_zeta", False)
    NFP = check_posint(NFP, "NFP", False)
    errorif(
        n_rho < 3,
        ValueError,
        f"n_rho must be at least 3, got {n_rho}: the innermost and outermost "
        "radial shells are Dirichlet-constrained, so fewer than three leaves "
        "nothing free.",
    )

    x_lob, _ = leggauss_lob(n_rho)
    D_rho, W_rho = legendre_diffmat(n_rho)
    if automorphism is None:
        # Lobatto nodes live on [-1, 1]; map affinely to [0, 1] and scale the
        # operators by the same jacobian, d(rho)/dx = 1/2.
        rho = 0.5 * (jnp.asarray(x_lob) + 1.0)
        dfa = jnp.full((n_rho,), 0.5)
    else:
        rho = automorphism_staircase1(x_lob, **automorphism)
        dfa = jax.vmap(
            lambda x: jax.grad(automorphism_staircase1, argnums=0)(x, **automorphism)
        )(jnp.asarray(x_lob))

    theta = fourier_pts(n_theta)
    D_theta, W_theta = fourier_diffmat(n_theta)

    if n_zeta == 1:
        # The axisymmetric level. A single toroidal node carries no derivative
        # across it, so `D_zeta` is the 1x1 zero matrix and the toroidal
        # dependence is supplied analytically by `AssemblyConfig.n_mode_axisym`,
        # which turns d/dphi into i*n. The one weight is the full toroidal
        # extent of the domain. Without this branch `fourier_diffmat(1)` raises
        # and the tokamak case cannot be built from the public API at all.
        zeta = jnp.zeros((1,))
        D_zeta_scaled = jnp.zeros((1, 1))
        W_zeta_scaled = jnp.asarray([2.0 * jnp.pi / NFP])
    else:
        zeta = fourier_pts(n_zeta, domain=[0.0, 2.0 * jnp.pi / NFP])
        D_zeta, W_zeta = fourier_diffmat(n_zeta)
        D_zeta_scaled = D_zeta * NFP
        W_zeta_scaled = jnp.diagonal(W_zeta / NFP)

    diffmat = DiffMat(
        D_rho=D_rho / dfa[:, None],
        W_rho=jnp.diagonal(W_rho * dfa[:, None]),
        D_theta=D_theta,
        W_theta=jnp.diagonal(W_theta),
        D_zeta=D_zeta_scaled,
        W_zeta=W_zeta_scaled,
    )
    return {"rho": jnp.asarray(rho), "theta": theta, "zeta": zeta}, diffmat


def fourier_pts(n, domain=None):
    """Equally spaced nodes on a periodic domain.

    Parameters
    ----------
    n : int
        Number of points.
    domain : sequence of 2 floats, optional
        Physical interval ``[a, b]``. Defaults to ``[0, 2*pi]``.

    Returns
    -------
    jax.Array, shape (n,)
        Nodes with spacing ``(b - a) / n``; the right endpoint is excluded.
    """
    if domain is None:
        domain = [0, 2 * jnp.pi]
    return jnp.linspace(domain[0], domain[1], n, endpoint=False)


def fourier_diffmat(n):
    """Skew-symmetric first-derivative matrix on a periodic Fourier grid.

    Parameters
    ----------
    n : int
        Grid size, at least 2.

    Returns
    -------
    D, W : tuple of jax.Array
        ``(n, n)`` matrices. ``D`` is exact for every complex exponential below
        the Nyquist limit.

    Notes
    -----
    The formula is Fornberg (1998) §3.2: the denominator is ``tan`` for even
    ``n`` and ``sin`` for odd ``n``. Since the weights are uniform, SBP reduces
    to ``D`` being skew-symmetric, i.e. ``B = 0`` -- there is no boundary.
    """
    n = check_posint(n, "n", False)
    errorif(n < 2, ValueError, f"n must be at least 2, got {n}.")
    i, j = jnp.mgrid[0:n, 0:n]
    angle = (i - j) * jnp.pi / n
    if n % 2 == 0:
        D = jnp.where(i != j, 0.5 * (-1.0) ** (i - j) / jnp.tan(angle), 0.0)
    else:
        D = jnp.where(i != j, 0.5 * (-1.0) ** (i - j) / jnp.sin(angle), 0.0)
    W = jnp.zeros((n, n)).at[jnp.diag_indices(n)].set(2 * jnp.pi / n)
    return D, W


def fourier_diffmat_truncated(n, M=None):
    """Fourier differentiation matrix with modes above ``M`` mapped to zero.

    The collocation grid and quadrature weights are identical to
    :func:`fourier_diffmat`; only the retained spectral content differs.
    Omitting ``M`` keeps every resolvable non-Nyquist mode and reproduces
    :func:`fourier_diffmat`.

    Parameters
    ----------
    n : int
        Number of equally spaced collocation points on ``[0, 2*pi)``.
    M : int, optional
        Highest retained wavenumber, ``1 <= M <= (n - 1) // 2``.

    Returns
    -------
    D, W : tuple of jax.Array, each shape (n, n)

    Raises
    ------
    ValueError
        If ``n < 3`` or ``M`` exceeds the Nyquist limit.
    """
    n = check_posint(n, "n", False)
    max_mode = (n - 1) // 2
    errorif(max_mode < 1, ValueError, "n must be at least 3.")
    M = max_mode if M is None else check_posint(M, "M", False)
    errorif(M > max_mode, ValueError, f"M must not exceed (n - 1) // 2 = {max_mode}.")

    i, j = jnp.mgrid[0:n, 0:n]
    modes = jnp.arange(1, M + 1)
    phase = (2.0 * jnp.pi / n) * (i - j)[:, :, None] * modes[None, None, :]
    D = -(2.0 / n) * jnp.sum(modes[None, None, :] * jnp.sin(phase), axis=-1)
    W = jnp.diag(jnp.full(n, 2.0 * jnp.pi / n))
    return D, W


def finite_difference_diffmat(N, h, dtype=None):
    """Fourth-order diagonal-norm SBP finite-difference matrices.

    Fourth order in the interior (five-point central stencil), second order at
    the boundary, with the Carpenter-Nordstrom closures that make the pair
    satisfy SBP exactly.

    Parameters
    ----------
    N : int
        Number of uniformly spaced nodes. At least 8 -- the boundary closures
        occupy four rows at each end.
    h : float
        Node spacing.
    dtype : dtype, optional
        Floating dtype. Defaults to float64.

    Returns
    -------
    D, W : tuple of jax.Array, each shape (N, N)

    Raises
    ------
    ValueError
        If ``N < 8``.
    """
    N = check_posint(N, "N", False)
    errorif(
        N < 8,
        ValueError,
        f"N must be at least 8 for the fourth-order boundary closures, got {N}.",
    )
    dtype = jnp.float64 if dtype is None else dtype
    D = jnp.zeros((N, N), dtype)
    H = jnp.ones((N,), dtype)
    W = jnp.zeros((N, N), dtype)

    rows = jnp.arange(4, N - 4, dtype=jnp.int32)
    offsets = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.int32)
    stencil = jnp.array([1, -8, 0, 8, -1], dtype) / 12.0
    row_idx = jnp.repeat(rows, 5)
    col_idx = (rows[:, None] + offsets).reshape(-1)
    D = D.at[row_idx, col_idx].set(jnp.tile(stencil, rows.size))

    f0 = jnp.array([-24 / 17, 59 / 34, -4 / 17, -3 / 34], dtype)
    f1 = jnp.array([-1 / 2, 0.0, 1 / 2], dtype)
    f2 = jnp.array([4 / 43, -59 / 86, 0.0, 59 / 86, -4 / 43], dtype)
    f3 = jnp.array([3 / 98, 0.0, -59 / 98, 0.0, 32 / 49, -4 / 49], dtype)

    # Lower boundary rows follow by SBP antisymmetry: D[N-1-i, N-1-j] = -D[i, j].
    D = (
        D.at[0, :4]
        .set(f0)
        .at[1, :3]
        .set(f1)
        .at[2, :5]
        .set(f2)
        .at[3, :6]
        .set(f3)
        .at[-1, -4:]
        .set(-f0[::-1])
        .at[-2, -3:]
        .set(-f1[::-1])
        .at[-3, -5:]
        .set(-f2[::-1])
        .at[-4, -6:]
        .set(-f3[::-1])
    )

    edge = jnp.array([17 / 48, 59 / 48, 43 / 48, 49 / 48], dtype)
    for k in range(4):
        H = H.at[k].set(edge[k]).at[-(k + 1)].set(edge[k])

    W = W.at[jnp.diag_indices(N)].set(H * h)
    return D / h, W


def jacobi_diffmat(N, alpha=0.0, beta=1.0):
    """Differentiation matrix on left-Gauss-Radau-Jacobi nodes.

    The default ``(alpha, beta) = (0, 1)`` is the cylindrical radial weight. This
    is AGNI's **recommended radial basis for the coupled Zernike path**: with
    coupled Zernike-Fourier operators, the Jacobi radial basis is the trusted
    converged ground truth, while a uniform radial grid produces *spurious*
    modes. See ``docs/resolution.md``.

    Parameters
    ----------
    N : int
        Number of nodes, at least 2.
    alpha, beta : float
        Jacobi weight exponents, both greater than -1.

    Returns
    -------
    D, W : tuple of jax.Array, each shape (N, N)
    """
    nodes, weights = gauss_radau_jacobi(N, alpha, beta)
    D = _barycentric_diffmat(nodes)
    return D, jnp.diag(weights)


def _bspline_basis_and_deriv(x, knots, degree):
    """Evaluate a B-spline basis and its first derivative at ``x``."""
    x = jnp.atleast_1d(x)
    n_basis = knots.size - degree - 1
    left = knots[:-1]
    right = knots[1:]

    basis = (x[:, None] >= left[None, :]) & (x[:, None] < right[None, :])
    final_nonempty = (right == knots[-1]) & (left < right)
    basis = basis | ((x[:, None] == knots[-1]) & final_nonempty[None, :])
    basis = basis.astype(x.dtype)

    def safe_divide(num, den):
        safe_den = jnp.where(den == 0, 1, den)
        return jnp.where(den == 0, 0.0, num / safe_den)

    def elevate(current, current_degree):
        i = jnp.arange(current.shape[1] - 1)
        left_den = knots[i + current_degree] - knots[i]
        right_den = knots[i + current_degree + 1] - knots[i + 1]
        left_coef = safe_divide(x[:, None] - knots[i][None, :], left_den[None, :])
        right_coef = safe_divide(
            knots[i + current_degree + 1][None, :] - x[:, None], right_den[None, :]
        )
        return left_coef * current[:, :-1] + right_coef * current[:, 1:]

    lower = basis
    for current_degree in range(1, degree):
        lower = elevate(lower, current_degree)
    basis = elevate(lower, degree)

    i = jnp.arange(n_basis)
    left_den = knots[i + degree] - knots[i]
    right_den = knots[i + degree + 1] - knots[i + 1]
    derivative = degree * (
        safe_divide(lower[:, :-1], left_den[None, :])
        - safe_divide(lower[:, 1:], right_den[None, :])
    )
    return basis, derivative


def bspline_diffmat(N, degree=4):
    """B-spline collocation differentiation matrix and quadrature weights.

    Greville abscissae of a clamped-uniform knot vector give one node per basis
    function. The derivative matrix collocates the analytic B-spline derivative;
    the diagonal weights are the exact basis integrals.

    Parameters
    ----------
    N : int
        Number of basis functions and collocation nodes.
    degree : int
        Polynomial degree. ``N`` must be at least ``degree + 1``.

    Returns
    -------
    D, W : tuple of jax.Array, each shape (N, N)

    Notes
    -----
    This pair does **not** satisfy SBP -- collocation at Greville points is not
    a diagonal-norm SBP scheme. It is included for comparison studies, not as a
    production radial basis, and ``tests/test_diffmat.py`` asserts SBP only for
    the bases that have it.
    """
    nodes, weights = bspline_nodes_weights(N, degree)
    knots = bspline_clamped_uniform_knots(N, degree)
    basis, derivative = _bspline_basis_and_deriv(nodes, knots, degree)
    D = jnp.linalg.solve(basis.T, derivative.T).T
    return D, jnp.diag(weights)


@jax.tree_util.register_pytree_node_class
class DiffMat:
    """Differentiation and quadrature matrices for a tensor-product grid.

    Holds one ``(D, W)`` pair per coordinate. At least one pair is required. The
    matrices must have been built for the nodes they will be used on -- nothing
    here can check that, and a mismatch produces a wrong eigenvalue rather than
    an error.

    **This class is the seam for any equilibrium code.** It accepts matrices as
    plain arrays, so a consumer is free to build them with AGNI's bases, with
    its own, or with a third library.

    Parameters
    ----------
    D_rho, D_theta, D_zeta : array-like, optional
        Square first-derivative matrices for each coordinate.
    W_rho, W_theta, W_zeta : array-like, optional
        Matching quadrature weights: either a 1-D weight vector or a square 2-D
        diagonal matrix. Must be supplied with, and only with, the corresponding
        ``D``.
    zernike_penalty_alpha : float, optional
        Coupled Zernike-Fourier de-aliasing penalty strength. Default 0 (off).
        The converged production runs used
        ``DEFAULT_ZERNIKE_PENALTY_ALPHA = 0.05``. If positive and no projector
        is supplied, one is built from ``D_rho`` and ``D_theta``.
    zernike_penalty_svd_tol : float, optional
        Relative SVD cutoff for building the projector.
    zernike_penalty_projector : array-like, optional
        Precomputed projector onto unrepresented nodal content.
    zernike_penalty_rank : int, optional
        Represented rank, for reporting.

    Notes
    -----
    **Coupled mode.** When the coupled Zernike-Fourier path is used, ``D_rho``
    and ``D_theta`` are the full non-separable ``(n_rho * n_theta)`` square
    operators rather than per-direction ones, while ``W_rho`` and ``W_theta``
    stay 1-D per-direction weight vectors -- the assembly still tensors the
    three weight vectors together. So a 1-D ``W`` whose length does not match
    its ``D`` is legal and expected; a 2-D ``W`` must be square and match.

    **Static vs traced.** ``DiffMat`` is a JAX pytree whose *matrices* are
    dynamic leaves and whose *scalar knobs* are static. The knobs have to be
    static because they drive Python branches: ``zernike_penalty_alpha > 0``
    decides whether the penalty is applied at all. Left as ordinary leaves they
    become traced scalars under ``jit`` and that comparison raises
    ``TracerBoolConversionError``.
    """

    def __init__(
        self,
        *,
        D_rho=None,
        D_theta=None,
        D_zeta=None,
        W_rho=None,
        W_theta=None,
        W_zeta=None,
        zernike_penalty_alpha=0.0,
        zernike_penalty_svd_tol=1e-10,
        zernike_penalty_projector=None,
        zernike_penalty_rank=None,
    ):
        self.D_rho = None if D_rho is None else jnp.asarray(D_rho)
        self.D_theta = None if D_theta is None else jnp.asarray(D_theta)
        self.D_zeta = None if D_zeta is None else jnp.asarray(D_zeta)
        self.W_rho = None if W_rho is None else jnp.asarray(W_rho)
        self.W_theta = None if W_theta is None else jnp.asarray(W_theta)
        self.W_zeta = None if W_zeta is None else jnp.asarray(W_zeta)
        self.zernike_penalty_alpha = float(zernike_penalty_alpha)
        self.zernike_penalty_svd_tol = float(zernike_penalty_svd_tol)
        self.zernike_penalty_projector = (
            None
            if zernike_penalty_projector is None
            else jnp.asarray(zernike_penalty_projector)
        )
        self.zernike_penalty_rank = (
            None if zernike_penalty_rank is None else int(zernike_penalty_rank)
        )
        if self.zernike_penalty_alpha > 0.0 and self.zernike_penalty_projector is None:
            projector, rank = zernike_penalty_projector_from_diffmat(
                self.D_rho, self.D_theta, self.zernike_penalty_svd_tol
            )
            self.zernike_penalty_projector = projector
            self.zernike_penalty_rank = rank
        self._validate()

    def _validate(self):
        """Check shapes and pairing, and build the static structure token."""
        pairs = (
            ("rho", self.D_rho, self.W_rho),
            ("theta", self.D_theta, self.W_theta),
            ("zeta", self.D_zeta, self.W_zeta),
        )
        errorif(
            all(D is None and W is None for _, D, W in pairs),
            ValueError,
            "DiffMat requires at least one differentiation/quadrature pair.",
        )
        for coord, D, W in pairs:
            errorif(
                (D is None) != (W is None),
                ValueError,
                f"D_{coord} and W_{coord} must be provided together.",
            )
            if D is None:
                continue
            errorif(
                D.ndim != 2 or D.shape[0] != D.shape[1],
                ValueError,
                f"D_{coord} must be a square matrix, got shape {D.shape}.",
            )
            errorif(
                W.ndim == 2 and W.shape != D.shape,
                ValueError,
                f"a 2-D W_{coord} must be square and match D_{coord}.",
            )
            errorif(
                W.ndim not in (1, 2),
                ValueError,
                f"W_{coord} must be a 1-D weight vector or a 2-D matrix.",
            )
        if self.zernike_penalty_projector is not None:
            Q = self.zernike_penalty_projector
            errorif(
                Q.ndim != 2 or Q.shape[0] != Q.shape[1],
                ValueError,
                "zernike_penalty_projector must be a square matrix.",
            )
            errorif(
                self.D_rho is not None and Q.shape != self.D_rho.shape,
                ValueError,
                "zernike_penalty_projector must match D_rho's shape.",
            )

        # Only STRUCTURE, never values: equal-shaped matrices then share
        # compiled code instead of forcing a retrace on every new equilibrium.
        def _shape(x):
            return None if x is None else tuple(x.shape)

        self._token = (
            "DiffMat",
            _shape(self.D_rho),
            _shape(self.D_theta),
            _shape(self.D_zeta),
            _shape(self.W_rho),
            _shape(self.W_theta),
            _shape(self.W_zeta),
            _shape(self.zernike_penalty_projector),
            self.zernike_penalty_alpha,
            self.zernike_penalty_svd_tol,
        )

    # -- convenience constructors -----------------------------------------

    @classmethod
    def from_zeta_grid(cls, zeta):
        """Build a fourth-order SBP finite-difference pair for a uniform zeta grid.

        Parameters
        ----------
        zeta : array-like
            One-dimensional, uniformly spaced nodes. At least 8 are required by
            the boundary stencil.

        Returns
        -------
        DiffMat

        Raises
        ------
        ValueError
            If ``zeta`` is not 1-D, has fewer than 8 nodes, or is not uniform.
        """
        zeta = jnp.asarray(zeta)
        errorif(zeta.ndim != 1, ValueError, "zeta must be one-dimensional.")
        errorif(zeta.size < 8, ValueError, "At least 8 zeta nodes are required.")
        spacing = np.diff(np.asarray(zeta))
        errorif(
            not np.allclose(spacing, spacing[0]),
            ValueError,
            "zeta nodes must be uniformly spaced.",
        )
        D_zeta, W_zeta = finite_difference_diffmat(zeta.size, spacing[0])
        return cls(D_zeta=D_zeta, W_zeta=W_zeta)

    # -- serialization -----------------------------------------------------

    def save(self, path):
        """Write to a ``.npz`` file. Own format -- no external IO framework.

        Parameters
        ----------
        path : str or pathlib.Path

        Returns
        -------
        str
            The path written.
        """
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        out = {
            "zernike_penalty_alpha": np.asarray(self.zernike_penalty_alpha),
            "zernike_penalty_svd_tol": np.asarray(self.zernike_penalty_svd_tol),
        }
        if self.zernike_penalty_rank is not None:
            out["zernike_penalty_rank"] = np.asarray(self.zernike_penalty_rank)
        for key in (
            "D_rho",
            "D_theta",
            "D_zeta",
            "W_rho",
            "W_theta",
            "W_zeta",
            "zernike_penalty_projector",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = np.asarray(val)
        np.savez_compressed(path, **out)
        return path

    @classmethod
    def load(cls, path):
        """Read a file written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path

        Returns
        -------
        DiffMat
        """
        with np.load(str(path)) as f:
            kwargs = {k: f[k] for k in f.files if k.startswith(("D_", "W_"))}
            if "zernike_penalty_projector" in f.files:
                kwargs["zernike_penalty_projector"] = f["zernike_penalty_projector"]
            if "zernike_penalty_rank" in f.files:
                kwargs["zernike_penalty_rank"] = int(f["zernike_penalty_rank"])
            kwargs["zernike_penalty_alpha"] = float(f["zernike_penalty_alpha"])
            kwargs["zernike_penalty_svd_tol"] = float(f["zernike_penalty_svd_tol"])
            return cls(**kwargs)

    # -- pytree ------------------------------------------------------------

    def tree_flatten(self):
        """Matrices are dynamic leaves; the scalar knobs are static aux data."""
        keys = tuple(
            k
            for k in (
                "D_rho",
                "D_theta",
                "D_zeta",
                "W_rho",
                "W_theta",
                "W_zeta",
                "zernike_penalty_projector",
            )
            if getattr(self, k) is not None
        )
        leaves = tuple(getattr(self, k) for k in keys)
        aux = (
            keys,
            self.zernike_penalty_alpha,
            self.zernike_penalty_svd_tol,
            self.zernike_penalty_rank,
        )
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        """Rebuild without re-validating: leaves may be tracers."""
        keys, alpha, svd_tol, rank = aux
        obj = object.__new__(cls)
        for key in (
            "D_rho",
            "D_theta",
            "D_zeta",
            "W_rho",
            "W_theta",
            "W_zeta",
            "zernike_penalty_projector",
        ):
            setattr(obj, key, None)
        for key, leaf in zip(keys, leaves):
            setattr(obj, key, leaf)
        obj.zernike_penalty_alpha = alpha
        obj.zernike_penalty_svd_tol = svd_tol
        obj.zernike_penalty_rank = rank
        obj._token = ("DiffMat-unflattened", keys, alpha, svd_tol)
        return obj

    def __hash__(self):
        """Hash the static structure only."""
        return hash(self._token)

    def __eq__(self, other):
        """Compare static structure only."""
        return isinstance(other, DiffMat) and self._token == other._token

    def __repr__(self):
        """Short summary of which pairs are present and their shapes."""

        def _s(x):
            return "-" if x is None else "x".join(str(d) for d in x.shape)

        return (
            f"DiffMat(rho={_s(self.D_rho)}, theta={_s(self.D_theta)}, "
            f"zeta={_s(self.D_zeta)}, zpen={self.zernike_penalty_alpha:g})"
        )
