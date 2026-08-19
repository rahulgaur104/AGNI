"""Quadrature nodes, weights, and the radial clustering automorphisms.

Everything here is **construction-time** machinery: it produces the fixed node
sets and weights a :class:`~agnimhd.basis.diffmat.DiffMat` is built on. None of
it sits in the differentiated path -- the growth rate is differentiated with
respect to the *equilibrium arrays*, not the grid -- so these routines are free
to use host NumPy and SciPy where that is clearer or more accurate, and they
return concrete arrays.

The Legendre machinery uses ``numpy.polynomial.legendre``. DESC reaches the same
values through ``orthax``; that is an extra dependency AGNI does not take, and
for concrete construction-time nodes the two agree to roundoff. The test suite
pins AGNI's Gauss-Lobatto nodes against the values DESC produced when the
fixture was exported, so the agreement is measured rather than assumed.

References
----------
Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM.
Canuto et al. (2006). *Spectral Methods -- Fundamentals in Single Domains*.
"""

import numpy as np
from numpy.polynomial.legendre import legder, legval
from scipy.special import gammaln

from .backend import check_posint, errorif, jnp

__all__ = [
    "automorphism_staircase1",
    "automorphism_staircase2",
    "bspline_nodes_weights",
    "gauss_radau_jacobi",
    "leggauss_lob",
    "zernike_nodes_weights",
]


def leggauss_lob(deg, interior_only=False):
    """Lobatto-Gauss-Legendre quadrature on ``[-1, 1]``.

    Returns points ``x_k`` and weights ``w_k`` for
    ``integral_{-1}^{1} f(x) dx ~= sum_k w_k f(x_k)``. The rule includes both
    endpoints and is exact for polynomials of degree ``2 * deg - 3``.

    Parameters
    ----------
    deg : int
        Number of quadrature points, at least 2.
    interior_only : bool
        Exclude the points and weights at -1 and +1; useful when
        ``f(-1) = f(1) = 0``. ``deg`` points are still returned -- they are the
        interior points of the ``deg + 2`` point Lobatto rule.

    Returns
    -------
    x, w : tuple of jax.Array
        Nodes and weights, each of shape ``(deg,)``.

    Notes
    -----
    Interior nodes are the roots of ``P'_{N-1}``, found by the Golub-Welsch
    algorithm on the Jacobi matrix and then improved by a single Newton step.
    The single step is enough because Golub-Welsch already lands within the
    quadratic convergence basin.
    """
    N = deg + 2 * bool(interior_only)
    errorif(N < 2, ValueError, f"deg must be at least 2, got {deg}.")

    # Golub-Welsch on the symmetric tridiagonal Jacobi matrix.
    n = np.arange(2, N - 1, dtype=float)
    off = np.sqrt((n**2 - 1) / (4 * n**2 - 1))
    if off.size:
        M = np.diag(off, 1) + np.diag(off, -1)
        x = np.linalg.eigvalsh(M)
    else:
        x = np.zeros(0)

    c0 = np.zeros(N)
    c0[-1] = 1.0  # the Legendre polynomial P_{N-1}

    # One Newton step on P'_{N-1}, whose roots are the interior Lobatto nodes.
    c = legder(c0)
    dy = legval(x, c)
    df = legval(x, legder(c))
    x = x - dy / df

    w = 2 / (N * (N - 1) * legval(x, c0) ** 2)

    if not interior_only:
        x = np.hstack([-1.0, x, 1.0])
        w_end = 2 / (deg * (deg - 1))
        w = np.hstack([w_end, w, w_end])

    assert x.size == w.size == deg
    return jnp.asarray(x), jnp.asarray(w)


def automorphism_staircase1(x, x_0=0.5, m_1=2.0, m_2=2.0, eps=0.0):
    """Map ``[-1, 1] -> [eps, 1]``, clustering nodes around ``x_0``.

    Plotted, it looks like a staircase with a single step. Composed with
    Gauss-Lobatto nodes this is how AGNI concentrates radial resolution where
    the mode is sharp without giving up spectral accuracy: the differentiation
    matrix and weights are corrected by the map's Jacobian (chain rule), so the
    scheme stays exact on the mapped polynomials.

    Parameters
    ----------
    x : ndarray
        Points in ``[-1, 1]``.
    x_0 : float
        Point around which node density is concentrated.
    m_1, m_2 : float
        Density control to the left and right of ``x_0``.
    eps : float
        Lower bound of the transformed interval. The default ``0`` preserves the
        map to ``[0, 1]``. Set ``eps > 0`` to keep a node off the magnetic axis;
        the image is then ``[eps, 1]``.

    Returns
    -------
    y : ndarray
        Transformed points.

    Notes
    -----
    Differentiable in ``x`` under ``jax.grad``, which is how callers obtain the
    Jacobian for the chain-rule correction. See ``examples/`` for the pattern.
    """
    lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
    upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
    return eps + (1 - eps) * (lower + upper)


def automorphism_staircase2(x, x_0=0.0, x_1=0.5, m_1=1.0, m_2=1.0, m_3=10.0, m_4=10.0):
    """Map ``[-1, 1] -> [0, 1]`` with a three-step staircase profile.

    Like :func:`automorphism_staircase1` but with two clustering points and
    additional terms (``m_3``, ``m_4``) that keep the spacing more uniform away
    from the endpoints.

    Parameters
    ----------
    x : ndarray
        Points in ``[-1, 1]``.
    x_0, x_1 : float
        Points around which node density is concentrated.
    m_1, m_2 : float
        Density control around ``x_0`` and ``x_1``.
    m_3, m_4 : float
        Uniformity control away from the endpoints.

    Returns
    -------
    y : ndarray
        Transformed points.
    """
    a = 0.5 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
    b = 0.5 * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
    c = 0.5 * (jnp.tanh(m_3 * (x - x_0)) + jnp.tanh(m_3 * (1 + x_0)))
    d = 0.5 * (jnp.tanh(m_4 * (x - x_1)) + jnp.tanh(m_4 * (1 + x_1)))
    y = a + b + 0.0 * (c + d)
    return y


def _jacobi_diag_offdiag(N, alpha, beta):
    """Jacobi-matrix diagonals for the orthonormal Jacobi polynomials."""
    a0 = (beta - alpha) / (alpha + beta + 2.0)
    n = np.arange(1, N, dtype=float)
    two_n_ab = 2.0 * n + alpha + beta
    a_rest = (beta**2 - alpha**2) / (two_n_ab * (two_n_ab + 2.0))
    a = np.concatenate([np.array([a0]), a_rest])

    # The usual expression has a removable 0/0 at n=1 when alpha + beta == -1.
    at_n1 = n == 1
    ratio = np.where(at_n1, 1.0, (n + alpha + beta) / (two_n_ab - 1.0))
    b_squared = (
        4.0 * n * (n + alpha) * (n + beta) * ratio / (two_n_ab**2 * (two_n_ab + 1.0))
    )
    return a, np.sqrt(b_squared)


def _gauss_jacobi(N, alpha, beta):
    """``N``-point Gauss-Jacobi nodes and weights on ``[-1, 1]``.

    Exact for polynomials through degree ``2N - 1`` against the weight
    ``(1 - x)**alpha * (1 + x)**beta``.
    """
    N = check_posint(N, "N", False)
    errorif(alpha <= -1 or beta <= -1, ValueError, "alpha and beta must exceed -1.")

    diagonal, off_diagonal = _jacobi_diag_offdiag(N, alpha, beta)
    matrix = (
        np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)
    )
    nodes, eigenvectors = np.linalg.eigh(matrix)

    log_mu0 = (
        (alpha + beta + 1.0) * np.log(2.0)
        + gammaln(alpha + 1.0)
        + gammaln(beta + 1.0)
        - gammaln(alpha + beta + 2.0)
    )
    weights = np.exp(log_mu0) * eigenvectors[0] ** 2
    return nodes, weights


def gauss_radau_jacobi(N, alpha=0.0, beta=1.0):
    """Left-Gauss-Radau-Jacobi quadrature on ``[-1, 1]``.

    The left endpoint is fixed at ``x[0] = -1``; the rule is exact for
    polynomials through degree ``2N - 2`` against
    ``(1 - x)**alpha * (1 + x)**beta``.

    Parameters
    ----------
    N : int
        Number of nodes, at least 2.
    alpha, beta : float
        Jacobi weight exponents, both greater than -1. The default ``(0, 1)``
        is the cylindrical radial weight after shifting to ``[0, 1]``.

    Returns
    -------
    x, w : tuple of jax.Array
        Ascending nodes and positive weights, each of shape ``(N,)``.
    """
    N = check_posint(N, "N", False)
    errorif(N < 2, ValueError, "N must be at least 2.")
    errorif(alpha <= -1 or beta <= -1, ValueError, "alpha and beta must exceed -1.")

    interior, _ = _gauss_jacobi(N - 1, alpha, beta + 1.0)
    nodes = np.concatenate([np.array([-1.0]), interior])

    # Integrate the nodal Lagrange basis with an auxiliary Gauss-Jacobi rule.
    aux_nodes, aux_weights = _gauss_jacobi(2 * N + 5, alpha, beta)
    aux_diff = aux_nodes[:, None] - nodes[None, :]
    full_product = np.prod(aux_diff, axis=1)
    denominator_matrix = nodes[:, None] - nodes[None, :] + np.eye(N)
    denominator = np.prod(denominator_matrix, axis=1)
    lagrange = full_product[:, None] / aux_diff / denominator[None, :]
    weights = np.sum(aux_weights[:, None] * lagrange, axis=0)
    return jnp.asarray(nodes), jnp.asarray(weights)


def zernike_nodes_weights(n_rho, n_theta):
    """Radial and poloidal nodes and weights for the unit disc.

    Radial nodes are shifted Gauss-Jacobi ``(alpha=0, beta=1)`` nodes strictly
    inside ``(0, 1)`` -- which is what keeps the magnetic axis off the grid.
    Poloidal nodes are equally spaced on ``[0, 2*pi)``.

    Parameters
    ----------
    n_rho, n_theta : int
        Number of radial and poloidal nodes.

    Returns
    -------
    rho, w_rho, theta, w_theta : tuple of jax.Array
        One-dimensional node and weight arrays for the two coordinates.
    """
    n_rho = check_posint(n_rho, "n_rho", False)
    n_theta = check_posint(n_theta, "n_theta", False)
    x, w = _gauss_jacobi(n_rho, 0.0, 1.0)
    rho = (1.0 + x) / 2.0
    w_rho = w / (4.0 * rho)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    w_theta = np.full(n_theta, 2.0 * np.pi / n_theta)
    return (
        jnp.asarray(rho),
        jnp.asarray(w_rho),
        jnp.asarray(theta),
        jnp.asarray(w_theta),
    )


def bspline_clamped_uniform_knots(N, degree):
    """Clamped-uniform B-spline knot vector on ``[-1, 1]``.

    Parameters
    ----------
    N : int
        Number of basis functions.
    degree : int
        Polynomial degree. ``N`` must be at least ``degree + 1``.

    Returns
    -------
    jax.Array, shape (N + degree + 1,)
    """
    N = check_posint(N, "N", False)
    degree = check_posint(degree, "degree", False)
    errorif(N < degree + 1, ValueError, "N must be at least degree + 1.")
    interior = np.linspace(-1.0, 1.0, N - degree + 1)[1:-1]
    return jnp.asarray(
        np.concatenate([np.full(degree + 1, -1.0), interior, np.full(degree + 1, 1.0)])
    )


def bspline_nodes_weights(N, degree=4):
    """Greville abscissae and exact B-spline integration weights on ``[-1, 1]``.

    Parameters
    ----------
    N : int
        Number of basis functions and collocation nodes.
    degree : int
        Polynomial degree. ``N`` must be at least ``degree + 1``.

    Returns
    -------
    x, w : tuple of jax.Array
        Greville abscissae and exact basis-integral weights, shape ``(N,)``.
    """
    knots = np.asarray(bspline_clamped_uniform_knots(N, degree))
    i = np.arange(N)
    indices = i[:, None] + np.arange(1, degree + 1)[None, :]
    nodes = np.mean(knots[indices], axis=1)
    weights = (knots[i + degree + 1] - knots[i]) / (degree + 1)
    return jnp.asarray(nodes), jnp.asarray(weights)
