"""AGNI's copy of DESC's boundary-integral machinery (``agnimhd.adapters._vacuum``).

Ported from the ``TestSingularities`` cases of the DESC branch ``ma/AGNI_ext``
(``tests/test_integrals.py``) that cover the copied code: the FFT interpolator and
the double-layer kernel through ``singular_integral``; plus the least-squares solve.
Needs DESC (master is enough); skipped where it is absent. DESC is reached through
``pytest.importorskip`` only: the test suite may not ``import desc`` (see
``tools/check_no_desc.sh``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

dgrid = pytest.importorskip("desc.grid")
Grid, LinearGrid = dgrid.Grid, dgrid.LinearGrid
dsing = pytest.importorskip("desc.integrals.singularities")

from agnimhd.adapters._vacuum.laplace import _lstsq_qr  # noqa: E402
from agnimhd.adapters._vacuum.singularities import (  # noqa: E402
    FFTInterpolator,
    singular_integral,
)


def _c_2d(x, y):
    """Non-separable test function (DESC's ``tests/test_interp_utils.py``)."""
    return (
        jnp.cos(3 * x) * jnp.sin(2 * x + y)
        + jnp.sin(5 * y) * jnp.cos(x + 3 * y)
        - 33.2
        + jnp.cos(x)
        + jnp.cos(y)
    )


def _polar_shifts(source, st, sz, q):
    """The (theta, zeta) offsets of the q x q polar nodes around each eval node."""
    h_t = 2 * np.pi / source.num_theta
    h_z = 2 * np.pi / source.num_zeta / source.NFP
    r, w, _, _ = dsing._get_quadrature_nodes(q)
    return st / 2 * h_t * r * np.sin(w), sz / 2 * h_z * r * np.cos(w)


@pytest.mark.parametrize(
    "settings",
    [
        (_c_2d, 5, 8, 1),  # the Nyquist frequencies of _c_2d
        (lambda x, y: np.cos(6 * x) ** 2 + np.sin(90 * y) + 1, 12, 3, 30),
    ],
)
def test_fft_interpolator_reproduces_a_resolved_function(settings):
    """At every polar node, the interpolant equals the function itself, with the
    eval grid about half the source grid: the source's full spectrum is kept. (DESC
    master's ``FFTInterpolator`` truncates it to the eval grid's and fails this.)"""
    f, M, N, NFP = settings
    s_grid = LinearGrid(M=M, N=N, NFP=NFP)
    st, sz, q = 3, 5, 4
    dt, dz = _polar_shifts(s_grid, st, sz, q)
    e_grid = LinearGrid(
        theta=s_grid.num_theta // 2 + 1, zeta=s_grid.num_zeta // 2 + 1, NFP=NFP
    )
    interp = FFTInterpolator(e_grid, s_grid, st, sz, q, warn_fft=False)
    theta, zeta = e_grid.nodes[:, 1], e_grid.nodes[:, 2]
    _f = f(s_grid.nodes[:, 1], s_grid.nodes[:, 2])
    for i in range(dt.size):
        np.testing.assert_allclose(interp(_f, i), f(theta + dt[i], zeta + dz[i]))


@pytest.mark.parametrize("source", [(9, 7), (18, 14)])
def test_interpolated_phi_basis_is_the_basis(source):
    """The Phi basis interpolated to the polar nodes equals the basis evaluated there,
    to rounding: why ``singular_integral`` needs no ``known_map`` for it. The basis
    is what ``phi_scaffold`` builds, capped at the eval grid's resolution, so the
    source grid resolves every column."""
    DoubleFourierSeries = pytest.importorskip("desc.basis").DoubleFourierSeries

    nfp = 4
    e_grid = LinearGrid(rho=1.0, theta=9, zeta=7, NFP=nfp, sym=False)
    s_grid = LinearGrid(rho=1.0, theta=source[0], zeta=source[1], NFP=nfp, sym=False)
    basis = DoubleFourierSeries(M=e_grid.M, N=e_grid.N, NFP=nfp)
    st, sz, q = 5, 3, 3
    interp = FFTInterpolator(e_grid, s_grid, st, sz, q, warn_fft=False)
    V = interp.fourier(basis.evaluate(s_grid))
    dt, dz = _polar_shifts(s_grid, st, sz, q)
    for i in range(dt.size):
        at = np.asarray(e_grid.nodes).copy()
        at[:, 1] += dt[i]
        at[:, 2] += dz[i]
        direct = basis.evaluate(Grid(at, NFP=nfp, sort=False))
        np.testing.assert_allclose(
            interp(V, i, is_fourier=True), direct, rtol=0, atol=1e-12
        )


def test_fft_interpolator_refuses_a_different_surface():
    """Grids on two different surfaces are refused (master's base-class check)."""
    source = LinearGrid(rho=1.0, theta=24, zeta=16, NFP=3, sym=False)
    other = LinearGrid(rho=0.5, theta=24, zeta=16, NFP=3, sym=False)
    with pytest.raises(AssertionError):
        FFTInterpolator(other, source, 12, 7, 9)


@pytest.mark.slow
def test_double_layer_of_a_constant_is_minus_one_half():
    """Green's identity: on the surface, the double layer of Phi = 1 is -1/2.

    DESC's ``test_singular_integral_greens_id`` checks ``int n.(x-y)/|x-y|^3 da =
    -2 pi`` with its ``nr_over_r3`` kernel; ``_kernel_dipole`` is that kernel over
    ``4 pi`` times Phi, so the same cases and errors (over ``4 pi``) apply. The errors
    are Dhairya Malhotra's convergence table, quoted in the DESC test.
    """
    Equilibrium = pytest.importorskip("desc.equilibrium").Equilibrium
    eq = Equilibrium()  # circular-cross-section torus
    keys = ["R", "phi", "Z", "e_theta", "e_zeta"]  # all on DESC master
    for Nu, Nv, s, q, err in [
        (13, 30, 13, 10, 0.4),
        (13, 45, 13, 12, 2e-2),
        (13, 60, 13, 12, 3e-3),
        (19, 90, 19, 16, 5e-5),
    ]:
        grid = LinearGrid(M=Nu // 2, N=Nv // 2, NFP=eq.NFP)
        data = dict(eq.compute(keys, grid=grid))
        data["e_theta x e_zeta"] = jnp.cross(data["e_theta"], data["e_zeta"])
        data["|e_theta x e_zeta|"] = jnp.linalg.norm(data["e_theta x e_zeta"], axis=-1)
        data["Phi (periodic)"] = jnp.ones(grid.num_nodes)
        interp = FFTInterpolator(grid, grid, s, s, q)
        D1 = singular_integral(data, data, interp, kernel="dipole", chunk_size=50)
        np.testing.assert_array_less(np.abs(0.5 + np.ravel(D1)), err / (4 * np.pi))


def _random_full_rank(rng, shape, complex_=False):
    """A random matrix of full rank, complex if asked."""
    A = rng.standard_normal(shape)
    if complex_:
        A = A + 1j * rng.standard_normal(shape)
    return A


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("shape", [(48, 63), (63, 48), (40, 40)])
def test_lstsq_qr_matches_lstsq(shape, complex_):
    """``_lstsq_qr`` is ``lstsq``: the least-squares solution for a tall matrix, the
    minimum-norm one for a wide matrix (the even-shell vacuum solve, 48 x 63 for an
    8 x 6 shell), the solution for a square one; real and complex."""
    rng = np.random.default_rng(0)
    D = _random_full_rank(rng, shape, complex_)
    B = _random_full_rank(rng, (shape[0], 5), complex_)
    x = np.asarray(_lstsq_qr(jnp.asarray(D), jnp.asarray(B)))
    ref = np.linalg.lstsq(D, B, rcond=None)[0]
    np.testing.assert_allclose(x, ref, rtol=0, atol=1e-12 * np.abs(ref).max())
    if shape[0] < shape[1]:
        # minimum norm: solves D x = B exactly, and lies in the row space of D
        np.testing.assert_allclose(D @ x, B, rtol=0, atol=1e-12 * np.abs(B).max())
        y = np.linalg.lstsq(D.conj().T, x, rcond=None)[0]
        np.testing.assert_allclose(
            D.conj().T @ y, x, rtol=0, atol=1e-12 * np.abs(x).max()
        )


@pytest.mark.parametrize("degenerate", [False, True])
@pytest.mark.parametrize("shape", [(48, 63), (63, 48)])
def test_lstsq_qr_derivative_matches_finite_differences(shape, degenerate):
    """The derivative of ``_lstsq_qr`` with respect to the matrix and the right-hand
    side matches central differences, also when every singular value of ``D`` is the
    same -- where an SVD-based derivative divides by their differences, and QR's is
    unaffected. (Measured with JAX 0.6.2: ``jnp.linalg.lstsq``'s JVP is all NaN on
    both degenerate cases here.)"""
    rng = np.random.default_rng(1)
    D = rng.standard_normal(shape)
    if degenerate:  # orthonormal columns or rows, times 2: all singular values 2
        U, _, Vh = np.linalg.svd(D, full_matrices=False)
        D = 2.0 * U @ Vh
    B = rng.standard_normal((shape[0], 3))
    dD, dB = rng.standard_normal(D.shape), rng.standard_normal(B.shape)

    def f(t):
        return _lstsq_qr(jnp.asarray(D) + t * dD, jnp.asarray(B) + t * dB)

    _, jvp = jax.jvp(f, (0.0,), (1.0,))
    h = 1e-6
    fd = (np.asarray(f(h)) - np.asarray(f(-h))) / (2 * h)
    jvp = np.asarray(jvp)
    assert np.isfinite(jvp).all()
    np.testing.assert_allclose(jvp, fd, rtol=0, atol=1e-7 * np.abs(fd).max())
