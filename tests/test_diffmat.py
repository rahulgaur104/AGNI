"""Differentiation and quadrature matrices.

The tensor-product mixed-derivative test, the finite-difference convergence
test and the summation-by-parts test are ported from ``tests/test_diffmatrices.py``
in PlasmaControl/DESC#1789, the differentiation-matrix pull request this package
extracts. They are kept as close to the originals as the DESC-free API allows.

Deliberately **not** ported: the ``teardown_module`` convergence-plotting block.
It needs ``matplotlib``, which is not a dependency of this package; it is gated
behind ``PLOT_CONVERGENCE`` so it never ran in CI; and being a plot it asserts
nothing and contributes no coverage. The convergence data it plotted is what the
parametrized resolutions in ``test_finite_difference_convergence`` assert
directly.
"""

import numpy as np
import pytest

from agnimhd.backend import jax, jnp
from agnimhd.basis import (
    DiffMat,
    bspline_diffmat,
    finite_difference_diffmat,
    fourier_diffmat,
    fourier_diffmat_truncated,
    fourier_pts,
    jacobi_diffmat,
    legendre_diffmat,
)
from agnimhd.quadrature import automorphism_staircase1, leggauss_lob

NFP = 5


# ---------------------------------------------------------------------------
# Ported from DESC#1789
# ---------------------------------------------------------------------------


def _eval_3D(f, x, y, z, nfp):
    """Evaluate ``f`` on the tensor grid, in (z, y, x) order."""
    return jax.vmap(
        lambda z_val: jax.vmap(
            lambda y_val: jax.vmap(lambda x_val: f(x_val, y_val, z_val, nfp))(x)
        )(y)
    )(z)


def _test_function(x, y, z, nfp):
    """Smooth, and effectively periodic in y and z."""
    return (
        jnp.exp(-100 * ((x - 0.8) ** 2))
        * jnp.sin(3 * x * 2 * jnp.pi)
        * (jnp.sin(4 * y) + jnp.cos(3 * y))
        * jnp.cos(nfp * z)
    )


_dx_f = jax.grad(_test_function, argnums=0)
_dy_f = jax.grad(_test_function, argnums=1)
_dz_f = jax.grad(_test_function, argnums=2)
_dxx_f = jax.grad(_dx_f, argnums=0)
_dxy_f = jax.grad(_dx_f, argnums=1)
_dxz_f = jax.grad(_dx_f, argnums=2)
_dyy_f = jax.grad(_dy_f, argnums=1)
_dyz_f = jax.grad(_dy_f, argnums=2)
_dzz_f = jax.grad(_dz_f, argnums=2)


def _tensor_product_derivative_3D(nx, ny, nz, dx_order, dy_order, dz_order, nfp):
    """Tensor-product differentiation matrix in 3D, with the staircase radial map.

    Legendre-Lobatto in x pushed through ``automorphism_staircase1``, Fourier in
    y and z. The first and second derivative chain-rule factors come from
    ``jax.grad`` of the automorphism, which is the same route production code
    uses to build a clustered radial operator.
    """
    x_cheb, _ = leggauss_lob(nx)
    y = fourier_pts(ny)
    z = fourier_pts(nz) / nfp

    auto_kw = dict(x_0=0.8, m_1=2.0, m_2=3.0)
    x = automorphism_staircase1(x_cheb, **auto_kw)
    d1 = jax.vmap(lambda v: jax.grad(automorphism_staircase1, argnums=0)(v, **auto_kw))(
        x_cheb
    )[:, None]
    d2 = jax.vmap(
        lambda v: jax.grad(jax.grad(automorphism_staircase1, argnums=0), argnums=0)(
            v, **auto_kw
        )
    )(x_cheb)[:, None]
    scale_x1 = 1 / d1
    scale_x2 = d2 / d1

    if dx_order == 0:
        Dx = jnp.eye(nx)
    elif dx_order == 1:
        D, _ = legendre_diffmat(nx)
        Dx = D * scale_x1
    else:
        D, _ = legendre_diffmat(nx)
        Dx = (D @ D - D * scale_x2) * scale_x1**2

    if dy_order == 0:
        Dy = jnp.eye(ny)
    elif dy_order == 1:
        Dy, _ = fourier_diffmat(ny)
    else:
        D, _ = fourier_diffmat(ny)
        Dy = D @ D

    if dz_order == 0:
        Dz = jnp.eye(nz)
    elif dz_order == 1:
        D, _ = fourier_diffmat(nz)
        Dz = D * nfp
    else:
        D, _ = fourier_diffmat(nz)
        Dz = (D @ D) * nfp**2

    if dx_order > 0 and dy_order > 0 and dz_order > 0:
        D = jnp.kron(Dz, jnp.kron(Dy, Dx))
    elif dx_order > 0 and dy_order > 0:
        D = jnp.kron(Dx, Dy)
    elif dx_order > 0 and dz_order > 0:
        D = jnp.kron(Dx, Dz)
    elif dy_order > 0 and dz_order > 0:
        D = jnp.kron(Dz, Dy)
    elif dx_order > 0:
        D = Dx
    elif dy_order > 0:
        D = Dy
    elif dz_order > 0:
        D = Dz
    else:
        D = jnp.kron(jnp.eye(nz), jnp.kron(jnp.eye(ny), jnp.eye(nx)))

    return jnp.where(jnp.abs(D) < 1e-12, 0.0, D), x, y, z


_TENSOR_CASES = [
    (1, 0, 0, _dx_f, 4e-3),
    (2, 0, 0, _dxx_f, 6e-3),
    (0, 1, 0, _dy_f, 1e-7),
    (0, 2, 0, _dyy_f, 1e-5),
    (0, 0, 1, _dz_f, 1e-7),
    (0, 0, 2, _dzz_f, 1e-5),
    (1, 1, 0, _dxy_f, 2e-3),
    (0, 1, 1, _dyz_f, 2e-3),
    (1, 0, 1, _dxz_f, 2e-3),
]


@pytest.mark.regression
@pytest.mark.parametrize("dx_order,dy_order,dz_order,analytic_fn,tol", _TENSOR_CASES)
def test_tensor_mixed_derivative(dx_order, dy_order, dz_order, analytic_fn, tol):
    """3D tensor-product derivatives match JAX's analytic derivatives.

    Covers pure and mixed first and second derivatives in every direction.
    """
    n = 48
    D, x, y, z = _tensor_product_derivative_3D(
        n, n, n, dx_order, dy_order, dz_order, NFP
    )
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    f0 = _test_function(X, Y, Z, NFP)

    if dx_order > 0 and dy_order > 0 and dz_order > 0:
        pytest.skip("full 3D mixed case builds an n^3 x n^3 operator; not run at n=48")
    elif dx_order > 0 and dy_order > 0:
        df = jnp.reshape(D @ jnp.reshape(f0, (n * n, n)), (n, n, n))
    elif dx_order > 0 and dz_order > 0:
        f = jnp.transpose(f0, (0, 2, 1))
        df = jnp.reshape(D @ jnp.reshape(f, (n * n, n)), (n, n, n))
        df = jnp.transpose(df, (0, 2, 1))
    elif dy_order > 0 and dz_order > 0:
        f = jnp.transpose(f0, (2, 1, 0))
        df = jnp.reshape(D @ jnp.reshape(f, (n * n, n)), (n, n, n))
        df = jnp.transpose(df, (2, 1, 0))
    elif dx_order > 0:
        df = jnp.reshape(D @ jnp.reshape(f0, (n, n * n)), (n, n, n))
    elif dy_order > 0:
        f = jnp.transpose(f0, (1, 0, 2))
        df = jnp.reshape(D @ jnp.reshape(f, (n, n * n)), (n, n, n))
        df = jnp.transpose(df, (1, 0, 2))
    else:
        f = jnp.transpose(f0, (2, 0, 1))
        df = jnp.reshape(D @ jnp.reshape(f, (n, n * n)), (n, n, n))
        df = jnp.transpose(df, (1, 2, 0))

    exact = _eval_3D(analytic_fn, x, y, z, NFP).transpose(2, 1, 0)
    error = float(jnp.max(jnp.abs(df - exact)))
    assert error < tol, (
        f"dx={dx_order}, dy={dy_order}, dz={dz_order}: "
        f"error {error:.2e} exceeds tol {tol}"
    )


@pytest.mark.parametrize(
    "N, alpha, x0, tol",
    [(48, 100.0, 0.7, 8.0e-2), (96, 100.0, 0.7, 9.0e-3), (192, 100.0, 0.7, 9.0e-4)],
)
def test_finite_difference_convergence(N, alpha, x0, tol):
    """The 4th-order FD matrix converges on an oscillating Gaussian.

    The three resolutions together are the convergence measurement: the error
    falls by roughly 10x per doubling, which is the expected fourth-order rate
    once the second-order boundary closures are included in the max norm.
    """
    a, b = 0.0, 1.0
    x = jnp.linspace(a, b, N)
    h = (b - a) / (N - 1)
    D, _ = finite_difference_diffmat(N, h)

    def f_scalar(v):
        return jnp.exp(-alpha * (v - x0) ** 2) * jnp.cos(4 * jnp.pi * (v - 0.5))

    f = f_scalar(x)
    df_true = jax.vmap(jax.grad(f_scalar))(x)
    err = float(jnp.max(jnp.abs(D @ f - df_true)))
    assert err < tol, f"max|err|={err:.2e} (N={N}, alpha={alpha})"


def test_summation_by_parts_nonperiodic():
    """SBP for the non-periodic bases: ``W D + (W D)^T == diag(-1, 0..0, 1)``.

    SBP is the discrete analogue of integration by parts, and it is what makes
    the discretized energy functional match the continuous one. A basis that
    quietly loses it yields a plausible, wrong eigenvalue -- so this is checked
    directly rather than inferred from convergence.
    """
    a, b = 0.0, 1.0
    N = 100
    h = (b - a) / (N - 1)

    D0, W0 = finite_difference_diffmat(N, h)
    D1, W1 = legendre_diffmat(N)

    B = jnp.zeros_like(D0).at[0, 0].set(-1).at[N - 1, N - 1].set(1)

    np.testing.assert_allclose(W0 @ D0 + (W0 @ D0).T, B, atol=1e-15)
    np.testing.assert_allclose(W1 @ D1 + (W1 @ D1).T, B, atol=5e-13)


# ---------------------------------------------------------------------------
# Added coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [8, 9, 16, 17])
def test_summation_by_parts_periodic(n):
    """On a periodic grid there is no boundary term, so ``W D`` is skew.

    Checked for both even and odd ``n``, which take different branches of the
    Fourier formula (``tan`` vs ``sin``).
    """
    D, W = fourier_diffmat(n)
    np.testing.assert_allclose(W @ D + (W @ D).T, jnp.zeros((n, n)), atol=1e-13)


@pytest.mark.parametrize("n", [8, 9, 16])
def test_fourier_diffmat_exact_on_representable_modes(n):
    """The Fourier matrix differentiates every sub-Nyquist mode exactly."""
    D, _ = fourier_diffmat(n)
    x = fourier_pts(n)
    for k in range(1, (n - 1) // 2 + 1):
        np.testing.assert_allclose(D @ jnp.sin(k * x), k * jnp.cos(k * x), atol=1e-11)
        np.testing.assert_allclose(D @ jnp.cos(k * x), -k * jnp.sin(k * x), atol=1e-11)


@pytest.mark.parametrize("N", [6, 10, 15])
def test_legendre_diffmat_exact_on_polynomials(N):
    """LGL differentiation is exact for polynomials of degree < N."""
    x, _ = leggauss_lob(N)
    D, _ = legendre_diffmat(N)
    for p in range(N):
        np.testing.assert_allclose(
            D @ x**p, p * x ** max(p - 1, 0) if p > 0 else jnp.zeros_like(x), atol=1e-9
        )


@pytest.mark.parametrize("N", [6, 10, 15])
def test_jacobi_diffmat_exact_on_polynomials(N):
    """Gauss-Radau-Jacobi differentiation is exact for polynomials of degree < N."""
    from agnimhd.quadrature import gauss_radau_jacobi

    x, _ = gauss_radau_jacobi(N)
    D, _ = jacobi_diffmat(N)
    for p in range(N):
        expect = p * x ** max(p - 1, 0) if p > 0 else jnp.zeros_like(x)
        np.testing.assert_allclose(D @ x**p, expect, atol=1e-8)


def test_all_diffmats_annihilate_constants():
    """Every basis sends a constant to zero -- a constant has no derivative.

    Any leak here shows up directly in the eigenvalue, because a rigid
    displacement is in the null space of the energy functional.
    """
    mats = {
        "legendre": legendre_diffmat(12)[0],
        "fourier": fourier_diffmat(10)[0],
        "fourier_truncated": fourier_diffmat_truncated(11, 3)[0],
        "finite_difference": finite_difference_diffmat(20, 0.05)[0],
        "jacobi": jacobi_diffmat(12)[0],
        "bspline": bspline_diffmat(12, 4)[0],
    }
    for name, D in mats.items():
        ones = jnp.ones(D.shape[0])
        residual = float(jnp.max(jnp.abs(D @ ones)))
        assert residual < 1e-9, f"{name} does not annihilate constants: {residual:.2e}"


def test_fourier_truncated_reduces_to_full():
    """Omitting M keeps every resolvable mode and reproduces fourier_diffmat."""
    for n in (7, 8, 11):
        D_full, W_full = fourier_diffmat(n)
        D_tr, W_tr = fourier_diffmat_truncated(n)
        if n % 2 == 1:
            np.testing.assert_allclose(D_tr, D_full, atol=1e-12)
        np.testing.assert_allclose(W_tr, W_full, atol=1e-15)


def test_fourier_truncated_kills_high_modes():
    """Modes above M are mapped to zero, which is the point of truncation."""
    n, M = 16, 3
    D, _ = fourier_diffmat_truncated(n, M)
    x = fourier_pts(n)
    np.testing.assert_allclose(D @ jnp.sin(1 * x), jnp.cos(x), atol=1e-11)
    assert float(jnp.max(jnp.abs(D @ jnp.sin(5 * x)))) < 1e-10


def test_fourier_truncated_rejects_over_nyquist_toroidal_cap():
    """NTOR must not exceed the Fourier Nyquist content of the zeta grid."""
    with pytest.raises(ValueError, match=r"M must not exceed"):
        fourier_diffmat_truncated(8, 4)


def test_quadrature_weights_integrate_constants():
    """Each W integrates a constant to the length of its interval."""
    _, W = legendre_diffmat(12)
    assert abs(float(jnp.sum(jnp.diag(W))) - 2.0) < 1e-12
    _, W = fourier_diffmat(10)
    assert abs(float(jnp.sum(jnp.diag(W))) - 2 * np.pi) < 1e-12
    _, W = finite_difference_diffmat(20, 1.0 / 19)
    assert abs(float(jnp.sum(jnp.diag(W))) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# DiffMat container
# ---------------------------------------------------------------------------


def _simple_diffmat():
    """A minimal valid three-coordinate DiffMat."""
    Dr, Wr = legendre_diffmat(5)
    Dt, Wt = fourier_diffmat(4)
    Dz, Wz = fourier_diffmat(4)
    return DiffMat(
        D_rho=Dr,
        W_rho=jnp.diagonal(Wr),
        D_theta=Dt,
        W_theta=jnp.diagonal(Wt),
        D_zeta=Dz,
        W_zeta=jnp.diagonal(Wz),
    )


def test_diffmat_requires_at_least_one_pair():
    """An empty DiffMat is refused rather than silently doing nothing."""
    with pytest.raises(ValueError, match="at least one"):
        DiffMat()


def test_diffmat_requires_paired_D_and_W():
    """A D without its W is refused: they are only meaningful together."""
    D, _ = legendre_diffmat(5)
    with pytest.raises(ValueError, match="must be provided together"):
        DiffMat(D_rho=D)


def test_diffmat_rejects_nonsquare_D():
    """D must be square."""
    with pytest.raises(ValueError, match="must be a square matrix"):
        DiffMat(D_rho=jnp.ones((3, 4)), W_rho=jnp.ones(3))


def test_diffmat_allows_1d_W_shorter_than_D():
    """In coupled mode D is the full 2D operator while W stays per-direction.

    A 1-D W whose length does not match D is therefore legal, and must not be
    rejected -- the assembly tensors the three weight vectors separately.
    """
    dm = DiffMat(D_rho=jnp.eye(12), W_rho=jnp.ones(4))
    assert dm.D_rho.shape == (12, 12)
    assert dm.W_rho.shape == (4,)


def test_diffmat_hash_depends_on_structure_not_values():
    """Two DiffMats of equal shape share a hash, so jit does not retrace.

    This is what keeps the compiled objective alive across optimizer steps: the
    matrices are values, and only their shapes are part of the signature.
    """
    a = _simple_diffmat()
    Dr, Wr = legendre_diffmat(5)
    Dt, Wt = fourier_diffmat(4)
    b = DiffMat(
        D_rho=Dr * 2.0,
        W_rho=jnp.diagonal(Wr),
        D_theta=Dt,
        W_theta=jnp.diagonal(Wt),
        D_zeta=Dt,
        W_zeta=jnp.diagonal(Wt),
    )
    assert hash(a) == hash(b)
    assert a == b


def test_diffmat_hash_differs_on_penalty_setting():
    """The penalty strength drives a Python branch, so it IS part of the key."""
    Dr, _ = legendre_diffmat(6)
    Dt, _ = fourier_diffmat(6)
    kw = dict(D_rho=Dr, W_rho=jnp.ones(6), D_theta=Dt, W_theta=jnp.ones(6))
    plain = DiffMat(**kw)
    penalized = DiffMat(**kw, zernike_penalty_alpha=0.05)
    assert hash(plain) != hash(penalized)
    assert plain != penalized


def test_diffmat_penalty_projector_is_built_when_alpha_positive():
    """A positive alpha with no supplied projector builds one from D_rho/D_theta."""
    from agnimhd.basis import zernike_fourier_diffmat
    from agnimhd.quadrature import zernike_nodes_weights

    rho, w_rho, theta, w_theta = zernike_nodes_weights(4, 6)
    D_rho, D_theta = zernike_fourier_diffmat(rho, theta)
    dm = DiffMat(
        D_rho=D_rho,
        W_rho=w_rho,
        D_theta=D_theta,
        W_theta=w_theta,
        zernike_penalty_alpha=0.05,
    )
    assert dm.zernike_penalty_projector is not None
    assert dm.zernike_penalty_projector.shape == (24, 24)
    assert dm.zernike_penalty_rank > 0


def test_diffmat_roundtrip(tmp_path):
    """save/load reproduces every matrix and knob. Own format, no IO framework."""
    dm = _simple_diffmat()
    path = dm.save(tmp_path / "dm")
    back = DiffMat.load(path)
    for key in ("D_rho", "D_theta", "D_zeta", "W_rho", "W_theta", "W_zeta"):
        np.testing.assert_allclose(
            np.asarray(getattr(back, key)), np.asarray(getattr(dm, key))
        )
    assert back.zernike_penalty_alpha == dm.zernike_penalty_alpha
    assert hash(back) == hash(dm)


def test_diffmat_is_a_pytree():
    """Matrices flatten as leaves; the scalar knobs stay static.

    If the knobs were leaves they would become tracers under jit, and
    ``alpha > 0`` would raise TracerBoolConversionError.
    """
    dm = _simple_diffmat()
    leaves, aux = jax.tree_util.tree_flatten(dm)
    assert len(leaves) == 6
    assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves)
    back = jax.tree_util.tree_unflatten(aux, leaves)
    assert isinstance(back.zernike_penalty_alpha, float)
    np.testing.assert_allclose(np.asarray(back.D_rho), np.asarray(dm.D_rho))


def test_diffmat_from_zeta_grid():
    """The convenience constructor builds a matching FD pair for uniform zeta."""
    zeta = jnp.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    dm = DiffMat.from_zeta_grid(zeta)
    assert dm.D_zeta.shape == (16, 16)
    assert float(jnp.max(jnp.abs(dm.D_zeta @ jnp.ones(16)))) < 1e-10


def test_diffmat_from_zeta_grid_rejects_nonuniform():
    """Non-uniform nodes are refused: the stencil assumes constant spacing."""
    zeta = jnp.asarray(np.concatenate([np.linspace(0, 1, 8), [1.5, 2.4]]))
    with pytest.raises(ValueError, match="uniformly spaced"):
        DiffMat.from_zeta_grid(zeta)
    with pytest.raises(ValueError, match="At least 8"):
        DiffMat.from_zeta_grid(jnp.linspace(0, 1, 4))
