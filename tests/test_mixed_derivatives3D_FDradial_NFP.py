#!/usr/bin/env python3
"""
Test file for higher-order mixed derivatives in a single field period.

Using tensor product approach in 3D:
- Chebyshev methods in x dimension
- Fourier methods in y dimension
- Fourier methods in z dimension
"""
import jax
import jax.numpy as jnp
import pytest
from jax import grad, vmap

from create_fourier_diffmatrix import fourier_diffmat, fourier_pts
from create_Pade_Lele_diffmatrix import create_lele_D1_6_matrix, create_lele_D2_6_matrix

jax.config.update("jax_platform_name", "cpu")  # same effect as the env-var

### Enable 64-bit precision
## --no-verify jax.config.update("jax_enable_x64", True)

NFP = 5


# === Helper Functions ===
def _eval_1D(f, x):
    """Evaluate function f at all grid points (x[i])."""
    return vmap(lambda x_val: f(x_val))(x)


def _eval_2D(f, x, y):
    """Evaluate function f at all grid points (x[i], y[j])."""
    return vmap(lambda y_val: vmap(lambda x_val: f(x_val, y_val))(x))(y)


def _eval_3D(f, x, y, z, NFP):
    """Evaluate function f at grid points (x[i], y[j], z[k])."""
    return vmap(
        lambda z_val: vmap(
            lambda y_val: vmap(lambda x_val: f(x_val, y_val, z_val, NFP))(x)
        )(y)
    )(z)


def _map_domain(x, option):
    """
    Map points from one domain to another nonlinearly.

    option = 0 is to move points to the middle
    option = 1 is to move points towards the edge
    """
    if option == 0:  # always bunch near the middle

        def _f(x):
            a = 0.015  # [0.0, 0.1]
            return 0.5 * (x + 1) + a * jnp.sin(jnp.pi * (x + 1))

    elif option == 1:  # bunch colocation nodes near a specific point

        def _f(x):
            x_0 = 0.8
            m_1 = 2.0
            m_2 = 3.0
            lower = x_0 * (
                1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1)
            )
            upper = (1 - x_0) * (
                jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2)
            )
            return lower + upper

    elif option == 3:

        def _f(x):
            return 0.5 * (x + 1) - 0.3 * (1 - x**16) * jnp.cos(0.5 * jnp.pi * (x - 1.8))

    elif option == 4:

        def _f(x):
            map_term1 = 0.5 * (x + 1)
            exp_term = jnp.exp(0.3 * (x - 1) ** 2)
            map_term2 = 0.3 * (1 - x**16) * exp_term * jnp.cos(0.5 * jnp.pi * (x - 1.8))
            return map_term1 - map_term2

    elif option == 5:

        def _f(x):
            x_0 = 0.8
            x_1 = 1.0
            m_1 = 2.0
            m_2 = 3.0
            m_3 = 20
            m_4 = 30
            lower = x_0 * (
                1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1)
            )
            upper = (1 - x_0) * (
                jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2)
            )
            axis = x_0 * (
                -1
                + 2 / (1 + jnp.exp(-m_3 * (x + 1)))
                - (x + 1) / (1 + jnp.exp(-2 * m_3))
            )
            edge = (1 - x_0) * (
                -1 + 2 / (1 + jnp.exp(m_4 * (x - 1))) + (x - 1) / (1 + jnp.exp(2 * m_4))
            )

            return x_1 * (lower + upper) + (1 - x_1) * (axis + edge)

    elif option == 6:

        def _f(x):
            return x

    elif option == 7:

        def _f(x):
            return x + 0.1 * jnp.sin(2 * jnp.pi * (x + 1))

    else:

        def _f(x):
            return (x + 1) / 2

    dx_f = grad(_f)
    dxx_f = grad(dx_f)

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    scale_vector2 = (_eval_1D(dxx_f, x)) * scale_vector1

    one_matrix = jnp.ones((len(x), len(x)))

    scale_matrix1 = one_matrix * scale_vector1[:, None]
    scale_matrix2 = one_matrix * scale_vector2[:, None]

    return _f(x), scale_matrix1, scale_matrix2


def _tensor_product_derivative_3D(  # noqa: C901
    nx, ny, nz, dx_order, dy_order, dz_order, NFP
):
    """
    Create a tensor product differentiation matrix in 3D.

    Parameters
    ----------
    nx : int
        Number of Chebyshev points in x dimension
    ny : int
        Number of Fourier points in y dimension
    nz : int
        Number of Fourier points in z dimension
    dx_order : int
        Order of x derivative (Chebyshev), 0 <= dx_order <= 2
    dy_order : int
        Order of y derivative (Fourier), 0 <= dy_order <= 2
    dz_order : int
        Order of z derivative (Fourier), 0 <= dz_order <= 2

    Returns
    -------
    D : array
        Tensor product differentiation matrix
    x : array
        x collocation points
    y : array
        y collocation points
    z : array
        z collocation points
    """
    # Generate collocation points
    x_cheb = jnp.linspace(0.0, 1.0, nx)
    y_four = fourier_pts(ny)
    z_four = fourier_pts(nz)

    # Map Chebyshev points to desired domain
    x, scale_x1, scale_x2 = _map_domain(x_cheb, option=6)
    y = y_four  # Already in [0, 2π]
    z = z_four / NFP  # Already in [0, 2π/NFP]

    x = jnp.linspace(0.0, 1.0, nx)  # uniform in physical ρ
    h = x[1] - x[0]

    A, B = create_lele_D1_6_matrix(nx, h)
    D = jnp.linalg.solve(A, B)

    A, B = create_lele_D2_6_matrix(nx, h)
    D2 = jnp.linalg.solve(A, B)

    if dx_order == 0:
        Dx = jnp.eye(nx)
    elif dx_order == 1:
        Dx = D * scale_x1
    elif dx_order == 2:
        Dx = (D2 - D * scale_x2) * scale_x1**2

    # Get y differentiation matrix (Fourier)
    if dy_order == 0:
        Dy = jnp.eye(ny)
    elif dy_order == 1:
        Dy = fourier_diffmat(ny)
    elif dy_order == 2:
        Dy = fourier_diffmat(ny) @ fourier_diffmat(ny)

    # Get z differentiation matrix (Fourier)
    if dz_order == 0:
        Dz = jnp.eye(nz)
    elif dz_order == 1:
        Dz = fourier_diffmat(nz) * NFP
    elif dz_order == 2:
        Dz = fourier_diffmat(nz) @ fourier_diffmat(nz) * NFP**2

    # Create identity matrices for tensor product
    Ix = jnp.eye(nx)
    Iy = jnp.eye(ny)
    Iz = jnp.eye(nz)

    # Tensor product approach using Kronecker products
    # Following the approach in the 2D code

    # Construct the appropriate matrix based on the derivative order

    if dx_order > 0 and dy_order > 0 and dz_order > 0:
        # Full 3D mixed derivative (x, y, z)
        D = jnp.kron(Dz, jnp.kron(Dy, Dx))

    elif dx_order > 0 and dy_order > 0:
        # Mixed derivative (x, y)
        D = jnp.kron(Dx, Dy)

    elif dx_order > 0 and dz_order > 0:
        # Mixed derivative (x, z)
        D = jnp.kron(Dx, Dz)

    elif dy_order > 0 and dz_order > 0:
        # Mixed derivative (y, z)
        D = jnp.kron(Dz, Dy)

    elif dx_order > 0:
        # Pure x derivative
        D = Dx

    elif dy_order > 0:
        # Pure y derivative
        D = Dy

    elif dz_order > 0:
        # Pure z derivative
        D = Dz
    else:
        # Identity (no derivative)
        D = jnp.kron(Iz, jnp.kron(Iy, Ix))

    # --no-verify print("condition number=", jnp.linalg.cond(Dx))
    # Clean up small values
    D = jnp.where(jnp.abs(D) < 1e-12, 0.0, D)

    return D, x, y, z


# === Test Function ===
def _test_function(x, y, z, NFP):
    """Test function that is smooth and effectively periodic in y and z."""
    return (
        jnp.exp(-100 * ((x - 0.8) ** 2))
        * jnp.sin(3 * x * 2 * jnp.pi)
        * (jnp.sin(4 * y) + jnp.cos(3 * y))
        * jnp.cos(NFP * z)
    )


# === Analytical Derivatives via Automatic Differentiation ===

# First derivatives
dx_f = grad(_test_function, argnums=0)
dy_f = grad(_test_function, argnums=1)
dz_f = grad(_test_function, argnums=2)

# Second derivatives
dxx_f = grad(dx_f, argnums=0)
dxy_f = grad(dx_f, argnums=1)
dxz_f = grad(dx_f, argnums=2)
dzx_f = dxz_f
dyy_f = grad(dy_f, argnums=1)
dyz_f = grad(dy_f, argnums=2)
dzz_f = grad(dz_f, argnums=2)


# --- Test Cases Configuration ---
# List of (dx_order, dy_order, analytic_fn, tolerance)
test_cases = [
    # Pure x derivatives
    (1, 0, 0, dx_f, 4e-3, NFP),
    (2, 0, 0, dxx_f, 8.0e-3, NFP),  # increased relative to the spectral
    # Pure y derivatives
    (0, 1, 0, dy_f, 1e-7, NFP),
    (0, 2, 0, dyy_f, 1e-5, NFP),
    # Pure z derivatives
    (0, 0, 1, dz_f, 1e-7, NFP),
    (0, 0, 2, dzz_f, 1e-5, NFP),
    # Mixed derivatives
    (1, 1, 0, dxy_f, 2e-3, NFP),
    (0, 1, 1, dyz_f, 2e-3, NFP),
    (1, 0, 1, dzx_f, 2e-3, NFP),
]

# a module‐level list to stash all of our (dx, dy, dz, n, error) tuples
collected_errors = []


@pytest.mark.regression
@pytest.mark.parametrize("n", [48])
@pytest.mark.parametrize("dx_order,dy_order,dz_order,analytic_fn,tol,NFP", test_cases)
def test_tensor_mixed_derivative(
    dx_order, dy_order, dz_order, analytic_fn, tol, NFP, n
):
    """Validate 3D tensor-product differentiation against analytical JAX derivatives."""
    # Grid resolution
    nx, ny, nz = 10 * n + 1, n, n

    D, x, y, z = _tensor_product_derivative_3D(
        nx, ny, nz, dx_order, dy_order, dz_order, NFP=NFP
    )

    # Create meshgrid for evaluation
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    f_vals0 = _test_function(X, Y, Z, NFP)

    if dx_order > 0 and dy_order > 0 and dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (2, 1, 0))
        f_flat = f_vals.flatten()

    elif dx_order > 0 and dy_order > 0:
        f_flat = jnp.reshape(f_vals0, (nx * ny, nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nx, ny, nz))

    elif dx_order > 0 and dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (0, 2, 1))
        f_flat = jnp.reshape(f_vals, (nx * nz, ny))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nx, nz, ny))
        df_grid = jnp.transpose(df_grid, (0, 2, 1))

    elif dy_order > 0 and dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (2, 1, 0))
        f_flat = jnp.reshape(f_vals, (nz * ny, nx))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nz, ny, nx))
        df_grid = jnp.transpose(df_grid, (2, 1, 0))

    elif dx_order > 0:
        f_flat = jnp.reshape(f_vals0, (nx, ny * nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nx, ny, nz))

    elif dy_order > 0:
        f_vals = jnp.transpose(f_vals0, (1, 0, 2))
        f_flat = jnp.reshape(f_vals, (ny, nx * nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (ny, nx, nz))
        df_grid = jnp.transpose(df_grid, (1, 0, 2))

    elif dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (2, 0, 1))
        f_flat = jnp.reshape(f_vals, (nz, nx * ny))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nz, nx, ny))
        df_grid = jnp.transpose(df_grid, (1, 2, 0))
    else:
        # Identity (no derivative)
        f_flat = f_vals0

    # Compute exact derivative
    df_exact = _eval_3D(analytic_fn, x, y, z, NFP).transpose(2, 1, 0)

    error = jnp.max(jnp.abs(df_grid - df_exact))

    # record it (pytest will still assert below)
    collected_errors.append((dx_order, dy_order, dz_order, n, error))

    assert (
        error < tol
    ), f"dx={dx_order}, dy={dy_order}, dz={dz_order}: error {error:.2e} exceeds tol {tol}"


# Unusual but works
# To view the plots, run pytest -s
@pytest.mark.skip
def _teardown_module(module):
    """For making convergence plots."""
    print("\n\n=== Collected errors ===")
    for dx, dy, dz, n, err in collected_errors:
        print(f"  dx={dx}, dy={dy}, dz={dz}, n={n} → error={err:.2e}")

    data = {}
    for dx, dy, dz, n, err in collected_errors:
        data.setdefault((dx, dy, dz), []).append((n, err))

    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    slopes = {}
    for (dx, dy, dz), pts in data.items():
        # sort by grid size
        pts_sorted = sorted(pts, key=lambda t: t[0])
        ns = jnp.array([t[0] for t in pts_sorted])
        errs = jnp.array([t[1] for t in pts_sorted])

        # fit a line: log(err) = slope*log(n) + intercept
        slope, intercept = jnp.polyfit(jnp.log(ns), jnp.log(errs), 1)
        slopes[(dx, dy, dz)] = slope

        # label with orders and slope
        label = f"dx={dx},dy={dy} (slope={slope:.2f})"
        plt.loglog(ns, errs, "o-", label=label)

    ref_ns = jnp.array(
        sorted({n for _, _, n, _ in collected_errors}), dtype=jnp.float64
    )
    plt.loglog(ref_ns, ref_ns**-4, "k--", label="O(n⁻⁴)")
    plt.loglog(ref_ns, ref_ns**-8, "k-.", label="O(n⁻⁸)")

    plt.xlabel("n (grid points per dim)")
    plt.ylabel("max error")
    plt.title("Convergence from collected_errors")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.savefig("tensor_product_convergence.png")

    print("\nEmpirical convergence slopes:")
    for (dx, dy, dz), m in slopes.items():
        print(f"  dx={dx}, dy={dy}, dz={dz} → slope ≃ {m:.2f}")
