#!/usr/bin/env python3
"""Pytest module for testing pure and mixed Chebyshev-Fourier derivatives (up to 4th order)."""
import jax
import jax.numpy as jnp
import pytest
from jax import grad, vmap
from matplotlib import pyplot as plt

from create_cheb_diffmatrix import cheb_D1, cheb_D2, cheb_D3, cheb_D4, chebpts_lobatto
from create_fourier_diffmatrix import fourier_diffmat, fourier_pts

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# === Helper Functions ===
def eval_1D(f, x):
    """Evaluate function f at all grid points (x[i])."""
    return vmap(lambda x_val: f(x_val))(x)


def eval_2D(f, x, y):
    """Evaluate function f at all grid points (x[i], y[j])."""
    return vmap(lambda y_val: vmap(lambda x_val: f(x_val, y_val))(x))(y)


def map_domain(x, option=1):
    """
    Map points from one domain to another nonlinearly.

    option = 0 is to move points to the middle
    option = 1 is to move points towards the edge
    """
    if option == 0:  # always bunch near the middle

        def _f(x):
            a = 0.11  # [0.0, 0.14]
            m = 1
            return 0.5 * (x + 1) + a * jnp.sin(m * jnp.pi * (x + 1))

    elif option == 1:  # bunch about a specific point

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

    elif option == 2:

        def _f(x):
            return 0.5 * (x + 1) - 0.3 * (1 - x**16) * jnp.cos(0.5 * jnp.pi * (x - 1.8))

    elif option == 3:

        def _f(x):
            return 0.5 * (x + 1) - 0.3 * (1 - x**16) * jnp.exp(
                0.3 * (x - 1) ** 2
            ) * jnp.cos(0.5 * jnp.pi * (x - 1.8))

    elif option == 4:  # Tal-Ezer transformation

        def _f(x):
            N = jnp.size(x)
            alpha = 1 / jnp.cosh(jnp.log(jnp.finfo("float").resolution) / N)
            return 0.5 + 0.5 * jnp.arcsin(alpha * x) / jnp.arcsin(alpha)

    else:

        def _f(x):
            return (x + 1) / 2

    dx_f = grad(_f)
    dxx_f = grad(dx_f)
    dxxx_f = grad(dxx_f)
    dxxxx_f = grad(dxxx_f)

    scale_vector1 = (eval_1D(dx_f, x)) ** -1
    scale_vector2 = (eval_1D(dxx_f, x)) * scale_vector1
    scale_vector3 = (eval_1D(dxxx_f, x)) * scale_vector1
    scale_vector4 = (eval_1D(dxxxx_f, x)) * scale_vector1

    one_matrix = jnp.ones((len(x), len(x)))

    scale_matrix1 = one_matrix * scale_vector1[:, None]
    scale_matrix2 = one_matrix * scale_vector2[:, None]
    scale_matrix3 = one_matrix * scale_vector3[:, None]
    scale_matrix4 = one_matrix * scale_vector4[:, None]

    return _f(x), scale_matrix1, scale_matrix2, scale_matrix3, scale_matrix4


def tensor_product_derivative(nx, ny, dx_order, dy_order):
    """
    Create a tensor product differentiation matrix for the specified orders.

    Parameters
    ----------
    nx : int
        Number of Chebyshev points in x dimension
    ny : int
        Number of Fourier points in y dimension
    dx_order : int
        Order of x derivative (Chebyshev)
    dy_order : int
        Order of y derivative (Fourier)

    Returns
    -------
    D : array
        Tensor product differentiation matrix
    x : array
        x collocation points
    y : array
        y collocation points
    """
    # Generate collocation points
    x_cheb = chebpts_lobatto(nx)
    y_four = fourier_pts(ny)

    ## Map Chebyshev points to desired domain
    x, scale_x1, scale_x2, scale_x3, scale_x4 = map_domain(x_cheb, option=1)

    y = y_four  # Already in [0, 2π]

    # Get x differentiation matrix (Chebyshev)
    if dx_order == 0:
        Dx = jnp.eye(nx)
    else:
        if dx_order == 1:
            Dx = cheb_D1(nx) * scale_x1
        elif dx_order == 2:
            Dx = (cheb_D2(nx) - cheb_D1(nx) * scale_x2) * scale_x1**2
        elif dx_order == 3:
            Dx = (
                cheb_D3(nx)
                - (cheb_D2(nx) - cheb_D1(nx) * scale_x2) * 3 * scale_x2
                - cheb_D1(nx) * scale_x3
            ) * scale_x1**3
        else:  # dx_order == 4
            Dx = (
                cheb_D4(nx)
                - cheb_D3(nx) * 6 * scale_x2
                + cheb_D2(nx) * (15 * scale_x2**2 - 4 * scale_x3)
                + cheb_D1(nx)
                * (5 * scale_x2 * (2 * scale_x3 - 3 * scale_x2**2) - scale_x4)
            ) * scale_x1**4

    # Get y differentiation matrix (Fourier)
    if dy_order == 0:
        Dy = jnp.eye(ny)
    elif dy_order == 1:
        Dy = fourier_diffmat(ny)
    elif dy_order == 2:
        Dy = fourier_diffmat(ny) @ fourier_diffmat(ny)
    elif dy_order == 3:
        Dy = fourier_diffmat(ny) @ fourier_diffmat(ny) @ fourier_diffmat(ny)
    else:
        Dy = (
            fourier_diffmat(ny)
            @ fourier_diffmat(ny)
            @ fourier_diffmat(ny)
            @ fourier_diffmat(ny)
        )

    # Create identity matrices for tensor product
    Ix = jnp.eye(nx)
    Iy = jnp.eye(ny)

    # Tensor product approach using Kronecker products
    # For ∂^(i+j)/∂x^i∂y^j, we use:
    # D = kron(Iy, Dx) + kron(Dy, Ix) for i=j=1
    # or just kron(Iy, Dx) for pure x derivatives, etc.
    if dx_order > 0 and dy_order > 0:
        # Mixed derivative
        # --no0verifyD = jnp.kron(Iy, Dx) @ jnp.kron(Dy, Ix)
        D = jnp.kron(Dy, Dx)  # Same as above

    elif dx_order > 0:
        # Pure x derivative
        D = jnp.kron(Iy, Dx)
    elif dy_order > 0:
        # Pure y derivative
        D = jnp.kron(Dy, Ix)
    else:
        # No derivative (identity)
        D = jnp.kron(Iy, Ix)

    D.at[jnp.where(jnp.abs(D) < 1e-12)].set(0.0)
    return D, x, y


# --- Test Function ---
def _function(x, y):
    """2D test function: localized and oscillatory in x, periodic in y."""
    return (
        2
        * jnp.exp(-100 * (x - 0.8) ** 2)
        * jnp.sin(6 * jnp.pi * x)
        * (jnp.sin(4 * y) + jnp.cos(3 * y))
    )


# --- Analytical Derivatives via JAX ---
# Pure x
dx_f = grad(_function, argnums=0)
dxx_f = grad(dx_f, argnums=0)
dxxx_f = grad(dxx_f, argnums=0)
dxxxx_f = grad(dxxx_f, argnums=0)
# Pure y
dy_f = grad(_function, argnums=1)
dyy_f = grad(dy_f, argnums=1)
dyyy_f = grad(dyy_f, argnums=1)
dyyyy_f = grad(dyyy_f, argnums=1)
# Mixed
dxy_f = grad(dx_f, argnums=1)
dxxy_f = grad(dxx_f, argnums=1)
dxyy_f = grad(dxy_f, argnums=1)
dxxxy_f = grad(dxxx_f, argnums=1)
dxxyy_f = grad(dxxy_f, argnums=1)
dxyyy_f = grad(dxyy_f, argnums=1)

# --- Test Cases Configuration ---
# List of (dx_order, dy_order, analytic_fn, tolerance)
test_cases = [
    # Pure x derivatives
    (1, 0, dx_f, 5e-12),
    (2, 0, dxx_f, 3e-09),
    (3, 0, dxxx_f, 2e-05),
    (4, 0, dxxxx_f, 3e-02),
    # Pure y derivatives
    (0, 1, dy_f, 1e-11),
    (0, 2, dyy_f, 1e-11),
    (0, 3, dyyy_f, 2e-10),
    (0, 4, dyyyy_f, 6e-09),
    # Mixed derivatives
    (1, 1, dxy_f, 3e-11),
    (2, 1, dxxy_f, 2e-08),
    (1, 2, dxyy_f, 5e-10),
    (3, 1, dxxxy_f, 5e-05),
    (1, 3, dxyyy_f, 1e-08),
    (2, 2, dxxyy_f, 2e-07),
]

# a module‐level list to stash all of our (dx, dy, n, error) tuples
collected_errors = []


@pytest.mark.regression
@pytest.mark.parametrize("n", [80])
@pytest.mark.parametrize("dx_order,dy_order,analytic_fn,tol", test_cases)
def test_tensor_mixed_derivative(dx_order, dy_order, analytic_fn, tol, n):
    """Validate tensor-product differentiation against analytical JAX derivatives."""
    # Grid resolution
    nx, ny = n, n

    D, x, y = tensor_product_derivative(nx, ny, dx_order, dy_order)

    # --no-verify print("condition number=", jnp.linalg.cond(D))
    # Evaluate function on mesh
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    f_vals = _function(X, Y)
    # Flatten (column-major) to match kron ordering
    f_flat = jnp.transpose(f_vals, (1, 0)).flatten()

    # Numerical derivative
    df_flat = D @ f_flat
    df_num = df_flat.reshape(nx, ny)

    # Analytical derivative (JAX auto-vectorized)
    df_exact = eval_2D(analytic_fn, x, y)

    error = float(jnp.max(jnp.abs(df_num - df_exact)))

    # record it (pytest will still assert below)
    collected_errors.append((dx_order, dy_order, n, error))

    assert (
        error < tol
    ), f"dx={dx_order}, dy={dy_order}: error {error:.2e} exceeds tol {tol}"


# Unusual but works
# To view the plots, run pytest -s
def teardown_module(module):
    """For making convergence plots."""
    print("\n\n=== Collected errors ===")
    for dx, dy, n, err in collected_errors:
        print(f"  dx={dx}, dy={dy}, n={n} → error={err:.2e}")

    data = {}
    for dx, dy, n, err in collected_errors:
        data.setdefault((dx, dy), []).append((n, err))

    plt.figure(figsize=(8, 6))
    slopes = {}
    for (dx, dy), pts in data.items():
        # sort by grid size
        pts_sorted = sorted(pts, key=lambda t: t[0])
        ns = jnp.array([t[0] for t in pts_sorted])
        errs = jnp.array([t[1] for t in pts_sorted])

        # fit a line: log(err) = slope*log(n) + intercept
        slope, intercept = jnp.polyfit(jnp.log(ns), jnp.log(errs), 1)
        slopes[(dx, dy)] = slope

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
    plt.show()

    print("\nEmpirical convergence slopes:")
    for (dx, dy), m in slopes.items():
        print(f"  dx={dx}, dy={dy} → slope ≃ {m:.2f}")
