#!/usr/bin/env python3
"""A set of functions that test fourier differentiation."""
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import grad, jit, vmap

# Import the necessary functions from your module
from create_fourier_diffmatrix import fourier_diffmat, fourier_pts

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def eval_1D(f, x):
    """Evaluate function f at all grid points (x[i])."""
    return vmap(lambda x_val: f(x_val))(x)


# Define test functions and derivatives in a jittable way
@jit
def _function(x):
    m = 0.5
    x0 = jnp.pi
    return jnp.sin(3 * x) * jnp.cos(2 * x) + jnp.exp(-((x - x0) ** 2) / m)


# === Analytical Derivatives via Automatic Differentiation ===
dx_f = grad(_function)
dxx_f = grad(dx_f)


@jit
def compute_derivatives(f, D1):
    """Compute numerical derivatives using differentiation matrices."""
    df_numeric = D1 @ f
    d2f_numeric = D1 @ (D1 @ f)  # Second derivative by applying D1 twice
    return df_numeric, d2f_numeric


@jit
def compute_errors(df_numeric, df_exact, d2f_numeric, d2f_exact):
    """Compute max errors for first and second derivatives."""
    error_1st = jnp.max(jnp.abs(df_numeric - df_exact))
    error_2nd = jnp.max(jnp.abs(d2f_numeric - d2f_exact))
    return error_1st, error_2nd


# Create jittable test function for a specific n value
@partial(jit, static_argnums=(0,))
def run_single_test(n):
    """Test the first and second order derivatives."""
    # Generate points on [0, 2π]
    x = fourier_pts(n)

    # Compute function values
    f = _function(x)

    # Get differentiation matrix
    D1 = fourier_diffmat(n)

    # Compute numerical derivatives
    df_numeric, d2f_numeric = compute_derivatives(f, D1)

    # Exact derivatives
    df_exact = eval_1D(dx_f, x)
    d2f_exact = eval_1D(dxx_f, x)

    # Compute errors
    error_1st, error_2nd = compute_errors(df_numeric, df_exact, d2f_numeric, d2f_exact)

    return error_1st, error_2nd, x, f, df_numeric, df_exact, d2f_numeric, d2f_exact


@pytest.mark.unit
@pytest.mark.parametrize("n", [16, 32, 64])
def test_fourier_diff_accuracy_even(n):
    """Test accuracy of Fourier differentiation for a specific grid size."""
    error_1st, error_2nd, _, _, _, _, _, _ = run_single_test(n)

    print(f"n = {n}: Max error (1st derivative): {error_1st:.2e}")
    print(f"n = {n}: Max error (2nd derivative): {error_2nd:.2e}")

    # Test for passing criteria (adjust thresholds as needed)
    assert error_1st < 1e-3, f"First derivative error too large: {error_1st:.2e}"
    assert error_2nd < 1e-2, f"Second derivative error too large: {error_2nd:.2e}"


@pytest.mark.unit
@pytest.mark.parametrize("n", [17, 33, 65])  # Test odd number of points
def test_fourier_diff_accuracy_odd(n):
    """Test Fourier differentiation with odd number of points."""
    error_1st, error_2nd, _, _, _, _, _, _ = run_single_test(n)

    print(f"n = {n} (odd): Max error (1st derivative): {error_1st:.2e}")
    print(f"n = {n} (odd): Max error (2nd derivative): {error_2nd:.2e}")

    # Test for passing criteria
    assert error_1st < 1e-3, f"First derivative error too large: {error_1st:.2e}"
    assert error_2nd < 1e-2, f"Second derivative error too large: {error_2nd:.2e}"


@pytest.mark.regression
def test_fourier_diff_convergence():
    """Test convergence of Fourier differentiation."""
    # Test with both odd and even number of points
    n_values = jnp.array([8, 16, 32, 64, 128])

    # Arrays to store results
    errors_1st = []
    errors_2nd = []

    # Run tests for each n value
    for n in n_values:
        error_1st, error_2nd, _, _, _, _, _, _ = run_single_test(int(n))
        errors_1st.append(float(error_1st))
        errors_2nd.append(float(error_2nd))

        print(f"n = {n}: Max error (1st derivative): {error_1st:.2e}")
        print(f"n = {n}: Max error (2nd derivative): {error_2nd:.2e}")

    # Generate plot for visualization
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, errors_1st, "bo-", label="1st derivative")
    plt.loglog(n_values, errors_2nd, "ro-", label="2nd derivative")
    plt.loglog(n_values, [1 / n**2 for n in n_values], "k--", label="O(1/n²)")
    plt.loglog(n_values, [1 / n**4 for n in n_values], "k-.", label="O(1/n⁴)")
    plt.title("Convergence Rate of Fourier Differentiation")
    plt.xlabel("Number of points (n)")
    plt.ylabel("Maximum Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("fourier_diff_convergence.png")

    # Check convergence rate
    if len(n_values) >= 3:
        conv_rate_1st = np.polyfit(np.log(n_values[:]), np.log(errors_1st[:]), 1)[0]
        conv_rate_2nd = np.polyfit(np.log(n_values[:]), np.log(errors_2nd[:]), 1)[0]

        print(f"Estimated convergence rate (1st derivative): {conv_rate_1st:.2f}")
        print(f"Estimated convergence rate (2nd derivative): {conv_rate_2nd:.2f}")

        assert conv_rate_1st < -1.5, "First derivative not converging fast enough"
        assert conv_rate_2nd < -1.5, "Second derivative not converging fast enough"


if __name__ == "__main__":
    test_fourier_diff_accuracy_even()
    test_fourier_diff_accuracy_odd()
    test_fourier_diff_convergence()
