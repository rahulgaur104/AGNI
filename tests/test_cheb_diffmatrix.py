#!/usr/bin/env python3
"""
Test module for Chebyshev differentiation matrices.

This module contains tests for validating the accuracy and convergence
properties of Chebyshev differentiation matrices on standard test functions.
All functions are designed to be jittable with JAX and avoid explicit loops
for better performance.
"""
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from jax import jit

from create_cheb_diffmatrix import chebpts_lobatto, diffmat_lobatto

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@jit
def function(x):
    """
    Test function for differentiation: sin(πx) + cos(πx).

    This function is smooth and has known analytical derivatives,
    making it suitable for testing differentiation accuracy.

    Parameters
    ----------
    x : jnp.ndarray
        Array of points where the function is evaluated

    Returns
    -------
    jnp.ndarray
        Function values at the provided points
    """
    return jnp.sin(jnp.pi * x) + jnp.cos(jnp.pi * x)


@jit
def first_derivative(x):
    """
    First analytical derivative of the test function.

    Parameters
    ----------
    x : jnp.ndarray
        Array of points where the derivative is evaluated

    Returns
    -------
    jnp.ndarray
        First derivative values at the provided points
    """
    return jnp.pi * (jnp.cos(jnp.pi * x) - jnp.sin(jnp.pi * x))


@jit
def second_derivative(x):
    """
    Second analytical derivative of the test function.

    Parameters
    ----------
    x : jnp.ndarray
        Array of points where the second derivative is evaluated

    Returns
    -------
    jnp.ndarray
        Second derivative values at the provided points
    """
    return -((jnp.pi) ** 2) * (jnp.sin(jnp.pi * x) + jnp.cos(jnp.pi * x))


@jit
def compute_derivatives(x, f, D):
    """
    Compute numerical derivatives using differentiation matrix.

    Parameters
    ----------
    x : jnp.ndarray
        Array of points where derivatives are computed
    f : jnp.ndarray
        Function values at the points in x
    D : jnp.ndarray
        Differentiation matrix

    Returns
    -------
    tuple
        Tuple containing first and second numerical derivatives
    """
    df_numeric = D @ f
    d2f_numeric = D @ (D @ f)
    return df_numeric, d2f_numeric


@jit
def compute_errors(df_numeric, df_exact, d2f_numeric, d2f_exact):
    """
    Compute maximum errors between numerical and exact derivatives.

    Parameters
    ----------
    df_numeric : jnp.ndarray
        Numerical first derivative values
    df_exact : jnp.ndarray
        Exact first derivative values
    d2f_numeric : jnp.ndarray
        Numerical second derivative values
    d2f_exact : jnp.ndarray
        Exact second derivative values

    Returns
    -------
    tuple
        Maximum absolute errors for first and second derivatives
    """
    error_1st = jnp.max(jnp.abs(df_numeric - df_exact))
    error_2nd = jnp.max(jnp.abs(d2f_numeric - d2f_exact))
    return error_1st, error_2nd


@partial(jit, static_argnums=(0,))
def run_single_test(n):
    """
    Run a differentiation test for a specific grid size.

    This function computes numerical derivatives using Chebyshev
    differentiation matrices and compares them with exact derivatives.

    Parameters
    ----------
    n : int
        Number of grid points

    Returns
    -------
    tuple
        Results including errors and arrays for plotting
    """
    x = chebpts_lobatto(n)
    D = diffmat_lobatto(n)

    f = function(x)

    df_numeric, d2f_numeric = compute_derivatives(x, f, D)

    # Exact derivatives
    df_exact = first_derivative(x)
    d2f_exact = second_derivative(x)

    # Compute errors
    error_1st, error_2nd = compute_errors(df_numeric, df_exact, d2f_numeric, d2f_exact)

    return error_1st, error_2nd, x, f, df_numeric, df_exact, d2f_numeric, d2f_exact


@pytest.mark.unit
@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_cheb_diff_accuracy(n):
    """
    Test accuracy of Chebyshev differentiation for specific grid sizes.

    This test evaluates whether the differentiation matrices achieve
    acceptable accuracy for the first and second derivatives of the
    test function. It uses different grid sizes to ensure stability.

    Parameters
    ----------
    n : int
        Number of grid points
    """
    error_1st, error_2nd, _, _, _, _, _, _ = run_single_test(n)

    print(f"n = {n}: Max error (1st derivative): {error_1st:.2e}")
    print(f"n = {n}: Max error (2nd derivative): {error_2nd:.2e}")

    if n <= 8:
        assert error_1st < 1e-1, f"First derivative error too large: {error_1st:.2e}"
        assert error_2nd < 2e-0, f"Second derivative error too large: {error_2nd:.2e}"
    else:
        assert error_1st < 1e-4, f"First derivative error too large: {error_1st:.2e}"
        assert error_2nd < 1e-2, f"Second derivative error too large: {error_2nd:.2e}"


@pytest.mark.regression
def test_cheb_diff_convergence():
    """
    Test convergence rate of Chebyshev differentiation.

    This regression test verifies that the Chebyshev differentiation
    matrices exhibit the expected spectral convergence properties as
    the number of grid points increases. It computes errors for a series
    of grid resolutions and estimates the convergence rate.
    """
    # Test convergence with different n values
    # 64 bit precision begins to affect accuracty of the diffmatrix
    # beyond n_values ~> 80
    n_values = jnp.array([8, 16, 32, 64, 80])

    errors_1st = []
    errors_2nd = []

    for n in n_values:
        error_1st, error_2nd, _, _, _, _, _, _ = run_single_test(int(n))
        errors_1st.append(float(error_1st))
        errors_2nd.append(float(error_2nd))

        print(f"n = {n}: Max error (1st derivative): {error_1st:.2e}")
        print(f"n = {n}: Max error (2nd derivative): {error_2nd:.2e}")

    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, errors_1st, "bo-", label="1st derivative")
    plt.loglog(n_values, errors_2nd, "ro-", label="2nd derivative")
    plt.loglog(n_values, [1 / n**2 for n in n_values], "k--", label="O(1/n²)")
    plt.title("Convergence Rate of Chebyshev Differentiation")
    plt.xlabel("Number of points (n)")
    plt.ylabel("Maximum Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("cheb_diff_convergence.png")

    if len(n_values) >= 3:
        # Estimate convergence rate from the last 3 points
        conv_rate_1st = jnp.polyfit(
            jnp.log(n_values[:]), jnp.log(jnp.array(errors_1st[:])), 1
        )[0]
        conv_rate_2nd = jnp.polyfit(
            jnp.log(n_values[:]), jnp.log(jnp.array(errors_2nd[:])), 1
        )[0]

        print(f"Estimated convergence rate (1st derivative): {conv_rate_1st:.2f}")
        print(f"Estimated convergence rate (2nd derivative): {conv_rate_2nd:.2f}")

        assert conv_rate_1st < -1.5, "First derivative not converging fast enough"
        assert conv_rate_2nd < -1.5, "Second derivative not converging fast enough"


# if the user wants to run it directly do:
# >>> python tests/test_cheb_diffmatrix.py
if __name__ == "__main__":
    test_cheb_diff_convergence()
