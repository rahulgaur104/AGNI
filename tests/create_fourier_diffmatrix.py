#!/usr/bin/env python3
"""A set of functions that create fourier differentiation."""
import jax
import jax.numpy as jnp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def fourier_pts(n, domain=None):
    """
    Generate equally spaced points for Fourier spectral methods.

    Parameters
    ----------
    n : int
        Number of points
    domain : list, optional
        Domain [a, b], default is [0, 2π]

    Returns
    -------
    x : array
        Equally spaced points
    """
    if domain is None:
        domain = [0, 2 * jnp.pi]

    a, b = domain
    h = (b - a) / n
    x = jnp.arange(n) * h + a

    return x


def fourier_diffmat(n):
    """
    First derivative matrix using Fourier spectral method.

    Parameters
    ----------
    n : int
        Number of points

    Returns
    -------
    D : array
        Differentiation matrix (n × n)
    """
    # Initialize with zeros
    D = jnp.zeros((n, n))

    # Set up indices
    i, j = jnp.mgrid[0:n, 0:n]

    # Compute off-diagonal elements differently for odd/even n
    if n % 2 == 0:  # even
        # Explicit formula for even number of points
        D = jnp.where(
            i != j, 0.5 * (-1) ** (i - j) / jnp.tan((i - j) * jnp.pi / n), 0.0
        )
    else:  # odd
        # Explicit formula for odd number of points
        D = jnp.where(
            i != j, 0.5 * (-1) ** (i - j) / jnp.sin((i - j) * jnp.pi / n), 0.0
        )

    # Get rid of essentially 0.s
    # Throws a jax dype error!
    # --no-verify D.at[jnp.where(jnp.abs(D) < 1e-12)].set(0.0)

    # Diagonal elements are zero (rows sum to zero)
    return D


def fourier_diffmat1(N):
    """
    First-order Fourier differentiation matrix.

    Handles both even and odd N cases correctly using the appropriate
    formulas for each case.

    Parameters
    ----------
    N : int
        Size of differentiation matrix

    Returns
    -------
    D1 : array
        First-order differentiation matrix (N × N)
    """
    h = 2.0 * jnp.pi / N

    # Create the first column
    col1 = jnp.zeros(N)

    # The diagonal element (j=0) is always 0 for first derivative
    # This is already set in our zeros initialization

    if N % 2 == 0:  # even N
        # For even N, we use cot(jh/2) formula
        j_indices = jnp.arange(1, N)

        ## Calculate cot(jh/2) directly
        # --no-verify sin_term = jnp.sin(j_indices * h / 2.0)
        # --no-verify cos_term = jnp.cos(j_indices * h / 2.0)
        # --no-verify cot_term = cos_term / sin_term
        cot_term = 1 / jnp.tan(j_indices * h / 2.0)

        # Apply the formula with alternating sign: 0.5 * (-1)^j * cot(jh/2)
        values = 0.5 * (-1.0) ** j_indices * cot_term

        # Set off-diagonal elements
        col1 = col1.at[j_indices].set(values)

    else:  # odd N
        # For odd N, we use csc(jh/2) formula
        j_indices = jnp.arange(1, N)

        # Calculate csc(jh/2) = 1/sin(jh/2)
        sin_term = jnp.sin(j_indices * h / 2.0)
        csc_term = 1.0 / sin_term

        # Apply the formula with alternating sign: 0.5 * (-1)^j * csc(jh/2)
        values = 0.5 * (-1.0) ** j_indices * csc_term

        # Set off-diagonal elements
        col1 = col1.at[j_indices].set(values)

    # First row is negative of first column (skew-symmetry)
    row1 = -col1

    # Construct the full matrix using circular shifts
    D1 = jax.vmap(lambda i: jnp.roll(col1, i))(jnp.arange(N))

    # Set the first row correctly for skew-symmetry
    D1 = D1.at[0].set(row1)

    # Get rid of essentially 0.s. Throws a jax dtpe error!
    # --no-verify D.at[jnp.where(jnp.abs(D) < 1e-12)].set(0.0)

    return D1


def fourier_diffmat2(N):
    """
    Second-order Fourier differentiation matrix.

    Handles both even and odd N cases correctly.

    Parameters
    ----------
    N : int
        Size of differentiation matrix

    Returns
    -------
    D2 : array
        Second-order differentiation matrix (N × N)
    """
    h = 2.0 * jnp.pi / N

    # Create the first column
    col1 = jnp.zeros(N)

    if N % 2 == 0:  # even N
        # Set the diagonal element (j=0)
        col1 = col1.at[0].set(-jnp.pi**2 / (3.0 * h**2) - 1.0 / 6.0)

        # Set the off-diagonal elements (j ≠ 0)
        j_indices = jnp.arange(1, N)
        sin_term = jnp.sin(j_indices * h / 2.0)
        sin_squared = sin_term**2
        values = -((-1.0) ** j_indices) / (2.0 * sin_squared)
        col1 = col1.at[j_indices].set(values)

    else:  # odd N
        # Set the diagonal element (j=0)
        col1 = col1.at[0].set(
            -jnp.pi**2 / (3.0 * h**2) + 1.0 / 12.0
        )  # Note the + 1/12 instead of - 1/6

        # Set the off-diagonal elements (j ≠ 0)
        j_indices = jnp.arange(1, N)
        sin_term = jnp.sin(j_indices * h / 2.0)
        cot_term = jnp.cos(j_indices * h / 2.0) / sin_term  # cot(x) = cos(x)/sin(x)

        # Formula for odd N: -(-1)^j * csc(jh/2) * cot(jh/2) / 2
        values = -((-1.0) ** j_indices) * (1.0 / sin_term) * cot_term / 2.0
        col1 = col1.at[j_indices].set(values)

    # Construct the full matrix (same for both even and odd N)
    # For second-order derivatives, the matrix is symmetric
    indices = jnp.arange(N)
    D2 = jax.vmap(lambda i: jnp.roll(col1, i))(indices)

    return D2
