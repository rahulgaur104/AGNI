#!/usr/bin/env python3
"""Pade and Lele high-order differentiation matrices reconstruction."""
import jax.numpy as jnp


def create_pade_D1_4_matrix(n, h=1.0):
    """
    Create 4th-order Padé approximation matrix for first derivative.

    Parameters
    ----------
    n : int
        Size of the matrix (number of grid points)
    h : float
        Grid spacing

    Returns
    -------
    A : jax.numpy.ndarray
        LHS matrix (for coefficients of derivatives)
    B : jax.numpy.ndarray
        RHS matrix (for coefficients of function values)
    """
    # Create LHS matrix A (tridiagonal)
    A = jnp.zeros((n, n))

    # Diagonal elements
    A = A.at[jnp.arange(n), jnp.arange(n)].set(jnp.ones(n))

    # Boundary points special treatment
    A = A.at[0, 1].set(2.0)
    A = A.at[n - 1, n - 2].set(2.0)

    # Off-diagonal elements for interior points (i-1 and i+1)
    interior = jnp.arange(1, n - 1)
    A = A.at[interior, interior - 1].set(jnp.ones(n - 2))
    A = A.at[interior, interior + 1].set(jnp.ones(n - 2))

    # Create RHS matrix B (maps function values to RHS vector)
    B = jnp.zeros((n, n))

    # Boundary points
    B = B.at[0, 0].set(-5.0 / 2.0 / h)
    B = B.at[0, 1].set(2.0 / h)
    B = B.at[0, 2].set(0.5 / h)

    B = B.at[n - 1, n - 1].set(5.0 / 2.0 / h)
    B = B.at[n - 1, n - 2].set(-2.0 / h)
    B = B.at[n - 1, n - 3].set(-0.5 / h)

    # Interior points
    interior = jnp.arange(1, n - 1)
    B = B.at[interior, interior + 1].set(3.0 / (2.0 * h))
    B = B.at[interior, interior - 1].set(-3.0 / (2.0 * h))

    return A, B


def create_pade_D2_4_matrix(n, h=1.0):
    """
    Create 4th-order Padé approximation matrix for second derivative.

    Parameters
    ----------
    n : int
        Size of the matrix (number of grid points)
    h : float
        Grid spacing

    Returns
    -------
    A : jax.numpy.ndarray
        LHS matrix (for coefficients of derivatives)
    B : jax.numpy.ndarray
        RHS matrix (for coefficients of function values)
    """
    # Create LHS matrix A (tridiagonal)
    A = jnp.zeros((n, n))

    # Diagonal elements
    A = A.at[jnp.arange(n), jnp.arange(n)].set(jnp.ones(n))

    # Interior points off-diagonal elements (i-1 and i+1)
    interior = jnp.arange(1, n - 1)
    A = A.at[interior, interior - 1].set(jnp.ones(n - 2) * 0.1)
    A = A.at[interior, interior + 1].set(jnp.ones(n - 2) * 0.1)

    # Create RHS matrix B (maps function values to RHS vector)
    B = jnp.zeros((n, n))

    # Boundary points (using second-order scheme)
    B = B.at[0, 0].set(2.0 / h**2)
    B = B.at[0, 1].set(-5.0 / h**2)
    B = B.at[0, 2].set(4.0 / h**2)
    B = B.at[0, 3].set(-1.0 / h**2)

    B = B.at[n - 1, n - 1].set(2.0 / h**2)
    B = B.at[n - 1, n - 2].set(-5.0 / h**2)
    B = B.at[n - 1, n - 3].set(4.0 / h**2)
    B = B.at[n - 1, n - 4].set(-1.0 / h**2)

    # Interior points
    interior = jnp.arange(1, n - 1)
    B = B.at[interior, interior + 1].set(6.0 / 5.0 / h**2)
    B = B.at[interior, interior].set(-12.0 / 5.0 / h**2)
    B = B.at[interior, interior - 1].set(6.0 / 5.0 / h**2)

    return A, B


def create_lele_D1_6_matrix(n, h=1.0):
    """
    Create 6th-order Lele compact approximation matrix for first derivative.

    Parameters
    ----------
    n : int
        Size of the matrix (number of grid points)
    h : float
        Grid spacing

    Returns
    -------
    A : jax.numpy.ndarray
        LHS matrix (for coefficients of derivatives)
    B : jax.numpy.ndarray
        RHS matrix (for coefficients of function values)
    """
    # Create LHS matrix A (tridiagonal)
    A = jnp.zeros((n, n))

    # Diagonal elements
    A = A.at[jnp.arange(n), jnp.arange(n)].set(jnp.ones(n))

    # Boundary and near-boundary treatment
    A = A.at[0, 1].set(2.0)
    A = A.at[n - 1, n - 2].set(2.0)

    # Second point from boundary
    A = A.at[1, 0].set(0.25)
    A = A.at[1, 2].set(0.25)
    A = A.at[n - 2, n - 1].set(0.25)
    A = A.at[n - 2, n - 3].set(0.25)

    # Interior points
    interior = jnp.arange(2, n - 2)
    A = A.at[interior, interior - 1].set(jnp.ones(n - 4) * (1.0 / 3.0))
    A = A.at[interior, interior + 1].set(jnp.ones(n - 4) * (1.0 / 3.0))

    # Create RHS matrix B (maps function values to RHS vector)
    B = jnp.zeros((n, n))

    # Boundary points
    B = B.at[0, 0].set(-5.0 / (2.0 * h))
    B = B.at[0, 1].set(4.0 / (2.0 * h))
    B = B.at[0, 2].set(1.0 / (2.0 * h))

    B = B.at[n - 1, n - 1].set(5.0 / (2.0 * h))
    B = B.at[n - 1, n - 2].set(-4.0 / (2.0 * h))
    B = B.at[n - 1, n - 3].set(-1.0 / (2.0 * h))

    # Second point from boundary
    B = B.at[1, 0].set(-3.0 / (4.0 * h))
    B = B.at[1, 2].set(3.0 / (4.0 * h))

    B = B.at[n - 2, n - 3].set(-3.0 / (4.0 * h))
    B = B.at[n - 2, n - 1].set(3.0 / (4.0 * h))

    # Interior points
    interior = jnp.arange(2, n - 2)
    B = B.at[interior, interior + 1].set(14.0 / (18.0 * h))
    B = B.at[interior, interior - 1].set(-14.0 / (18.0 * h))
    B = B.at[interior, interior + 2].set(1.0 / (36.0 * h))
    B = B.at[interior, interior - 2].set(-1.0 / (36.0 * h))

    return A, B


def create_lele_D2_6_matrix(n, h=1.0):
    """
    Create 6th-order Lele compact approximation matrix for second derivative.

    Parameters
    ----------
    n : int
        Size of the matrix (number of grid points)
    h : float
        Grid spacing

    Returns
    -------
    A : jax.numpy.ndarray
        LHS matrix (for coefficients of derivatives)
    B : jax.numpy.ndarray
        RHS matrix (for coefficients of function values)
    """
    # Create LHS matrix A (tridiagonal)
    A = jnp.zeros((n, n))

    # Diagonal elements
    A = A.at[jnp.arange(n), jnp.arange(n)].set(jnp.ones(n))

    # Boundary conditions
    A = A.at[0, 1].set(11.0)
    A = A.at[n - 1, n - 2].set(11.0)

    # Second point from boundary
    A = A.at[1, 0].set(0.1)
    A = A.at[1, 2].set(0.1)
    A = A.at[n - 2, n - 1].set(0.1)
    A = A.at[n - 2, n - 3].set(0.1)

    # Interior points
    interior = jnp.arange(2, n - 2)
    A = A.at[interior, interior - 1].set(jnp.ones(n - 4) * (2.0 / 11.0))
    A = A.at[interior, interior + 1].set(jnp.ones(n - 4) * (2.0 / 11.0))

    # Create RHS matrix B (maps function values to RHS vector)
    B = jnp.zeros((n, n))

    # Boundary points
    B = B.at[0, 0].set(13.0 / h**2)
    B = B.at[0, 1].set(-27.0 / h**2)
    B = B.at[0, 2].set(15.0 / h**2)
    B = B.at[0, 3].set(-1.0 / h**2)

    B = B.at[n - 1, n - 1].set(13.0 / h**2)
    B = B.at[n - 1, n - 2].set(-27.0 / h**2)
    B = B.at[n - 1, n - 3].set(15.0 / h**2)
    B = B.at[n - 1, n - 4].set(-1.0 / h**2)

    # Second point from boundary
    B = B.at[1, 0].set(6.0 / (5.0 * h**2))
    B = B.at[1, 1].set(-12.0 / (5.0 * h**2))
    B = B.at[1, 2].set(6.0 / (5.0 * h**2))

    B = B.at[n - 2, n - 1].set(6.0 / (5.0 * h**2))
    B = B.at[n - 2, n - 2].set(-12.0 / (5.0 * h**2))
    B = B.at[n - 2, n - 3].set(6.0 / (5.0 * h**2))

    # Interior points
    interior = jnp.arange(2, n - 2)
    B = B.at[interior, interior].set(-2.0 * (12.0 / 11.0 + 3.0 / 44.0) / h**2)
    B = B.at[interior, interior + 1].set((12.0 / 11.0) / h**2)
    B = B.at[interior, interior - 1].set((12.0 / 11.0) / h**2)
    B = B.at[interior, interior + 2].set((3.0 / 44.0) / h**2)
    B = B.at[interior, interior - 2].set((3.0 / 44.0) / h**2)

    return A, B


def apply_compact_derivative(A, B, f):
    """
    Apply compact finite difference approximation to function values.

    Parameters
    ----------
    A : jax.numpy.ndarray
        LHS matrix (for coefficients of derivatives)
    B : jax.numpy.ndarray
        RHS matrix (for coefficients of function values)
    f : jax.numpy.ndarray
        Function values at grid points

    Returns
    -------
    df : jax.numpy.ndarray
        Derivative approximation
    """
    b = B @ f
    df = jnp.linalg.solve(A, b)
    return df
