#!/usr/bin/env python3

import jax
import jax.numpy as jnp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

###############################
#--CHEBYSHEV BASIS FUNCTIONS--#
###############################

def chebpts(n, domain=None):
    """Generate Chebyshev points of the first kind on the domain."""
    if n <= 0:
        return jnp.array([])
    if domain is None:
        domain = [-1, 1]
    k = jnp.arange(n)
    x = jnp.cos((2*k + 1) * jnp.pi / (2*n))
    # Scale to domain
    if domain[0] != -1 or domain[1] != 1:
        x = (domain[1] - domain[0]) / 2 * (x + 1) + domain[0]
    return x

def diffmat(n):
    """Differentiation matrix using Chebyshev points (vectorized, no loops)."""
    if n <= 0:
        return jnp.array([])
    
    # Chebyshev points
    x = chebpts(n)
    
    # Create meshgrid for computing differences
    X = jnp.tile(x[:, None], (1, n))  # Repeat x along columns
    dX = X - X.T  # Element-wise differences (x_i - x_j)
    
    # Set up the coefficients for boundaries
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[n-1].set(2.0)
    
    # Compute c_i/c_j factors
    C = jnp.outer(c, 1.0/c)
    
    # Compute (-1)^(i+j) matrix
    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0)**(I + J)
    
    # Compute off-diagonal elements
    D = jnp.where(dX != 0, C * S / dX, 0.0)
    
    # Compute diagonal elements (negative sum of rows)
    D = D.at[jnp.diag_indices(n)].set(0.0)  # Clear diagonal first
    row_sums = jnp.sum(D, axis=1)
    D = D.at[jnp.diag_indices(n)].set(-row_sums)
    
    return D


def diffmat2_standard(n):
    """
    Second derivative matrix using standard Chebyshev points (not Lobatto).
    """
    if n <= 1:
        return jnp.array([[0.0]])
    
    # Get standard Chebyshev points (first kind)
    x = chebpts(n)
    
    # Method 2: Directly compute the second derivative matrix
    # This is more accurate but more complicated
    # Create matrix of differences
    X = jnp.tile(x[:, None], (1, n))
    dX = X - X.T
    
    # Set up the coefficients for boundaries
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[n-1].set(2.0)
    
    # Compute c_i/c_j factors
    C = jnp.outer(c, 1.0/c)
    
    # Compute (-1)^(i+j) matrix
    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0)**(I + J)
    
    # Initialize the second derivative matrix
    D2_direct = jnp.zeros((n, n))
    
    # Off-diagonal elements
    mask = I != J
    # Formula for D2[i,j] with i≠j in Chebyshev differentiation
    # D2[i,j] = 2*C[i,j]*S[i,j]/(dX[i,j]^2) * (x[i]/(1-x[i]^2) - 2/(dX[i,j]))
    
    # This is a simplified version - for better accuracy, consult specialized literature
    D2_direct = jnp.where(mask, 
                          2 * C * S / (dX**2) * (X / (1 - X**2) - 2 / dX),
                          D2_direct)
    
    # Diagonal elements (for standard Chebyshev points)
    # These formulas are approximations based on the requirement that rows sum to zero
    D2_diag = -jnp.sum(D2_direct.at[jnp.diag_indices(n)].set(0.0), axis=1)
    D2_direct = D2_direct.at[jnp.diag_indices(n)].set(D2_diag)
    
    # Return both methods for comparison
    return D2_direct

def chebpts_lobatto(n, domain=None):
    """Generate Chebyshev-Lobatto points (includes endpoints)."""
    if n <= 1:
        return jnp.array([0.0])
    
    # Chebyshev-Lobatto points on [-1, 1]
    k = jnp.arange(n)
    x = jnp.cos(k * jnp.pi / (n - 1))
    
    # Scale to domain if needed
    if domain is not None:
        if domain[0] != -1 or domain[1] != 1:
            x = (domain[1] - domain[0]) / 2 * (x + 1) + domain[0]
    
    return x

def diffmat_lobatto(n):
    """Differentiation matrix using Chebyshev-Lobatto points (vectorized)."""
    if n <= 1:
        return jnp.array([[0.0]])
    
    # Get Lobatto points
    x = chebpts_lobatto(n)
    
    # Create matrix of differences
    X = jnp.tile(x[:, None], (1, n))  # Repeat x along columns
    dX = X - X.T  # Element-wise differences (x_i - x_j)
    
    # Handle diagonal separately (to avoid division by zero)
    D = jnp.zeros((n, n))
    
    # Off-diagonal elements
    i, j = jnp.mgrid[0:n, 0:n]
    mask = i != j
    
    # Compute c_j values (weights at the endpoints)
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[n-1].set(2.0)
    
    # Compute D[i,j] for i≠j
    # For Lobatto points: D[i,j] = (c_i/c_j) * (-1)^(i+j) / (x_i - x_j)
    D = jnp.where(mask, 
                 (c[i] / c[j]) * (-1.0)**(i+j) / dX, 
                 D)
    
    # Special case for corners
    D = D.at[0, 0].set((2 * (n-1)**2 + 1) / 6)
    D = D.at[n-1, n-1].set(-(2 * (n-1)**2 + 1) / 6)
    
    # Other diagonal elements
    for i in range(1, n-1):
        D = D.at[i, i].set(-x[i] / (2 * (1 - x[i]**2)))
    
    # Alternative diagonal computation: negative sum of rows
    # Instead of specialized formulas, we can use the fact that each row sums to zero
    D_diag = -jnp.sum(D.at[jnp.diag_indices(n)].set(0.0), axis=1)
    D = D.at[jnp.diag_indices(n)].set(D_diag)
    
    return D


#############################
#--FOURIER BASIS FUNCTIONS--#
#############################

def fourier_pts(n, domain=None):
    """
    Generate equally spaced points for Fourier spectral methods.
    
    Parameters:
    -----------
    n : int
        Number of points
    domain : list, optional
        Domain [a, b], default is [0, 2π]
        
    Returns:
    --------
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
    
    Parameters:
    -----------
    n : int
        Number of points
        
    Returns:
    --------
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
            i != j,
            0.5 * (-1)**(i-j) / jnp.tan((i-j) * jnp.pi / n),
            0.0
        )
    else:  # odd
        # Explicit formula for odd number of points
        D = jnp.where(
            i != j,
            0.5 * (-1)**(i-j) / jnp.sin((i-j) * jnp.pi / n),
            0.0
        )
    
    # Diagonal elements are zero (rows sum to zero)
    return D
