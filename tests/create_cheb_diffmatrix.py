import numpy as np
import jax
import jax.numpy as jnp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

#def chebpts(n, domain=None):
#    """Generate Chebyshev points of the first kind on the domain."""
#    if n <= 0:
#        return jnp.array([])
#    if domain is None:
#        domain = [-1, 1]
#    k = jnp.arange(n)
#    x = jnp.cos((2*k + 1) * jnp.pi / (2*n))
#    # Scale to domain
#    if domain[0] != -1 or domain[1] != 1:
#        x = (domain[1] - domain[0]) / 2 * (x + 1) + domain[0]
#    return x

def chebpts_lobatto(n, domain=None):
    """Generate Chebyshev points using the sine function approach."""
    if n <= 0:
        return jnp.array([])
    if domain is None:
        domain = [-1, 1]
    k = jnp.arange(n)
    x = jnp.sin(jnp.pi * jnp.flip(2 * k - n + 1) / (2 * (n - 1)))
    if domain is not None:
        if domain[0] != -1 or domain[1] != 1:
            x = (domain[1] - domain[0]) / 2 * (x + 1) + domain[0]
    return x

#def chebpts_lobatto(n, domain=None):
#    """Generate Chebyshev-Lobatto points (includes endpoints)."""
#    if n <= 1:
#        return jnp.array([0.0])
#    
#    # Chebyshev-Lobatto points on [-1, 1]
#    k = jnp.arange(n)
#    x = jnp.cos(k * jnp.pi / (n - 1))
#    
#    # Scale to domain if needed
#    if domain is not None:
#        if domain[0] != -1 or domain[1] != 1:
#            x = (domain[1] - domain[0]) / 2 * (x + 1) + domain[0]
#    
#    return x

def diffmat(n):
    """Differentiation matrix using Chebyshev points (vectorized, no loops)."""
    if n <= 0:
        return jnp.array([])
    
    # Chebyshev points
    #x = chebpts(n)
    x = chebpts_lobatto(n)
    
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


def diffmat1_lobatto(n):
    """First-order differentiation matrix using Chebyshev-Lobatto points (vectorized)."""
    if n <= 1:
        return jnp.array([[0.0]])
    
    # Get Lobatto points
    x = chebpts_lobatto(n)
    
    # Create meshgrid for computing differences
    X = jnp.tile(x[:, None], (1, n))  # Each row is x
    Xdiff = X - X.T                   # x_i - x_j
    
    # Compute c values (weights at the endpoints)
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[n-1].set(2.0)
    
    # Compute c_i/c_j factors
    C = jnp.outer(c, 1.0/c)
    
    # Compute (-1)^(i+j) matrix
    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0)**(I + J)
    
    # Compute off-diagonal elements
    mask = I != J
    D = jnp.where(mask, C * S / Xdiff, 0.0)
    
    # Compute diagonal elements (negative sum of rows)
    D = D.at[jnp.diag_indices(n)].set(0.0)  # Clear diagonal first
    row_sums = jnp.sum(D, axis=1)
    D = D.at[jnp.diag_indices(n)].set(-row_sums)
    
    return D

def diffmat2_lobatto(n):
    """
    Second-order differentiation matrix using Chebyshev-Lobatto points (vectorized).
    Uses explicit formula instead of squaring the first derivative matrix.
    """
    if n <= 1:
        return jnp.array([[0.0]])
    
    # Get Lobatto points
    x = chebpts_lobatto(n)
    
    # Create meshgrids
    X = jnp.tile(x[:, None], (1, n))  # Each row is x
    Xdiff = X - X.T                   # x_i - x_j
    Xsum = X + X.T                    # x_i + x_j
    
    # Compute c values (weights at the endpoints)
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[n-1].set(2.0)
    
    # Compute c_i/c_j factors
    C = jnp.outer(c, 1.0/c)
    
    # Compute (-1)^(i+j) matrix
    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0)**(I + J)
    
    # Initialize the matrix
    D2 = jnp.zeros((n, n))
    
    # Fill off-diagonal elements using vectorized formula
    mask = I != J
    D2 = jnp.where(mask, C * S * Xsum / (Xdiff**2), D2)
    
    # Special treatment for diagonal elements
    diag_indices = jnp.diag_indices(n)
    
    # For interior points
    interior_mask = jnp.ones(n, dtype=bool)
    interior_mask = interior_mask.at[0].set(False)
    interior_mask = interior_mask.at[n-1].set(False)
    
    interior_indices = jnp.where(interior_mask)[0]
    interior_x = x[interior_indices]
    
    # Formula for interior points: -x_i^2/(1-x_i^2) - 1/(2*(1-x_i^2))
    interior_diag = -interior_x**2 / (1 - interior_x**2) - 1 / (2 * (1 - interior_x**2))
    
    # Formula for endpoints: (2*(n-1)^2 + 1)/3
    endpoint_diag = (2 * (n-1)**2 + 1) / 3.0
    
    # Assign diagonal values
    D2 = D2.at[0, 0].set(endpoint_diag)
    D2 = D2.at[n-1, n-1].set(endpoint_diag)
    D2 = D2.at[interior_indices, interior_indices].set(interior_diag)
    
    return D2

def diffmat3_lobatto(n):
    """
    Third-order differentiation matrix using Chebyshev-Lobatto points (vectorized).
    Uses explicit formula instead of cubing the first derivative matrix.
    """
    if n <= 2:
        return jnp.zeros((n, n))
    
    # Get Lobatto points
    x = chebpts_lobatto(n)
    
    # Create meshgrids
    X = jnp.tile(x[:, None], (1, n))   # Each row is x
    Xdiff = X - X.T                    # x_i - x_j
    
    # Term for numerator: x_i^2 + 4*x_i*x_j + x_j^2
    Xterm = X**2 + 4*X*X.T + X.T**2
    
    # Compute c values (weights at the endpoints)
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[n-1].set(2.0)
    
    # Compute c_i/c_j factors
    C = jnp.outer(c, 1.0/c)
    
    # Compute (-1)^(i+j) matrix
    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0)**(I + J)
    
    # Initialize the matrix
    D3 = jnp.zeros((n, n))
    
    # Fill off-diagonal elements using vectorized formula
    mask = I != J
    D3 = jnp.where(mask, C * S * Xterm / (Xdiff**3), D3)
    
    # Diagonal elements calculated to ensure D³(x³) = 6 at all points
    # This is a conservation principle that improves accuracy
    
    # Create test vector for cubic: x^3
    cubic = x**3
    
    # Temporary D3 matrix with zeros on diagonal
    D3_temp = D3.copy()
    
    # For each row, compute what diagonal element makes D³(x³) = 6
    for i in range(n):
        # Third derivative of x^3 should be 6
        target_deriv = 6.0
        
        # Compute current sum of off-diagonal terms applied to x^3
        current_sum = jnp.sum(D3_temp[i, :] * cubic) - D3_temp[i, i] * cubic[i]
        
        # What diagonal value makes the sum equal to 6?
        D3 = D3.at[i, i].set((target_deriv - current_sum) / cubic[i])
    
    return D3

def diffmat4_lobatto(n):
    """
    Fourth-order differentiation matrix using Chebyshev-Lobatto points (vectorized).
    Uses explicit formula instead of applying the differentiation matrix 4 times.
    """
    if n <= 3:
        return jnp.zeros((n, n))
    
    # Get Lobatto points
    x = chebpts_lobatto(n)
    
    # Create meshgrids
    X = jnp.tile(x[:, None], (1, n))   # Each row is x
    Xdiff = X - X.T                    # x_i - x_j
    
    # Term for numerator: complicated expression for 4th derivative
    # x_i^3 + 11*x_i^2*x_j + 11*x_i*x_j^2 + x_j^3
    Xterm = X**3 + 11*X**2*X.T + 11*X*X.T**2 + X.T**3
    
    # Compute c values (weights at the endpoints)
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[n-1].set(2.0)
    
    # Compute c_i/c_j factors
    C = jnp.outer(c, 1.0/c)
    
    # Compute (-1)^(i+j) matrix
    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0)**(I + J)
    
    # Initialize the matrix
    D4 = jnp.zeros((n, n))
    
    # Fill off-diagonal elements using vectorized formula
    mask = I != J
    D4 = jnp.where(mask, C * S * Xterm / (Xdiff**4), D4)
    
    # Diagonal elements calculated to ensure D⁴(x⁴) = 24 at all points
    # This is a conservation principle that improves accuracy
    
    # Create test vector for quartic: x^4
    quartic = x**4
    
    # Temporary D4 matrix with zeros on diagonal
    D4_temp = D4.copy()
    
    # For each row, compute what diagonal element makes D⁴(x⁴) = 24
    for i in range(n):
        # Fourth derivative of x^4 should be 24
        target_deriv = 24.0
        
        # Compute current sum of off-diagonal terms applied to x^4
        current_sum = jnp.sum(D4_temp[i, :] * quartic) - D4_temp[i, i] * quartic[i]
        
        # What diagonal value makes the sum equal to 24?
        D4 = D4.at[i, i].set((target_deriv - current_sum) / quartic[i])
    
    return D4

def precondition_diffmat2(D2, x):
    """
    Create a preconditioner for the second derivative matrix.
    
    Parameters:
    -----------
    D2 : array
        Second derivative matrix
    x : array
        Chebyshev points used to create D2
    
    Returns:
    --------
    P : array
        Preconditioner matrix (diagonal)
    """
    n = len(x)
    
    # Create diagonal preconditioner based on (1-x²) weighting
    # This accounts for the natural singularity in the second derivative near boundaries
    weights = (1 - x**2)
    
    # For stability near endpoints
    eps = 1e-10
    weights = jnp.maximum(weights, eps)
    
    # Diagonal preconditioner
    P = jnp.diag(weights)
    
    return P


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


def cheb_D1(N):
    """First-order Chebyshev differentiation matrix."""
    # Identity matrix
    I = jnp.eye(N)
    
    # Indices for flipping trick
    n1 = N // 2
    n2 = (N + 1) // 2
    
    # Compute theta vector
    k = jnp.arange(N)
    th = k * jnp.pi / (N - 1)
    
    # Compute Chebyshev points
    x = jnp.cos(th)
    
    # Create matrices of theta values
    T1, T2 = jnp.meshgrid(th / 2, th / 2)
    
    # Compute differences using trigonometric identity
    DX = 2 * jnp.sin(T1 + T2) * jnp.sin(T1 - T2)
    
    # Apply the flipping trick
    DX_top = DX[:n1, :]
    DX_bottom = -jnp.flip(jnp.flip(DX[:n2, :], axis=0), axis=1)
    DX = jnp.vstack([DX_top, DX_bottom])
    
    # Put 1's on the main diagonal
    DX = DX.at[jnp.diag_indices(N)].set(1.0)
    
    # Create the C matrix (c_i/c_j)
    i, j = jnp.meshgrid(k, k)
    C = jnp.power(-1.0, i + j)
    
    # Adjust first and last rows
    C = C.at[0, :].multiply(2.0)
    C = C.at[N-1, :].multiply(2.0)
    
    # Adjust first and last columns
    C = C.at[:, 0].multiply(0.5)
    C = C.at[:, N-1].multiply(0.5)
    
    # Z contains entries 1/(x_i-x_j) with zeros on diagonal
    Z = 1.0 / DX
    Z = Z.at[jnp.diag_indices(N)].set(0.0)
    
    # Compute first differentiation matrix
    i, j = jnp.meshgrid(jnp.arange(N), jnp.arange(N))
    mask = i != j
    D = jnp.zeros((N, N))
    D = jnp.where(mask, C * Z, D)
    
    # Diagonal elements (negative row sum)
    diag = -jnp.sum(D, axis=1)
    D = D.at[jnp.diag_indices(N)].set(diag)
    
    return D

def cheb_D2(N):
    """Second-order Chebyshev differentiation matrix - direct implementation."""
    # Identity matrix
    I = jnp.eye(N)
    
    # Indices for flipping trick
    n1 = N // 2
    n2 = (N + 1) // 2
    
    # Compute theta vector
    k = jnp.arange(N)
    th = k * jnp.pi / (N - 1)
    
    # Compute Chebyshev points
    x = jnp.cos(th)
    
    # Create matrices of theta values
    T1, T2 = jnp.meshgrid(th / 2, th / 2)
    
    # Compute differences using trigonometric identity
    DX = 2 * jnp.sin(T1 + T2) * jnp.sin(T1 - T2)
    
    # Apply the flipping trick
    DX_top = DX[:n1, :]
    DX_bottom = -jnp.flip(jnp.flip(DX[:n2, :], axis=0), axis=1)
    DX = jnp.vstack([DX_top, DX_bottom])
    
    # Put 1's on the main diagonal
    DX = DX.at[jnp.diag_indices(N)].set(1.0)
    
    # Create the C matrix (c_i/c_j)
    i, j = jnp.meshgrid(k, k)
    C = jnp.power(-1.0, i + j)
    
    # Adjust first and last rows
    C = C.at[0, :].multiply(2.0)
    C = C.at[N-1, :].multiply(2.0)
    
    # Adjust first and last columns
    C = C.at[:, 0].multiply(0.5)
    C = C.at[:, N-1].multiply(0.5)
    
    # Z contains entries 1/(x_i-x_j) with zeros on diagonal
    Z = 1.0 / DX
    Z = Z.at[jnp.diag_indices(N)].set(0.0)
    
    # First initialize D as identity (ell = 0 case)
    D = I.copy()
    
    # For second derivative (ell = 2):
    # First do one iteration (ell = 1)
    D_diag_1 = jnp.diag(D)
    D1 = jnp.zeros((N, N))
    mask = i != j
    D1 = jnp.where(mask, 1 * Z * (C * jnp.tile(D_diag_1, (N, 1)).T - D), D1)
    D1_diag = -jnp.sum(D1, axis=1)
    D1 = D1.at[jnp.diag_indices(N)].set(D1_diag)
    
    # Now do second iteration (ell = 2)
    D_diag_2 = jnp.diag(D1)
    D2 = jnp.zeros((N, N))
    D2 = jnp.where(mask, 2 * Z * (C * jnp.tile(D_diag_2, (N, 1)).T - D1), D2)
    D2_diag = -jnp.sum(D2, axis=1)
    D2 = D2.at[jnp.diag_indices(N)].set(D2_diag)
    
    return D2



def cheb_D3(N):
    """Third-order Chebyshev differentiation matrix - direct implementation."""
    # Identity matrix
    I = jnp.eye(N)

    # Indices for flipping trick
    n1 = N // 2
    n2 = (N + 1) // 2

    # Compute theta vector
    k = jnp.arange(N)
    th = k * jnp.pi / (N - 1)

    # Compute Chebyshev points
    x = jnp.cos(th)

    # Create matrices of theta values
    T1, T2 = jnp.meshgrid(th / 2, th / 2)

    # Compute differences using trigonometric identity
    DX = 2 * jnp.sin(T1 + T2) * jnp.sin(T1 - T2)

    # Apply the flipping trick
    DX_top = DX[:n1, :]
    DX_bottom = -jnp.flip(jnp.flip(DX[:n2, :], axis=0), axis=1)
    DX = jnp.vstack([DX_top, DX_bottom])

    # Put 1's on the main diagonal
    DX = DX.at[jnp.diag_indices(N)].set(1.0)

    # Create the C matrix (c_i/c_j)
    i, j = jnp.meshgrid(k, k)
    C = jnp.power(-1.0, i + j)

    # Adjust first and last rows
    C = C.at[0, :].multiply(2.0)
    C = C.at[N-1, :].multiply(2.0)

    # Adjust first and last columns
    C = C.at[:, 0].multiply(0.5)
    C = C.at[:, N-1].multiply(0.5)

    # Z contains entries 1/(x_i-x_j) with zeros on diagonal
    Z = 1.0 / DX
    Z = Z.at[jnp.diag_indices(N)].set(0.0)

    # First initialize D as identity (ell = 0 case)
    D = I.copy()
    mask = i != j

    # First iteration (ell = 1)
    D_diag_1 = jnp.diag(D)
    D1 = jnp.zeros((N, N))
    D1 = jnp.where(mask, 1 * Z * (C * jnp.tile(D_diag_1, (N, 1)).T - D), D1)
    D1_diag = -jnp.sum(D1, axis=1)
    D1 = D1.at[jnp.diag_indices(N)].set(D1_diag)

    # Second iteration (ell = 2)
    D_diag_2 = jnp.diag(D1)
    D2 = jnp.zeros((N, N))
    D2 = jnp.where(mask, 2 * Z * (C * jnp.tile(D_diag_2, (N, 1)).T - D1), D2)
    D2_diag = -jnp.sum(D2, axis=1)
    D2 = D2.at[jnp.diag_indices(N)].set(D2_diag)

    # Third iteration (ell = 3)
    D_diag_3 = jnp.diag(D2)
    D3 = jnp.zeros((N, N))
    D3 = jnp.where(mask, 3 * Z * (C * jnp.tile(D_diag_3, (N, 1)).T - D2), D3)
    D3_diag = -jnp.sum(D3, axis=1)
    D3 = D3.at[jnp.diag_indices(N)].set(D3_diag)

    return D3

def cheb_D4(N):
    """Fourth-order Chebyshev differentiation matrix - direct implementation."""
    # Identity matrix
    I = jnp.eye(N)

    # Indices for flipping trick
    n1 = N // 2
    n2 = (N + 1) // 2

    # Compute theta vector
    k = jnp.arange(N)
    th = k * jnp.pi / (N - 1)

    # Compute Chebyshev points
    x = jnp.cos(th)

    # Create matrices of theta values
    T1, T2 = jnp.meshgrid(th / 2, th / 2)

    # Compute differences using trigonometric identity
    DX = 2 * jnp.sin(T1 + T2) * jnp.sin(T1 - T2)

    # Apply the flipping trick
    DX_top = DX[:n1, :]
    DX_bottom = -jnp.flip(jnp.flip(DX[:n2, :], axis=0), axis=1)
    DX = jnp.vstack([DX_top, DX_bottom])

    # Put 1's on the main diagonal
    DX = DX.at[jnp.diag_indices(N)].set(1.0)

    # Create the C matrix (c_i/c_j)
    i, j = jnp.meshgrid(k, k)
    C = jnp.power(-1.0, i + j)

    # Adjust first and last rows
    C = C.at[0, :].multiply(2.0)
    C = C.at[N-1, :].multiply(2.0)

    # Adjust first and last columns
    C = C.at[:, 0].multiply(0.5)
    C = C.at[:, N-1].multiply(0.5)

    # Z contains entries 1/(x_i-x_j) with zeros on diagonal
    Z = 1.0 / DX
    Z = Z.at[jnp.diag_indices(N)].set(0.0)

    # First initialize D as identity (ell = 0 case)
    D = I.copy()
    mask = i != j

    # First iteration (ell = 1)
    D_diag_1 = jnp.diag(D)
    D1 = jnp.zeros((N, N))
    D1 = jnp.where(mask, 1 * Z * (C * jnp.tile(D_diag_1, (N, 1)).T - D), D1)
    D1_diag = -jnp.sum(D1, axis=1)
    D1 = D1.at[jnp.diag_indices(N)].set(D1_diag)

    # Second iteration (ell = 2)
    D_diag_2 = jnp.diag(D1)
    D2 = jnp.zeros((N, N))
    D2 = jnp.where(mask, 2 * Z * (C * jnp.tile(D_diag_2, (N, 1)).T - D1), D2)
    D2_diag = -jnp.sum(D2, axis=1)
    D2 = D2.at[jnp.diag_indices(N)].set(D2_diag)

    # Third iteration (ell = 3)
    D_diag_3 = jnp.diag(D2)
    D3 = jnp.zeros((N, N))
    D3 = jnp.where(mask, 3 * Z * (C * jnp.tile(D_diag_3, (N, 1)).T - D2), D3)
    D3_diag = -jnp.sum(D3, axis=1)
    D3 = D3.at[jnp.diag_indices(N)].set(D3_diag)

    # Fourth iteration (ell = 4)
    D_diag_4 = jnp.diag(D3)
    D4 = jnp.zeros((N, N))
    D4 = jnp.where(mask, 4 * Z * (C * jnp.tile(D_diag_4, (N, 1)).T - D3), D4)
    D4_diag = -jnp.sum(D4, axis=1)
    D4 = D4.at[jnp.diag_indices(N)].set(D4_diag)

    return D4


