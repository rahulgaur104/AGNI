#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from create_fourier_diffmatrix import *

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)

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


#def fourier_diffmat2(N):
#    """
#    Second-order Fourier differentiation matrix.
#    
#    This implementation exactly follows the formula from equation (3.11) in the text:
#    
#    S''_N(x_j) = {
#        -π²/(3h²) - 1/6,         j ≡ 0 (mod N),
#        -(-1)^j/(2sin²(jh/2)),   j ≠ 0 (mod N)
#    }
#    
#    Parameters:
#    -----------
#    N : int
#        Size of differentiation matrix
#        
#    Returns:
#    --------
#    D2 : array
#        Second-order differentiation matrix (N × N)
#    """
#    h = 2.0 * jnp.pi / N
#    
#    # Create the first column
#    col1 = jnp.zeros(N)
#    
#    # Set the diagonal element (j=0)
#    col1 = col1.at[0].set(-jnp.pi**2 / (3.0 * h**2) - 1.0/6.0)
#    
#    # Set the off-diagonal elements (j ≠ 0)
#    j_indices = jnp.arange(1, N)
#    sin_term = jnp.sin(j_indices * h / 2.0)
#    sin_squared = sin_term**2
#    values = -(-1.0)**j_indices / (2.0 * sin_squared)
#    col1 = col1.at[j_indices].set(values)
#    
#    # Construct the full matrix
#    indices = jnp.arange(N)
#    D2 = jax.vmap(lambda i: jnp.roll(col1, i))(indices)
#    
#    return D2


def test_fourier_diffmat2():
    """
    Test the second-order differentiation matrix with analytical functions.
    
    1. sin(x): second derivative is -sin(x)
    2. cos(2x): second derivative is -4*cos(2x)
    """
    # Parameters
    N = 1024  # Number of grid points
    domain = [0, 2 * jnp.pi]
    #domain = [-jnp.pi,  jnp.pi]
    
    # Create grid points
    x = fourier_pts(N, domain)
    
    # Create differentiation matrix
    D2 = fourier_diffmat2(N)
    
    # Test Case 1: sin(x)
    print("\nTest with sin(x):")
    a = 50
    f1 = jnp.cos(4*x) * jnp.exp(-a * (x-jnp.pi)**2)
    exp_term = jnp.exp(-a * (x-jnp.pi)**2)
    sin_term = 16 * a * (x-jnp.pi) * jnp.sin(4 * x)
    cos_term = (4 * a**2 * (x-jnp.pi)**2 - 2 * a - 16) * jnp.cos(4 * (x-jnp.pi))

    f1_xx_exact = exp_term * (sin_term + cos_term)
    f1_xx_numerical = D2 @ f1
    
    max_error1 = jnp.max(jnp.abs(f1_xx_exact - f1_xx_numerical))
    print(f"  Maximum error for sin(x): {max_error1:.4e}")
    
    # Test Case 2: cos(2x)
    print("\nTest with cos(2x):")
    f2 = jnp.cos(2*x)
    f2_xx_exact = -4 * jnp.cos(2*x)
    f2_xx_numerical = D2 @ f2
    
    max_error2 = jnp.max(jnp.abs(f2_xx_exact - f2_xx_numerical))
    print(f"  Maximum error for cos(2x): {max_error2:.4e}")
    
    # Test Case 3: sin(3x)
    print("\nTest with sin(3x):")
    f3 = jnp.sin(3*x)
    f3_xx_exact = -9 * jnp.sin(3*x)
    f3_xx_numerical = D2 @ f3
    
    max_error3 = jnp.max(jnp.abs(f3_xx_exact - f3_xx_numerical))
    print(f"  Maximum error for sin(3x): {max_error3:.4e}")
    
    # Plot the results for visualization
    plt.figure(figsize=(12, 8))
    
    # Plot sin(x) results
    plt.subplot(3, 1, 1)
    plt.plot(x, f1_xx_exact, 'b-', label='Exact')
    plt.plot(x, f1_xx_numerical, 'ro', label='Numerical')
    plt.title('Second Derivative of sin(x)')
    plt.legend()
    plt.grid(True)
    
    # Plot cos(2x) results
    plt.subplot(3, 1, 2)
    plt.plot(x, f2_xx_exact, 'b-', label='Exact')
    plt.plot(x, f2_xx_numerical, 'ro', label='Numerical')
    plt.title('Second Derivative of cos(2x)')
    plt.legend()
    plt.grid(True)
    
    # Plot sin(3x) results
    plt.subplot(3, 1, 3)
    plt.plot(x, f3_xx_exact, 'b-', label='Exact')
    plt.plot(x, f3_xx_numerical, 'ro', label='Numerical')
    plt.title('Second Derivative of sin(3x)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return max_error1, max_error2, max_error3

# Main execution
if __name__ == "__main__":
    # Run the test
    errors = test_fourier_diffmat2()
    print("\nSummary of maximum errors:")
    print(f"  sin(x): {errors[0]:.4e}")
    print(f"  cos(2x): {errors[1]:.4e}")
    print(f"  sin(3x): {errors[2]:.4e}")
