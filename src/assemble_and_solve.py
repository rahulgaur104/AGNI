#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from create_fourier_diffmatrix import *
from create_chebyshev_diffmatrix import *

"""
Build the full operator matrix for the spectral problem 
using tensor products.
+-----------+-----------+----------+
|           |           |          |
| A_ρρ      | A_ρθ      | Aρζ      |
|           |           |          |
+-----------+-----------+----------+
|           |           |          |
| A_θρ      | A_θθ      | A_θζ     |
| =  A_ρθ   |           |          |
+-----------+-----------+----------+
|           |           |          |
| A_ζρ      | A_ζθ      | A_ζζ     |
| = A_θρ    | = A_θζ    |          |
+-----------+-----------+----------+
Parameters:
- D_rho: 1D differentiation matrix for rho
- D_theta: 1D differentiation matrix for theta
- D_zeta: 1D differentiation matrix for zeta
- E: Equilibrium quantities at collocation points
"""
# Get dimensions from input matrices
n_rho = D_rho.shape[0]
n_theta = D_theta.shape[0]
n_zeta = D_zeta.shape[0]
n_total = n_rho * n_theta * n_zeta

rho_pts = chebpts_lobatto(n_rho)

# Differentiation matrix size (n_rho x n_rho)
D_rho0 = diffmat_lobatto(n_rho)

# Generate points on [0, 2π] 
theta_pts = fourier_pts(n_theta)
zeta_pts = fourier_pts(n_zeta)

# Get differentiation matrices
D_theta0 = fourier_diffmat(n_theta)

# Get differentiation matrices
D_zeta0 = fourier_diffmat(n_zeta)

I_rho0 = jnp.eye(n_rho)
I_theta0 = jnp.eye(n_theta)
I_zeta0 = jnp.eye(n_zeta)

D_rho0_T = D_rho0.T
D_theta0_T = D_theta0.T
D_zeta0_T = D_zeta0.T

D_rho = jnp.kron(I_zeta0, jnp.kron(I_theta0, D_rho0))
D_rho_T = jnp.kron(I_zeta0, jnp.kron(I_theta0, D_rho0_T))

D_theta = jnp.kron(I_zeta0, jnp.kron(D_theta0, I_rho0))
D_theta_T = jnp.kron(I_zeta0, jnp.kron(D_theta0_T, I_rho0))

D_zeta = jnp.kron(D_zeta0, jnp.kron(I_theta0, I_rho0))
D_zeta_T = jnp.kron(D_zeta0_T, jnp.kron(I_theta0, I_rho0))

# Compute all operator building A using tensor products
# Same coordinate operators (x = x')
D_rho_T_D_rho     = D_rho0_T @ D_rho0        # (n_ρ n_ρ x n_ρ n_ρ)
D_theta_T_D_theta = D_theta0_T @ D_theta0
D_zeta_T_D_zeta   = D_zeta0_T @ D_zeta0

# Mixed coordinate operators(x != x')
# Cannot calculate the kronecker product in all three dimensions
# as storing it will take all the GPU memory and throw an OOM!
D_rho_T_D_theta  = jnp.kron(D_theta0, D_rho0_T)     # (n_θ n_ρ x n_θ n_ρ)  
D_theta_T_D_rho  = jnp.kron(D_theta0_T, D_rho0)

D_rho_T_D_zeta   = jnp.kron(D_zeta0,  D_rho0.T)     # (n_ζ n_ρ x n_ζ n_ρ)
D_zeta_T_D_rho   = jnp.kron(D_zeta0_T, D_rho0))

D_theta_T_D_zeta = jnp.kron(D_zeta0, D_theta0_T)    # (n_ζ n_θ x n_ζ n_θ)
D_zeta_T_D_theta = jnp.kron(D_zeta0_T, D_theta0)


# We calcate the product of the giant matrix  (3LMN, 3LMN) with
# the the vector v (3 LMN, 1) without explicitly storing A
def A_matvec(v):
    """
    Return the product A @ v without ever materializing the matrix A.
    """
    v = jnp.transpose(v, (0, 2, 1))
    v0 = jnp.reshape(v, (n_rho, n_theta * n_zeta))

    v = jnp.transpose(v, (0, 2, 1))
    v1 = jnp.reshape(v, (n_theta, n_zeta * n_rho))

    v = jnp.transpose(v, (0, 2, 1))
    v2 = jnp.reshape(v, (n_zeta, n_rho * n_theta))

    v = jnp.transpose(v, (0, 2, 1))
    v3 = jnp.reshape(v, (n_rho * n_theta, n_zeta))

    v = jnp.transpose(v, (0, 2, 1))
    v4 = jnp.reshape(v, (n_rho * n_theta, n_zeta))
    
    v = jnp.transpose(v, (0, 2, 1))
    v5 = jnp.reshape(v, (n_rho * n_theta, n_zeta))

    # Now we calcualte all the terms of the form geometric 
    # coefficients x differentation matrix x v 

    return matvec_product


@partial(jax.jit, static_argnums=(1, 2, 3))
def randomized_svd(A_matvec, n, k=5, p=10, q=2):
    """
    Randomized SVD for computing top-k eigenvalues/eigenvectors of symmetric matrix.
    
    Uses randomized subspace iteration with optional initial guess vectors to
    accelerate convergence.
    
    Parameters
    ----------
    A_matvec : callable
       Function that computes matrix-vector product A @ v
       Input shape: (n,), Output shape: (n,)
    n : int
       Dimension of the matrix A
    k : int, default=5
       Number of eigenvalues/eigenvectors to compute
    p : int, default=10
       Oversampling parameter. Total subspace size is k + p
    q : int, default=2
       Number of power iterations for improved accuracy
    V0 : array_like, shape (n,) or (n, m), optional
       Initial guess vectors. If shape (n,), treated as single vector.
       If shape (n, m), uses first m vectors as initial guesses.
       Remaining k+p-m vectors initialized randomly if m < k+p.
    
    Returns
    -------
    eigvals : ndarray, shape (k,)
       Top k eigenvalues in ascending order
    eigvecs : ndarray, shape (n, k)
       Corresponding eigenvectors as columns
    
    Notes
    -----
    - Algorithm uses QR-based power iteration for numerical stability
    - For symmetric positive definite matrices, eigenvalues equal squared singular values
    - Initial guesses can significantly reduce iterations needed for convergence
    - All operations are JAX-jittable with no Python loops
    """
    key = random.PRNGKey(0)
    l = k + p  #Oversample

    #Initialize with guess or random
    if V0 is None:
        key = random.PRNGKey(0)
        Omega = random.normal(key, (n, l))
    else:
        # ensure V0 is 2D
        V0 = jnp.atleast_2d(V0.T).T  # shape (n, num_vecs)
        if V0.shape[1] >= l:
            # Have enough initial vectors
            Omega = V0[:, :l]
        else:
            # Need to add random vectors
            key = random.PRNGKey(0)
            n_random = l - V0.shape[1]
            Omega_random = random.normal(key, (n, n_random))
            Omega = jnp.hstack([V0, Omega_random])

    # Gaussian random matrix (3LMN × l)
    Omega = random.normal(key, (n, l))

    # Batched Power iterations loopless
    def power_iter(Y):
        Y = vmap(A_matvec)(Y.T).T  # Should have a batching size optn.
        Q, _ = jnp.linalg.qr(Y)
        return Q

    # Scan through power iterations w/o accumulating gradients
    Y, _ = lax.scan(
        lambda Y, _: (power_iter(Y), None),
        Omega,
        None,
        length=q
    )

    # Project A to low-rank basis
    Q, _ = jnp.linalg.qr(Y)
    AQ = vmap(A_matvec)(Q.T).T  # Batched A @ Q
    S = Q.T @ AQ  # Small l × l matrix

    # Solve eigenvalue problem
    eigvals, V = jnp.linalg.eigh(S)
    return eigvals[-k:], Q @ V[:, -k:]  # Top k eigenvalues/vectors


residuals = jnp.linalg.norm(A_matvec(eigvecs) - eigvals * eigvecs, axis=0)
print("Max residual:", jnp.max(residuals))  # Should be < 1e-6




