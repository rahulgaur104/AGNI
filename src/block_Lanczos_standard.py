#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import lax, vmap, random
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def block_lanczos(A_matvec, n, k=10, block_size=20, max_iter=50, V0=None):
    """
    Block Lanczos algorithm for computing top-k eigenvalues/eigenvectors of symmetric matrix.
    
    Constructs a block Krylov subspace K = span{V, AV, A²V, ..., A^(m-1)V} where V is 
    a block of vectors, then projects A onto this subspace to form a block tridiagonal 
    matrix T whose eigenvalues approximate the extreme eigenvalues of A.
    
    Parameters
    ----------
    A_matvec : callable
       Function that computes matrix-vector product A @ v
       Input shape: (n,), Output shape: (n,)
    n : int
       Dimension of the matrix A
    k : int, default=10
       Number of eigenvalues/eigenvectors to compute
    block_size : int, default=20
       Size of vector blocks. Larger blocks improve convergence but increase memory
    max_iter : int, default=50
       Maximum number of Lanczos iterations. Total subspace size is block_size * max_iter
    V0 : array_like, shape (n,) or (n, m), optional
       Initial guess vectors. If shape (n,), treated as single vector.
       If shape (n, m), uses first m vectors. Remaining block_size-m vectors 
       initialized randomly if m < block_size.
    
    Returns
    -------
    eigvals : ndarray, shape (k,)
       Top k eigenvalues in ascending order
    eigvecs : ndarray, shape (n, k)
       Corresponding eigenvectors as columns
    
    Notes
    -----
    Block tridiagonal matrix T has structure:
       
       ┌─────┬─────┬─────┬─────┐
       │ α₀  │ β₀ᵀ │  0  │  0  │
       ├─────┼─────┼─────┼─────┤
       │ β₀  │ α₁  │ β₁ᵀ │  0  │
       ├─────┼─────┼─────┼─────┤
       │  0  │ β₁  │ α₂  │ β₂ᵀ │
       ├─────┼─────┼─────┼─────┤
       │  0  │  0  │ β₂  │ α₃  │
       └─────┴─────┴─────┴─────┘
       
    where αᵢ are block_size × block_size symmetric matrices and 
    βᵢ are block_size × block_size matrices.
    
    - Three-term recurrence: V_{j+1} = (A V_j - V_j α_j - V_{j-1} β_{j-1}ᵀ) R_j^{-1}
    - Uses QR factorization for numerical stability
    - All operations are JAX-jittable with no Python loops
    - More efficient than randomized methods for computing few extreme eigenvalues
    """
    # Initialize with guess or random
    if V0 is None:
        key = random.PRNGKey(0)
        V = random.normal(key, (n, block_size))
    else:
        # V0 shape: (n,) or (n, num_vecs)
        V0 = jnp.atleast_2d(V0.T).T  # ensure (n, num_vecs)
        if V0.shape[1] < block_size:
            key = random.PRNGKey(0)
            V_rand = random.normal(key, (n, block_size - V0.shape[1]))
            V = jnp.hstack([V0, V_rand])
        else:
            V = V0[:, :block_size]
    
    V, _ = jnp.linalg.qr(V)
    
    def step(carry, _):
        V_curr, V_prev, beta_prev = carry
        
        AV = vmap(A_matvec)(V_curr.T).T
        alpha = V_curr.T @ AV
        W = AV - V_curr @ alpha - V_prev @ beta_prev.T
        
        V_next, beta = jnp.linalg.qr(W)
        
        return (V_next, V_curr, beta), (alpha, beta, V_curr)
    
    _, (alphas, betas, Vs) = lax.scan(
        step, 
        (V, jnp.zeros_like(V), jnp.zeros((block_size, block_size))), 
        None, 
        max_iter
    )
 
    # Build T using advanced indexing
    T = jnp.zeros((max_iter * block_size, max_iter * block_size))
    
    # Diagonal blocks
    i, j = jnp.meshgrid(jnp.arange(max_iter), jnp.arange(block_size), indexing='ij')
    diag_idx = (i * block_size + j, i * block_size + jnp.arange(block_size))
    T = T.at[diag_idx].set(alphas)
    
    # Off-diagonal blocks
    i, j = jnp.meshgrid(jnp.arange(max_iter-1), jnp.arange(block_size), indexing='ij')
    upper_idx = (i * block_size + j, (i + 1) * block_size + jnp.arange(block_size))
    lower_idx = ((i + 1) * block_size + j, i * block_size + jnp.arange(block_size))
    T = T.at[upper_idx].set(betas[:-1].transpose(0, 2, 1))
    T = T.at[lower_idx].set(betas[:-1])
    
    # Solve
    eigvals, eigvecs = jnp.linalg.eigh(T)
    idx = jnp.argsort(eigvals)[-k:]
    
    V_full = Vs.transpose(1, 0, 2).reshape(n, -1)
    return eigvals[idx], V_full @ eigvecs[:, idx]
