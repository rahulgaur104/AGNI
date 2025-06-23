import jax
import jax.numpy as jnp
from jax import random, lax, vmap
from functools import partial

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
    - operations are JAX-jittable with no Python loops
    """
    key = random.PRNGKey(0)
    l = k + p  # Oversampling

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

    #random matrix (3LMN × l)
    Omega = random.normal(key, (n, l))

    # Batched Power iterations loopless
    def power_iter(Y):
        Y = vmap(A_matvec)(Y.T).T  #add batching?
        Q, _ = jnp.linalg.qr(Y)
        return Q

    #Power iterations w/o accumulating gradients
    Y, _ = lax.scan(
        lambda Y, _: (power_iter(Y), None),
        Omega,
        None,
        length=q
    )

    #Projectto low-rank basis
    Q, _ = jnp.linalg.qr(Y)
    AQ = vmap(A_matvec)(Q.T).T  # Batched A @ Q
    S = Q.T @ AQ  # Small l × l matrix

    # Solve eigenvalue problem
    eigvals, V = jnp.linalg.eigh(S)
    return eigvals[-k:], Q @ V[:, -k:]  # Top k eigenvalues/vectors



residuals = jnp.linalg.norm(A_matvec(eigvecs) - eigvals * eigvecs, axis=0)
print("Max residual:", jnp.max(residuals))  # Should be < 1e-6
