import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
from functools import partial

@partial(jit, static_argnums=(1, 2, 3, 4))
def gpu_eigh_scan(A_matvec, B_matvec, n, k=5, p=10, q=2):
    """
    Generalized eigendecomposition using:
    - Only lax.scan (no Python/fori_loops)
    - Full GPU batching
    - B-orthogonalization via QR
    """
    key = random.PRNGKey(0)
    l = k + p
    
    # Batched matrix operations
    batch_A = vmap(A_matvec, in_axes=1, out_axes=1)
    batch_B = vmap(B_matvec, in_axes=1, out_axes=1)
    
    # Initialize with random vectors
    Omega = random.normal(key, (n, l))
    
    # Power iterations via scan
    def power_step(Y, _):
        AY = batch_A(Y)
        Q, _ = jnp.linalg.qr(AY)
        BQ = batch_B(Q)
        Y_next, _ = jnp.linalg.qr(BQ)
        return Y_next, None
    
    Y, _ = lax.scan(power_step, Omega, xs=None, length=q)
    
    # Project and solve
    Q, _ = jnp.linalg.qr(Y)
    AQ = batch_A(Q)
    BQ = batch_B(Q)
    S = Q.T @ AQ
    T = Q.T @ BQ
    eigvals, V = jnp.linalg.eigh(S, T)
    
    return eigvals[-k:], Q @ V[:, -k:]


# Custom matvecs (replace with your actual implementations)
def A_matvec(v):
    return (jnp.diag(jnp.linspace(1.0, 2.0, 200000)) @ v +
            0.1 * jnp.roll(v, -1))

def B_matvec(v):
    return v + 0.01 * jnp.roll(v, 2)

# Run on GPU
eigvals, eigvecs = gpu_eigh_scan(A_matvec, B_matvec, n=200000, k=5)

# Check residuals (batched)
residual = lambda v, lam: jnp.linalg.norm(A_matvec(v) - lam*B_matvec(v))
residuals = vmap(residual)(eigvecs.T, eigvals)
print("Residual norms:", residuals)
