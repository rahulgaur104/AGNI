import jax
import jax.numpy as jnp


def top_k_eigenvalues_dense(A, k=5, num_iterations=100, key=None):
    """
    Compute top-k eigenvalues of a dense symmetric matrix using power iteration.
    This version works with an explicitly provided matrix.

    Args:
        A: Symmetric matrix (N x N)
        k: Number of eigenvalues to compute
        num_iterations: Number of power iterations
        key: JAX random key

    Returns:
        Top k eigenvalues (largest magnitude first)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = A.shape[0]

    # Initialize random vectors
    vectors = jax.random.normal(key, (k, n))

    # Orthonormalize
    vectors, _ = jnp.linalg.qr(vectors.T)
    vectors = vectors.T

    # Power iteration
    def body_fun(i, v):
        # Matrix-vector multiplication with the provided matrix
        v_new = v @ A

        # QR decomposition for orthogonalization
        q, _ = jnp.linalg.qr(v_new.T)
        return q.T

    # Run the iterations
    vectors = jax.lax.fori_loop(0, num_iterations, body_fun, vectors)

    # Compute Rayleigh quotients
    rayleigh = jnp.diag(vectors @ A @ vectors.T)

    # Sort by magnitude
    idx = jnp.argsort(-jnp.abs(rayleigh))
    eigenvalues = rayleigh[idx]

    return eigenvalues


def test_tridiagonal_toeplitz(n=50000, k=5):
    """
    Test with a tridiagonal Toeplitz matrix.

    Args:
        n: Matrix dimension
        k: Number of top eigenvalues to compute
    """
    print(f"\n=== Testing with Tridiagonal Toeplitz Matrix (n={n}) ===")
    key = jax.random.PRNGKey(42)

    # For the tridiagonal case, we'll use a specialized approach
    # We know the eigenvalues analytically without forming the matrix

    # Analytical eigenvalues (largest k)
    j_values = jnp.arange(1, n + 1)
    analytical_evals = 2 + 2 * jnp.cos(j_values * jnp.pi / (n + 1))
    analytical_evals = jnp.sort(analytical_evals)[::-1][:k]

    # For validation, we'll create a small version of the matrix
    # and test our algorithm against it
    small_n = 1000  # Small enough to form explicitly
    A_small = (
        jnp.diag(jnp.ones(small_n) * 2)
        + jnp.diag(jnp.ones(small_n - 1) * -1, 1)
        + jnp.diag(jnp.ones(small_n - 1) * -1, -1)
    )

    # Run our algorithm on the small matrix
    computed_evals_small = top_k_eigenvalues_dense(A_small, k, 100, key)

    # Compute exact eigenvalues for the small matrix
    exact_evals_small = jnp.linalg.eigvalsh(A_small)[::-1][:k]

    # Analytical eigenvalues for the small matrix
    j_values_small = jnp.arange(1, small_n + 1)
    analytical_evals_small = 2 + 2 * jnp.cos(j_values_small * jnp.pi / (small_n + 1))
    analytical_evals_small = jnp.sort(analytical_evals_small)[::-1][:k]

    print(f"Small matrix (n={small_n}):")
    print(f"  Analytical top-{k} eigenvalues:", analytical_evals_small)
    print(f"  Exact (eigvalsh) top-{k} eigenvalues:", exact_evals_small)
    print(f"  Our method top-{k} eigenvalues:", computed_evals_small)

    rel_error_small = jnp.abs(
        (computed_evals_small - exact_evals_small) / exact_evals_small
    )
    print(f"  Relative errors vs exact:", rel_error_small)
    print(f"  Max relative error:", jnp.max(rel_error_small))

    print(f"\nLarge matrix (n={n}):")
    print(f"  Analytical top-{k} eigenvalues:", analytical_evals)
    print(f"  Note: For the large matrix, we rely on the analytical solution")
    print(f"  since the matrix is too large to form explicitly.")

    return computed_evals_small, exact_evals_small, analytical_evals_small


def test_explicitly_constructed_matrix(n=5000, k=5):
    """
    Test with an explicitly constructed matrix with known eigenvalues.
    """
    print(f"\n=== Testing with Explicitly Constructed Matrix (n={n}) ===")
    key = jax.random.PRNGKey(45)
    key1, key2 = jax.random.split(key)

    # Create orthogonal matrix for similarity transform
    Q, _ = jnp.linalg.qr(jax.random.normal(key1, (n, n)))

    # Create diagonal matrix with known eigenvalues
    known_eigenvalues = jnp.concatenate(
        [
            jnp.array([1000.0, 900.0, 800.0, 700.0, 600.0][:k]),
            jnp.linspace(10.0, 1.0, n - k),
        ]
    )
    D = jnp.diag(known_eigenvalues)

    # Create the test matrix: A = Q D Q^T
    A = Q @ D @ Q.T

    # Ensure symmetry (might have numerical issues)
    A = (A + A.T) / 2

    # Compute top-k eigenvalues using our method
    computed_evals = top_k_eigenvalues_dense(A, k, 100, key2)

    # Compute exact eigenvalues using JAX's eigvalsh for comparison
    exact_evals = jnp.linalg.eigvalsh(A)[::-1][:k]  # Descending order

    print(f"Known top-{k} eigenvalues:", known_eigenvalues[:k])
    print(f"JAX eigvalsh top-{k} eigenvalues:", exact_evals)
    print(f"Our method top-{k} eigenvalues:", computed_evals)

    # Relative error compared to eigvalsh
    rel_error = jnp.abs((computed_evals - exact_evals) / exact_evals)
    print("Relative errors (vs eigvalsh):", rel_error)
    print("Max relative error:", jnp.max(rel_error))

    return computed_evals, exact_evals, known_eigenvalues[:k]


if __name__ == "__main__":
    # Run tests with various matrix types and sizes
    test_tridiagonal_toeplitz(n=50000, k=5)
    test_explicitly_constructed_matrix(n=5000, k=5)
