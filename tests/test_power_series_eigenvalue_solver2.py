import jax
import jax.numpy as jnp


def top_k_eigenvalues(A, k=5, num_iterations=100):
    """
    Compute top-k eigenvalues of a symmetric matrix using power iteration.
    Simple implementation with no fancy decorators.

    Args:
        A: Symmetric matrix (N x N)
        k: Number of eigenvalues to compute
        num_iterations: Number of power iterations

    Returns:
        Top k eigenvalues (largest magnitude first)
    """
    n = A.shape[0]
    key = jax.random.PRNGKey(0)

    # Initialize random vectors
    vectors = jax.random.normal(key, (k, n))

    # Orthonormalize initial vectors
    vectors, _ = jnp.linalg.qr(vectors.T)
    vectors = vectors.T  # Shape: (k, dim)

    # Power iteration
    for _ in range(num_iterations):
        # Matrix-vector multiplication
        new_vectors = vectors @ A

        # QR decomposition for orthogonalization
        q, _ = jnp.linalg.qr(new_vectors.T)
        vectors = q.T

    # Compute Rayleigh quotients
    rayleigh = jnp.diag(vectors @ A @ vectors.T)

    # Sort by magnitude (largest first)
    idx = jnp.argsort(-jnp.abs(rayleigh))
    eigenvalues = rayleigh[idx]

    return eigenvalues


def test_tridiagonal_toeplitz(k=5):
    """
    Test with a tridiagonal Toeplitz matrix.
    We'll use a modest size that can be formed explicitly.

    Args:
        k: Number of top eigenvalues to compute
    """
    n = 20000  # Modest size that we can form explicitly
    print(f"\n=== Testing with Tridiagonal Toeplitz Matrix (n={n}) ===")

    # Create the tridiagonal Toeplitz matrix
    A = (
        jnp.diag(jnp.ones(n) * 2)
        + jnp.diag(jnp.ones(n - 1) * -1, 1)
        + jnp.diag(jnp.ones(n - 1) * -1, -1)
    )

    # Analytical eigenvalues (largest k)
    j_values = jnp.arange(1, n + 1)
    analytical_evals = 2 + 2 * jnp.cos(j_values * jnp.pi / (n + 1))
    analytical_evals = jnp.sort(analytical_evals)[::-1][:k]

    # Compute eigenvalues using our method
    computed_evals = top_k_eigenvalues(A, k, 200)

    # Compute exact eigenvalues using JAX's eigvalsh for comparison
    exact_evals = jnp.linalg.eigvalsh(A)[::-1][:k]  # Descending order

    print(f"Analytical top-{k} eigenvalues:", analytical_evals)
    print(f"Exact (eigvalsh) top-{k} eigenvalues:", exact_evals)
    print(f"Our method top-{k} eigenvalues:", computed_evals)

    # Relative error compared to exact
    rel_error = jnp.abs((computed_evals - exact_evals) / exact_evals)
    print("Relative errors (vs exact):", rel_error)
    print("Max relative error:", jnp.max(rel_error))

    return computed_evals, exact_evals, analytical_evals


def test_explicitly_constructed_matrix(n=5000, k=5):
    """
    Test with an explicitly constructed matrix with known eigenvalues.
    """
    print(f"\n=== Testing with Explicitly Constructed Matrix (n={n}) ===")
    key = jax.random.PRNGKey(45)

    # Create orthogonal matrix for similarity transform
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))

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
    computed_evals = top_k_eigenvalues(A, k, 100)

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
    test_tridiagonal_toeplitz(k=5)
    test_explicitly_constructed_matrix(n=5000, k=5)
