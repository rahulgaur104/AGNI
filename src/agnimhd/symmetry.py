"""Stellarator symmetry: the even and odd parity subspaces of the operator.

On a stellarator-symmetric equilibrium the reflection
``(rho, theta, zeta) -> (rho, -theta, -zeta)``, with the displacement
components ``(xi^rho, xi^theta, xi^zeta)`` taking the signs ``(+, -, -)``,
commutes with the real-family operator (measured ``2.4e-15`` on the shipped
case). The operator then splits into an even and an odd block of about half
the size, and the lowest eigenvalue of the full problem is the lower of the
two blocks' lowest eigenvalues. ``AssemblyConfig(parity="even" | "odd")``
solves one block.

The parity basis is a sparse orthonormal matrix ``C`` with one column per
node pair, ``(e_i + s e_j) / sqrt(2)``; a node the reflection maps to itself
survives in the parity its component sign selects. ``C`` is held as two
gathers (``idx``, ``w``), so ``C y`` and ``C^T x`` are a scatter-add and a
gather, and the reduced matrix ``C^T A C`` is formed from ``A`` by gathers.
"""

import numpy as np

from .assemble import keep_indices
from .backend import errorif, jax, jnp

__all__ = ["ParityBasis", "parity_basis", "symmetry_error"]

SIGNS = np.array([1.0, -1.0, -1.0])  # component signs under the reflection


class ParityBasis:
    """``C`` of one parity: ``n`` kept DOFs to ``m`` parity DOFs."""

    def __init__(self, idx, w, src, sign, n):
        self.idx, self.w, self.src, self.sign, self.n = idx, w, src, sign, n
        self.m = int(idx.shape[0])

    def reduce(self, x):
        """``C^T x``: a vector (or the columns of a matrix) to the parity space."""
        return (
            self.w[:, 0, None] * x[self.idx[:, 0]]
            + self.w[:, 1, None] * x[self.idx[:, 1]]
        )

    def expand(self, y):
        """``C y``: a parity-space vector back to the kept DOFs."""
        out = jnp.zeros((self.n,) + y.shape[1:], y.dtype)
        out = out.at[self.idx[:, 0]].add(self.w[:, 0, None] * y)
        return out.at[self.idx[:, 1]].add(self.w[:, 1, None] * y)

    def reduce_vector(self, x):
        """``C^T x`` for one vector."""
        return self.reduce(x[:, None])[:, 0]

    def expand_vector(self, y):
        """``C y`` for one vector."""
        return self.expand(y[:, None])[:, 0]

    def reduce_matrix(self, A):
        """``C^T A C``: the parity block of the dense reduced matrix ``A``."""
        AC = self.reduce(A.T).T  # A C, by gathering columns
        return self.reduce(AC)

    def reflect(self, x):
        """``P x``: the signed reflection on the kept DOFs."""
        return self.sign * x[self.src]


def parity_basis(n_rho, n_theta, n_zeta, parity):
    """The :class:`ParityBasis` of ``parity`` (``"even"`` or ``"odd"``).

    Node order is rho-major, then theta, then zeta, both angles equispaced from
    0, so the reflection maps node ``(i, j, k)`` to ``(i, -j, -k)`` modulo the
    counts. The Dirichlet mask (:func:`~agnimhd.assemble.keep_indices`) is
    preserved by it.
    """
    s = {"even": 1.0, "odd": -1.0}[parity]
    n_total = n_rho * n_theta * n_zeta
    r, t, z = np.meshgrid(
        np.arange(n_rho), np.arange(n_theta), np.arange(n_zeta), indexing="ij"
    )
    node = ((r * n_theta + (-t) % n_theta) * n_zeta + (-z) % n_zeta).ravel()
    keep = keep_indices(n_rho, n_theta, n_zeta)
    pos = np.full(3 * n_total, -1)
    pos[keep] = np.arange(keep.size)
    src = pos[np.concatenate([c * n_total + node for c in range(3)])[keep]]
    sign = SIGNS[keep // n_total]
    i = np.arange(keep.size)
    j = src
    first = i <= j  # one column per pair; fixed points have i == j
    pair = first & (i != j)
    fixed = (i == j) & (sign == s)
    idx = np.stack([i, j], axis=1)[pair | fixed]
    w = np.where(
        (i != j)[:, None],
        np.stack([np.full(i.size, 1 / np.sqrt(2)), s * sign / np.sqrt(2)], axis=1),
        np.stack([np.ones(i.size), np.zeros(i.size)], axis=1),
    )[pair | fixed]
    # NumPy, not jax arrays: the basis is built inside `jit` and read from a host
    # callback, and a traced constant captured there is an escaped tracer.
    return ParityBasis(idx, w, src, sign, int(keep.size))


def symmetry_error(op, basis, seed=0):
    """``|P A P x - A x| / |A x|`` on a random ``x``; zero when the equilibrium is
    stellarator-symmetric."""
    x = jax.random.normal(jax.random.PRNGKey(seed), (basis.n,), dtype=jnp.float64)
    Ax = op["Ax"](x)
    PAPx = basis.reflect(op["Ax"](basis.reflect(x)))
    return jnp.linalg.norm(PAPx - Ax) / jnp.linalg.norm(Ax)


def require_symmetric(error, tol=1e-8):
    """Raise unless the measured commutation ``error`` is below ``tol``."""
    errorif(
        float(error) > tol,
        ValueError,
        f"parity needs a stellarator-symmetric equilibrium: the operator's "
        f"commutation error with the reflection is {float(error):.1e} "
        f"(limit {tol:.0e}). Use parity=None.",
    )
