"""Dense shift-invert on several GPUs with JAXMg (``eigensolver="dense_mg"``).

``A + sigma I`` is assembled in row blocks, one per device
(:func:`agnimhd.assemble.assemble_rows`), so no device ever holds the whole
matrix, and padded with an identity block to a multiple of ``mg_tile`` times
the device count. Block inverse iteration then alternates one JAXMg Cholesky
solve with a block of ``mg_block`` vectors and a Rayleigh-Ritz step against the
exact matrix-free operator. JAXMg cannot reuse a factor between calls, so each
iteration rebuilds and refactors the matrix. ``jaxmg`` is imported only here.
"""

import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .assemble import assemble_rows, matfree_operator, operator_dtype
from .backend import errorif, jax, jnp
from .objective import _squared_growth_rate

__all__ = ["dense_mg", "shifted_rows", "solve_shifted"]

AXIS = "gpus"
ROWS, REP = P(AXIS, None), P()


def shifted_rows(eq, diffmat, assembly, sigma, mesh, tile, density=None):
    """Row-sharded ``blockdiag(A - sigma I, I)`` and the unpadded size ``n``;
    ``sigma`` here is the shift of ``A``, ``SolverConfig.shift``."""
    n = matfree_operator(eq, diffmat, assembly, density=density)["n_keep"]
    step = mesh.devices.size * tile
    n_pad = -(-n // step) * step
    rows = n_pad // mesh.devices.size

    def block(eq, diffmat):
        r = jax.lax.axis_index(AXIS) * rows + jnp.arange(rows)
        return assemble_rows(eq, diffmat, assembly, r, 256, density, sigma, n_pad)

    f = jax.shard_map(block, mesh=mesh, in_specs=(REP, REP), out_specs=ROWS)
    return jax.jit(f)(eq, diffmat), n


def solve_shifted(M, B, mesh, tile):
    """``M^-1 B`` for the SPD row-sharded ``M`` (donated) and a small block ``B``."""
    import jaxmg

    B = jax.device_put(B, NamedSharding(mesh, REP))  # jaxmg < 1.0 wants B replicated
    return jaxmg.potrs(M, B, tile, mesh, ROWS)


def dense_mg(eq, diffmat, assembly, solver, v0=None, density=None, log=None):
    """``(v, gamma2)`` by block inverse iteration with Rayleigh-Ritz on the exact ``A``.

    ``gamma2 = -lambda`` is the squared growth rate, positive when unstable.
    Stops after ``mg_iters`` iterations or once the eigenpair residual
    ``||A v - lam v|| / |lam|`` is below ``mg_tol`` (checked only when not
    traced). ``log(it, gamma2, residual, v)`` is called after every iteration; a
    true return value stops the iteration there. Real operators only: JAXMg's
    complex solve is not verified, so a complex toroidal family (or
    ``axisym=True``) raises; solve those with ``eigsh``, ``jax_lanczos`` or ``jd``.
    """
    errorif(
        operator_dtype(assembly, diffmat) != jnp.float64,
        ValueError,
        "dense_mg solves real operators only (toroidal families 0 and NFP/2); "
        "JAXMg's complex solve is not verified. Use eigensolver 'eigsh', "
        "'jax_lanczos' or 'jd' for this family.",
    )
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    Ax = jax.vmap(matfree_operator(eq, diffmat, assembly, density=density)["Ax"], 1, 1)
    V = None
    for it in range(solver.mg_iters):
        M, n = shifted_rows(
            eq, diffmat, assembly, solver.shift, mesh, solver.mg_tile, density
        )
        if V is None:
            V = np.random.default_rng(solver.seed).standard_normal((n, solver.mg_block))
            V = jnp.asarray(V)
            V = V if v0 is None else V.at[:, 0].set(v0)
        B = jnp.pad(V, ((0, M.shape[0] - n), (0, 0)))
        W = solve_shifted(M, B, mesh, solver.mg_tile)
        Q = jnp.linalg.qr(W[:n])[0]
        AQ = Ax(Q)
        H = Q.conj().T @ AQ
        lam, Y = jnp.linalg.eigh((H + H.conj().T) / 2)
        V, lam, AV = Q @ Y, lam[0], AQ @ Y[:, 0]
        res = jnp.linalg.norm(AV - lam * V[:, 0]) / jnp.abs(lam)
        gamma2 = _squared_growth_rate(V[:, 0], AV)
        if log is not None and log(it, gamma2, res, V[:, 0]):
            break
        if not isinstance(res, jax.core.Tracer) and float(res) < solver.mg_tol:
            break
    return V[:, 0], gamma2
