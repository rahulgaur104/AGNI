"""Dense shift-invert on several GPUs with JAXMg (``eigensolver="dense_mg"``).

``A - sigma I`` is assembled in row blocks, one per device
(:func:`agnimhd.assemble.assemble_rows`), so no device ever holds the whole
matrix. It is padded with an identity block to a multiple of ``mg_tile`` times
the device count, inverted once by JAXMg's Cholesky routines, and Lanczos then
runs on products with the sharded inverse. ``jaxmg`` is imported only here.
"""

import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .assemble import assemble_rows, matfree_operator, operator_dtype
from .backend import jax, jnp
from .solvers import lanczos_shift_invert

__all__ = ["dense_mg", "shifted_rows", "inverse"]

AXIS = "gpus"


def shifted_rows(eq, diffmat, assembly, sigma, mesh, tile):
    """Row-sharded ``blockdiag(A - sigma I, I)`` and the unpadded size ``n``."""
    n = matfree_operator(eq, diffmat, assembly)["n_keep"]
    step = mesh.devices.size * tile
    n_pad = -(-n // step) * step
    rows = n_pad // mesh.devices.size

    def block(eq, diffmat):
        r = jax.lax.axis_index(AXIS) * rows + jnp.arange(rows)
        A = assemble_rows(eq, diffmat, assembly, jnp.minimum(r, n - 1))
        A = jnp.where((r < n)[:, None], jnp.pad(A, ((0, 0), (0, n_pad - n))), 0)
        return A.at[jnp.arange(rows), r].add(jnp.where(r < n, -sigma, 1.0))

    spec = (P(), P())  # equilibrium and operators replicated on every device
    f = jax.shard_map(block, mesh=mesh, in_specs=spec, out_specs=P(AXIS, None))
    return jax.jit(f)(eq, diffmat), n


def inverse(M, mesh, tile):
    """Inverse of the SPD row-sharded ``M`` with JAXMg, row-sharded like ``M``."""
    import jaxmg

    spec = P(AXIS, None)
    if hasattr(jaxmg, "potri"):  # jaxmg < 1.0: cusolverMg explicit inverse
        return jaxmg.potri(M, tile, mesh, spec)
    eye = jax.jit(
        lambda: jnp.eye(M.shape[0], dtype=M.dtype),
        out_shardings=NamedSharding(mesh, spec),
    )()
    return jaxmg.potrs(M, eye, tile, mesh, spec)


def dense_mg(eq, diffmat, assembly, solver, v0=None):
    """``(v, lam)`` by shift-invert Lanczos on the JAXMg inverse."""
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    M, n = shifted_rows(eq, diffmat, assembly, solver.sigma, mesh, solver.mg_tile)
    X = inverse(M, mesh, solver.mg_tile)
    pad = X.shape[0] - n

    def opinv(b):
        return (X @ jnp.pad(b, (0, pad)))[:n]

    dtype = operator_dtype(assembly)
    return lanczos_shift_invert(
        opinv, n, dtype, solver.sigma, solver.num_matvecs, solver.seed, v0
    )
