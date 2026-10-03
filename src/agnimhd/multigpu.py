"""Dense shift-invert on several GPUs with JAXMg (``eigensolver="dense_mg"``).

``A - sigma I`` is assembled in row blocks, one per device
(:func:`agnimhd.assemble.assemble_rows`), so no device ever holds the whole
matrix. It is padded with an identity block to a multiple of ``mg_tile`` times
the device count, inverted once by JAXMg's Cholesky routines, and Lanczos then
runs on products with the sharded inverse. ``jaxmg`` is imported only here.

With jaxmg < 1.0 the inverse comes from ``potri`` without its symmetrization
step, which would make transposed full-size copies: only the upper triangle is
valid, and :func:`inverse_matvec` reads only that triangle, tile by tile.
"""

import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .assemble import assemble_rows, matfree_operator, operator_dtype
from .backend import jax, jnp
from .solvers import lanczos_shift_invert

__all__ = ["dense_mg", "inverse", "inverse_matvec", "shifted_rows"]

AXIS = "gpus"
ROWS, REP = P(AXIS, None), P()


def shifted_rows(eq, diffmat, assembly, sigma, mesh, tile, density=None):
    """Row-sharded ``blockdiag(A - sigma I, I)`` and the unpadded size ``n``."""
    n = matfree_operator(eq, diffmat, assembly, density=density)["n_keep"]
    step = mesh.devices.size * tile
    n_pad = -(-n // step) * step
    rows = n_pad // mesh.devices.size

    def block(eq, diffmat):
        r = jax.lax.axis_index(AXIS) * rows + jnp.arange(rows)
        return assemble_rows(eq, diffmat, assembly, r, 256, density, sigma, n_pad)

    f = jax.shard_map(block, mesh=mesh, in_specs=(REP, REP), out_specs=ROWS)
    return jax.jit(f)(eq, diffmat), n


def inverse(M, mesh, tile):
    """``(X, upper)``: the inverse of the SPD row-sharded ``M``, row-sharded.

    ``upper`` is True when only the upper triangle of ``X`` is valid.
    ``M`` is donated.
    """
    import jaxmg

    if hasattr(jaxmg, "potri_shardmap_ctx"):  # jaxmg < 1.0 (cusolverMg)
        f = jax.shard_map(
            lambda a: jaxmg.potri_shardmap_ctx(a, tile, pad=False)[0],
            mesh=mesh,
            in_specs=ROWS,
            out_specs=ROWS,
            check_vma=False,
        )
        return jax.jit(f, donate_argnums=0)(M), True
    eye = jax.jit(
        lambda: jnp.eye(M.shape[0], dtype=M.dtype),
        out_shardings=NamedSharding(mesh, ROWS),
    )()
    return jaxmg.potrs(M, eye, tile, mesh, ROWS), False


def inverse_matvec(X, upper, mesh, tile):
    """``b -> X b`` for the row-sharded ``X``, from its upper triangle if ``upper``."""
    if not upper:
        return jax.jit(lambda b: X @ b)
    n_pad, rows = X.shape[0], X.shape[0] // mesh.devices.size

    def local(U, b):
        r0 = jax.lax.axis_index(AXIS) * rows
        i, c = r0 + jnp.arange(rows), jnp.arange(n_pad)
        b_blk = jax.lax.dynamic_slice(b, (r0,), (rows,))
        right = c >= r0 + rows  # columns right of this device's diagonal block
        y = U @ jnp.where(right, b, 0)
        z = jnp.where(right, jnp.conj(jnp.conj(b_blk) @ U), 0)

        def tile_k(k, yz):  # the diagonal block, one column tile at a time
            c0 = (r0 + k * tile).astype(jnp.int32)
            ck = c0 + jnp.arange(tile)
            D = jax.lax.dynamic_slice(U, (jnp.int32(0), c0), (rows, tile))
            yk = jnp.where(ck[None, :] >= i[:, None], D, 0) @ b[ck]
            Ds = jnp.where(ck[None, :] > i[:, None], D, 0)
            zk = jax.lax.dynamic_slice(yz[1], (c0,), (tile,))
            zk = zk + jnp.conj(jnp.conj(b_blk) @ Ds)
            return yz[0] + yk, jax.lax.dynamic_update_slice(yz[1], zk, (c0,))

        y, z = jax.lax.fori_loop(0, rows // tile, tile_k, (y, z))
        return y, jax.lax.psum(z, AXIS)

    f = jax.shard_map(local, mesh=mesh, in_specs=(ROWS, REP), out_specs=(P(AXIS), REP))
    return jax.jit(lambda b: sum(f(X, b)))


def dense_mg(eq, diffmat, assembly, solver, v0=None, density=None):
    """``(v, lam)`` by shift-invert Lanczos on the JAXMg inverse."""
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    tile = solver.mg_tile
    M, n = shifted_rows(eq, diffmat, assembly, solver.sigma, mesh, tile, density)
    X, upper = inverse(M, mesh, tile)
    Xb = inverse_matvec(X, upper, mesh, tile)
    pad = X.shape[0] - n

    def opinv(b):
        return Xb(jnp.pad(b, (0, pad)))[:n]

    dtype = operator_dtype(assembly)
    return lanczos_shift_invert(
        opinv, n, dtype, solver.sigma, solver.num_matvecs, solver.seed, v0
    )
