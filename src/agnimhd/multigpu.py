"""Dense shift-invert on several GPUs with JAXMg (``eigensolver="dense_mg"``).

``A + sigma I`` is assembled in row blocks, one per device
(:func:`agnimhd.assemble.assemble_rows`), so no device ever holds the whole
matrix, and padded with an identity block to a multiple of ``mg_tile`` times
the device count. Block inverse iteration then alternates one JAXMg Cholesky
solve with a block of ``mg_block`` vectors and a Rayleigh-Ritz step against the
exact matrix-free operator. JAXMg cannot reuse a factor between calls, so each
iteration rebuilds and refactors the matrix. ``jaxmg`` is imported only here.

With ``jaxmg`` 1.x (cuSOLVERMp) there is one process per GPU, possibly on
several nodes, after ``jax.distributed.initialize()``: every process runs the
same calls, the matrix is split over all GPUs of all processes, and the block of
vectors and the Rayleigh-Ritz step are repeated on every process.
"""

import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .assemble import assemble_rows, matfree_operator, operator_dtype
from .backend import jax, jnp
from .objective import _squared_growth_rate

__all__ = ["dense_mg", "shifted_rows", "solve_shifted"]

AXIS = "gpus"
ROWS, REP = P(AXIS, None), P()


def _replicated(tree, mesh):
    """The arrays of ``tree`` replicated over ``mesh``. With several processes
    each process passes its own copy, the same on all of them."""
    sharding = NamedSharding(mesh, REP)

    def put(x):
        if jax.process_count() == 1:
            return jax.device_put(x, sharding)
        x = np.asarray(x)
        return jax.make_array_from_callback(x.shape, sharding, lambda i: x[i])

    return jax.tree.map(put, tree)


def _row_builder(eq, diffmat, assembly, mesh, tile, density=None):
    """``(build, n)``: ``build(shift)`` is the row-sharded ``blockdiag(A - shift I,
    I)``, compiled once for every shift; ``n`` the unpadded size."""
    n = matfree_operator(eq, diffmat, assembly, density=density)["n_keep"]
    step = mesh.devices.size * tile
    n_pad = -(-n // step) * step
    rows = n_pad // mesh.devices.size

    def block(eq, diffmat, density, shift):
        r = jax.lax.axis_index(AXIS) * rows + jnp.arange(rows)
        return assemble_rows(eq, diffmat, assembly, r, 256, density, shift, n_pad)

    f = jax.jit(jax.shard_map(block, mesh=mesh, in_specs=(REP,) * 4, out_specs=ROWS))
    args = (eq, diffmat, density)
    if jax.process_count() > 1:
        args = _replicated(args, mesh)

    def build(shift):
        shift = jnp.asarray(shift, dtype=jnp.float64)
        if jax.process_count() > 1:
            shift = _replicated(shift, mesh)
        return f(*args, shift)

    return build, n


def shifted_rows(eq, diffmat, assembly, sigma, mesh, tile, density=None):
    """Row-sharded ``blockdiag(A - sigma I, I)`` and the unpadded size ``n``;
    ``sigma`` here is the shift of ``A``, ``SolverConfig.shift``."""
    build, n = _row_builder(eq, diffmat, assembly, mesh, tile, density)
    return build(sigma), n


def _any_process(flag):
    """``flag`` (a bool) true on any process: one decision for all of them, so
    no process leaves a collective solve the others still enter."""
    if jax.process_count() == 1:
        return flag
    from jax.experimental import multihost_utils

    return bool(np.any(multihost_utils.process_allgather(np.asarray(flag))))


def _failed(status):
    """Whether JAXMg's Cholesky solve failed: ``jaxmg`` 0.0.9 returns one status
    int, 1.x a status vector per rank with the status code first. A failed 1.x
    solve still returns a finite solution."""
    s = jnp.ravel(status)
    try:
        from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE as size
    except ImportError:  # jaxmg 0.0.9
        size = 1
    if s.size % size:  # one status int
        size = 1
    return jnp.any(s.reshape(-1, size)[:, 0] != 0)


def solve_shifted(M, B, mesh, tile):
    """``M^-1 B`` for the row-sharded ``M`` (donated) and a small block ``B``,
    replicated: with several processes, this process's copy. NaN when ``M`` is
    not positive definite (the shift is above the softest eigenvalue), as in
    the one-GPU dense solve."""
    import jaxmg

    W, status = jaxmg.potrs(
        M, _replicated(B, mesh), tile, mesh, ROWS, return_status=True
    )
    if jax.process_count() > 1:
        if not W.sharding.is_fully_replicated:
            raise RuntimeError(f"jaxmg returned the solution sharded as {W.sharding}")
        W, status = W.addressable_data(0), status.addressable_data(0)
        return jnp.where(_any_process(bool(_failed(status))), jnp.nan, W)
    return jnp.where(_failed(status), jnp.nan, W)


def dense_mg(eq, diffmat, assembly, solver, v0=None, density=None, log=None):
    """``(v, gamma2)`` by block inverse iteration with Rayleigh-Ritz on the exact
    ``A`` and a shift that moves towards ``gamma^2``.

    ``gamma2 = -lambda`` is the squared growth rate, positive when unstable.
    Every iteration factors ``A + s I`` anew (JAXMg keeps no factor), so the
    shift ``s`` costs nothing to change: it starts at ``solver.sigma`` and moves
    halfway towards the Ritz value ``gamma2`` after each successful factor.
    ``gamma2`` never exceeds the largest ``gamma^2``, which lies below every
    shift that factors, so the move keeps the shift above ``gamma2``; a shift
    that lands below the largest ``gamma^2`` fails to factor and is replaced by
    the midpoint between it and the last one that factored. NaN when
    ``solver.sigma`` itself does not factor, as in the one-GPU dense solve.

    Stops after ``mg_iters`` iterations, once ``gamma2`` changes by at most
    ``mg_theta_tol`` relative over a successful iteration, or once the residual
    ``||A v - lam v|| / |lam|`` is below ``mg_tol`` (both checked only when not
    traced; at a large Zernike penalty the residual scales with the penalty and
    the ``gamma2`` change is the test that ends the run). ``log(it, gamma2,
    residual, v)`` is called after every iteration; a true return value stops
    the iteration there. A complex toroidal family (or ``axisym=True``) runs the
    same way on a complex Hermitian matrix: JAXMg's Cholesky solve accepts it
    (``jaxmg`` 0.0.9 on four GPUs agrees with a one-GPU solve to 1e-15).
    """
    dtype = operator_dtype(assembly, diffmat)
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    Ax = jax.vmap(matfree_operator(eq, diffmat, assembly, density=density)["Ax"], 1, 1)
    build, n = _row_builder(eq, diffmat, assembly, mesh, solver.mg_tile, density)
    V = np.random.default_rng(solver.seed).standard_normal((n, solver.mg_block))
    V = jnp.asarray(V, dtype=dtype)
    V = V if v0 is None else V.at[:, 0].set(v0)
    s_try = s_good = jnp.asarray(solver.sigma, dtype=jnp.float64)
    gamma2 = jnp.asarray(jnp.nan)
    for it in range(solver.mg_iters):
        M = build(-s_try)
        W = solve_shifted(
            M, jnp.pad(V, ((0, M.shape[0] - n), (0, 0))), mesh, solver.mg_tile
        )
        ok = jnp.isfinite(W[:n]).all()
        Q = jnp.linalg.qr(jnp.where(ok, W[:n], V))[0]
        AQ = Ax(Q)
        H = Q.conj().T @ AQ
        lam, Y = jnp.linalg.eigh((H + H.conj().T) / 2)
        Vn, lam, AV = Q @ Y, lam[0], AQ @ Y[:, 0]
        res = jnp.linalg.norm(AV - lam * Vn[:, 0]) / jnp.abs(lam)
        g_new = _squared_growth_rate(Vn[:, 0], AV)
        g_prev = gamma2
        # The first shift failing leaves gamma2 NaN for good, as in "dense".
        V = jnp.where(ok, Vn, V)
        gamma2 = jnp.where(ok, g_new, gamma2)
        s_good = jnp.where(ok, s_try, s_good)
        s_try = jnp.where(ok, 0.5 * (gamma2 + s_good), 0.5 * (s_try + s_good))
        stop = log is not None and log(it, gamma2, res, V[:, 0])
        if isinstance(res, jax.core.Tracer):
            if stop:
                break
            continue
        g = float(gamma2)
        stop = stop or not np.isfinite(g)  # solver.sigma does not factor
        if bool(ok):
            tol = solver.mg_theta_tol
            theta_stop = tol > 0 and abs(g - float(g_prev)) <= tol * abs(g)
            stop = stop or theta_stop or float(res) < solver.mg_tol
        if _any_process(bool(stop)):
            break
    return V[:, 0], gamma2
