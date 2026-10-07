"""Numerical machinery for the AGNI eigensolver.

The block ("ring") preconditioner, Jacobi-Davidson, the coarse-to-fine
prolongation used to seed and deflate the fine solve, and the coarse
generalized eigensolve.

**All algorithms live here.** Drivers, examples and tests are thin: choose a
resolution and a basis, call in, compare the number that comes back. Nothing in
this module knows about resolution, basis, equilibrium or optimizer settings --
those are the caller's business. If a test has to reach inside the solver to
assemble its own linear algebra, this API is wrong.

Conventions
-----------
``meta``
    A dict describing one discretization level, from :func:`level_meta`.
``reduced`` vs ``physical``
    "Reduced" vectors carry only the kept degrees of freedom (length
    ``n_keep``). "Physical" arrays are ``(n_rho, n_theta, n_zeta, 3)``. The two
    differ by the Dirichlet mask *and* by the Cholesky-transform scaling, so
    they are never interchanged implicitly.

Node ordering is rho-major: the flat index of node ``(i, j, k)`` is
``(i * n_theta + j) * n_zeta + k``, and component ``c`` lives at
``c * n_total + ...``. Every index map here assumes it.
"""

import numpy as np

from .backend import jax, jnp

__all__ = [
    "adjoint_defect",
    "apply_space",
    "apply_space_t",
    "build_ring_blocks",
    "coarse_gen_modes",
    "coarse_seed_and_deflation",
    "deflation_Y",
    "factor_ring_blocks_traced",
    "fourier_interp_matrix",
    "from_phys",
    "from_phys_h",
    "jacobi_davidson",
    "lanczos_shift_invert",
    "level_meta",
    "make_block_precond",
    "make_transfer",
    "ring_index_maps",
    "ring_nodes",
    "to_phys",
    "to_phys_h",
]


# ---------------------------------------------------------------------------
# Reduced <-> physical
# ---------------------------------------------------------------------------


def _scatter_red(meta, q_red):
    """Reduced vector -> ``(n_total, 3)`` node array, zeros on dropped DOFs."""
    full = jnp.zeros((3 * meta["n_total"],), dtype=q_red.dtype)
    full = full.at[meta["keep"]].set(q_red, unique_indices=True)
    return full.reshape(3, meta["n_total"]).T


def _gather_red(meta, qnodes):
    """``(n_total, 3)`` node array -> reduced vector."""
    return qnodes.T.reshape(-1)[meta["keep"]]


def to_phys(meta, q_red):
    """Reduced solver coordinates -> physical ``(n_rho, n_theta, n_zeta, 3)``.

    Parameters
    ----------
    meta : dict
        From :func:`level_meta`.
    q_red : ndarray, shape (n_keep,)

    Returns
    -------
    jax.Array, shape (n_rho, n_theta, n_zeta, 3)
    """
    qnodes = _scatter_red(meta, q_red)
    u = meta["diag"] * jnp.einsum("nij,nj->ni", meta["linv_dt"], qnodes)
    return u.reshape(meta["n_rho"], meta["n_theta"], meta["n_zeta"], 3)


def from_phys(meta, u_full):
    """Physical field -> reduced solver coordinates. Inverse of :func:`to_phys`.

    Parameters
    ----------
    meta : dict
    u_full : ndarray, shape (n_rho, n_theta, n_zeta, 3)

    Returns
    -------
    jax.Array, shape (n_keep,)
    """
    unodes = u_full.reshape(meta["n_total"], 3)
    qnodes = jnp.einsum("nij,nj->ni", meta["inv_linv_dt"], unodes / meta["diag"])
    return _gather_red(meta, qnodes)


def to_phys_h(meta, u_full):
    """Transpose of :func:`to_phys`.

    Needed because the prolongation's adjoint is **not** its inverse: ``P^T``
    must be the true transpose or the deflated CG loses symmetry and stops being
    a valid Krylov method.

    Parameters
    ----------
    meta : dict
    u_full : ndarray, shape (n_rho, n_theta, n_zeta, 3)

    Returns
    -------
    jax.Array, shape (n_keep,)
    """
    unodes = u_full.reshape(meta["n_total"], 3)
    qnodes = jnp.einsum("nij,nj->ni", meta["linv_dt_h"], meta["diag"] * unodes)
    return _gather_red(meta, qnodes)


def from_phys_h(meta, q_red):
    """Transpose of :func:`from_phys`.

    Parameters
    ----------
    meta : dict
    q_red : ndarray, shape (n_keep,)

    Returns
    -------
    jax.Array, shape (n_rho, n_theta, n_zeta, 3)
    """
    qnodes = _scatter_red(meta, q_red)
    unodes = jnp.einsum("nij,nj->ni", meta["inv_linv_dt_h"], qnodes) / meta["diag"]
    return unodes.reshape(meta["n_rho"], meta["n_theta"], meta["n_zeta"], 3)


def level_meta(op):
    """Build a level ``meta`` dict from a matrix-free operator's output.

    The inverses of the per-node Cholesky transform and both transposes are
    formed once here, so every transfer call downstream is a pure einsum.

    Parameters
    ----------
    op : dict
        What :func:`agnimhd.assemble.matfree_operator` returns.

    Returns
    -------
    dict
        Keys ``linv_dt``, ``linv_dt_h``, ``inv_linv_dt``, ``inv_linv_dt_h``,
        ``diag``, ``keep``, ``n_total``, ``n_rho``, ``n_theta``, ``n_zeta``,
        ``n_keep``.
    """
    linv_dt = jnp.asarray(op["Linv_DT"])
    inv_linv_dt = jnp.linalg.inv(linv_dt)
    return dict(
        linv_dt=linv_dt,
        linv_dt_h=jnp.swapaxes(linv_dt, -1, -2),
        inv_linv_dt=inv_linv_dt,
        inv_linv_dt_h=jnp.swapaxes(inv_linv_dt, -1, -2),
        diag=jnp.asarray(op["diagBsqinv"]),
        keep=jnp.asarray(op["keep"]),
        n_total=int(op["n_total"]),
        n_rho=int(op["n_rho"]),
        n_theta=int(op["n_theta"]),
        n_zeta=int(op["n_zeta"]),
        n_keep=int(op["n_keep"]),
    )


# ---------------------------------------------------------------------------
# Prolongation: coarse level -> fine level
# ---------------------------------------------------------------------------


def fourier_interp_matrix(n_src, n_dst, period):
    """Exact Fourier interpolation matrix on a uniform periodic grid.

    Used for theta (period ``2*pi``) and zeta (period ``2*pi/NFP``). Exact, not
    approximate: both grids are uniform and periodic, so the trigonometric
    interpolant through the coarse samples reproduces every mode the coarse grid
    can represent.

    Parameters
    ----------
    n_src, n_dst : int
    period : float

    Returns
    -------
    ndarray, shape (n_dst, n_src)

    Notes
    -----
    The wavenumber scaling ``k = 2*pi/period`` is not optional. ``modes`` counts
    integer harmonics **of the period**, so the basis function is
    ``exp(i * m * k * x)``, not ``exp(i * m * x)``. Dropping ``k`` happens to be
    harmless at ``period = 2*pi``, where ``k = 1`` -- which is the poloidal case
    and therefore the one that gets looked at. At any other period the basis
    functions are no longer periodic on the grid, the matrix stops being an
    interpolation operator at all, and it fails even the trivial
    ``n_src == n_dst`` case: measured defect against the identity at ``n = 8``,
    ``period = 2*pi/4`` was **0.897**, versus 4e-16 with the scaling in place.
    That is the toroidal transfer, so every coarse-to-fine prolongation in the
    two-level solve is affected.
    """
    k = 2.0 * np.pi / period
    x = np.arange(n_src) * (period / n_src)
    y = np.arange(n_dst) * (period / n_dst)
    modes = np.fft.fftfreq(n_src) * n_src
    coeff = np.exp(-1j * k * np.outer(modes, x)) / n_src
    vals = np.exp(1j * k * np.outer(y, modes))
    return np.real_if_close(vals @ coeff, tol=1000).real


def apply_space(u, pr, pt, pz):
    """Separable tensor-product interpolation, coarse -> fine."""
    return jnp.einsum("ia,jb,kc,abcq->ijkq", pr, pt, pz, u)


def apply_space_t(u, pr, pt, pz, scale=1.0):
    """Transpose of :func:`apply_space`, fine -> coarse."""
    return scale * jnp.einsum("ia,jb,kc,ijkq->abcq", pr, pt, pz, u)


def make_transfer(meta_c, meta_f, pr, pt, pz):
    """Return ``(P, PT)`` as callables on reduced-coordinate vectors.

    ``PT`` is the exact transpose of ``P``, not an inverse and not a re-derived
    restriction. Check it with :func:`adjoint_defect` before trusting a deflated
    solve: if ``<P q_c, q_f> != <q_c, PT q_f>`` the deflation space is not what
    the CG thinks it is.

    Parameters
    ----------
    meta_c, meta_f : dict
    pr, pt, pz : ndarray

    Returns
    -------
    P, PT : tuple of callable
    """

    def P(q_c):
        return from_phys(meta_f, apply_space(to_phys(meta_c, q_c), pr, pt, pz))

    def PT(q_f):
        return to_phys_h(
            meta_c, apply_space_t(from_phys_h(meta_f, q_f), pr, pt, pz, 1.0)
        )

    return P, PT


def adjoint_defect(P, PT, n_c, n_f, trials=8, seed=0):
    """Worst relative ``<P x, y>`` vs ``<x, PT y>`` mismatch over random pairs.

    A number, not an assertion, so callers choose the tolerance. Values near
    machine epsilon (~1e-14) are expected; anything larger means ``PT`` is not
    the transpose of ``P``.

    Parameters
    ----------
    P, PT : callable
    n_c, n_f : int
    trials : int
    seed : int

    Returns
    -------
    float
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = jnp.asarray(rng.standard_normal(n_c))
        y = jnp.asarray(rng.standard_normal(n_f))
        lhs = float(jnp.vdot(P(x), y).real)
        rhs = float(jnp.vdot(x, PT(y)).real)
        scale = max(abs(lhs), abs(rhs), 1e-300)
        worst = max(worst, abs(lhs - rhs) / scale)
    return worst


# ---------------------------------------------------------------------------
# Block ("ring") preconditioner
# ---------------------------------------------------------------------------


def factor_ring_blocks_traced(blocks, ridge=0.0):
    """Cholesky at a FIXED ridge, safe under trace.

    Factors once at the given ridge and reports finiteness as a traced flag. A
    non-SPD block therefore yields NaN -- visible in the result, which is the
    safer failure inside a jitted solve.

    Parameters
    ----------
    blocks : ndarray, shape (m, b, b)
    ridge : float

    Returns
    -------
    L, ok, ridge : tuple
    """
    b = blocks.shape[-1]
    eye = jnp.eye(b, dtype=blocks.dtype)[None]
    L = jnp.linalg.cholesky(blocks + ridge * eye)
    return L, jnp.all(jnp.isfinite(L)), ridge


def make_block_precond(L, Gs, n):
    """Build ``M^-1`` from Cholesky factors and the group index map.

    ``M^-1 r`` gathers each group's entries out of ``r``, solves the group's
    Cholesky system, and scatters the result back. Padded slots (``Gs == -1``)
    are zeroed on both the gather and the scatter; their gather index is clamped
    to 0 purely to stay in bounds.

    Parameters
    ----------
    L : ndarray, shape (m, b, b)
    Gs : ndarray of int, shape (m, b)
    n : int
        Length of the reduced vector.

    Returns
    -------
    callable

    Notes
    -----
    The scatter uses ``.add`` rather than ``.set`` so overlapping partitions
    would accumulate. For the ring partition the groups are disjoint and the
    two coincide -- but ``add`` is the correct operation for the additive
    Schwarz form this is.
    """
    from jax.scipy.linalg import solve_triangular

    Gs = jnp.asarray(Gs)
    mask = (Gs >= 0).astype(jnp.result_type(float))
    idx = jnp.where(Gs >= 0, Gs, 0)

    LH = jnp.conj(jnp.swapaxes(L, -1, -2))  # H = L L^H: L^H, not L^T (axisym)

    def M(r):
        y = r[idx] * mask.astype(r.dtype)
        z = solve_triangular(L, y[..., None], lower=True)
        z = solve_triangular(LH, z, lower=False)[..., 0]
        z = z * mask.astype(z.dtype)
        return jnp.zeros((n,), dtype=r.dtype).at[idx].add(z)

    return M


# ---------------------------------------------------------------------------
# Jacobi-Davidson
# ---------------------------------------------------------------------------


def _herm(X):
    return jnp.conj(jnp.swapaxes(X, -1, -2))


def jacobi_davidson(
    Ax,
    precond,
    v0,
    Z=None,
    *,
    sigma=0.0,
    outer=200,
    inner=100,
    maxdim=60,
    keep=10,
    tol=0.0,
    theta_tol=1e-8,
):
    """Softest eigenpair of ``A`` by Jacobi-Davidson, matrix-free and jit-able.

    Rayleigh-Ritz on a basis grown by ``(I-uu^H)(A - sigma I)(I-uu^H) t = -r``,
    solved by ``inner`` projected PCG steps with ``precond`` plus the deflation
    ``Y Y^H`` of ``Z``; ``sigma`` must sit below the spectrum. Restart to ``keep``
    at ``maxdim``; stop after ``outer`` corrections, at the returned vector's
    eigen-residual ``||A v - theta v|| / |theta| <= tol`` or at Ritz change
    ``theta_tol`` (0 disables either). Returns ``(theta, v, {"iters", "resid"})``."""

    def Hx(x):
        return Ax(x) - sigma * x

    if Z is None:
        M = precond
    else:
        Y, _ = deflation_Y(Z, jax.vmap(Hx, in_axes=1, out_axes=1)(Z))

        def M(r):
            return precond(r) + Y @ (_herm(Y) @ r)

    m, slot = maxdim, jnp.arange(maxdim)
    u0 = v0 / jnp.linalg.norm(v0)
    V = jnp.zeros((v0.shape[0], m), dtype=v0.dtype).at[:, 0].set(u0)
    AV = jnp.zeros_like(V).at[:, 0].set(Ax(u0))

    def ritz(V, AV, j):
        # Unused slots get a diagonal above every Ritz value, so the lowest
        # eigenpairs live on the used block only.
        used = slot < j
        S = _herm(V) @ AV
        S = jnp.where(used[:, None] & used[None, :], 0.5 * (S + _herm(S)), 0.0)
        big = jnp.where(used, 0.0, jnp.linalg.norm(S) + 1.0)
        return jnp.linalg.eigh(S + jnp.diag(big).astype(S.dtype))

    def correction(u, r):
        def P(x):
            return x - u * jnp.vdot(u, x)

        def body(st):
            x, r, p, rz, k, _ = st
            Ap = P(Hx(P(p)))
            curv = jnp.real(jnp.vdot(p, Ap))
            # Past convergence r^H M r is roundoff and can hit 0: the next
            # beta is then 0/0, and x + 0 * NaN poisons x. Stop there.
            good = (curv > 0) & (rz > 0)
            a = jnp.where(good, rz / jnp.where(good, curv, 1.0), 0.0)
            x, r = x + a * p, r - a * Ap
            z = P(M(P(r)))
            rzn = jnp.real(jnp.vdot(r, z))
            return (x, r, z + (rzn / rz) * p, rzn, k + good.astype(k.dtype), good)

        r0 = P(-r)
        z0 = P(M(P(r0)))
        ok = jnp.array(True)
        st = (jnp.zeros_like(r0), r0, z0, jnp.real(jnp.vdot(r0, z0)), slot[0], ok)
        x = jax.lax.while_loop(lambda s: (s[4] < inner) & s[5], body, st)[0]
        return P(x)

    def lowest_ritz_pair(V, AV, j):
        """Lowest Ritz value and vector, all Ritz vectors, and ``A u - theta u``
        with ``A u`` applied afresh, so the stop tests the vector returned."""
        w, Y = ritz(V, AV, j)
        u = V @ Y[:, 0]
        u = u / jnp.linalg.norm(u)
        return w[0], Y, u, Ax(u) - w[0] * u

    def eigen_residual(theta, r):
        """``||A u - theta u|| / |theta|``."""
        return jnp.linalg.norm(r) / jnp.maximum(jnp.abs(theta), 1e-300)

    def step(state):
        V, AV, j, it, theta, Y, u, r, _ = state
        t = correction(u, r)

        def restart(a):
            V_, AV_ = a
            Vn = jnp.zeros_like(V_).at[:, :keep].set(V_ @ Y[:, :keep])
            AVn = jnp.zeros_like(AV_).at[:, :keep].set(AV_ @ Y[:, :keep])
            return Vn, AVn, jnp.full_like(j, keep)

        V, AV, j = jax.lax.cond(j >= m, restart, lambda a: (*a, j), (V, AV))
        for _ in range(2):
            t = t - V @ (_herm(V) @ t)
        tn = jnp.linalg.norm(t)
        good = jnp.isfinite(tn) & (tn > 1e-300)
        t = t / jnp.where(good, tn, 1.0)
        V = jnp.where(good, V.at[:, j].set(t), V)
        AV = jnp.where(good, AV.at[:, j].set(Ax(t)), AV)
        j = j + good.astype(j.dtype)
        return (V, AV, j, it + 1, *lowest_ritz_pair(V, AV, j), theta)

    def go(state):
        it, theta, r, theta_prev = state[3], state[4], state[7], state[8]
        done = jnp.array(False)
        if tol > 0:
            done = done | (eigen_residual(theta, r) <= tol)
        if theta_tol > 0:
            done = done | (jnp.abs(theta - theta_prev) <= theta_tol * jnp.abs(theta))
        return (it < outer) & ~done

    one = jnp.asarray(1)
    state = (V, AV, one, 0 * one, *lowest_ritz_pair(V, AV, one), jnp.asarray(jnp.inf))
    _, _, _, it, theta, _, v, r, _ = jax.lax.while_loop(go, step, state)
    return theta, v, {"iters": it, "resid": eigen_residual(theta, r)}


# ---------------------------------------------------------------------------
# Coarse generalized eigensolve and the deflation space it supplies
# ---------------------------------------------------------------------------


def coarse_gen_modes(Hc, blocks, Gs, k, num_matvecs, ridge=0.0, seed=3):
    """Softest ``k`` generalized modes of ``(Hc, M_block)`` on the coarse level.

    Solves the pencil by congruence: with ``M_block = L L^T`` from the block
    Cholesky, ``A = L^-1 Hc L^-T`` is similar to ``M^-1 Hc``, so a standard
    symmetric eigensolve on ``A`` gives the generalized modes, back-transformed
    by ``x = L^-T y``. ``A`` is never formed: its inverse is ``L^T Hc^-1 L``.
    Shift-invert Lanczos on ``A`` targets the SOFTEST end, which is the end
    that matters: those are the modes the fine solve struggles with and the
    ones worth deflating.

    Fully traceable -- safe inside ``jit``, no host round-trips.

    Parameters
    ----------
    Hc : ndarray, shape (n_c, n_c)
        Symmetric, already shifted by ``-sigma``.
    blocks : ndarray, shape (m, b, b)
        Coarse block-diagonal of the preconditioner.
    Gs : ndarray of int, shape (m, b)
        Group index map; padding may be ``-1``.
    k : int
        Number of modes retained. Static.
    num_matvecs : int
        Lanczos steps. Static.
    ridge : float
        Static Cholesky ridge.
    seed : int

    Returns
    -------
    lam : jax.Array, shape (k,)
        Coarse generalized eigenvalues, ascending (softest first).
    X : jax.Array, shape (n_c, k)
        Unit-norm modes.

    Notes
    -----
    **No ridge escalation.** ``ridge`` is a static argument, so a non-SPD block
    yields NaN -- visible in the result, which is the safer failure.

    **The sign of the returned coarse eigenvalue does not predict success.** It
    was positive at both an inadequate and an adequate coarse resolution, and
    both landed on the correct negative fine mode. The coarse space is a useful
    subspace even when its own lowest Ritz value has not resolved the
    instability. Do not use it as a pre-flight check.
    """
    from jax.scipy.linalg import solve_triangular
    from matfree import decomp, eig

    Gs = jnp.asarray(Gs)
    mask = (Gs >= 0).astype(blocks.dtype)
    idx = jnp.where(Gs >= 0, Gs, 0)

    b = Gs.shape[-1]
    eye = jnp.eye(b, dtype=blocks.dtype)[None]
    L = jnp.linalg.cholesky(blocks + ridge * eye)
    mask3 = mask[..., None]

    def blk_solve(Mat, lower):
        """``L^-1 Mat`` (lower) or ``L^-H Mat``, columns batched.

        The groups PARTITION the reduced indices, so ``Mat[idx]`` is a permuted
        copy and one batched triangular solve covers every block. Works for any
        number of columns, which is why the reduction and the back-transform
        share it.
        """
        Lu = L if lower else jnp.conj(jnp.swapaxes(L, -1, -2))
        Y = Mat[idx] * mask3
        Zb = solve_triangular(Lu, Y, lower=lower) * mask3
        return jnp.zeros_like(Mat).at[idx].add(Zb)

    def blk_mul(x, adjoint):
        """``L x`` or ``L^H x`` for one vector."""
        Lu = jnp.conj(jnp.swapaxes(L, -1, -2)) if adjoint else L
        y = jnp.einsum("mij,mj->mi", Lu, x[idx] * mask) * mask
        return jnp.zeros_like(x).at[idx].add(y)

    # Shift-invert Lanczos on A = L^-1 Hc L^-H needs only A^-1 = L^H Hc^-1 L, so
    # Hc is factored in place of A: the coarse solve holds Hc and its factor, not
    # the congruence, its Hermitian part and their LU (several n_c x n_c copies).
    lu = jax.scipy.linalg.lu_factor(0.5 * (Hc + jnp.conj(Hc.T)))
    tri = decomp.tridiag_sym(num_matvecs, reortho="full", materialize=True)
    alg = eig.eigh_partial(tri)
    v0 = jax.random.normal(jax.random.PRNGKey(seed), (Hc.shape[0],), dtype=Hc.dtype)
    v0 = v0 / jnp.linalg.norm(v0)

    def inv_A(rhs):
        """``A^-1 rhs = L^H Hc^-1 L rhs``."""
        y = jax.scipy.linalg.lu_solve(lu, blk_mul(rhs, False))
        return blk_mul(y, True)

    mu, vecs = alg(inv_A, v0)

    lam_all = 1.0 / mu
    order = jnp.argsort(lam_all)[:k]  # ascending: softest first
    lam = lam_all[order]
    X = blk_solve(jnp.swapaxes(vecs[order], 0, 1), False)  # x = L^-H y
    X = X / jnp.linalg.norm(X, axis=0, keepdims=True)
    return lam, X


def coarse_seed_and_deflation(
    Hc, blocks_c, Gs_c, meta_c, meta_f, pr, pt, pz, k, num_matvecs, ridge=0.0, seed=3
):
    """Softest coarse generalized modes, prolonged to the fine grid.

    This is what makes the fine solve tractable: the coarse level is small
    enough to solve nearly exactly, and its softest modes -- prolonged -- are
    both a good starting vector and a deflation space that removes the fine
    operator's worst-conditioned directions.

    Parameters
    ----------
    Hc : ndarray, shape (n_c, n_c)
    blocks_c : ndarray, shape (m, b, b)
    Gs_c : ndarray of int, shape (m, b)
    meta_c, meta_f : dict
    pr, pt, pz : ndarray
    k, num_matvecs : int
    ridge : float
    seed : int

    Returns
    -------
    v0 : jax.Array, shape (n_f,)
        Unit-norm prolonged softest mode; the start vector.
    Z : jax.Array, shape (n_f, k)
        Prolonged deflation basis. Column 0 is ``v0`` up to scaling.
    lam_c : jax.Array, shape (k,)
        Coarse generalized eigenvalues, for reporting only.
    """
    lam_c, X_c = coarse_gen_modes(
        Hc, blocks_c, Gs_c, k, num_matvecs, ridge=ridge, seed=seed
    )
    P, _ = make_transfer(meta_c, meta_f, pr, pt, pz)
    # X_c is (n_c, k): vmap P over the k columns, then put k back on axis 1.
    Z = jnp.swapaxes(jax.vmap(P)(jnp.swapaxes(X_c, 0, 1)), 0, 1)
    v0 = Z[:, 0]
    v0 = v0 / jnp.linalg.norm(v0)
    return v0, Z, lam_c


def deflation_Y(Z, HZ, rcond=1e-12):
    """``Y`` for ``M^-1 = M_ring^-1 + Y Y^T``, fully traced, fixed shape.

    The obvious implementation selects surviving directions with BOOLEAN MASKS
    -- ``Z[:, live] @ Q[:, keep] / sqrt(w[keep])`` -- which is a variable-size
    gather plus an ``int(keep.sum())`` Python branch. Neither can be traced, so
    that form cannot be used under ``jit``.

    Same result at fixed shape: keep all ``k`` columns and ZERO the rejected
    ones. ``Y Y^T`` is unchanged, because a zero column contributes nothing to
    the outer product.

    Dead directions (``diag(Z^T H Z) <= 0``) are handled by zeroing those
    COLUMNS OF Z before the mixing, so whatever the eigenvectors do with them
    afterwards they multiply a zero column and cannot re-enter ``Y``.

    Parameters
    ----------
    Z : ndarray, shape (n, k)
    HZ : ndarray, shape (n, k)
    rcond : float

    Returns
    -------
    Y : jax.Array, shape (n, k)
    rank : jax.Array
        Number of directions that survived the ``rcond`` cut, as a traced
        scalar. Kept on device deliberately -- an ``int()`` here forces a host
        sync on every solve.
    """
    k = Z.shape[1]
    A2 = jnp.conj(jnp.swapaxes(Z, 0, 1)) @ HZ  # Z^H H Z: real diagonal
    A2 = 0.5 * (A2 + jnp.conj(jnp.swapaxes(A2, 0, 1)))
    dg = jnp.real(jnp.diagonal(A2))
    live = dg > 0.0
    d = jnp.where(live, jnp.sqrt(jnp.where(live, dg, 1.0)), 1.0)
    Hh = (A2 / d[:, None]) / d[None, :]
    eye = jnp.eye(k, dtype=A2.dtype)
    both = live[:, None] & live[None, :]
    # Dead rows/cols become identity so eigh stays well posed. Harmless: the
    # matching columns of Z are zeroed below.
    Hh = jnp.where(both, 0.5 * (Hh + jnp.conj(jnp.swapaxes(Hh, 0, 1))), eye)
    w, Q = jnp.linalg.eigh(Hh)
    keep = w > rcond * jnp.max(w)
    scale = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, w, 1.0)), 0.0)
    Zs = jnp.where(live[None, :], Z / d[None, :], 0.0)
    return (Zs @ Q) * scale[None, :], jnp.sum(keep)


# ---------------------------------------------------------------------------
# Ring block assembly
# ---------------------------------------------------------------------------


def ring_nodes(n_rho, n_theta, n_zeta, i, k):
    """Node indices of the poloidal ring at ``(rho_i, zeta_k)``, rho-major.

    Parameters
    ----------
    n_rho, n_theta, n_zeta : int
    i, k : int
        Radial and toroidal index of the ring.

    Returns
    -------
    ndarray of int, shape (n_theta,)
    """
    return np.array(
        [(i * n_theta + j) * n_zeta + k for j in range(n_theta)], dtype=np.int64
    )


def ring_index_maps(keep, res):
    """Static index arrays for the ring build. Grid structure only.

    ``alive`` depends only on which reduced DOFs exist -- the keep mask drops
    ``xi^rho`` on the first and last radial shell -- so it is a property of the
    GRID and can be computed once on the host. That is what turns the per-ring
    masking, a variable-size gather, into a fixed-shape traced gather that
    ``vmap`` can batch over all rings at once.

    Parameters
    ----------
    keep : ndarray of int
    res : tuple of int
        ``(n_rho, n_theta, n_zeta)``.

    Returns
    -------
    sel : jax.Array of int, shape (m, b)
        Positions WITHIN the ``3 * n_theta`` ring ordering that survive the keep
        mask, padded with 0.
    pad : jax.Array, shape (m, b)
        1.0 on real entries, 0.0 on padding.
    G : ndarray of int, shape (m, b)
        The reduced indices, ``-1`` padded.
    """
    n_rho, n_theta, n_zeta = res
    n_total = n_rho * n_theta * n_zeta
    keep = np.asarray(keep)
    full_to_red = -np.ones(3 * n_total, dtype=np.int64)
    full_to_red[keep] = np.arange(keep.size)

    raw = []
    for i in range(n_rho):
        for k in range(n_zeta):
            nodes = ring_nodes(n_rho, n_theta, n_zeta, i, k)
            raw.append(
                np.concatenate([full_to_red[c * n_total + nodes] for c in range(3)])
            )
    raw = np.asarray(raw, dtype=np.int64)

    b = int(max((r >= 0).sum() for r in raw))
    m = raw.shape[0]
    sel = np.zeros((m, b), dtype=np.int64)
    pad = np.zeros((m, b))
    G = -np.ones((m, b), dtype=np.int64)
    for gi, r in enumerate(raw):
        pos = np.flatnonzero(r >= 0)
        sel[gi, : pos.size] = pos
        pad[gi, : pos.size] = 1.0
        G[gi, : pos.size] = r[pos]
    return jnp.asarray(sel), jnp.asarray(pad), G


def build_ring_blocks(
    eq, diffmat, config, res, sel, pad, sigma, density=None, batch=24
):
    """Ring blocks of ``H = A - sigma I``, ``batch`` rings at a time.

    The eager form of the tail::

        sub = blk[ix_(alive, alive)];  blocks[gi, :na, :na] = sub - sigma*I
        blocks[gi, t, t] = 1 for t >= na

    is written here, with ``w = pad_i * pad_j``, as::

        blocks = sub*w - sigma*diag(pad) + diag(1 - pad)

    which is the same matrix: on real entries it is ``sub - sigma*I``; on padded
    rows ``w = 0`` kills ``sub``, ``diag(pad)`` kills the shift, and
    ``diag(1 - pad)`` leaves the inert identity the padding needs so the
    Cholesky stays defined.

    Parameters
    ----------
    eq : EquilibriumData
    diffmat : DiffMat
    config : AssemblyConfig
    res : tuple of int
        ``(n_rho, n_theta, n_zeta)``.
    sel, pad : ndarray
        From :func:`ring_index_maps`.
    sigma : float
        Shift. Pass 0 to assemble unshifted, so an adaptive second pass costs a
        diagonal subtraction rather than a full reassembly.
    density : ndarray, optional
    batch : int
        Rings assembled at once; the peak memory of the build scales with it.

    Returns
    -------
    jax.Array, shape (m, b, b)

    Notes
    -----
    This is the **traced** build: one vmapped assembly per batch of rings. A host
    loop over rings cannot survive a trace -- it needs a device round-trip and a
    variable-size boolean gather per ring. It reproduces the dense matrix's
    sub-blocks to ~5e-16.
    """
    from .assemble import assemble_dense, finish_ring_block

    n_rho, n_theta, n_zeta = res
    m, b = sel.shape
    nodes_all = jnp.asarray(
        np.stack(
            [
                ring_nodes(n_rho, n_theta, n_zeta, i, k)
                for i in range(n_rho)
                for k in range(n_zeta)
            ]
        )
    )

    def one_ring(nodes):
        out = assemble_dense(eq, diffmat, config, density=density, ring_nodes=nodes)
        return finish_ring_block(out["A"], out["Linv"], out["au_diag"], n_theta)

    blk = jax.lax.map(one_ring, nodes_all, batch_size=min(batch, m))
    rows = sel[:, :, None]
    cols = sel[:, None, :]
    ar = jnp.arange(m)[:, None, None]
    sub = blk[ar, rows, cols]
    sub = 0.5 * (sub + jnp.conj(jnp.swapaxes(sub, -1, -2)))  # Hermitian, not T
    w = pad[:, :, None] * pad[:, None, :]
    eye = jnp.eye(b, dtype=sub.dtype)[None]
    return sub * w - sigma * (pad[:, :, None] * eye) + (1.0 - pad)[:, :, None] * eye


def lanczos_shift_invert(opinv, n, dtype, sigma, num_matvecs, seed=0, v0=None):
    """Lanczos on ``opinv = (A - sigma I)^-1``: the eigenpair nearest ``sigma``.

    Returns ``(v, lam)`` with ``lam = sigma + 1 / mu`` for the largest ``|mu|``,
    the softest mode when ``sigma`` sits below the spectrum.
    """
    from matfree import decomp, eig

    tri = decomp.tridiag_sym(num_matvecs, reortho="full", materialize=True)
    if v0 is None:
        v0 = np.random.default_rng(seed).standard_normal(n)
    v0 = jnp.asarray(v0, dtype=dtype)
    mu, vecs = eig.eigh_partial(tri)(opinv, v0 / jnp.linalg.norm(v0))
    idx = jnp.argmax(jnp.abs(mu))
    return vecs[idx], sigma + 1.0 / jnp.where(mu[idx] == 0, jnp.inf, mu[idx])
