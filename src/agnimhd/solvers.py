"""Numerical machinery for the AGNI eigensolver.

The block ("ring") preconditioner, preconditioned CG with and without deflation,
the coarse-to-fine prolongation used to seed and deflate the fine solve, and the
coarse generalized eigensolve.

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

from .backend import errorif, jax, jnp

__all__ = [
    "GROUP_PARTITIONS",
    "adjoint_defect",
    "apply_space",
    "apply_space_t",
    "barycentric_matrix",
    "build_ring_blocks",
    "coarse_gen_modes",
    "coarse_seed_and_deflation",
    "deflation_Y",
    "factor_ring_blocks",
    "fourier_interp_matrix",
    "from_phys",
    "from_phys_h",
    "group_index_matrix",
    "level_meta",
    "make_block_precond",
    "make_transfer",
    "pcg",
    "pcg_deflated",
    "ring_index_maps",
    "ring_nodes",
    "to_phys",
    "to_phys_h",
    "transfer_matrices",
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


def barycentric_matrix(x_src, x_dst):
    """Barycentric interpolation matrix from ``x_src`` nodes to ``x_dst``.

    Spectrally accurate on the Gauss-Jacobi/Lobatto radial nodes, which is why
    radial transfer uses this rather than linear interpolation: the coarse mode
    has to be represented well enough to be a useful seed, and a low-order
    radial transfer would inject error exactly where the mode is sharpest.

    Parameters
    ----------
    x_src, x_dst : array-like

    Returns
    -------
    ndarray, shape (x_dst.size, x_src.size)

    Notes
    -----
    Rows where ``x_dst`` coincides with a source node are set to the exact
    delta, avoiding the 0/0 in the barycentric weights.
    """
    x_src = np.asarray(x_src, dtype=float)
    x_dst = np.asarray(x_dst, dtype=float)
    n = x_src.size
    w = np.ones(n)
    for j in range(n):
        w[j] = 1.0 / np.prod(x_src[j] - np.delete(x_src, j))
    mat = np.empty((x_dst.size, n), dtype=float)
    for i, x in enumerate(x_dst):
        hit = np.where(np.isclose(x, x_src, rtol=0.0, atol=1e-14))[0]
        if hit.size:
            mat[i, :] = 0.0
            mat[i, hit[0]] = 1.0
        else:
            tmp = w / (x - x_src)
            mat[i, :] = tmp / np.sum(tmp)
    return mat


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


def transfer_matrices(rho_c, rho_f, res_c, res_f, nfp):
    """The three 1-D interpolation matrices, coarse -> fine.

    Returned separately rather than as one Kronecker product because the tensor
    structure is what keeps the transfer cheap.

    Parameters
    ----------
    rho_c, rho_f : array-like
        Radial nodes of the two levels.
    res_c, res_f : tuple of int
        ``(n_rho, n_theta, n_zeta)`` of each level.
    nfp : int
        Field periods, setting the toroidal period.

    Returns
    -------
    pr, pt, pz : tuple of jax.Array
    """
    pr = barycentric_matrix(rho_c, rho_f)
    pt = fourier_interp_matrix(res_c[1], res_f[1], 2.0 * np.pi)
    pz = fourier_interp_matrix(res_c[2], res_f[2], 2.0 * np.pi / nfp)
    return jnp.asarray(pr), jnp.asarray(pt), jnp.asarray(pz)


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

#: Node groupings the block preconditioner can use. Each maps to one dense
#: block that gets factorized exactly.
#:
#: ``theta_line``
#:     One block per ``(rho, zeta)``, spanning all theta and all 3 components:
#:     block size ``3 * n_theta``. The poloidal "ring" -- the production choice,
#:     and the cheapest grouping that still captures the dominant poloidal
#:     coupling.
#: ``shell``
#:     One block per ``rho``, spanning all ``(theta, zeta)``: block size
#:     ``3 * n_theta * n_zeta``. Adds the toroidal coupling the rings drop, but
#:     block cost grows as ``n_zeta^3``, so it pays only when toroidal coupling
#:     actually limits convergence.
#: ``zeta_line``, ``radial_line``, ``node3``
#:     Toroidal lines, radial lines and the pointwise 3x3 -- diagnostics for
#:     locating which direction dominates the conditioning.
GROUP_PARTITIONS = ("theta_line", "shell", "zeta_line", "radial_line", "node3")


def _groups_theta_line(red, n_rho, n_theta, n_zeta):
    for i in range(n_rho):
        for k in range(n_zeta):
            yield [red(c, i, j, k) for c in range(3) for j in range(n_theta)]


def _groups_shell(red, n_rho, n_theta, n_zeta):
    for i in range(n_rho):
        yield [
            red(c, i, j, k)
            for c in range(3)
            for j in range(n_theta)
            for k in range(n_zeta)
        ]


def _groups_zeta_line(red, n_rho, n_theta, n_zeta):
    for i in range(n_rho):
        for j in range(n_theta):
            yield [red(c, i, j, k) for c in range(3) for k in range(n_zeta)]


def _groups_radial_line(red, n_rho, n_theta, n_zeta):
    for j in range(n_theta):
        for k in range(n_zeta):
            yield [red(c, i, j, k) for c in range(3) for i in range(n_rho)]


def _groups_node3(red, n_rho, n_theta, n_zeta):
    for i in range(n_rho):
        for j in range(n_theta):
            for k in range(n_zeta):
                yield [red(c, i, j, k) for c in range(3)]


_GROUP_BUILDERS = {
    "theta_line": _groups_theta_line,
    "shell": _groups_shell,
    "zeta_line": _groups_zeta_line,
    "radial_line": _groups_radial_line,
    "node3": _groups_node3,
}


def group_index_matrix(keep, res, partition="theta_line"):
    """``(m, b)`` reduced indices per group; ``-1`` pads groups short of ``b``.

    Dropped DOFs are **compacted out**, not left in place. The Dirichlet mask
    removes ``xi^rho`` on the innermost and outermost radial shells, so a
    boundary group has fewer live DOFs than an interior one. Rather than leave
    holes where they fell, each group's live indices are packed to the front and
    the row is padded with ``-1`` at the END; ``b`` is the longest LIVE group,
    which can be narrower than the nominal group size. Groups with no live DOFs
    are dropped entirely.

    This convention is load-bearing, not cosmetic: block assembly, the
    preconditioner apply and every recorded conditioning number were produced
    with it. Leaving the holes in place instead yields a different ``b``, a
    different block layout, and a preconditioner that silently misaligns against
    the blocks -- it does not error, it just makes CG worse.

    Parameters
    ----------
    keep : ndarray
        Indices of retained DOFs within the length-``3 * n_total`` vector.
    res : tuple of int
        ``(n_rho, n_theta, n_zeta)``.
    partition : str
        One of :data:`GROUP_PARTITIONS`.

    Returns
    -------
    Gs : ndarray of int, shape (m, b)
        Live reduced indices packed to the front of each row, ``-1`` after.

    Raises
    ------
    ValueError
        For an unknown partition, or if it produces no live groups.
    """
    errorif(
        partition not in GROUP_PARTITIONS,
        ValueError,
        f"unknown partition {partition!r}; expected one of {GROUP_PARTITIONS}",
    )
    n_rho, n_theta, n_zeta = res
    n_total = n_rho * n_theta * n_zeta
    keep = np.asarray(keep)
    full_to_red = -np.ones(3 * n_total, dtype=np.int64)
    full_to_red[keep] = np.arange(keep.size)

    def red(c, i, j, k):
        return full_to_red[c * n_total + (i * n_theta + j) * n_zeta + k]

    groups = list(_GROUP_BUILDERS[partition](red, n_rho, n_theta, n_zeta))
    groups = [[x for x in g if x >= 0] for g in groups]
    groups = [g for g in groups if g]
    errorif(not groups, ValueError, f"partition {partition!r} produced no live groups")
    b = max(len(g) for g in groups)
    Gs = -np.ones((len(groups), b), dtype=np.int64)
    for gi, g in enumerate(groups):
        Gs[gi, : len(g)] = g
    return Gs


def factor_ring_blocks(blocks, ridge=0.0, verbose=False):
    """Cholesky-factorize the block diagonal, escalating a ridge if needed.

    The blocks are the exact diagonal sub-blocks of ``H = A - sigma I``. When
    ``sigma`` sits below the spectrum ``H`` is SPD and so is every principal
    submatrix, so ``ridge=0`` should succeed. A ridge becomes necessary only
    when ``sigma`` has drifted into the spectrum.

    The ridge actually needed is returned, because it is a direct measure of how
    far the blocks are from positive definite. **A large ridge is a warning that
    the shift is wrong, not a knob to turn.**

    Parameters
    ----------
    blocks : ndarray, shape (m, b, b)
    ridge : float
        Fixed ridge to use. Non-positive means escalate automatically.
    verbose : bool

    Returns
    -------
    L : jax.Array or None
        Lower Cholesky factors, shape ``(m, b, b)``. None if every trial failed.
    ok : bool
    ridge_used : float or None

    Notes
    -----
    Chooses its ridge by reading a concrete bool off the factor, so this cannot
    be traced. Use :func:`factor_ring_blocks_traced` inside ``jit``.
    """
    b = blocks.shape[-1]
    eye = jnp.eye(b, dtype=blocks.dtype)[None]
    scale = float(jnp.mean(jnp.abs(jnp.diagonal(blocks, axis1=-2, axis2=-1))))
    trials = [ridge] if ridge > 0 else [0.0]
    if ridge <= 0:
        trials += [
            scale * f
            for f in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 1e3)
        ]
    for r in trials:
        L = jnp.linalg.cholesky(blocks + r * eye)
        if bool(jnp.all(jnp.isfinite(L))):
            if verbose:
                print(
                    f"[factor] cholesky ok with ridge={r:.6e} "
                    f"(mean |block diag| = {scale:.6e}, ratio {r / scale:.2e})",
                    flush=True,
                )
            return L, True, r
        if verbose:
            print(f"[factor] ridge={r:.6e} failed, escalating", flush=True)
    return None, False, None


def factor_ring_blocks_traced(blocks, ridge=0.0):
    """Cholesky at a FIXED ridge, safe under trace.

    :func:`factor_ring_blocks` selects its ridge by reading ``bool(...)`` off a
    traced array, which cannot be traced. This variant factors once at the given
    ridge and reports finiteness as a traced flag instead. A non-SPD block
    therefore yields NaN rather than escalating -- visible in the result, which
    is the safer failure inside a jitted solve.

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
    would accumulate. For the partitions in :data:`GROUP_PARTITIONS` the groups
    are disjoint and the two coincide -- but ``add`` is the correct operation
    for the additive Schwarz form this is.
    """
    from jax.scipy.linalg import solve_triangular

    Gs = jnp.asarray(Gs)
    mask = (Gs >= 0).astype(jnp.result_type(float))
    idx = jnp.where(Gs >= 0, Gs, 0)

    def M(r):
        y = r[idx] * mask.astype(r.dtype)
        z = solve_triangular(L, y[..., None], lower=True)
        z = solve_triangular(jnp.swapaxes(L, -1, -2), z, lower=False)[..., 0]
        z = z * mask.astype(z.dtype)
        return jnp.zeros((n,), dtype=r.dtype).at[idx].add(z)

    return M


# ---------------------------------------------------------------------------
# Preconditioned CG, with and without deflation
# ---------------------------------------------------------------------------


def pcg(Hf, b_rhs, M, tol, maxiter):
    """Preconditioned CG that reports its own iteration count.

    Parameters
    ----------
    Hf : callable
        Applies ``H``. Must be symmetric, and SPD for CG to be legal.
    b_rhs : ndarray
    M : callable
        Preconditioner apply.
    tol : float
        Relative-residual tolerance.
    maxiter : int
        Iteration cap. Hitting it is **not** an error.

    Returns
    -------
    x, iters, relres
        As *traced* arrays, not Python scalars. Do not convert here: this runs
        as the ``OPinv`` inside a Lanczos iteration that is itself under a jit
        trace, and forcing concretization breaks that caller.

    Warnings
    --------
    **Never read ``relres`` as a quality proxy on the AGNI operator.** It is
    anti-correlated with accuracy: a run with relative residual 1.42 gave an
    answer 0.10% from truth while one at 0.91 gave 7.9%. Neither converged in
    the residual sense.
    """
    bnorm = jnp.linalg.norm(b_rhs)

    def body(state):
        x, r, p, rz, k, _ = state
        Ap = Hf(p)
        alpha = rz / jnp.vdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M(r)
        rz_new = jnp.vdot(r, z)
        p = z + (rz_new / rz) * p
        return (x, r, p, rz_new, k + 1, jnp.linalg.norm(r) / bnorm)

    def cond(state):
        _, _, _, _, k, relres = state
        return (k < maxiter) & (relres > tol)

    x0 = jnp.zeros_like(b_rhs)
    r0 = b_rhs
    z0 = M(r0)
    state = (x0, r0, z0, jnp.vdot(r0, z0), 0, jnp.array(1.0))
    x, r, p, rz, k, relres = jax.lax.while_loop(cond, body, state)
    return x, k, relres


def _make_deflation(Hf, Z):
    """Build the ``H``-orthogonal projector pair for a deflation space ``Z``.

    Returns ``(project, correct, deflate_x)``.

    ``project(v) = v - H Z (Z^T H Z)^-1 Z^T v`` removes, from a **residual**,
    the components CG cannot resolve. ``correct(b)`` supplies the exact
    solution inside ``span(Z)``. ``deflate_x(x) = x - Z (Z^T H Z)^-1 (H Z)^T x``
    is the matching projector on the **solution** side: it returns the part of
    ``x`` that is ``H``-orthogonal to ``Z``, satisfying ``Z^T H deflate_x(x) =
    0``, and it is what an initial guess has to pass through -- see
    :func:`pcg_deflated`.

    The point of deflation: the modes that make ``H`` ill-conditioned are the
    softest ones, and those are exactly what the coarse level resolves well. If
    they are solved exactly and projected out, CG only has to work on the rest,
    where the preconditioned spectrum is clustered.
    """
    HZ = jax.vmap(Hf, in_axes=1, out_axes=1)(Z)
    ZtHZ = Z.T @ HZ
    # Z^T H Z is symmetric in exact arithmetic; forcing it keeps the Cholesky
    # below well posed.
    ZtHZ = 0.5 * (ZtHZ + ZtHZ.T)
    chol = jax.scipy.linalg.cho_factor(ZtHZ, lower=True)

    def coarse_solve(rhs):
        return jax.scipy.linalg.cho_solve(chol, rhs)

    def project(v):
        return v - HZ @ coarse_solve(Z.T @ v)

    def correct(b_rhs):
        return Z @ coarse_solve(Z.T @ b_rhs)

    def deflate_x(x):
        return x - Z @ coarse_solve(HZ.T @ x)

    return project, correct, deflate_x


def pcg_deflated(Hf, b_rhs, M, tol, maxiter, Z=None, x0=None):
    """Preconditioned CG with optional deflation by a coarse space ``Z``.

    With ``Z=None`` this is exactly :func:`pcg`. With ``Z`` supplied the solve
    splits: the component in ``span(Z)`` comes from a direct solve of the small
    ``Z^T H Z`` system, and CG runs only on the ``H``-orthogonal complement.

    Parameters
    ----------
    Hf : callable
        Applies ``H = A - sigma I``. Must be symmetric, and SPD for CG to be
        legal -- which requires ``sigma`` below the spectrum.
    b_rhs : ndarray
    M : callable
        Preconditioner apply, e.g. from :func:`make_block_precond`.
    tol : float
    maxiter : int
    Z : ndarray, shape (n, k), optional
        Deflation basis, typically prolonged coarse modes.
    x0 : ndarray, optional
        Initial guess, typically a prolonged coarse eigenvector.

    Returns
    -------
    x, iters, relres
        Traced arrays; see :func:`pcg`.
    """
    if Z is None:
        if x0 is None:
            return pcg(Hf, b_rhs, M, tol, maxiter)
        # Shift to a zero initial guess by solving for the correction.
        r0 = b_rhs - Hf(x0)
        dx, k, relres = pcg(Hf, r0, M, tol, maxiter)
        return x0 + dx, k, relres

    project, correct, deflate_x = _make_deflation(Hf, Z)

    x_coarse = correct(b_rhs)
    # The initial guess must be H-orthogonalized against Z before use. Adding a
    # raw x0 to x_coarse counts the span(Z) component of the solution TWICE --
    # once exactly, from the coarse solve, and once from the guess -- and CG,
    # which now runs only on the complement, has no way to remove it. The
    # symptom is a converged-looking solve (relative residual 9.6e-12) whose
    # answer is 89% wrong; deflation alone and a seed alone are both correct,
    # so nothing short of comparing against a direct solve catches it.
    x_seed = 0.0 if x0 is None else deflate_x(x0)
    r0 = project(b_rhs - Hf(x_coarse + x_seed))

    def MP(r):
        return project(M(r))

    def HP(v):
        return project(Hf(v))

    dx, k, relres = pcg(HP, r0, MP, tol, maxiter)
    return x_coarse + x_seed + dx, k, relres


# ---------------------------------------------------------------------------
# Coarse generalized eigensolve and the deflation space it supplies
# ---------------------------------------------------------------------------


def coarse_gen_modes(Hc, blocks, Gs, k, num_matvecs, ridge=0.0, seed=3):
    """Softest ``k`` generalized modes of ``(Hc, M_block)`` on the coarse level.

    Solves the pencil by congruence: with ``M_block = L L^T`` from the block
    Cholesky, ``A = L^-1 Hc L^-T`` is similar to ``M^-1 Hc``, so a standard
    symmetric eigensolve on ``A`` gives the generalized modes, back-transformed
    by ``x = L^-T y``. Shift-invert Lanczos on ``A`` targets the SOFTEST end,
    which is the end that matters: those are the modes the fine solve struggles
    with and the ones worth deflating.

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
    **No ridge escalation.** :func:`factor_ring_blocks` chooses its ridge by
    reading a concrete bool off a traced array, which cannot be traced, so
    ``ridge`` is a static argument here. A non-SPD block therefore yields NaN
    rather than silently escalating -- visible in the result, which is the safer
    failure.

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
        """``L^-1 Mat`` (lower) or ``L^-T Mat``, columns batched.

        The groups PARTITION the reduced indices, so ``Mat[idx]`` is a permuted
        copy and one batched triangular solve covers every block. Works for any
        number of columns, which is why the reduction and the back-transform
        share it.
        """
        Lu = L if lower else jnp.swapaxes(L, -1, -2)
        Y = Mat[idx] * mask3
        Zb = solve_triangular(Lu, Y, lower=lower) * mask3
        return jnp.zeros_like(Mat).at[idx].add(Zb)

    # A = L^-1 Hc L^-T, formed as L^-1 (L^-1 Hc)^T using the symmetry of Hc.
    A = blk_solve(jnp.swapaxes(blk_solve(Hc, True), 0, 1), True)
    A = 0.5 * (A + jnp.swapaxes(A, 0, 1))

    lu = jax.scipy.linalg.lu_factor(A)
    tri = decomp.tridiag_sym(num_matvecs, reortho="full", materialize=True)
    alg = eig.eigh_partial(tri)
    v0 = jax.random.normal(jax.random.PRNGKey(seed), (A.shape[0],), dtype=Hc.dtype)
    v0 = v0 / jnp.linalg.norm(v0)
    mu, vecs = alg(lambda rhs: jax.scipy.linalg.lu_solve(lu, rhs), v0)

    lam_all = 1.0 / mu
    order = jnp.argsort(lam_all)[:k]  # ascending: softest first
    lam = lam_all[order]
    X = blk_solve(jnp.swapaxes(vecs[order], 0, 1), False)  # x = L^-T y
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
        Unit-norm prolonged softest mode; the Lanczos/CG start vector.
    Z : jax.Array, shape (n_f, k)
        Prolonged deflation basis. Column 0 is ``v0`` up to scaling.
    lam_c : jax.Array, shape (k,)
        Coarse generalized eigenvalues, for reporting only.

    Warnings
    --------
    **The coarse radial resolution is a correctness threshold, not a cost
    knob.** Below the floor the fine solve does not return a less accurate
    eigenvalue -- it returns the wrong mode, with the opposite sign. Measured at
    fine 24x12x8: coarse 8 gave ``+2.070e-03`` against a true ``-1.337622e-04``,
    coarse 12 gave -1.2323e-04 (7.9% off), coarse 16 gave -1.33623e-04 (0.10%
    off). **The floor is 16, and it costs nothing** -- coarse 16 ran in 238 s
    against coarse 12's 274 s. See ``docs/resolution.md``.
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
    A2 = jnp.swapaxes(Z, 0, 1) @ HZ
    A2 = 0.5 * (A2 + jnp.swapaxes(A2, 0, 1))
    dg = jnp.diagonal(A2)
    live = dg > 0.0
    d = jnp.where(live, jnp.sqrt(jnp.where(live, dg, 1.0)), 1.0)
    Hh = (A2 / d[:, None]) / d[None, :]
    eye = jnp.eye(k, dtype=A2.dtype)
    both = live[:, None] & live[None, :]
    # Dead rows/cols become identity so eigh stays well posed. Harmless: the
    # matching columns of Z are zeroed below.
    Hh = jnp.where(both, 0.5 * (Hh + jnp.swapaxes(Hh, 0, 1)), eye)
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
        The reduced indices, ``-1`` padded, as :func:`group_index_matrix`
        returns.
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


def build_ring_blocks(eq, diffmat, config, res, sel, pad, sigma, density=None):
    """Ring blocks of ``H = A - sigma I``, all rings at once under ``vmap``.

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

    Returns
    -------
    jax.Array, shape (m, b, b)

    Notes
    -----
    This is the **traced** build: one vmapped assembly over all rings. A host
    loop over rings cannot survive a trace -- it needs a device round-trip and a
    variable-size boolean gather per ring -- which is what made the deflated
    path unusable under ``jit``, and ``jit`` is the production path. The two
    builds are numerically identical; both reproduce the dense matrix's
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

    blk = jax.vmap(one_ring)(nodes_all)
    rows = sel[:, :, None]
    cols = sel[:, None, :]
    ar = jnp.arange(m)[:, None, None]
    sub = blk[ar, rows, cols]
    sub = 0.5 * (sub + jnp.swapaxes(sub, -1, -2))
    w = pad[:, :, None] * pad[:, None, :]
    eye = jnp.eye(b, dtype=sub.dtype)[None]
    return sub * w - sigma * (pad[:, :, None] * eye) + (1.0 - pad)[:, :, None] * eye
