"""Linear-algebra machinery behind the matrix-free path.

These pieces are what let AGNI run above the resolution where a dense matrix
fits, and they fail *quietly*. A misaligned preconditioner does not error, it
just makes CG slower. A restriction that is not the exact transpose of the
prolongation does not error, it silently stops the deflated iteration from
being a legal Krylov method. So each property is asserted directly rather than
inferred from an end-to-end answer.

Wherever a small synthetic SPD problem can carry the property, it is used --
those are fast and exact. The pieces that only mean something on the real
operator (ring blocks, the transfer between two real levels) are checked
against the shipped equilibrium.
"""

import numpy as np
import pytest
import scipy.linalg

from agnimhd.assemble import keep_indices, matfree_operator
from agnimhd.backend import jax, jnp
from agnimhd.solvers import (
    adjoint_defect,
    build_ring_blocks,
    coarse_gen_modes,
    deflation_Y,
    factor_ring_blocks_traced,
    fourier_interp_matrix,
    from_phys,
    jacobi_davidson,
    level_meta,
    make_block_precond,
    make_transfer,
    ring_index_maps,
    ring_nodes,
    to_phys,
)

# ---------------------------------------------------------------------------
# Synthetic SPD problems
# ---------------------------------------------------------------------------


def _spd(n, cond=1e4, seed=0):
    """A symmetric positive definite matrix with a prescribed condition number."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    w = np.logspace(0, np.log10(cond), n)
    return (Q * w) @ Q.T


def _block_diag_spd(m, b, seed=0):
    """``(m, b, b)`` stack of SPD blocks, and the dense matrix they form."""
    blocks = np.stack([_spd(b, cond=50.0, seed=seed + i) for i in range(m)])
    return blocks, scipy.linalg.block_diag(*blocks)


# ---------------------------------------------------------------------------
# Interpolation matrices
# ---------------------------------------------------------------------------


def test_fourier_interp_is_exact_on_representable_modes():
    """Both grids are uniform and periodic, so this is exact, not approximate."""
    n_src, n_dst, period = 8, 20, 2.0 * np.pi
    P = fourier_interp_matrix(n_src, n_dst, period)
    x = np.arange(n_src) * period / n_src
    y = np.arange(n_dst) * period / n_dst
    for m in range(-(n_src // 2) + 1, n_src // 2):
        for fn in (np.cos, np.sin):
            assert np.max(np.abs(P @ fn(m * x) - fn(m * y))) < 1e-12, f"mode {m}"


def test_fourier_interp_is_real_and_respects_the_field_period():
    """The toroidal period is ``2*pi/NFP``, not ``2*pi``."""
    P = fourier_interp_matrix(6, 15, 2.0 * np.pi / 4)
    assert P.dtype == np.float64
    x = np.arange(6) * (2.0 * np.pi / 4) / 6
    y = np.arange(15) * (2.0 * np.pi / 4) / 15
    # NFP = 4, so mode n on the reduced grid is toroidal mode 4n.
    assert np.max(np.abs(P @ np.cos(4 * x) - np.cos(4 * y))) < 1e-12


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


def test_partition_tiles_every_dof_exactly_once():
    """Every live reduced DOF appears in exactly one ring, and none twice.

    This is what makes the block preconditioner a valid additive Schwarz
    operator. A DOF covered twice would be preconditioned twice; one covered
    never would not be preconditioned at all. Neither errors.
    """
    res = (5, 6, 4)
    keep = keep_indices(*res)
    _, pad, G = ring_index_maps(keep, res)
    live = G[G >= 0]
    assert live.size == np.asarray(keep).size, "the rings do not cover every DOF"
    assert np.array_equal(np.sort(live), np.arange(live.size)), "a DOF repeats"
    np.testing.assert_array_equal(np.asarray(pad) > 0, G >= 0)


def test_ring_nodes_is_rho_major():
    """Ring node indices follow the same ordering as everything else."""
    n_rho, n_theta, n_zeta = 5, 6, 4
    got = ring_nodes(n_rho, n_theta, n_zeta, 2, 3)
    want = [(2 * n_theta + j) * n_zeta + 3 for j in range(n_theta)]
    assert got.tolist() == want


# ---------------------------------------------------------------------------
# Block preconditioner
# ---------------------------------------------------------------------------


def test_block_precond_inverts_the_block_diagonal_exactly():
    """``M`` is the exact inverse of the block-diagonal it was built from."""
    m, b = 4, 5
    blocks, dense = _block_diag_spd(m, b)
    Gs = np.arange(m * b).reshape(m, b)
    L, ok, _ = factor_ring_blocks_traced(jnp.asarray(blocks))
    assert ok, "SPD blocks should factor with no ridge"
    M = make_block_precond(L, Gs, m * b)
    rng = np.random.default_rng(0)
    r = rng.standard_normal(m * b)
    got = np.asarray(M(jnp.asarray(r)))
    want = np.linalg.solve(dense, r)
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-12


def test_block_precond_is_symmetric():
    """CG requires an SPD preconditioner; asymmetry breaks the method silently."""
    m, b = 3, 4
    blocks, _ = _block_diag_spd(m, b)
    Gs = np.arange(m * b).reshape(m, b)
    L, ok, _ = factor_ring_blocks_traced(jnp.asarray(blocks))
    assert ok
    M = make_block_precond(L, Gs, m * b)
    n = m * b
    Mmat = np.stack(
        [np.asarray(M(jnp.zeros(n).at[j].set(1.0))) for j in range(n)], axis=1
    )
    assert np.max(np.abs(Mmat - Mmat.T)) < 1e-13
    assert np.min(np.linalg.eigvalsh(0.5 * (Mmat + Mmat.T))) > 0.0


def test_block_precond_ignores_padded_slots():
    """A ``-1`` slot contributes nothing on either the gather or the scatter."""
    m, b = 3, 4
    blocks, _ = _block_diag_spd(m, b)
    Gs = np.arange(m * b).reshape(m, b).astype(np.int64)
    Gs[1, -1] = -1  # drop one DOF from the middle block
    L, ok, _ = factor_ring_blocks_traced(jnp.asarray(blocks))
    assert ok
    M = make_block_precond(L, Gs, m * b)
    out = np.asarray(M(jnp.ones(m * b)))
    dropped = 1 * b + (b - 1)
    assert out[dropped] == 0.0, "a padded slot received a contribution"


def test_factor_ring_blocks_traced_reports_nan_instead_of_escalating():
    """Under trace the failure must be visible in the result, not a branch."""
    blocks, _ = _block_diag_spd(3, 4)
    bad = blocks - 2.0 * np.max(np.linalg.eigvalsh(blocks[0])) * np.eye(4)[None]
    _, ok, _ = factor_ring_blocks_traced(jnp.asarray(bad), ridge=0.0)
    assert not bool(ok)
    L, ok_good, _ = factor_ring_blocks_traced(jnp.asarray(blocks), ridge=0.0)
    assert bool(ok_good)
    assert np.all(np.isfinite(np.asarray(L)))


def test_factor_ring_blocks_traced_survives_jit():
    """It is the variant meant for the production jitted path."""
    blocks, _ = _block_diag_spd(3, 4)
    fn = jax.jit(lambda B: factor_ring_blocks_traced(B, ridge=0.0)[0])
    assert np.all(np.isfinite(np.asarray(fn(jnp.asarray(blocks)))))


# ---------------------------------------------------------------------------
# Deflation
# ---------------------------------------------------------------------------


def test_jacobi_davidson_finds_the_softest_pair():
    """Spectrum shaped like the fixture's (two negative modes, a null cluster,
    a long positive tail), identity preconditioner, fixed shift below the
    spectrum. With and without deflation by the next modes, under jit, and with
    the Ritz-value stop instead of the residual stop."""
    rng = np.random.default_rng(0)
    n = 300
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    w = np.sort(np.r_[-1.3e-4, -6e-5, 1e-11 + 1e-3 * rng.random(n - 2)])
    A = jnp.asarray((Q * w) @ Q.T)
    v0 = jnp.asarray(rng.standard_normal(n))
    kw = dict(sigma=-1e-3, inner=50, tol=1e-8, theta_tol=0.0)
    for Z in (None, jnp.asarray(Q[:, 1:4])):
        th, v, info = jacobi_davidson(lambda x: A @ x, lambda x: x, v0, Z, **kw)
        assert abs(float(th) - w[0]) / abs(w[0]) < 1e-8
        assert float(info["resid"]) < 1e-8 and int(info["iters"]) < 200
        assert abs(np.vdot(np.asarray(v), Q[:, 0])) > 1.0 - 1e-10
    jd = jax.jit(lambda u: jacobi_davidson(lambda x: A @ x, lambda x: x, u, **kw)[0])
    assert abs(float(jd(v0)) - w[0]) / abs(w[0]) < 1e-8
    kw.update(tol=0.0, theta_tol=1e-12)
    th, _, _ = jacobi_davidson(lambda x: A @ x, lambda x: x, v0, **kw)
    assert abs(float(th) - w[0]) / abs(w[0]) < 1e-8


def test_deflation_Y_reproduces_the_masked_construction():
    """The fixed-shape form equals the boolean-mask form it replaced.

    The readable version selects surviving directions with boolean masks, which
    cannot be traced. This version keeps all ``k`` columns and zeroes the
    rejected ones; ``Y Y^T`` must be identical, since a zero column contributes
    nothing to the outer product.
    """
    n, k = 50, 6
    A = _spd(n, cond=1e5, seed=13)
    rng = np.random.default_rng(14)
    Z = jnp.asarray(np.linalg.qr(rng.standard_normal((n, k)))[0])
    HZ = jnp.asarray(A) @ Z

    Y, rank = deflation_Y(Z, HZ)
    assert int(rank) == k, "a well-conditioned space should keep every direction"

    # Reference: Y Y^T must be the inverse of Z^T H Z pulled back through Z.
    A2 = np.asarray(Z).T @ np.asarray(HZ)
    A2 = 0.5 * (A2 + A2.T)
    want = np.asarray(Z) @ np.linalg.inv(A2) @ np.asarray(Z).T
    got = np.asarray(Y) @ np.asarray(Y).T
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-8


def test_deflation_Y_zeroes_dead_directions():
    """A direction with ``z^T H z <= 0`` cannot re-enter ``Y``.

    Dead directions are removed by **zeroing their columns of Z**, before the
    eigen-mixing, so no later rotation can bring them back. They are not
    removed by the ``rcond`` cut, and the returned ``rank`` counts only that
    cut -- dead rows are replaced by identity rows to keep ``eigh`` well posed,
    and those contribute eigenvalue 1, which survives ``rcond``. So ``rank`` is
    not a count of live directions, and the property to assert is the rank of
    ``Y`` itself.
    """
    n, k = 40, 5
    A = _spd(n, cond=1e3, seed=17)
    rng = np.random.default_rng(18)
    Z = np.linalg.qr(rng.standard_normal((n, k)))[0]
    HZ = A @ Z
    HZ[:, 2] = -A @ Z[:, 2]  # force diag(Z^T H Z)[2] < 0
    Y, rank = deflation_Y(jnp.asarray(Z), jnp.asarray(HZ))
    assert np.all(np.isfinite(np.asarray(Y)))
    assert (
        np.linalg.matrix_rank(np.asarray(Y), tol=1e-8) == k - 1
    ), "the dead direction still spans a direction of Y"
    # Shape is fixed regardless -- that is what makes it traceable.
    assert np.asarray(Y).shape == (n, k)
    assert int(rank) <= k


def test_deflation_Y_survives_jit():
    """Fixed shapes are the whole point of this construction."""
    n, k = 30, 4
    A = _spd(n, cond=1e3, seed=19)
    rng = np.random.default_rng(20)
    Z = jnp.asarray(np.linalg.qr(rng.standard_normal((n, k)))[0])
    Y, rank = jax.jit(deflation_Y)(Z, jnp.asarray(A) @ Z)
    assert np.asarray(Y).shape == (n, k)
    assert int(rank) == k


# ---------------------------------------------------------------------------
# Coarse generalized eigensolve
# ---------------------------------------------------------------------------


def test_coarse_gen_modes_matches_a_dense_generalized_eigensolve():
    """The congruence route reproduces ``scipy.linalg.eigh(Hc, M)``.

    ``coarse_gen_modes`` solves the pencil by Cholesky congruence rather than
    calling a generalized eigensolver, because the congruence route is
    traceable. It has to give the same modes.
    """
    m, b = 5, 4
    n = m * b
    blocks, block_dense = _block_diag_spd(m, b, seed=23)
    Hc = _spd(n, cond=1e3, seed=24)
    Gs = np.arange(n).reshape(m, b)
    k = 3

    lam, X = coarse_gen_modes(
        jnp.asarray(Hc), jnp.asarray(blocks), Gs, k, num_matvecs=n, seed=1
    )
    want = scipy.linalg.eigh(Hc, block_dense, eigvals_only=True)[:k]
    np.testing.assert_allclose(np.asarray(lam), want, rtol=1e-8)

    # And the returned vectors really solve the pencil.
    X = np.asarray(X)
    for j in range(k):
        resid = Hc @ X[:, j] - want[j] * (block_dense @ X[:, j])
        assert np.linalg.norm(resid) / np.linalg.norm(Hc @ X[:, j]) < 1e-7


def test_coarse_gen_modes_returns_the_softest_end_ascending():
    """Shift-invert targets the softest modes; the order is the contract."""
    m, b = 4, 4
    n = m * b
    blocks, _ = _block_diag_spd(m, b, seed=31)
    Hc = _spd(n, cond=1e3, seed=32)
    lam, X = coarse_gen_modes(
        jnp.asarray(Hc),
        jnp.asarray(blocks),
        np.arange(n).reshape(m, b),
        4,
        num_matvecs=n,
        seed=1,
    )
    lam = np.asarray(lam)
    assert np.all(np.diff(lam) >= -1e-12), "eigenvalues are not ascending"
    np.testing.assert_allclose(np.linalg.norm(np.asarray(X), axis=0), 1.0, atol=1e-10)


def test_coarse_gen_modes_survives_jit():
    """It runs inside the jitted two-level solve."""
    m, b = 4, 3
    n = m * b
    blocks, _ = _block_diag_spd(m, b, seed=41)
    Hc = _spd(n, cond=1e2, seed=42)
    Gs = np.arange(n).reshape(m, b)
    fn = jax.jit(lambda H, B: coarse_gen_modes(H, B, Gs, 2, num_matvecs=n, seed=1)[0])
    assert np.all(np.isfinite(np.asarray(fn(jnp.asarray(Hc), jnp.asarray(blocks)))))


# ---------------------------------------------------------------------------
# On the real operator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fine_op(eq_data, diffmat, config):
    """The matrix-free operator on the shipped case."""
    return matfree_operator(eq_data, diffmat, config)


def test_reduced_and_physical_coordinates_round_trip(fine_op):
    """``from_phys(to_phys(q)) == q`` on the retained DOFs.

    The transform is a per-node Cholesky congruence, so its inverse is formed
    rather than solved. Round-tripping is how a wrong inverse gets caught.
    """
    meta = level_meta(fine_op)
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal(meta["n_keep"]))
    back = np.asarray(from_phys(meta, to_phys(meta, q)))
    rel = np.max(np.abs(back - np.asarray(q))) / np.max(np.abs(np.asarray(q)))
    assert rel < 1e-10, f"round trip is off by {rel:.3e}"


def test_prolongation_and_restriction_are_exact_adjoints(
    eq_data, diffmat, config, fine_op
):
    """``<P q_c, q_f> == <q_c, PT q_f>`` to machine precision.

    ``PT`` is the true transpose of ``P``, not an inverse and not a separately
    derived restriction. If they drift apart the deflated CG loses symmetry and
    stops being a valid Krylov method -- and nothing about the run announces
    that. Values near 1e-14 are expected.
    """
    from agnimhd.assemble import matfree_operator as _op

    n_rho_f, n_theta, n_zeta = eq_data.resolution

    # A coarse level on the same grid is the degenerate case, and it is the one
    # that can be built without a second equilibrium export: the transfer is
    # then the identity in space but still exercises the full reduced <->
    # physical machinery on both sides.
    meta_f = level_meta(fine_op)
    pr = np.eye(n_rho_f)
    pt = fourier_interp_matrix(n_theta, n_theta, 2.0 * np.pi)
    pz = fourier_interp_matrix(n_zeta, n_zeta, 2.0 * np.pi / eq_data.NFP)
    P, PT = make_transfer(meta_f, meta_f, pr, pt, pz)
    defect = adjoint_defect(P, PT, meta_f["n_keep"], meta_f["n_keep"], trials=6)
    assert defect < 1e-12, f"P and PT are not adjoint: defect {defect:.3e}"
    del _op


def test_identity_transfer_is_the_identity(fine_op, eq_data):
    """Same-grid prolongation must not perturb the vector it carries."""
    meta = level_meta(fine_op)
    n_rho, n_theta, n_zeta = eq_data.resolution
    pr = np.eye(n_rho)
    pt = fourier_interp_matrix(n_theta, n_theta, 2.0 * np.pi)
    pz = fourier_interp_matrix(n_zeta, n_zeta, 2.0 * np.pi / eq_data.NFP)
    P, _ = make_transfer(meta, meta, pr, pt, pz)
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal(meta["n_keep"]))
    rel = np.max(np.abs(np.asarray(P(q)) - np.asarray(q))) / np.max(
        np.abs(np.asarray(q))
    )
    assert rel < 1e-10, f"identity transfer moved the vector by {rel:.3e}"


def test_ring_blocks_are_the_dense_diagonal_blocks(eq_data, diffmat, config, dense):
    """The vmapped ring build reproduces the dense matrix's own sub-blocks.

    Recorded agreement is ~1e-16 relative. This is the check that the
    preconditioner is built from the operator actually being solved -- a ring
    block assembled from a different expression would still be SPD, still
    factor, and still make CG converge to the wrong problem's answer.
    """
    A = np.asarray(dense["A"])
    res = eq_data.resolution
    keep = keep_indices(*res)
    sel, pad, G = ring_index_maps(keep, res)
    blocks = np.asarray(
        build_ring_blocks(eq_data, diffmat, config, res, sel, pad, sigma=0.0)
    )
    assert blocks.shape[0] == G.shape[0]

    scale = np.max(np.abs(A))
    worst = 0.0
    for gi in (0, 1, G.shape[0] // 2, G.shape[0] - 1):
        idx = G[gi][G[gi] >= 0]
        want = A[np.ix_(idx, idx)]
        want = 0.5 * (want + want.T)
        got = blocks[gi][: idx.size, : idx.size]
        worst = max(worst, np.max(np.abs(got - want)) / scale)
    assert worst < 1e-14, f"ring blocks differ from the dense sub-blocks: {worst:.3e}"


def test_ring_block_padding_is_an_inert_identity(eq_data, diffmat, config):
    """Padded rows carry a 1 on the diagonal so the Cholesky stays defined."""
    res = eq_data.resolution
    keep = keep_indices(*res)
    sel, pad, G = ring_index_maps(keep, res)
    blocks = np.asarray(
        build_ring_blocks(eq_data, diffmat, config, res, sel, pad, sigma=0.0)
    )
    padded = np.flatnonzero(np.asarray(pad)[0] == 0.0)
    if padded.size == 0:
        pytest.fail("no padded ring in the shipped case; the test cannot run")
    for t in padded:
        row = blocks[0, t]
        assert row[t] == 1.0
        assert np.max(np.abs(np.delete(row, t))) == 0.0


def test_ring_blocks_apply_the_shift_only_to_live_entries(eq_data, diffmat, config):
    """``sigma`` shifts the real diagonal; padding stays at 1."""
    res = eq_data.resolution
    keep = keep_indices(*res)
    sel, pad, G = ring_index_maps(keep, res)
    kwargs = dict(res=res, sel=sel, pad=pad)
    b0 = np.asarray(build_ring_blocks(eq_data, diffmat, config, sigma=0.0, **kwargs))
    b1 = np.asarray(build_ring_blocks(eq_data, diffmat, config, sigma=-0.1, **kwargs))
    diff = b1 - b0
    live = np.asarray(pad) > 0
    on_diag = np.stack([np.diagonal(d) for d in diff])
    np.testing.assert_allclose(on_diag[live], 0.1, atol=1e-12)
    np.testing.assert_allclose(on_diag[~live], 0.0, atol=0.0)
    # Nothing off-diagonal moved.
    off = diff - np.stack([np.diag(np.diagonal(d)) for d in diff])
    assert np.max(np.abs(off)) == 0.0


def test_ring_preconditioner_helps_on_the_real_operator(axisym_case):
    """On the actual AGNI operator (the complex one-plane case), Jacobi-Davidson
    with the ring preconditioner converges from a random start; without it, it
    has not converged after 200 outer iterations."""
    from agnimhd.assemble import assemble_dense

    eq, dm, cfg = axisym_case
    A = np.asarray(assemble_dense(eq, dm, cfg)["A"])
    sigma = 1.3 * float(np.linalg.eigvalsh(A)[0])
    op = matfree_operator(eq, dm, cfg)
    sel, pad, G = ring_index_maps(keep_indices(*eq.resolution), eq.resolution)
    blocks = build_ring_blocks(eq, dm, cfg, eq.resolution, sel, pad, sigma=sigma)
    M = make_block_precond(factor_ring_blocks_traced(blocks)[0], G, op["n_keep"])
    v0 = jnp.asarray(np.random.default_rng(0).standard_normal(op["n_keep"]) + 0j)
    kw = dict(sigma=sigma, tol=1e-8, theta_tol=0.0)
    for precond, converges in ((M, True), (lambda x: x, False)):
        _, _, info = jacobi_davidson(op["Ax"], precond, v0, **kw)
        assert (float(info["resid"]) < 1e-6) == converges, info


# ---------------------------------------------------------------------------
# On the complex (axisymmetric, n != 0) operator: every factor is Hermitian
# ---------------------------------------------------------------------------
# conftest's ``axisym_case`` (one zeta plane of the fixture) is complex
# Hermitian: there a plain transpose is the wrong adjoint (DESC cacf77de7).


@pytest.fixture(scope="module")
def axisym_dense(axisym_case):
    """``(H = A - sigma I, ring blocks, G)``, ``sigma`` below the spectrum."""
    from agnimhd.assemble import assemble_dense

    eq, dm, cfg = axisym_case
    A = np.asarray(assemble_dense(eq, dm, cfg)["A"])
    sigma = float(np.min(np.linalg.eigvalsh(A))) - 1.0
    sel, pad, G = ring_index_maps(keep_indices(*eq.resolution), eq.resolution)
    blocks = build_ring_blocks(eq, dm, cfg, eq.resolution, sel, pad, sigma=sigma)
    return A - sigma * np.eye(A.shape[0]), blocks, G


def _block_diag(H, G):
    """Dense block diagonal of ``H`` over the groups ``G``."""
    M = np.zeros_like(H)
    for row in G:
        idx = row[row >= 0]
        M[np.ix_(idx, idx)] = H[np.ix_(idx, idx)]
    return M


def test_ring_blocks_and_preconditioner_are_hermitian(axisym_dense):
    """Ring blocks equal the dense Hermitian sub-blocks; ``M^-1`` is their
    exact Hermitian inverse (``H = L L^H`` needs ``L^H``, not ``L^T``)."""
    H, blocks, G = axisym_dense
    n, scale, blocks = H.shape[0], np.max(np.abs(H)), np.asarray(blocks)
    for gi, row in enumerate(G):
        idx = row[row >= 0]
        got = blocks[gi][: idx.size, : idx.size]
        assert np.max(np.abs(got - H[np.ix_(idx, idx)])) / scale < 1e-13, gi
    L, ok, _ = factor_ring_blocks_traced(jnp.asarray(blocks))
    assert ok
    M = make_block_precond(L, G, n)
    Mmat = np.stack([np.asarray(M(jnp.eye(n, dtype=H.dtype)[j])) for j in range(n)], 1)
    want = np.linalg.inv(_block_diag(H, G))
    assert np.max(np.abs(Mmat - Mmat.conj().T)) / np.max(np.abs(want)) < 1e-12
    assert np.max(np.abs(Mmat - want)) / np.max(np.abs(want)) < 1e-10


def test_deflation_is_hermitian_on_the_complex_operator(axisym_dense):
    """``Y Y^H == Z (Z^H H Z)^-1 Z^H`` for complex ``Z``."""
    H, _, _ = axisym_dense
    n, k = H.shape[0], 6
    rng = np.random.default_rng(5)
    Zc = np.linalg.qr(rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k)))[0]
    Y, rank = deflation_Y(jnp.asarray(Zc), jnp.asarray(H @ Zc))
    want = Zc @ np.linalg.inv(Zc.conj().T @ H @ Zc) @ Zc.conj().T
    got = np.asarray(Y) @ np.asarray(Y).conj().T
    assert int(rank) == k and np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-8


def test_coarse_gen_modes_on_the_complex_pencil(axisym_dense):
    """Hermitian congruence ``A = L^-1 H L^-H``: pencil values and vectors
    against ``scipy.linalg.eigh(H, M)``. The complex Lanczos on ``A`` needs
    matfree >= 0.6.2 (see pyproject)."""
    H, blocks, G = axisym_dense
    k, M = 3, _block_diag(H, G)
    want = scipy.linalg.eigh(H, M, eigvals_only=True)[:k]
    lam, X = coarse_gen_modes(jnp.asarray(H), blocks, G, k, num_matvecs=100, seed=1)
    np.testing.assert_allclose(np.asarray(lam), want, rtol=1e-8)
    X = np.asarray(X)
    resid = np.linalg.norm(H @ X - M @ X * want[None, :], axis=0)
    assert np.max(resid / np.linalg.norm(H @ X, axis=0)) < 1e-7
