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

import os

import numpy as np
import pytest
import scipy.linalg

from agnimhd.assemble import keep_indices, matfree_operator, ring_block
from agnimhd.backend import jax, jnp
from agnimhd.solvers import (
    GROUP_PARTITIONS,
    adjoint_defect,
    barycentric_matrix,
    build_ring_blocks,
    coarse_gen_modes,
    coarse_seed_and_deflation,
    deflation_Y,
    factor_ring_blocks,
    factor_ring_blocks_traced,
    fourier_interp_matrix,
    from_phys,
    group_index_matrix,
    level_meta,
    make_block_precond,
    make_transfer,
    pcg,
    pcg_deflated,
    ring_index_maps,
    ring_nodes,
    to_phys,
    transfer_matrices,
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


def test_barycentric_is_exact_on_polynomials():
    """Interpolation through ``n`` nodes reproduces degree ``n-1`` exactly.

    Radial transfer between levels uses this rather than a low-order
    interpolation, because a coarse mode that is not represented well is not a
    useful seed.
    """
    n = 9
    x_src = np.cos(np.pi * np.arange(n) / (n - 1))[::-1]
    x_dst = np.linspace(-0.9, 0.9, 23)
    P = barycentric_matrix(x_src, x_dst)
    for deg in range(n):
        f = x_src**deg
        want = x_dst**deg
        assert np.max(np.abs(P @ f - want)) < 1e-12, f"degree {deg}"


def test_barycentric_reproduces_coincident_nodes_exactly():
    """A destination node sitting on a source node gets the exact delta row."""
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    P = barycentric_matrix(x, x)
    np.testing.assert_allclose(P, np.eye(x.size), atol=0.0, rtol=0.0)


def test_barycentric_rows_sum_to_one():
    """Constants are reproduced, which is partition of unity."""
    x_src = np.linspace(0.0, 1.0, 7)
    P = barycentric_matrix(x_src, np.linspace(0.05, 0.95, 13))
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-13)


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


@pytest.mark.parametrize("partition", GROUP_PARTITIONS)
def test_partition_tiles_every_dof_exactly_once(partition):
    """Every live reduced DOF appears in exactly one group, and none twice.

    This is what makes the block preconditioner a valid additive Schwarz
    operator. A DOF covered twice would be preconditioned twice; one covered
    never would not be preconditioned at all. Neither errors.
    """
    res = (5, 6, 4)
    keep = keep_indices(*res)
    Gs = group_index_matrix(keep, res, partition=partition)
    live = Gs[Gs >= 0]
    assert live.size == np.asarray(keep).size, f"{partition} does not cover every DOF"
    assert np.array_equal(np.sort(live), np.arange(live.size)), "a DOF repeats"


@pytest.mark.parametrize("partition", GROUP_PARTITIONS)
def test_partition_pads_at_the_end(partition):
    """Live indices are packed to the front; ``-1`` padding only follows them.

    The convention is load-bearing: leaving holes where the Dirichlet mask
    dropped a DOF yields a different block width and a preconditioner that
    misaligns against the blocks.
    """
    res = (5, 6, 4)
    Gs = group_index_matrix(keep_indices(*res), res, partition=partition)
    for row in Gs:
        first_pad = np.argmax(row < 0) if (row < 0).any() else row.size
        assert (row[:first_pad] >= 0).all()
        assert (row[first_pad:] < 0).all()


def test_theta_line_block_width_is_three_n_theta_in_the_interior():
    """The production partition's interior block is the full poloidal ring."""
    res = (5, 6, 4)
    Gs = group_index_matrix(keep_indices(*res), res, partition="theta_line")
    assert Gs.shape[1] == 3 * res[1]
    # Boundary rings lose their n_theta xi^rho DOFs to the Dirichlet mask.
    widths = np.sort(np.unique((Gs >= 0).sum(axis=1)))
    assert widths.tolist() == [2 * res[1], 3 * res[1]]


def test_unknown_partition_is_rejected():
    """A typo in the partition name must not fall through to a default."""
    res = (5, 6, 4)
    with pytest.raises(ValueError, match="unknown partition"):
        group_index_matrix(keep_indices(*res), res, partition="poloidal")


def test_ring_nodes_is_rho_major():
    """Ring node indices follow the same ordering as everything else."""
    n_rho, n_theta, n_zeta = 5, 6, 4
    got = ring_nodes(n_rho, n_theta, n_zeta, 2, 3)
    want = [(2 * n_theta + j) * n_zeta + 3 for j in range(n_theta)]
    assert got.tolist() == want


def test_ring_index_maps_agree_with_the_theta_line_partition():
    """The ring build and the generic partition describe the same groups.

    They are computed by different code -- one walks rings, one walks the
    generic group builder -- so agreement is a real cross-check rather than a
    restatement.
    """
    res = (5, 6, 4)
    keep = keep_indices(*res)
    _, pad, G = ring_index_maps(keep, res)
    Gs = group_index_matrix(keep, res, partition="theta_line")
    assert G.shape == Gs.shape
    np.testing.assert_array_equal(np.sort(G[G >= 0]), np.sort(Gs[Gs >= 0]))
    np.testing.assert_array_equal(np.asarray(pad) > 0, G >= 0)


# ---------------------------------------------------------------------------
# Block preconditioner
# ---------------------------------------------------------------------------


def test_block_precond_inverts_the_block_diagonal_exactly():
    """``M`` is the exact inverse of the block-diagonal it was built from."""
    m, b = 4, 5
    blocks, dense = _block_diag_spd(m, b)
    Gs = np.arange(m * b).reshape(m, b)
    L, ok, ridge = factor_ring_blocks(jnp.asarray(blocks))
    assert ok and ridge == 0.0, "SPD blocks should factor with no ridge"
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
    L, ok, _ = factor_ring_blocks(jnp.asarray(blocks))
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
    L, ok, _ = factor_ring_blocks(jnp.asarray(blocks))
    assert ok
    M = make_block_precond(L, Gs, m * b)
    out = np.asarray(M(jnp.ones(m * b)))
    dropped = 1 * b + (b - 1)
    assert out[dropped] == 0.0, "a padded slot received a contribution"


def test_factor_ring_blocks_escalates_a_ridge_when_it_must():
    """Indefinite blocks force a ridge, and the ridge used is reported.

    A large ridge means the shift has drifted into the spectrum. It is a
    diagnostic, not a knob.
    """
    blocks, _ = _block_diag_spd(3, 4)
    blocks = blocks - 2.0 * np.max(np.linalg.eigvalsh(blocks[0])) * np.eye(4)[None]
    L, ok, ridge = factor_ring_blocks(jnp.asarray(blocks))
    assert ok, "escalation never reached a positive definite shift"
    assert ridge > 0.0


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
# PCG
# ---------------------------------------------------------------------------


def test_pcg_solves_an_spd_system():
    """Unpreconditioned, given enough iterations, CG reaches the exact answer."""
    n = 60
    A = _spd(n, cond=1e3, seed=1)
    rng = np.random.default_rng(2)
    b = rng.standard_normal(n)
    x, iters, relres = pcg(
        lambda v: jnp.asarray(A) @ v,
        jnp.asarray(b),
        lambda v: v,
        1e-12,
        500,
    )
    err = np.max(np.abs(np.asarray(x) - np.linalg.solve(A, b)))
    assert err / np.max(np.abs(np.linalg.solve(A, b))) < 1e-9
    assert float(relres) <= 1e-12
    # Termination in n steps is an exact-arithmetic statement. In floating
    # point, loss of conjugacy costs iterations -- 135 for n = 60 at condition
    # 1e3 -- so only the convergence itself is asserted, not the count.
    assert 0 < int(iters) < 500


def test_preconditioning_reduces_the_iteration_count():
    """The block preconditioner has to actually buy something.

    This is the property that justifies its cost. Measured on a block-dominant
    SPD system, it is a large factor, not a marginal one.
    """
    m, b = 8, 6
    n = m * b
    blocks, block_dense = _block_diag_spd(m, b, seed=5)
    rng = np.random.default_rng(7)
    coupling = rng.standard_normal((n, n)) * 0.02
    A = block_dense + coupling + coupling.T
    A = A + (abs(min(np.linalg.eigvalsh(A))) + 1.0) * np.eye(n)
    # Rebuild the blocks from the coupled matrix so the preconditioner is the
    # true block diagonal of the operator being solved, as in production.
    blocks = np.stack([A[i * b : (i + 1) * b, i * b : (i + 1) * b] for i in range(m)])
    Gs = np.arange(n).reshape(m, b)
    L, ok, _ = factor_ring_blocks(jnp.asarray(blocks))
    assert ok
    M = make_block_precond(L, Gs, n)

    rhs = jnp.asarray(rng.standard_normal(n))
    Hf = lambda v: jnp.asarray(A) @ v  # noqa: E731
    x_plain, it_plain, _ = pcg(Hf, rhs, lambda v: v, 1e-10, 2000)
    x_prec, it_prec, _ = pcg(Hf, rhs, M, 1e-10, 2000)

    want = np.linalg.solve(A, np.asarray(rhs))
    for x in (x_plain, x_prec):
        assert np.max(np.abs(np.asarray(x) - want)) / np.max(np.abs(want)) < 1e-7
    assert int(it_prec) < int(
        it_plain
    ), f"preconditioning did not help: {int(it_prec)} vs {int(it_plain)} iters"


def test_pcg_maxiter_is_a_cap_not_an_error():
    """Hitting the iteration cap returns the current iterate, quietly."""
    n = 80
    A = _spd(n, cond=1e8, seed=3)
    rng = np.random.default_rng(4)
    b = rng.standard_normal(n)
    x, iters, relres = pcg(
        lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda v: v, 1e-14, 5
    )
    assert int(iters) == 5
    assert float(relres) > 1e-14
    assert np.all(np.isfinite(np.asarray(x)))


# ---------------------------------------------------------------------------
# Deflation
# ---------------------------------------------------------------------------


def _deflation_setup(n=70, k=6, seed=11):
    A = _spd(n, cond=1e6, seed=seed)
    rng = np.random.default_rng(seed + 1)
    b = jnp.asarray(rng.standard_normal(n))
    # Deflate with the softest eigenvectors, which is what the coarse level
    # approximates in production.
    _, V = np.linalg.eigh(A)
    Z = jnp.asarray(V[:, :k])
    return A, b, Z


def test_deflated_pcg_with_no_space_is_plain_pcg():
    """``Z=None`` must be exactly :func:`pcg`, not an approximation of it."""
    A, b, _ = _deflation_setup()
    Hf = lambda v: jnp.asarray(A) @ v  # noqa: E731
    x1, i1, r1 = pcg(Hf, b, lambda v: v, 1e-10, 900)
    x2, i2, r2 = pcg_deflated(Hf, b, lambda v: v, 1e-10, 900, Z=None)
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
    assert int(i1) == int(i2) and float(r1) == float(r2)


def test_deflation_changes_the_work_not_the_answer():
    """Deflation is an acceleration. It must not move the solution.

    If it does, the projector pair is wrong and the deflated iteration is
    solving a different system than the caller asked for.
    """
    A, b, Z = _deflation_setup()
    Hf = lambda v: jnp.asarray(A) @ v  # noqa: E731
    want = np.linalg.solve(A, np.asarray(b))

    x_plain, it_plain, _ = pcg_deflated(Hf, b, lambda v: v, 1e-11, 3000)
    x_defl, it_defl, _ = pcg_deflated(Hf, b, lambda v: v, 1e-11, 3000, Z=Z)

    for name, x in (("plain", x_plain), ("deflated", x_defl)):
        rel = np.max(np.abs(np.asarray(x) - want)) / np.max(np.abs(want))
        assert rel < 1e-7, f"{name} solve is off by {rel:.3e}"
    assert int(it_defl) < int(
        it_plain
    ), f"deflation did not reduce work: {int(it_defl)} vs {int(it_plain)}"


def test_initial_guess_changes_the_work_not_the_answer():
    """A seed vector is a starting point, not a constraint on the answer."""
    A, b, Z = _deflation_setup()
    Hf = lambda v: jnp.asarray(A) @ v  # noqa: E731
    want = np.linalg.solve(A, np.asarray(b))
    x0 = jnp.asarray(0.9 * want)
    for Zarg in (None, Z):
        x, _, _ = pcg_deflated(Hf, b, lambda v: v, 1e-11, 3000, Z=Zarg, x0=x0)
        rel = np.max(np.abs(np.asarray(x) - want)) / np.max(np.abs(want))
        assert rel < 1e-7, f"seeded solve is off by {rel:.3e}"


def test_deflated_pcg_survives_jit():
    """The deflated solve is the production path and runs under ``jit``."""
    A, b, Z = _deflation_setup(n=40, k=4)
    Aj = jnp.asarray(A)

    @jax.jit
    def solve(rhs, Zm):
        x, _, _ = pcg_deflated(lambda v: Aj @ v, rhs, lambda v: v, 1e-10, 500, Z=Zm)
        return x

    want = np.linalg.solve(A, np.asarray(b))
    got = np.asarray(solve(b, Z))
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-7


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
    eq_data, diffmat, config, fine_op, eq_meta
):
    """``<P q_c, q_f> == <q_c, PT q_f>`` to machine precision.

    ``PT`` is the true transpose of ``P``, not an inverse and not a separately
    derived restriction. If they drift apart the deflated CG loses symmetry and
    stops being a valid Krylov method -- and nothing about the run announces
    that. Values near 1e-14 are expected.
    """
    from agnimhd.assemble import matfree_operator as _op

    n_rho_f, n_theta, n_zeta = eq_data.resolution
    rho_f = np.asarray(eq_meta["rho_nodes"])

    # A coarse level on the same grid is the degenerate case, and it is the one
    # that can be built without a second equilibrium export: the transfer is
    # then the identity in space but still exercises the full reduced <->
    # physical machinery on both sides.
    meta_f = level_meta(fine_op)
    pr, pt, pz = transfer_matrices(
        rho_f,
        rho_f,
        (n_rho_f, n_theta, n_zeta),
        (n_rho_f, n_theta, n_zeta),
        eq_data.NFP,
    )
    P, PT = make_transfer(meta_f, meta_f, pr, pt, pz)
    defect = adjoint_defect(P, PT, meta_f["n_keep"], meta_f["n_keep"], trials=6)
    assert defect < 1e-12, f"P and PT are not adjoint: defect {defect:.3e}"
    del _op


def test_identity_transfer_is_the_identity(fine_op, eq_data, eq_meta):
    """Same-grid prolongation must not perturb the vector it carries."""
    meta = level_meta(fine_op)
    rho = np.asarray(eq_meta["rho_nodes"])
    res = eq_data.resolution
    pr, pt, pz = transfer_matrices(rho, rho, res, res, eq_data.NFP)
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


@pytest.mark.slow
def test_ring_blocks_eager_and_vmapped_both_match_dense(
    eq_data, diffmat, config, dense
):
    """Both ring-block builds reproduce the dense sub-blocks, on every ring.

    The preconditioner's blocks are assembled two ways. ``build_ring_blocks``
    is one vmapped assembly over all rings, which is the only form that
    survives a trace and therefore the only form production uses.
    :func:`agnimhd.assemble.ring_block` is the eager per-ring build, which is
    what a person reads when checking the restriction is right.

    Comparing the two against EACH OTHER is not enough -- they share
    ``assemble_dense``, so a shared error passes. Both are compared against the
    dense matrix, and over every ring rather than a sample: the padded ring and
    the two radial-boundary rings are the ones whose index maps differ, and a
    sample that misses them tests the easy case.
    """
    A = np.asarray(dense["A"])
    res = eq_data.resolution
    n_rho, n_theta, n_zeta = res
    keep = keep_indices(*res)
    sel, pad, G = ring_index_maps(keep, res)
    scale = np.max(np.abs(A))

    vmapped = np.asarray(
        build_ring_blocks(eq_data, diffmat, config, res, sel, pad, sigma=0.0)
    )
    assert vmapped.shape[0] == G.shape[0] == n_rho * n_zeta

    sel_np, G_np = np.asarray(sel), np.asarray(G)
    worst_v = worst_e = worst_ve = 0.0
    for gi in range(G_np.shape[0]):
        i, k = divmod(gi, n_zeta)
        live = G_np[gi][G_np[gi] >= 0]
        na = live.size
        want = A[np.ix_(live, live)]
        want = 0.5 * (want + want.conj().T)

        got_v = vmapped[gi][:na, :na]
        nodes = ring_nodes(n_rho, n_theta, n_zeta, i, k)
        full = np.asarray(ring_block(eq_data, diffmat, config, nodes))
        take = sel_np[gi][:na]
        got_e = full[np.ix_(take, take)]
        got_e = 0.5 * (got_e + got_e.conj().T)

        worst_v = max(worst_v, np.max(np.abs(got_v - want)) / scale)
        worst_e = max(worst_e, np.max(np.abs(got_e - want)) / scale)
        worst_ve = max(worst_ve, np.max(np.abs(got_e - got_v)) / scale)

    # Measured ~1e-16 for all three; 1e-13 leaves room for a BLAS difference
    # without leaving room for a different expression.
    assert worst_v < 1e-13, f"vmapped build differs from dense by {worst_v:.3e}"
    assert worst_e < 1e-13, f"eager build differs from dense by {worst_e:.3e}"
    assert worst_ve < 1e-13, f"the two builds differ from each other: {worst_ve:.3e}"


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


@pytest.mark.slow
def test_pcg_deflated_two_level_matches_dense(
    eq_data, diffmat, config, dense, fine_op, eq_meta, coarse_case, coarse_meta
):
    """The full two-level solve lands on the dense answer's mode.

    This is the only end-to-end coverage of the composed ``pcg_deflated`` path:
    ring preconditioner, coarse generalized eigensolve, prolongation, deflation
    projector and shift-invert Lanczos, all against the operator the dense
    eigenvalue came from. Every piece has its own test above; none of them
    catches the pieces being wired together wrong.

    It is composed here rather than inside ``growth_rate`` on purpose. The
    two-level path needs a SECOND equilibrium at the coarse nodes, which the
    package cannot produce -- it ships no adapters -- so the caller supplies it.
    ``objective._primal`` says as much when handed
    ``SolverConfig(eigensolver="pcg_deflated")``.

    THE SOLVE IS JITTED, AND THAT IS NOT AN OPTIMIZATION. Eagerly, each of the
    ~300000 matrix-free applies pays Python dispatch: measured 92 minutes of CPU
    without finishing, against 240 s jitted at a quarter of this budget. Every
    routine composed here is documented as traceable precisely so this path can
    be jitted, and jit is the production path -- running it eagerly tests a
    configuration nobody uses.

    RESOLUTION IS A CORRECTNESS THRESHOLD, NOT A COST KNOB. Below the coarse
    floor the solve does not return a less accurate eigenvalue -- it returns the
    WRONG MODE, with the opposite sign. Measured at fine 24x12x8, k=50,
    num_matvecs=100, cg_maxiter=3000, against a dense -1.337622e-04:

      coarse  8 : +2.070e-03    SIGN FLIP -- unstable read as stable
      coarse 12 : -1.2323e-04   right sign, 7.9% off
      coarse 16 : -1.33623e-04  0.10% off

    so the shipped coarse level is 16, and it is not more expensive than 12.

    THE CG BUDGET IS THE BINDING AXIS, not the Krylov dimension. Convergence
    study on the shipped case, coarse 16, cg_tol=1e-6, against a dense
    -1.337627e-04:

      num_matvecs= 20, cg_maxiter= 300 : -1.231761e-04  ratio 0.9209   16.5 s
      num_matvecs= 30, cg_maxiter= 500 : -1.313037e-04  ratio 0.9816   25.4 s
      num_matvecs= 50, cg_maxiter= 800 : -1.332533e-04  ratio 0.9962   60.0 s
      num_matvecs=100, cg_maxiter=6000 : -1.337627e-04  ratio 1.0000  711.5 s

    Monotone, and CONVERGED at the last row -- it reproduces the dense
    eigenvalue exactly. That is the reference configuration; raise
    AGNI_TEST_NMV/AGNI_TEST_CG to it when checking this path for real. The
    default is the 60-second row, which is 0.38% off and still catches the
    failure this test exists for by a factor of 24 (see the deflation note in
    the body). It is a CI budget, not a claim of convergence.

    Two things that do NOT work as diagnostics here, both measured:

    * The sign of the coarse eigenvalue does not predict success. It is
      ``lam_c0 = +6.163e-08`` on the shipped coarse level, positive, and the
      fine solve lands on the correct negative mode regardless.
    * The CG residual is anti-correlated with accuracy. Coarse 16 had the worse
      relative residual and the better answer; neither run converged. Do not
      read it as a quality proxy on this operator.

    The test asserts the sign and the order of magnitude, not digits: it exists
    to cover the composed path, and the budget is the smallest that reaches the
    right mode.
    """
    from matfree import decomp, eig

    from agnimhd.assemble import assemble_dense

    # `dense` is requested so the comparison is against the SAME assembled
    # matrix the reference eigenvalue was measured from, not merely against a
    # number in the sidecar.
    np.testing.assert_allclose(
        float(np.linalg.eigvalsh(np.asarray(dense["A"]))[0]),
        float(eq_meta["dense_lambda3"]),
        rtol=1e-6,
    )
    lam_dense = float(eq_meta["dense_lambda3"])
    # Below the whole spectrum, which is what makes H = A - sigma I SPD and CG
    # a legal iteration at all.
    sigma = 1.3 * lam_dense
    k, cg_tol = 50, 1e-6
    num_matvecs = int(os.environ.get("AGNI_TEST_NMV", 50))
    cg_maxiter = int(os.environ.get("AGNI_TEST_CG", 800))

    eq_c, dm_c, cfg_c = coarse_case
    res_f, res_c = eq_data.resolution, eq_c.resolution
    assert res_c[0] == 16, "the coarse radial floor is 16; see the docstring"
    assert res_c[1:] == res_f[1:], (
        "theta and zeta must NOT be coarsened: the deflation space then stops "
        "resolving the mode and the fine solve collapses onto the wrong one"
    )

    # Index maps and interpolation matrices are static structure, so they are
    # built once outside the traced solve rather than retraced with it.
    meta_f = level_meta(fine_op)
    meta_c = level_meta(matfree_operator(eq_c, dm_c, cfg_c))
    sel_f, pad_f, G_f = ring_index_maps(keep_indices(*res_f), res_f)
    sel_c, pad_c, G_c = ring_index_maps(keep_indices(*res_c), res_c)
    pr, pt, pz = transfer_matrices(
        np.asarray(coarse_meta["rho_nodes"]),
        np.asarray(eq_meta["rho_nodes"]),
        res_c,
        res_f,
        eq_data.NFP,
    )

    P, PT = make_transfer(meta_c, meta_f, pr, pt, pz)
    defect = adjoint_defect(P, PT, meta_c["n_keep"], meta_f["n_keep"], trials=4)
    assert defect < 1e-12, (
        f"P and PT are not adjoint across the two levels: {defect:.3e}. The "
        "deflated CG is then not a valid Krylov method, and nothing in the run "
        "announces it."
    )

    @jax.jit
    def solve(eq_f, dm_f, eq_c, dm_c):
        Ax = matfree_operator(eq_f, dm_f, config)["Ax"]

        def Hf(v):
            return Ax(v) - sigma * v

        L_f, ok_f, _ = factor_ring_blocks_traced(
            build_ring_blocks(eq_f, dm_f, config, res_f, sel_f, pad_f, sigma)
        )
        M = make_block_precond(L_f, G_f, meta_f["n_keep"])

        Hc = assemble_dense(eq_c, dm_c, cfg_c)["A"]
        Hc = Hc - sigma * jnp.eye(Hc.shape[0], dtype=Hc.dtype)
        blocks_c = build_ring_blocks(eq_c, dm_c, cfg_c, res_c, sel_c, pad_c, sigma)
        v0, Z, lam_c = coarse_seed_and_deflation(
            Hc, blocks_c, G_c, meta_c, meta_f, pr, pt, pz, k, num_matvecs
        )

        # DEFLATE THROUGH THE PRECONDITIONER, NOT BY PROJECTION. `pcg_deflated`
        # runs CG on `project(H v) = H v - HZ (Z^T H Z)^-1 Z^T H v`, which
        # removes span(Z) from the operator itself. That is correct for solving
        # one linear system and WRONG as the inverse inside an eigensolve: Z is
        # the prolonged softest coarse modes, so projecting it out deletes the
        # very subspace the target eigenvector lives in, and shift-invert
        # Lanczos then faithfully returns the softest mode of the COMPLEMENT.
        # Measured: lam = +3.264e-03 against a dense -1.338e-04 -- wrong sign,
        # and STABLE under a 3x budget increase, because it was the exact answer
        # to a different problem. The additive form below leaves H alone; Y only
        # changes how fast CG gets there. docs/migration.md records that DESC's
        # production path does it this way.
        HZ = jax.vmap(Hf, in_axes=1, out_axes=1)(Z)
        Y, rank = deflation_Y(Z, HZ)

        def Mdefl(r):
            return M(r) + Y @ (Y.T @ r)

        def OPinv(b):
            x, _, _ = pcg(Hf, b, Mdefl, cg_tol, cg_maxiter)
            return x

        tri = decomp.tridiag_sym(num_matvecs, reortho="full", materialize=True)
        mu, vecs = eig.eigh_partial(tri)(OPinv, v0)
        v = vecs[jnp.argmax(jnp.abs(mu))]
        # The Rayleigh quotient against A itself, not against the shift-inverted
        # operator: the latter inherits CG's residual, the former does not.
        lam = jnp.real(jnp.vdot(v, Ax(v)) / jnp.vdot(v, v))
        return lam, ok_f, Z, lam_c[0], rank

    lam_pcg, ok_f, Z, lam_c0, rank = solve(eq_data, diffmat, eq_c, dm_c)
    lam_pcg = float(lam_pcg)

    assert bool(ok_f), "the fine ring blocks are not SPD -- sigma is in the spectrum"
    assert Z.shape == (meta_f["n_keep"], k)
    assert np.all(np.isfinite(np.asarray(Z))), "the coarse modes came back non-finite"
    assert np.isfinite(lam_pcg), "the two-level solve returned a non-finite eigenvalue"
    assert np.sign(lam_pcg) == np.sign(lam_dense), (
        f"the two-level solve flipped the sign: {lam_pcg:.6e} vs dense "
        f"{lam_dense:.6e} -- an unstable equilibrium reported as stable. "
        f"(coarse lam_c0={float(lam_c0):.3e}, which does NOT diagnose this)"
    )
    ratio = abs(lam_pcg / lam_dense)
    assert 0.2 < ratio < 5.0, (
        f"magnitude off by more than 5x: {lam_pcg:.6e} vs dense {lam_dense:.6e}. "
        "At this budget it need not converge, but it must land on the same mode."
    )


def test_ring_preconditioner_helps_on_the_real_operator(
    eq_data, diffmat, config, dense, fine_op
):
    """On the actual AGNI operator, ring preconditioning reduces CG work.

    Run against ``H = A - sigma I`` with ``sigma`` below the spectrum, which is
    the only regime where CG is legal here: above it, ``H`` is indefinite and
    the method is not a valid Krylov iteration at all.
    """
    A = np.asarray(dense["A"])
    sigma = float(np.min(np.linalg.eigvalsh(A))) - 1.0
    H = A - sigma * np.eye(A.shape[0])
    assert np.min(np.linalg.eigvalsh(H)) > 0.0, "H must be SPD for CG to be legal"

    res = eq_data.resolution
    keep = keep_indices(*res)
    sel, pad, G = ring_index_maps(keep, res)
    blocks = build_ring_blocks(eq_data, diffmat, config, res, sel, pad, sigma=sigma)
    L, ok, ridge = factor_ring_blocks(blocks)
    assert ok, "ring blocks of an SPD H failed to factor"
    assert ridge == 0.0, (
        f"ridge {ridge:.3e} was needed on blocks of an SPD H; the shift is "
        "the thing to check, not the ridge"
    )
    M = make_block_precond(L, G, A.shape[0])

    Hf = lambda v: jnp.asarray(H) @ v  # noqa: E731
    rng = np.random.default_rng(0)
    rhs = jnp.asarray(rng.standard_normal(A.shape[0]))
    _, it_plain, _ = pcg(Hf, rhs, lambda v: v, 1e-8, 4000)
    x_prec, it_prec, _ = pcg(Hf, rhs, M, 1e-8, 4000)

    want = np.linalg.solve(H, np.asarray(rhs))
    rel = np.max(np.abs(np.asarray(x_prec) - want)) / np.max(np.abs(want))
    assert rel < 1e-5, f"preconditioned solve is off by {rel:.3e}"
    assert int(it_prec) < int(
        it_plain
    ), f"ring preconditioning did not help: {int(it_prec)} vs {int(it_plain)}"
