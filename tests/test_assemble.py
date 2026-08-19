"""Assembly: the dense matrix, the ring blocks, and the matrix-free operator.

The dense ARPACK path is the ground truth. Every other path must reproduce it on
the same equilibrium, grid and shift.

Every tolerance here is a **measured** number, and the eigenvalue regression
reads the value out of ``tests/data/qh_lowres_24x12x8.json`` -- the sidecar
written by the export job -- rather than a constant typed from a document.
"""

import numpy as np
import pytest

from agnimhd.assemble import (
    assemble_dense,
    keep_indices,
    matfree_operator,
    ring_block,
)
from agnimhd.backend import jnp
from agnimhd.basis import DiffMat, fourier_diffmat, legendre_diffmat
from agnimhd.quadrature import automorphism_staircase1, leggauss_lob

# The `diffmat`, `config` and `dense` fixtures live in conftest.py: the solver
# tests need exactly the same three, and building the dense operator twice is
# the most expensive thing the suite does.
#
#: The radial clustering used when the fixture was exported. Must match, or the
#: differentiation matrices are built on different nodes than the geometry.
_AUTO_KW = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)


# ---------------------------------------------------------------------------
# Node set and grid bookkeeping
# ---------------------------------------------------------------------------


def test_quadrature_nodes_match_the_export(eq_data, eq_meta):
    """AGNI's Gauss-Lobatto nodes reproduce the ones the fixture was built on.

    If these drift, the differentiation matrices are built on different nodes
    than the geometry was evaluated at, and every downstream number is wrong in
    a way nothing else would catch.
    """
    x_lob, _ = leggauss_lob(eq_data.n_rho)
    rho = np.asarray(automorphism_staircase1(x_lob, **_AUTO_KW))
    assert np.max(np.abs(rho - np.array(eq_meta["rho_nodes"]))) < 1e-14


def test_keep_indices_tile_every_dof_once():
    """The Dirichlet mask drops exactly the boundary xi^rho DOFs, no others."""
    n_rho, n_theta, n_zeta = 5, 4, 3
    n_total = n_rho * n_theta * n_zeta
    keep = keep_indices(n_rho, n_theta, n_zeta)
    assert keep.size == 3 * n_total - 2 * n_theta * n_zeta
    assert len(set(keep.tolist())) == keep.size
    dropped = sorted(set(range(3 * n_total)) - set(keep.tolist()))
    shell = n_theta * n_zeta
    expected = list(range(shell)) + list(range(n_total - shell, n_total))
    assert dropped == expected


def test_assembly_rejects_mismatched_resolution(eq_data, config):
    """A DiffMat built for a different grid raises instead of assembling."""
    D_rho, W_rho = legendre_diffmat(eq_data.n_rho + 1)
    D_theta, W_theta = fourier_diffmat(eq_data.n_theta)
    D_zeta, W_zeta = fourier_diffmat(eq_data.n_zeta)
    bad = DiffMat(
        D_rho=D_rho,
        W_rho=jnp.diagonal(W_rho),
        D_theta=D_theta,
        W_theta=jnp.diagonal(W_theta),
        D_zeta=D_zeta,
        W_zeta=jnp.diagonal(W_zeta),
    )
    with pytest.raises(ValueError, match="nodes the equilibrium was evaluated on"):
        assemble_dense(eq_data, bad, config)


# ---------------------------------------------------------------------------
# Structure of the assembled matrix
# ---------------------------------------------------------------------------


def test_dense_matrix_shape_and_symmetry(dense, eq_data):
    """A is square on the kept DOFs and symmetric."""
    A = np.asarray(dense["A"])
    n_total = eq_data.n_nodes
    n_keep = 3 * n_total - 2 * eq_data.n_theta * eq_data.n_zeta
    assert A.shape == (n_keep, n_keep)
    # Symmetry is enforced by construction, but roundoff in the congruence
    # leaves a residual. Measured 2.9e-11 against ||A|| ~ 1e2.
    assert np.max(np.abs(A - A.T)) < 1e-9


def test_dense_matrix_is_finite(dense):
    """No NaN or inf survives the assembly."""
    A = np.asarray(dense["A"])
    assert np.all(np.isfinite(A))


def test_mass_matrix_cholesky_reproduces_it(eq_data, diffmat, config):
    """B's per-node 3x3 blocks are SPD and Linv is their inverse Cholesky.

    B is the kinetic matrix; positive definiteness is what makes the reduction
    to a standard symmetric eigenproblem legal in the first place.
    """
    op = matfree_operator(eq_data, diffmat, config)
    Linv_DT = np.asarray(op["Linv_DT"])
    # Linv_DT is the transpose of L^-1 per node; L L^T = B_scaled, so
    # (L^-1)(L^-1)^T = B_scaled^-1 and B_scaled must be SPD.
    Linv = np.swapaxes(Linv_DT, -1, -2)
    B_inv = Linv @ np.swapaxes(Linv, -1, -2)
    eigs = np.linalg.eigvalsh(B_inv)
    assert np.all(eigs > 0), "the whitened mass blocks are not positive definite"
    assert np.all(np.isfinite(Linv))


# ---------------------------------------------------------------------------
# The three assembly routes agree
# ---------------------------------------------------------------------------


def test_matfree_operator_matches_dense_matrix(eq_data, diffmat, config, dense):
    """The matrix-free operator equals the dense matrix, column for column.

    This is the check that keeps the two definitions of the operator from
    drifting apart: the matrix-free path is what runs at production resolution,
    where no dense matrix exists to compare against.

    The bound is **relative**, and that choice matters. The recorded historical
    agreement, 2e-11, is an absolute number measured on one particular case, and
    an absolute bound on this quantity is really a bound on ``eps * ||A||``. On
    the shipped case ``||A||_max = 8.2e5``, so a single ulp of the matrix is
    1.8e-10 -- the absolute agreement cannot be better than that, and it is not:
    measured max absolute column error is 2.9e-11, i.e. 0.16 ulp. Relative to
    the column being reproduced it is **4.8e-16**, which is the number that
    actually says the two operators are the same operator.

    Independent confirmation that the residual is the *dense* path's roundoff
    and not the matrix-free path's: the dense matrix's own asymmetry,
    ``max|A - A.T|``, is 2.9104e-11 on this case -- the same value, to every
    digit, as the largest column disagreement. The matrix-free operator is
    symmetric by construction; the dense congruence is only symmetric up to
    roundoff, so it is the noisier of the two.
    """
    A = np.asarray(dense["A"])
    op = matfree_operator(eq_data, diffmat, config)
    Ax = op["Ax"]
    n_keep = op["n_keep"]
    assert n_keep == A.shape[0]

    # Materializing all n_keep columns costs 6720 operator applications; a
    # random subset of columns is the same check at a fraction of the cost, and
    # a systematic disagreement cannot hide in the unsampled columns.
    rng = np.random.default_rng(0)
    cols = rng.choice(n_keep, size=24, replace=False)
    scale = np.max(np.abs(A))
    for j in cols:
        e = jnp.zeros(n_keep).at[j].set(1.0)
        got = np.asarray(Ax(e))
        err = np.max(np.abs(got - A[:, j]))
        assert err < 1e-14 * np.max(np.abs(A[:, j])), f"column {j} disagrees"
        # And it stays inside one ulp of the matrix scale, absolutely.
        assert err < np.finfo(float).eps * scale, f"column {j} exceeds one ulp"


def test_matfree_operator_is_symmetric(eq_data, diffmat, config):
    """<x, A y> == <A x, y> for random vectors, without forming A."""
    op = matfree_operator(eq_data, diffmat, config)
    Ax, n_keep = op["Ax"], op["n_keep"]
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = jnp.asarray(rng.standard_normal(n_keep))
        y = jnp.asarray(rng.standard_normal(n_keep))
        lhs = float(jnp.vdot(x, Ax(y)))
        rhs = float(jnp.vdot(Ax(x), y))
        assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300) < 1e-10


def test_ring_block_matches_dense_sub_block(eq_data, diffmat, config, dense):
    """A ring block equals the corresponding sub-block of the full matrix.

    Exactly, not approximately: every step after the ring restriction is
    node-diagonal or a permutation, so it restricts to a ring without error.
    Measured ~1e-16 relative on every ring.
    """
    from agnimhd.solvers import ring_nodes

    n_rho, n_theta, n_zeta = eq_data.resolution
    n_total = eq_data.n_nodes
    keep = keep_indices(n_rho, n_theta, n_zeta)
    full_to_red = -np.ones(3 * n_total, dtype=np.int64)
    full_to_red[keep] = np.arange(keep.size)
    A = np.asarray(dense["A"])

    # An interior ring (all three components alive) and a boundary ring (xi^rho
    # dropped), so both branches of the keep mask are covered.
    for i, k in ((n_rho // 2, 1), (0, 0)):
        nodes = ring_nodes(n_rho, n_theta, n_zeta, i, k)
        blk = np.asarray(ring_block(eq_data, diffmat, config, nodes))
        red = full_to_red[np.concatenate([nodes, nodes + n_total, nodes + 2 * n_total])]
        alive = red >= 0
        idx = red[alive]
        sub_ring = blk[np.ix_(alive, alive)]
        sub_dense = A[np.ix_(idx, idx)]
        scale = max(np.max(np.abs(sub_dense)), 1e-300)
        err = np.max(np.abs(sub_ring - sub_dense)) / scale
        assert err < 1e-12, f"ring (i={i}, k={k}) disagrees with dense: {err:.3e}"


# ---------------------------------------------------------------------------
# The eigenvalue
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.regression
def test_dense_eigenvalue_reproduces_reference(dense, eq_meta):
    """The dense ARPACK eigenvalue matches the value measured at export time.

    **Sign first.** A wrong sign means an unstable equilibrium was reported as
    stable, which is the failure mode that matters; magnitude is checked second.

    The tolerance is the measured eigenvalue noise floor, 2.8e-5 RELATIVE. Do
    not tighten it past that -- below the floor the comparison is measuring
    roundoff. Do not loosen it either: in this codebase a moved number has
    usually meant a wrong mode, not noise.
    """
    from scipy.sparse.linalg import eigsh

    A = np.asarray(dense["A"])
    w, _ = eigsh(A, k=1, sigma=-1e-1, which="LM", tol=1e-8)
    lam = float(w[0])
    ref = float(eq_meta["dense_lambda3"])

    assert np.sign(lam) == np.sign(ref), (
        f"SIGN FLIP: got {lam:.6e} against reference {ref:.6e}. An unstable "
        "equilibrium has been reported as stable."
    )
    rel = abs(lam - ref) / abs(ref)
    assert rel < 2.8e-5, f"lambda moved by {rel:.3e} relative: {lam:.9e} vs {ref:.9e}"


@pytest.mark.slow
def test_minor_radius_sensitivity(eq_data, diffmat, config):
    """A 3.76% change in `a` moves lambda far more than discretization error.

    This documents the trap rather than discovering it: two different formulas
    in DESC (QuadratureGrid vs LinearGrid) give values of `a` differing by
    3.76%, and `a` sets the whole normalization -- B_N = |Psi|/(pi a^2), with
    the operator's terms carrying a^2, a^3 and a^4. The resulting shift in
    lambda dwarfs the 2.8e-5 noise floor, so an adapter that picks the wrong
    formula produces a confidently wrong growth rate.
    """
    from scipy.sparse.linalg import eigsh

    def solve(eq):
        A = np.asarray(assemble_dense(eq, diffmat, config)["A"])
        return float(eigsh(A, k=1, sigma=-1e-1, which="LM", tol=1e-8)[0][0])

    lam0 = solve(eq_data)
    lam1 = solve(eq_data.replace(a=eq_data.a * 1.0376))

    rel = abs(lam1 - lam0) / abs(lam0)
    assert rel > 1e-2, (
        f"a 3.76% change in the minor radius moved lambda by only {rel:.3e}. "
        "That contradicts the recorded sensitivity; check that `a` is actually "
        "reaching the normalization."
    )
