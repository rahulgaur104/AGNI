"""Zernike basis: agreement with the frozen DESC values, and its own properties.

The frozen-value tests are the cross-check that would otherwise require
importing DESC. See ``tools/export_zernike_reference.py`` for how
``tests/data/zernike_reference.npz`` was produced.
"""

import numpy as np
import pytest

from agnimhd.basis.zernike import (
    fourier,
    zernike_eval_matrix,
    zernike_fourier_diffmat,
    zernike_modes,
    zernike_penalty_projector_from_diffmat,
    zernike_radial,
)

# ---------------------------------------------------------------------------
# Against the frozen DESC reference
# ---------------------------------------------------------------------------


def test_radial_polynomials_match_reference(zernike_reference):
    """R_l^m and dR/drho reproduce DESC's values to machine precision."""
    modes = zernike_reference["radial__modes"][:, :2].astype(int)
    rho = zernike_reference["radial__rho"]

    R = zernike_radial(rho, modes[:, 0], modes[:, 1], dr=0)
    dR = zernike_radial(rho, modes[:, 0], modes[:, 1], dr=1)

    assert np.max(np.abs(R - zernike_reference["radial__R"])) < 1e-13
    assert np.max(np.abs(dR - zernike_reference["radial__dRdrho"])) < 1e-12


def test_mode_sets_match_reference(zernike_reference, zernike_cases):
    """Both indexing conventions produce DESC's mode set, in DESC's order."""
    for case in zernike_cases:
        mine = zernike_modes(case["L_resolved"], case["M_resolved"], case["indexing"])
        ref = zernike_reference[case["tag"] + "__modes"][:, :2].astype(int)
        assert mine.shape == ref.shape, case["tag"]
        assert np.array_equal(mine, ref), case["tag"]


def test_evaluation_matrices_match_reference(zernike_reference, zernike_cases):
    """The basis matrix and both first derivatives match DESC on every case."""
    for case in zernike_cases:
        tag = case["tag"]
        rho = zernike_reference[tag + "__rho"]
        theta = zernike_reference[tag + "__theta"]
        modes = zernike_modes(case["L_resolved"], case["M_resolved"], case["indexing"])

        A = zernike_eval_matrix(rho, theta, modes, 0, 0)
        dA_drho = zernike_eval_matrix(rho, theta, modes, 1, 0)
        dA_dtheta = zernike_eval_matrix(rho, theta, modes, 0, 1)

        assert np.max(np.abs(A - zernike_reference[tag + "__A"])) < 1e-12, tag
        assert (
            np.max(np.abs(dA_drho - zernike_reference[tag + "__dA_drho"])) < 1e-11
        ), tag
        assert (
            np.max(np.abs(dA_dtheta - zernike_reference[tag + "__dA_dtheta"])) < 1e-12
        ), tag


def test_coupled_diffmats_match_reference(zernike_reference, zernike_cases):
    """The composed coupled operators D_rho, D_theta match DESC on every case.

    Including the rank-deficient cases: the pseudo-inverse itself is stable
    there, so the operators agree even when the penalty projector built from
    them does not (see the next test).
    """
    for case in zernike_cases:
        tag = case["tag"]
        D_rho, D_theta = zernike_fourier_diffmat(
            zernike_reference[tag + "__rho"],
            zernike_reference[tag + "__theta"],
            L=case["L"],
            M=case["M"],
            spectral_indexing=case["indexing"],
        )
        ref_r = zernike_reference[tag + "__D_rho"]
        ref_t = zernike_reference[tag + "__D_theta"]
        scale_r = max(np.max(np.abs(ref_r)), 1.0)
        scale_t = max(np.max(np.abs(ref_t)), 1.0)
        assert np.max(np.abs(np.asarray(D_rho) - ref_r)) / scale_r < 1e-12, tag
        assert np.max(np.abs(np.asarray(D_theta) - ref_t)) / scale_t < 1e-12, tag


def test_penalty_projector_matches_reference_where_well_posed(
    zernike_reference, zernike_cases
):
    """Q matches DESC exactly on every well-posed case.

    It deliberately does NOT match on the rank-deficient cases, and that is a
    fix, not a regression -- see
    ``test_penalty_projector_is_deterministic_when_rank_deficient``.
    """
    checked = 0
    for case in zernike_cases:
        if case["rank_deficient"]:
            continue
        tag = case["tag"]
        D_rho, D_theta = zernike_fourier_diffmat(
            zernike_reference[tag + "__rho"],
            zernike_reference[tag + "__theta"],
            L=case["L"],
            M=case["M"],
            spectral_indexing=case["indexing"],
        )
        Q, rank = zernike_penalty_projector_from_diffmat(D_rho, D_theta)
        assert (
            np.max(np.abs(np.asarray(Q) - zernike_reference[tag + "__penalty_Q"]))
            < 1e-12
        ), tag
        assert rank == int(zernike_reference[tag + "__penalty_rank"]), tag
        checked += 1
    assert checked >= 6, "expected at least six well-posed reference cases"


def test_penalty_projector_is_deterministic_when_rank_deficient(
    zernike_reference, zernike_cases
):
    """A constant already inside the row space must not be re-added as noise.

    The original implementation decided whether to append the constant mode with
    an ABSOLUTE test, ``||residual|| > 10 * eps``. When the constant already lies
    in the derivative row space -- which happens as soon as the Zernike basis is
    over-resolved relative to the nodes -- that residual is pure roundoff, the
    absolute test still passes, and normalizing it appends a UNIT-LENGTH NOISE
    VECTOR to the represented basis. Two implementations agreeing on
    ``D_rho``/``D_theta`` to 4e-15 then produced projectors differing by 0.35 in
    the sup norm, and the penalty silently failed to penalize an arbitrary
    direction.

    AGNI uses a relative floor instead. This test pins the consequence: on the
    rank-deficient cases AGNI's represented rank is exactly one LOWER than the
    frozen DESC value, that one being the spurious direction.
    """
    checked = 0
    for case in zernike_cases:
        if not case["rank_deficient"]:
            continue
        tag = case["tag"]
        D_rho, D_theta = zernike_fourier_diffmat(
            zernike_reference[tag + "__rho"],
            zernike_reference[tag + "__theta"],
            L=case["L"],
            M=case["M"],
            spectral_indexing=case["indexing"],
        )
        _, rank = zernike_penalty_projector_from_diffmat(D_rho, D_theta)
        assert rank == int(zernike_reference[tag + "__penalty_rank"]) - 1, (
            f"{tag}: expected AGNI to drop exactly the one spurious constant "
            "direction that DESC's absolute threshold admitted"
        )
        checked += 1
    assert checked == 2, "expected exactly two rank-deficient reference cases"


# ---------------------------------------------------------------------------
# Properties that hold regardless of any reference
# ---------------------------------------------------------------------------


def test_penalty_projector_is_hermitian_and_idempotent(
    zernike_reference, zernike_cases
):
    """Q is a projector: Hermitian and Q @ Q == Q, on every case."""
    for case in zernike_cases:
        tag = case["tag"]
        D_rho, D_theta = zernike_fourier_diffmat(
            zernike_reference[tag + "__rho"],
            zernike_reference[tag + "__theta"],
            L=case["L"],
            M=case["M"],
            spectral_indexing=case["indexing"],
        )
        Q = np.asarray(zernike_penalty_projector_from_diffmat(D_rho, D_theta)[0])
        assert np.max(np.abs(Q - Q.T)) < 1e-13, tag
        assert np.max(np.abs(Q @ Q - Q)) < 1e-12, tag


def test_penalty_projector_annihilates_represented_content(
    zernike_reference, zernike_cases
):
    """Q kills everything the basis represents -- that is what it is for."""
    for case in zernike_cases:
        tag = case["tag"]
        D_rho, D_theta = zernike_fourier_diffmat(
            zernike_reference[tag + "__rho"],
            zernike_reference[tag + "__theta"],
            L=case["L"],
            M=case["M"],
            spectral_indexing=case["indexing"],
        )
        Q = np.asarray(zernike_penalty_projector_from_diffmat(D_rho, D_theta)[0])
        scale = max(np.max(np.abs(np.asarray(D_rho))), 1.0)
        assert np.max(np.abs(Q @ np.asarray(D_rho).T)) / scale < 1e-11, tag
        assert np.max(np.abs(Q @ np.asarray(D_theta).T)) / scale < 1e-11, tag
        # ... and leaves a constant alone: a constant is represented.
        ones = np.ones(Q.shape[0])
        assert np.max(np.abs(Q @ ones)) < 1e-10, tag


def test_zernike_derivative_matches_finite_differences():
    """dR/drho is the derivative of R, checked against central differences."""
    modes = zernike_modes(8, 4, "ansi")
    rho = np.array([0.13, 0.37, 0.61, 0.88])
    h = 1e-6
    analytic = zernike_radial(rho, modes[:, 0], modes[:, 1], dr=1)
    numeric = (
        zernike_radial(rho + h, modes[:, 0], modes[:, 1], dr=0)
        - zernike_radial(rho - h, modes[:, 0], modes[:, 1], dr=0)
    ) / (2 * h)
    assert np.max(np.abs(analytic - numeric)) < 1e-6


def test_zernike_parity_modes_vanish():
    """Modes with (l - |m|) odd are identically zero, as the definition says."""
    ell = np.array([3, 4, 5])
    m = np.array([0, 1, 2])  # all have (l - |m|) odd
    rho = np.linspace(0.05, 1.0, 7)
    assert np.max(np.abs(zernike_radial(rho, ell, m, dr=0))) == 0.0


def test_zernike_radial_at_edge_is_unity():
    """R_l^m(1) == 1 for every valid mode -- the standard normalization."""
    modes = zernike_modes(10, 5, "ansi")
    valid = (modes[:, 0] - np.abs(modes[:, 1])) % 2 == 0
    R = zernike_radial(np.array([1.0]), modes[:, 0], modes[:, 1], dr=0)
    assert np.allclose(R[0, valid], 1.0, atol=1e-10)


def test_fourier_sign_convention():
    """m >= 0 is cosine, m < 0 is sine."""
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    m = np.array([0, 1, -1, 3, -3])
    got = fourier(theta, m, dt=0)
    expect = np.stack(
        [
            np.ones_like(theta),
            np.cos(theta),
            np.sin(theta),
            np.cos(3 * theta),
            np.sin(3 * theta),
        ],
        axis=-1,
    )
    assert np.max(np.abs(got - expect)) < 1e-13


def test_fourier_derivative():
    """The dt=1 branch is the theta derivative of the dt=0 branch."""
    theta = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    m = np.array([0, 2, -2, 4])
    h = 1e-7
    analytic = fourier(theta, m, dt=1)
    numeric = (fourier(theta + h, m, 0) - fourier(theta - h, m, 0)) / (2 * h)
    assert np.max(np.abs(analytic - numeric)) < 1e-6


@pytest.mark.parametrize("indexing", ["ansi", "fringe"])
def test_mode_sets_are_sorted_and_unique(indexing):
    """Modes are lexicographically sorted by (l, m) and contain no duplicates."""
    modes = zernike_modes(10, 5, indexing)
    order = np.lexsort((modes[:, 1], modes[:, 0]))
    assert np.array_equal(modes, modes[order])
    assert len({tuple(row) for row in modes}) == modes.shape[0]


def test_zernike_modes_rejects_unknown_indexing():
    """An unknown convention raises rather than silently choosing one."""
    with pytest.raises(ValueError, match="ansi.*fringe"):
        zernike_modes(4, 2, "noll")


def test_zernike_radial_refuses_higher_derivatives():
    """dr > 1 raises: AGNI's operators are first order, so it would be dead code."""
    with pytest.raises(NotImplementedError, match="first order"):
        zernike_radial(np.array([0.5]), np.array([2]), np.array([0]), dr=2)


def test_penalty_projector_rejects_bad_shapes():
    """Non-square or mismatched inputs raise with a message naming the matrix."""
    good = np.eye(4)
    with pytest.raises(ValueError, match="D_rho must be a square matrix"):
        zernike_penalty_projector_from_diffmat(np.ones((4, 5)), good)
    with pytest.raises(ValueError, match="same shape"):
        zernike_penalty_projector_from_diffmat(good, np.eye(5))
    with pytest.raises(ValueError, match="required"):
        zernike_penalty_projector_from_diffmat(None, good)


def test_diffmat_rejects_empty_or_2d_nodes():
    """Bad node arrays raise rather than producing an empty operator."""
    with pytest.raises(ValueError, match="cannot be empty"):
        zernike_fourier_diffmat(np.array([]), np.array([0.0, 1.0]))
