"""``Basis``: the nodes and derivative matrices of a solve, chosen by keyword.

A solve script starts with one::

    basis = Basis(40, 48, 16, mpol=8, ntor=2)
    eq, diffmat = from_desc("eq.h5", basis)          # nodes, geometry, matrices

or, for an equilibrium code without an adapter,
``nodes, diffmat = basis.nodes_and_diffmat(NFP)`` and the geometry evaluated on
the tensor product of ``nodes``.
"""

from dataclasses import replace

import numpy as np
import pytest
from conftest import fixture_basis, on_fewer_angles

from agnimhd import Basis

NFP = 4


def test_lobatto_basis_rebuilds_the_fixture_nodes(eq_meta, coarse_meta):
    """The fixtures' basis (Lobatto, their staircase map) gives their radial nodes.

    The suite's reference eigenvalues are computed with this basis's matrices
    (``conftest.build_diffmat``), so the two together pin it down.
    """
    for meta in (eq_meta, coarse_meta):
        nodes, _ = fixture_basis(meta["resolution"]).nodes_and_diffmat(meta["NFP"])
        assert np.max(np.abs(nodes["rho"] - np.array(meta["rho_nodes"]))) < 1e-14


@pytest.mark.parametrize(
    "basis, k",
    [(Basis(32, 12, 9, mpol=3, ntor=2), 1), (Basis(32, 12, 8, radial="lobatto"), NFP)],
    ids=["default_truncated_one_period", "lobatto_four_periods"],
)
def test_diffmat_differentiates_a_smooth_function(basis, k):
    """The DiffMat differentiates ``rho^3 cos(2 theta) sin(2 k zeta)`` to roundoff.

    ``k`` field periods, so the modes are within ``mpol`` and ``ntor``. The
    toroidal weights add up to one period, and a mode above ``mpol`` has no
    derivative.
    """
    nodes, diffmat = basis.nodes_and_diffmat(k)
    rho, theta, zeta = np.meshgrid(
        *(np.asarray(nodes[c]) for c in ("rho", "theta", "zeta")), indexing="ij"
    )
    f = rho**3 * np.cos(2 * theta) * np.sin(2 * k * zeta)
    derivatives = (
        3 * rho**2 * np.cos(2 * theta) * np.sin(2 * k * zeta),
        -2 * rho**3 * np.sin(2 * theta) * np.sin(2 * k * zeta),
        2 * k * rho**3 * np.cos(2 * theta) * np.cos(2 * k * zeta),
    )
    matrices = (diffmat.D_rho, diffmat.D_theta, diffmat.D_zeta)
    for axis, (D, expected) in enumerate(zip(matrices, derivatives)):
        df = np.moveaxis(np.tensordot(np.asarray(D), f, axes=(1, axis)), 0, axis)
        np.testing.assert_allclose(df, expected, atol=1e-10)
    assert np.sum(diffmat.w_zeta) == pytest.approx(2 * np.pi / k, rel=1e-14)

    if basis.mpol is not None:  # a mode above mpol is dropped, not differentiated
        above = np.cos((basis.mpol + 1) * np.asarray(nodes["theta"]))
        np.testing.assert_allclose(np.asarray(diffmat.D_theta) @ above, 0, atol=1e-12)


def test_coarse_level_keeps_the_radial_nodes_and_the_fourier_truncation():
    """``coarse()`` is the Jacobi-Davidson coarse level: fewer angular nodes, by
    default the fewest that hold ``mpol`` and ``ntor``; everything else is the
    fine basis. Without ``mpol`` and ``ntor`` the two levels could keep
    different modes, so it raises."""
    fine = Basis(48, 48, 16, mpol=8, ntor=2)
    assert fine.coarse() == replace(fine, n_theta=17, n_zeta=5)
    assert fine.coarse(20, 12) == replace(fine, n_theta=20, n_zeta=12)
    with pytest.raises(ValueError, match="mpol=..., ntor="):
        Basis(24, 12, 8).coarse()


@pytest.mark.parametrize("family", [0, 1, 2])
def test_coarse_level_interpolates_every_shared_mode_exactly(eq_data, family):
    """The coarse-to-fine transfer is Fourier interpolation in theta and zeta,
    with family ``x``'s phase ``exp(i x zeta)``: exact for every mode both levels
    keep. Families 0 and ``NFP / 2`` get a real matrix, as their operator is real."""
    basis = fixture_basis(eq_data.resolution, mpol=2, ntor=1)
    eq_coarse = on_fewer_angles(eq_data, theta_step=2, zeta=slice(None, None, 2))
    _, _, (theta, zeta) = basis.coarse_level(eq_coarse, family)
    nodes = [b.nodes_and_diffmat(NFP)[0] for b in (basis.coarse(6, 4), basis)]
    for m in range(-2, 3):
        mode = [np.exp(1j * m * np.asarray(level["theta"])) for level in nodes]
        np.testing.assert_allclose(theta @ mode[0], mode[1], atol=1e-13)
    for n in [family + k * NFP for k in (-1, 0, 1) if abs(family + k * NFP) <= NFP]:
        mode = [np.exp(1j * n * np.asarray(level["zeta"])) for level in nodes]
        np.testing.assert_allclose(zeta @ mode[0], mode[1], atol=1e-13)
    assert np.iscomplexobj(zeta) == (family == 1)


def test_basis_rejects_an_unknown_radial_basis_or_family():
    """Only known radial bases pass, and families ``0 ... NFP - 1``."""
    with pytest.raises(ValueError, match="radial"):
        Basis(24, 12, 8, radial="chebyshev")
    with pytest.raises(ValueError, match="family"):
        Basis(24, 12, 8).nodes_and_diffmat(NFP, family=NFP)
