"""``Basis``: the nodes and derivative matrices of a solve, chosen by keyword.

A solve script starts with one::

    basis = Basis(40, 48, 16, domain="field_period", mpol=8, ntor=2)
    eq, diffmat = from_desc("eq.h5", basis)          # nodes, geometry, matrices

or, for an equilibrium code without an adapter,
``nodes, diffmat = basis.nodes_and_diffmat(NFP)`` and the geometry evaluated on
the tensor product of ``nodes``.
"""

from dataclasses import replace

import numpy as np
import pytest
from conftest import fixture_basis

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
    "basis",
    [
        Basis(32, 12, 9, domain="full_torus", mpol=3, ntor=2),
        Basis(32, 12, 8, domain="field_period", radial="lobatto"),
    ],
    ids=["default_truncated_full_torus", "lobatto_field_period"],
)
def test_diffmat_differentiates_a_smooth_function(basis):
    """The DiffMat differentiates ``rho^3 cos(2 theta) sin(2 k zeta)`` to roundoff.

    ``k`` is the number of domains in the torus, so the modes are within
    ``mpol`` and ``ntor``. The toroidal weights add up to the domain's length,
    and a mode above ``mpol`` has no derivative.
    """
    nodes, diffmat = basis.nodes_and_diffmat(NFP)
    k = basis.nfp_mode(NFP)  # NFP for "field_period", 1 for "full_torus"
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


def test_coarse_level_reduces_only_the_radial_resolution():
    """``coarse()`` keeps everything but ``n_rho``, which drops to 2/3, rounded.

    It is the Jacobi-Davidson coarse level, which needs the fine level's angles,
    ``mpol`` and ``ntor``.
    """
    fine = Basis(40, 48, 16, domain="field_period", mpol=8, ntor=2)
    assert fine.coarse() == replace(fine, n_rho=27)
    assert fixture_basis((24, 12, 8)).coarse() == fixture_basis((16, 12, 8))


@pytest.mark.parametrize(
    "choices, error",
    [
        ({}, TypeError),  # no default: the right domain depends on the mode
        ({"domain": "torus"}, ValueError),
        ({"domain": "field_period", "radial": "chebyshev"}, ValueError),
    ],
    ids=["no_domain", "unknown_domain", "unknown_radial"],
)
def test_basis_rejects_a_missing_or_unknown_choice(choices, error):
    """``domain`` must be given, and only known domains and radial bases pass."""
    with pytest.raises(error):
        Basis(24, 12, 8, **choices)
