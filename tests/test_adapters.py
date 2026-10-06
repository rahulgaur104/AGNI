"""``from_desc``: a DESC file in, the fixture's EquilibriumData out.

Needs DESC; skipped where it is absent (CI). The DESC files are the ones the
fixtures ``tests/data/qh_lowres_24x12x8.npz`` and
``tests/data/dshape_zernike_16x48x1.npz`` were exported from.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import DSHAPE_FILE, fixture_basis
from test_dshape import GAMMA2_N1

from agnimhd import Basis, eigenpair, from_desc, growth_rate
from agnimhd.adapters.desc import is_desc_file
from agnimhd.config import AssemblyConfig, SolverConfig

DESC_FILE = Path(__file__).parent / "data" / "AGNI_QH_lowres.h5"


def test_is_desc_file_distinguishes_the_two_h5_layouts(tmp_path, eq_data):
    ours = eq_data.save_hdf5(tmp_path / "ours.h5")
    assert is_desc_file(DESC_FILE)
    assert not is_desc_file(ours)
    assert not is_desc_file(tmp_path / "missing.h5")


@pytest.mark.slow
def test_from_desc_reproduces_the_exported_fixture(eq_data, eq_meta):
    pytest.importorskip("desc")
    eq, diffmat = from_desc(str(DESC_FILE), fixture_basis(eq_data.resolution))
    assert eq.resolution == eq_data.resolution and eq.NFP == eq_data.NFP
    for key in ("g_rr", "sqrtg", "J_sup_zeta", "iota", "p", "J_cross_grad_rho"):
        np.testing.assert_allclose(
            np.asarray(getattr(eq, key)), np.asarray(getattr(eq_data, key)), rtol=1e-8
        )
    assert float(eq.a) == pytest.approx(float(eq_data.a), rel=1e-10)
    gamma2, _, _ = eigenpair(
        eq, diffmat, AssemblyConfig(gamma=eq_meta["gamma"]), SolverConfig()
    )
    assert float(gamma2) == pytest.approx(-eq_meta["dense_lambda3"], rel=2.8e-5)


@pytest.mark.slow
def test_from_desc_loads_the_dshape_tokamak_on_a_zernike_basis(dshape):
    """The paper's DSHAPE file on Zernike nodes (``tests/test_dshape.py``): the
    exported fixture, and ``n = 1`` near marginal with the paper's code's value."""
    pytest.importorskip("desc")
    basis = Basis(16, 48, 1, radial="zernike", mpol=4, zernike_penalty=1.0)
    eq, diffmat = from_desc(str(DSHAPE_FILE), basis)
    for key in ("g_rr", "g_vv", "sqrtg", "J_sup_zeta", "iota", "p"):
        np.testing.assert_allclose(
            np.asarray(getattr(eq, key)), np.asarray(getattr(dshape, key)), rtol=1e-8
        )
    assert float(eq.a) == pytest.approx(float(dshape.a), rel=1e-10)
    config = AssemblyConfig(
        axisym=True,
        n_mode_axisym=1,
        coupled_rt=True,
        n_rho_coupled=16,
        n_theta_coupled=48,
    )
    gamma2 = float(growth_rate(eq, diffmat, config, SolverConfig(sigma=1e-5)))
    assert gamma2 == pytest.approx(GAMMA2_N1, rel=2.8e-5)


@pytest.mark.slow
def test_desc_objective_reproduces_the_reference(eq_data, eq_meta):
    """AgniStability maps the grid at the given params and matches the dense value."""
    load = pytest.importorskip("desc.io").load
    from agnimhd.adapters.desc_objective import AgniStability

    eq = load(str(DESC_FILE))
    eq = eq[-1] if hasattr(eq, "__getitem__") else eq
    obj = AgniStability(eq, basis=fixture_basis(eq_data.resolution))
    obj.build(verbose=0)
    gamma2 = float(obj.compute(eq.params_dict)[0])
    assert gamma2 == pytest.approx(-eq_meta["dense_lambda3"], rel=2.8e-5)


@pytest.mark.slow
def test_desc_objective_solves_one_toroidal_family(period_case):
    """``AgniStability(..., family=1)`` is that family's ``gamma^2`` (a complex
    operator inside DESC's objective), as solved on the exported fixture."""
    load = pytest.importorskip("desc.io").load
    from agnimhd.adapters.desc_objective import AgniStability

    eq_data, _ = period_case
    basis = fixture_basis(eq_data.resolution)
    eq = load(str(DESC_FILE))
    eq = eq[-1] if hasattr(eq, "__getitem__") else eq
    obj = AgniStability(eq, basis=basis, family=1)
    obj.build(verbose=0)
    gamma2 = float(obj.compute(eq.params_dict)[0])
    diffmat = basis.nodes_and_diffmat(eq_data.NFP, family=1)[1]
    assert gamma2 == pytest.approx(float(growth_rate(eq_data, diffmat)), rel=2.8e-5)
