"""``from_desc``: a DESC file in, the fixture's EquilibriumData out.

Needs DESC; skipped where it is absent (CI). The DESC file is the one the
fixture ``tests/data/qh_lowres_24x12x8.npz`` was exported from.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import fixture_basis

from agnimhd import eigenpair, from_desc
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
