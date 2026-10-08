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

from agnimhd import Basis, eigenpair, from_desc, growth_rate, load, solve
from agnimhd.adapters.desc import is_desc_file
from agnimhd.cli import main
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
def test_desc_optimizer_jacobian_matches_finite_differences(period_case):
    """DESC's jitted reverse-mode Jacobian of ``AgniStability`` (the optimizer's
    path) is finite and matches central differences in the three equilibrium
    coefficients ``gamma^2`` moves most with (measured 1e-6 apart at this step)."""
    load = pytest.importorskip("desc.io").load
    objectives = pytest.importorskip("desc.objectives")
    from agnimhd.adapters.desc_objective import AgniStability

    eq_data, _ = period_case
    eq = load(str(DESC_FILE))
    eq = eq[-1] if hasattr(eq, "__getitem__") else eq
    stability = AgniStability(eq, basis=fixture_basis(eq_data.resolution))
    objective = objectives.ObjectiveFunction((stability,), deriv_mode="blocked")
    objective.build(verbose=0)
    x = objective.x(eq)
    jac = np.asarray(objective.jac_scaled_error(x))[0]
    assert np.all(np.isfinite(jac))
    h = 1e-6
    for i in np.argsort(-np.abs(jac))[:3]:
        step = np.zeros_like(x)
        step[i] = h
        plus, minus = (objective.compute_scaled_error(x + s)[0] for s in (step, -step))
        assert float(plus - minus) / (2 * h) == pytest.approx(jac[i], rel=1e-4)


@pytest.mark.slow
def test_desc_objective_with_jd_matches_the_dense_objective():
    """``AgniStability`` with ``eigensolver="jd"`` evaluates the equilibrium on
    its coarse level at every call; value and DESC's Jacobian equal the dense
    (eigsh) objective's, both weighted by the density (``density=True``), whose
    value is ``from_desc(..., density=True)``'s, also with ``warm_start=True`` (the
    second call starts from the first one's eigenvector) and with an adapted shift
    (``sigma_factor`` 2, and 1.05 where the solve is redone). Measured: 1.2e-11
    and 7.2e-8 apart for the density, 2.1e-11 from ``from_desc``; 7.8e-12 and
    1.1e-7 apart for the warm start. A coarse basis that is not
    ``basis.coarse(...)``, and the density of an equilibrium without a density
    profile, are refused."""
    load = pytest.importorskip("desc.io").load
    objectives = pytest.importorskip("desc.objectives")
    from agnimhd.adapters.desc_objective import AgniStability

    def last(path):
        eq = load(str(path))
        return eq[-1] if hasattr(eq, "__getitem__") else eq

    eq = last(DESC_FILE)
    basis = Basis(16, 12, 8, mpol=5, ntor=1)
    with pytest.raises(ValueError, match="electron_density"):
        AgniStability(last(DSHAPE_FILE), basis, density=True)

    def value_and_jacobian(stability):
        objective = objectives.ObjectiveFunction((stability,), deriv_mode="blocked")
        objective.build(verbose=0)
        x = objective.x(eq)
        value = float(objective.compute_scaled_error(x)[0])
        return value, np.asarray(objective.jac_scaled_error(x))[0]

    gamma2, jac = value_and_jacobian(AgniStability(eq, basis, density=True))
    eq_data, diffmat = from_desc(eq, basis, density=True)
    assert gamma2 == pytest.approx(float(growth_rate(eq_data, diffmat)), rel=1e-8)
    jd = SolverConfig(
        eigensolver="jd", sigma=1.3 * gamma2, jd_tol=1e-4, jd_theta_tol=0.0
    )
    with pytest.raises(ValueError, match="basis.coarse"):
        AgniStability(eq, basis, solver=jd, coarse=Basis(12, 12, 6, mpol=5, ntor=1))
    coarse = basis.coarse(12, 6)
    stability = AgniStability(eq, basis, solver=jd, coarse=coarse, density=True)
    gamma2_jd, jac_jd = value_and_jacobian(stability)
    assert gamma2_jd == pytest.approx(gamma2, rel=1e-8)
    np.testing.assert_allclose(jac_jd, jac, rtol=0, atol=1e-5 * np.abs(jac).max())
    # warm_start: DESC's Jacobian follows its value at the same point and starts
    # from the value's eigenvector; both are still the dense objective's.
    warm = AgniStability(
        eq,
        basis,
        solver=jd,
        coarse=basis.coarse(12, 6),
        density=True,
        warm_start=True,
    )
    gamma2_w, jac_w = value_and_jacobian(warm)
    assert gamma2_w == pytest.approx(gamma2, rel=1e-8)
    np.testing.assert_allclose(jac_w, jac, rtol=0, atol=1e-5 * np.abs(jac).max())
    assert warm._warm.hits >= 1 and warm._warm.hits < warm._warm.reads
    # sigma_factor: the shift follows the kept gamma^2 (2x), or is too close to it
    # (1.05x) and the solve is redone at the configured shift. Same dense answer.
    for factor in (2.0, 1.05):
        adapt = AgniStability(
            eq,
            basis,
            solver=jd,
            coarse=basis.coarse(12, 6),
            density=True,
            warm_start=True,
            sigma_factor=factor,
        )
        gamma2_a, jac_a = value_and_jacobian(adapt)
        assert gamma2_a == pytest.approx(gamma2, rel=1e-8), factor
        np.testing.assert_allclose(jac_a, jac, rtol=0, atol=1e-5 * np.abs(jac).max())
    with pytest.raises(ValueError, match="sigma_factor"):
        AgniStability(eq, basis, solver=jd, sigma_factor=2.0)
    with pytest.raises(ValueError, match="warm_start"):
        AgniStability(eq, basis, warm_start=True)


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


@pytest.mark.slow
def test_jd_with_its_coarse_level_matches_dense_on_the_default_basis():
    """``agnimhd.solve(file, basis, "jd", density=True)`` also evaluates the
    equilibrium, with its normalized ``ni``, on the coarse nodes; JD gives
    eigsh's density-weighted ``gamma^2`` on the default (Gauss-Radau-Jacobi)
    radial nodes. With the softest coarse mode alone as its start, JD returned
    another mode here (8.95e-5 for 3.554e-4): the parity trap of
    ``test_jd_matches_the_dense_eigenpair``."""
    pytest.importorskip("desc")
    basis = Basis(24, 12, 8, mpol=5, ntor=1)
    coarse = basis.coarse(12, 6)
    src = load(DESC_FILE)
    assert float(src.evaluate(coarse, density=True).density.min()) < 0.5
    gamma2, _, _ = solve(src, basis, density=True)
    gamma2_jd, _, resid = solve(
        src,
        basis,
        "jd",
        density=True,
        coarse=coarse,
        sigma=1.3 * float(gamma2),
        jd_tol=1e-3,
        jd_theta_tol=0.0,
        jd_outer=1000,
    )
    assert float(gamma2_jd) == pytest.approx(float(gamma2), rel=1e-7)
    assert float(resid) <= 1e-3


@pytest.mark.slow
def test_jd_on_the_zernike_basis_matches_dense():
    """On a coupled (Zernike) basis the JD coarse level is assembled with its own
    ``n_rho_coupled``, ``n_theta_coupled`` (before, the fine counts made the
    coarse operator refuse its matrices), and JD gives eigsh's ``gamma^2``.
    Measured against dense LAPACK: 4.7e-11 apart, 95 outer iterations."""
    pytest.importorskip("desc")
    basis = Basis(16, 18, 8, radial="zernike", mpol=5, ntor=1, zernike_penalty=0.05)
    eq, diffmat, coarse = from_desc(
        str(DESC_FILE), basis, density=True, coarse=basis.coarse(12, 4)
    )
    config = AssemblyConfig(coupled_rt=True, n_rho_coupled=16, n_theta_coupled=18)
    gamma2, _, _ = eigenpair(eq, diffmat, config)
    jd = SolverConfig(
        eigensolver="jd", sigma=1.3 * float(gamma2), jd_tol=1e-4, jd_theta_tol=0.0
    )
    gamma2_jd, _, resid = eigenpair(eq, diffmat, config, jd, coarse=coarse)
    assert float(gamma2_jd) == pytest.approx(float(gamma2), rel=1e-7)
    assert float(resid) <= jd.jd_tol


@pytest.mark.slow
def test_cli_solve_with_jd_on_a_desc_file(capsys):
    """``agnimhd solve eq.h5 --eigensolver jd`` builds the coarse level from the
    file (default angular nodes, 7 x 3 here) and, on the complex family 1,
    agrees with the default eigsh solve whose ``gamma^2`` sets its shift."""
    pytest.importorskip("desc")

    def solve(*options):
        argv = ["solve", str(DESC_FILE), "--res", "12,8,4", "--mpol", "3"]
        assert main([*argv, "--ntor", "1", "--family", "1", *options]) == 0
        return float(capsys.readouterr().out.split("gamma^2")[1].split()[0])

    gamma2 = solve()
    sigma = f"{1.3 * gamma2:.6e}"
    assert solve("--eigensolver", "jd", "--sigma", sigma) == pytest.approx(
        gamma2, rel=1e-7
    )
