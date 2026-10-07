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
def test_agni_stability_in_a_desc_optimization():
    """``AgniStability`` beside a standard DESC objective in a usual DESC
    optimization: ``proximal-lsq-exact`` over the eight lowest boundary modes
    with ``ForceBalance`` as the constraint and the profiles and ``Psi`` fixed.
    The shipped case is shrunk to ``L = 6, M = 6, N = 4`` so that one step fits
    on a CPU (measured 257 s, 6.4 GB): ``gamma^2`` falls from 1.58e-7 to
    -1.0e-6, the aspect ratio stays at its target, and the equilibrium is still
    in force balance. The shift of the dense solver is set from ``eigsh``'s
    value at the start and must stay above every ``gamma^2`` the optimizer
    meets."""
    load = pytest.importorskip("desc.io").load
    objectives = pytest.importorskip("desc.objectives")
    from agnimhd.adapters.desc_objective import AgniStability

    eq = load(str(DESC_FILE))
    eq = eq[-1] if hasattr(eq, "__getitem__") else eq
    eq.change_resolution(L=6, M=6, N=4, L_grid=12, M_grid=12, N_grid=8)
    eq.solve(verbose=0, maxiter=50)
    basis = Basis(12, 8, 6, mpol=3, ntor=1)
    start = AgniStability(eq, basis)
    start.build(verbose=0)
    gamma2_0 = float(start.compute(eq.params_dict)[0])
    dense = SolverConfig(
        eigensolver="jax_lanczos", factor="cholesky", sigma=max(2 * gamma2_0, 1e-5)
    )
    aspect = float(eq.compute("R0/a")["R0/a"])
    objective = objectives.ObjectiveFunction(
        (
            AgniStability(eq, basis, solver=dense, weight=1e3),
            objectives.AspectRatio(eq, target=aspect, weight=1.0),
        ),
        deriv_mode="blocked",
    )
    R = np.asarray(eq.surface.R_basis.modes)
    Z = np.asarray(eq.surface.Z_basis.modes)
    constraints = (
        objectives.ForceBalance(eq),
        objectives.FixBoundaryR(
            eq, modes=np.vstack(([0, 0, 0], R[np.abs(R).max(1) > 1]))
        ),
        objectives.FixBoundaryZ(eq, modes=Z[np.abs(Z).max(1) > 1]),
        objectives.FixIota(eq),
        objectives.FixPsi(eq),
        objectives.FixElectronDensity(eq),
        objectives.FixElectronTemperature(eq),
        objectives.FixIonTemperature(eq),
        objectives.FixAtomicNumber(eq),
    )
    eq_new, result = eq.optimize(
        objective,
        constraints,
        optimizer="proximal-lsq-exact",
        maxiter=1,
        verbose=0,
        copy=True,
        options={"solve_options": {"maxiter": 5, "verbose": 0}},
    )
    assert result["nit"] >= 1
    gamma2_1 = float(objective.compute_unscaled(objective.x(eq_new))[0])
    assert gamma2_1 < gamma2_0
    assert float(eq_new.compute("R0/a")["R0/a"]) == pytest.approx(aspect, rel=1e-2)
    force = {}
    for name, e in (("before", eq), ("after", eq_new)):
        fb = objectives.ForceBalance(e)
        fb.build(verbose=0)
        force[name] = float(np.linalg.norm(fb.compute_unscaled(*fb.xs(e))))
    assert force["after"] < 10 * force["before"]


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
    """``from_desc(..., density=True, coarse=basis.coarse(12, 6))`` also
    evaluates the equilibrium, with its normalized ``ni``, on the coarse angular
    nodes; JD with that coarse level gives eigsh's density-weighted ``gamma^2``
    on the default (Gauss-Radau-Jacobi) radial nodes. With the softest coarse
    mode alone as its start, JD returned another mode here (8.95e-5 for
    3.554e-4): the parity trap of ``test_jd_matches_the_dense_eigenpair``."""
    pytest.importorskip("desc")
    basis = Basis(24, 12, 8, mpol=5, ntor=1)
    eq, diffmat, coarse = from_desc(
        str(DESC_FILE), basis, density=True, coarse=basis.coarse(12, 6)
    )
    assert float(eq.density.min()) < 0.5 and coarse[0].density is not None
    gamma2, _, _ = eigenpair(eq, diffmat)
    jd = SolverConfig(
        eigensolver="jd",
        sigma=1.3 * float(gamma2),
        jd_tol=1e-3,
        jd_theta_tol=0.0,
        jd_outer=1000,
    )
    gamma2_jd, _, resid = eigenpair(eq, diffmat, solver=jd, coarse=coarse)
    assert float(gamma2_jd) == pytest.approx(float(gamma2), rel=1e-7)
    assert float(resid) <= jd.jd_tol


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
