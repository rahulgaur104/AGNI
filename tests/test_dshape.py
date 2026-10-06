"""DSHAPE tokamak, the AGNI paper's tokamak benchmark (arXiv:2608.01750v3, sec. 5.2).

The paper solves each toroidal mode ``n`` of the DSHAPE tokamak on its own, on a
coupled Zernike basis with 96x96 nodes and ``MPOL = 4 n``. As a script::

    basis = Basis(96, 96, 1, radial="zernike", mpol=4 * n, zernike_penalty=0.01)
    eq, diffmat = from_desc("dshape_imax0.98_1608.h5", basis)
    config = AssemblyConfig(axisym=True, n_mode_axisym=n, coupled_rt=True,
                            n_rho_coupled=96, n_theta_coupled=96)
    gamma2 = growth_rate(eq, diffmat, config, SolverConfig(sigma=1e-3))

The Zernike derivative matrices annihilate the nodal content the basis does not
represent, so only ``zernike_penalty`` holds that content against the pressure
drive, and its strength after the whitening grows with the node count. At the
16x48 nodes here every unstable eigenvalue at the benchmark's penalties (0.08
for ``n = 1``, 0.01 for ``n = 2 ... 5``) comes from the penalty: without that
content (the infinite-penalty limit) every ``n = 1 ... 5`` is stable. The
paper's verdict, ``n = 2 ... 5`` unstable, holds in that limit only from 48x96
(``n = 2, 3, 4``) and 96x96 (``n = 5``), too large for CI; ``benchmarks/dshape.py``
checks it. Measurements: ``docs/benchmarks.md``.
"""

import pytest

from agnimhd import AssemblyConfig, Basis, SolverConfig, growth_rate

#: The paper's code (DESC's AGNI) on the same 16x48 nodes with the same settings
#: (``MPOL = 4 n``, penalty 0.01), lowest eigenvalue of the full dense spectrum,
#: measured 2026-10-03. Penalty artifacts, not physics: see the module docstring.
GAMMA2_16x48 = {
    2: 3.5593695113e-4,
    3: 4.3613951250e-4,
    4: 1.2182511244e-3,
    5: 1.8159789842e-3,
}

#: ``n = 1`` at 16x48, ``MPOL = 4``, penalty 1, from the same code: within 1 % of
#: the infinite-penalty limit (-1.3610e-6) and of penalty 10 (-1.3565e-6).
GAMMA2_N1 = -1.3512550896e-6


def dshape_mode(dshape, n, zernike_penalty):
    """``gamma^2`` of toroidal mode ``n``, ``MPOL = 4 n``, on the fixture's nodes."""
    n_rho, n_theta, _ = dshape.resolution
    basis = Basis(
        n_rho, n_theta, 1, radial="zernike", mpol=4 * n, zernike_penalty=zernike_penalty
    )
    _, diffmat = basis.nodes_and_diffmat(dshape.NFP)
    config = AssemblyConfig(
        axisym=True,
        n_mode_axisym=n,
        coupled_rt=True,
        n_rho_coupled=n_rho,
        n_theta_coupled=n_theta,
    )
    sigma = 1e-2 if n > 1 else 1e-5
    return float(growth_rate(dshape, diffmat, config, SolverConfig(sigma=sigma)))


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_dshape_matches_the_papers_code_on_the_same_nodes(dshape, n):
    """Code agreement, not physics: agnimhd gives the paper's code's ``gamma^2``
    on the same nodes with the benchmark's settings for ``n = 2 ... 5``."""
    gamma2 = dshape_mode(dshape, n, zernike_penalty=0.01)
    assert gamma2 == pytest.approx(GAMMA2_16x48[n], rel=2.8e-5)


@pytest.mark.parametrize("zernike_penalty", [1.0, 10.0])
def test_dshape_n1_is_near_marginal_once_the_penalty_suffices(dshape, zernike_penalty):
    """``n = 1`` is near marginal and the same with penalty 1 and 10 to 1 %: the
    infinite-penalty limit, not a penalty artifact. (The benchmark's 0.08 gives
    2.65e-5 here, an artifact of this coarse grid.)"""
    gamma2 = dshape_mode(dshape, 1, zernike_penalty)
    assert abs(gamma2) < 1e-5
    assert gamma2 == pytest.approx(GAMMA2_N1, rel=1e-2)
