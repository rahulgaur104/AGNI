"""DSHAPE tokamak, the AGNI paper's tokamak benchmark (arXiv:2608.01750v3, sec. 5.2).

The paper solves each toroidal mode ``n`` of the DSHAPE tokamak on its own, on a
coupled Zernike basis with 96x96 nodes and ``MPOL = 4 n``. As a script::

    basis = Basis(96, 96, 1, mpol=4 * n)  # Zernike: the default
    eq, diffmat = from_desc("dshape_imax0.98_1608.h5", basis)
    config = AssemblyConfig(axisym=True, n_mode_axisym=n)
    gamma2 = growth_rate(eq, diffmat, config, SolverConfig(sigma=1e-3))

The Zernike derivative matrices annihilate the nodal content the basis does not
represent, and the solve leaves it out (``DiffMat.span_rt``). That is the
infinite-penalty limit of the paper's code, which held the content with a
penalty instead: there, at the 16x48 nodes here, every unstable eigenvalue at
the benchmark's penalties came from the penalty, and in the limit every
``n = 1 ... 5`` is stable. The paper's verdict, ``n = 2 ... 5`` unstable, holds
in that limit only from 48x96 (``n = 2, 3, 4``) and 96x96 (``n = 5``), too large
for CI; ``benchmarks/dshape.py`` checks it. Measurements: ``docs/benchmarks.md``.
"""

import pytest

from agnimhd import AssemblyConfig, Basis, SolverConfig, growth_rate

#: ``n = 1`` at 16x48, ``MPOL = 4``: the paper's code's infinite-penalty limit,
#: measured 2026-10-03 (penalty 1 and 10 gave -1.3513e-6 and -1.3565e-6).
GAMMA2_N1 = -1.3610e-6


def dshape_mode(dshape, n, solver=None):
    """``gamma^2`` of toroidal mode ``n``, ``MPOL = 4 n``, on the fixture's nodes:
    the default (Zernike) basis, and no coupled-mode setting in the assembly."""
    n_rho, n_theta, _ = dshape.resolution
    _, diffmat = Basis(n_rho, n_theta, 1, mpol=4 * n).nodes_and_diffmat(dshape.NFP)
    config = AssemblyConfig(axisym=True, n_mode_axisym=n)
    return float(
        growth_rate(dshape, diffmat, config, solver or SolverConfig(sigma=1e-5))
    )


def test_dshape_n1_is_the_papers_code_in_the_infinite_penalty_limit(dshape):
    """``n = 1`` is near marginal, at the value the paper's code approaches as
    its penalty grows: what the solve on the span computes without one."""
    assert dshape_mode(dshape, 1) == pytest.approx(GAMMA2_N1, rel=1e-4)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_dshape_is_stable_at_16x48_without_the_penalty_artifacts(dshape, n):
    """At 16x48 the modes ``n = 2 ... 5`` the paper's code found unstable here
    came from its penalty: on the span they are stable."""
    assert dshape_mode(dshape, n) < 0.0


def test_dense_agrees_with_eigsh_on_the_complex_operator(dshape):
    """``eigensolver="dense"`` (Cholesky of the complex Hermitian ``A + sigma I``)
    gives eigsh's ``n = 2``."""
    dense = SolverConfig(eigensolver="dense", sigma=1e-5)
    assert dshape_mode(dshape, 2, dense) == pytest.approx(
        dshape_mode(dshape, 2), rel=1e-8
    )
