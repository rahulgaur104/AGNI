"""Matrix-free JD on the coupled Zernike basis, through ``eigenpair``.

``eigenpair`` compiles the whole JD solve (``_eigenpair_jit``), so the Fourier
mode preconditioner is built from traced arrays: these tests run that path, on
the DSHAPE tokamak (complex operator, ``n = 2``) and the QH stellarator
(families 0 and 1, both stellarator parities), at a small and a large penalty,
against the dense eigenpair on the same nodes. The grids are the smallest whose
softest mode is clearly unstable: on coarser ones it sits in the cluster at
gamma^2 ~ 0, where JD and dense may pick different members. JD stops at its
default Ritz-value change; at a large penalty the eigen-residual is no
convergence measure (it scales with the penalty), so it is not tested. Needs
DESC (``from_desc``).
"""

from pathlib import Path

import pytest
from conftest import DSHAPE_FILE

from agnimhd import AssemblyConfig, Basis, SolverConfig, eigenpair, from_desc

QH_FILE = Path(__file__).parent / "data" / "AGNI_QH_lowres.h5"


def jd_and_dense(path, basis, config, family=0):
    """``(gamma2_jd, gamma2_dense)`` for one basis and assembly."""
    pytest.importorskip("desc")
    eq, diffmat, coarse = from_desc(
        str(path), basis, family=family, coarse=basis.coarse()
    )
    gamma2 = float(eigenpair(eq, diffmat, config, SolverConfig())[0])
    jd = SolverConfig(eigensolver="jd", sigma=1.3 * abs(gamma2))
    gamma2_jd = eigenpair(eq, diffmat, config, jd, coarse=coarse)[0]
    return float(gamma2_jd), gamma2


@pytest.mark.slow
@pytest.mark.parametrize("penalty", [1e-2, 1e4])
def test_zernike_jd_matches_dense_on_the_tokamak(penalty):
    basis = Basis(32, 48, 1, mpol=8, penalty=penalty)
    config = AssemblyConfig(axisym=True, n_mode_axisym=2)
    gamma2_jd, gamma2 = jd_and_dense(DSHAPE_FILE, basis, config)
    print(f"tokamak penalty {penalty:.0e}: JD {gamma2_jd:+.10e} dense {gamma2:+.10e}")
    assert gamma2 > 0
    assert gamma2_jd == pytest.approx(gamma2, rel=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("penalty", [1e-2, 1e4])
@pytest.mark.parametrize(
    "family, parity", [(0, None), (1, None), (0, "even"), (0, "odd")]
)
def test_zernike_jd_matches_dense_on_the_stellarator(penalty, family, parity):
    basis = Basis(16, 16, 8, mpol=4, ntor=2, penalty=penalty)
    config = AssemblyConfig(gamma=5.0 / 3.0, parity=parity)
    gamma2_jd, gamma2 = jd_and_dense(QH_FILE, basis, config, family)
    print(
        f"QH family {family} parity {parity} penalty {penalty:.0e}: "
        f"JD {gamma2_jd:+.10e} dense {gamma2:+.10e}"
    )
    assert gamma2 > 0
    assert gamma2_jd == pytest.approx(gamma2, rel=1e-6)
