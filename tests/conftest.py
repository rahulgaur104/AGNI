"""Shared fixtures.

Every fixture in the suite is a **version-controlled file under
``tests/data/``**. Nothing here reaches outside the repository: no absolute
paths, no ``sys.path`` injection, no environment variable pointing at a working
tree, no subprocess call to a script shipped elsewhere. That is what lets the
whole suite run in an environment where no equilibrium code is installed.

A missing fixture **fails**, it never skips. A skip is invisible and reads as
"not applicable"; a broken checkout must break the build. This is not
hypothetical -- a blanket ``*.h5`` rule in a ``.gitignore`` once kept the
equilibrium out of a repository, CI skipped 6 of 8 solver tests, and the entire
solver was reported as tested when none of it was.
"""

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"

#: The shipped low-resolution Patil QH case, exported from DESC by
#: ``tools/export_fixture.py``. See its ``.json`` sidecar for provenance.
EQ_FIXTURE = DATA / "qh_lowres_24x12x8.npz"
EQ_META = DATA / "qh_lowres_24x12x8.json"

#: Zernike values frozen from DESC by ``tools/export_zernike_reference.py``.
ZERNIKE_REFERENCE = DATA / "zernike_reference.npz"


def _require(path):
    """Return ``path``, failing the test if it is not there."""
    if not path.is_file():
        pytest.fail(
            f"missing test fixture: {path}\n"
            "This is a BROKEN CHECKOUT, not an inapplicable test. The file is "
            "version-controlled; verify with\n"
            f"    git ls-files --error-unmatch {path}\n"
            "and check .gitignore for a blanket binary-file rule."
        )
    return path


@pytest.fixture(scope="session")
def eq_meta():
    """dict : provenance and measured reference values for the shipped case."""
    return json.loads(_require(EQ_META).read_text())


@pytest.fixture(scope="session")
def eq_data():
    """EquilibriumData : the shipped low-resolution QH case at 24x12x8."""
    from agnimhd import EquilibriumData

    return EquilibriumData.load(_require(EQ_FIXTURE))


#: The radial clustering used when the fixture was exported. Must match, or the
#: differentiation matrices are built on different nodes than the geometry.
AUTO_KW = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)


@pytest.fixture(scope="session")
def diffmat(eq_data):
    """DiffMat on exactly the nodes the shipped fixture was exported on."""
    from agnimhd.backend import jax, jnp
    from agnimhd.basis import DiffMat, fourier_diffmat, legendre_diffmat
    from agnimhd.quadrature import automorphism_staircase1, leggauss_lob

    n_rho, n_theta, n_zeta = eq_data.resolution
    x_lob, _ = leggauss_lob(n_rho)
    dfa = jax.vmap(
        lambda x: jax.grad(automorphism_staircase1, argnums=0)(x, **AUTO_KW)
    )(x_lob)
    D_rho, W_rho = legendre_diffmat(n_rho)
    D_theta, W_theta = fourier_diffmat(n_theta)
    D_zeta, W_zeta = fourier_diffmat(n_zeta)
    return DiffMat(
        D_rho=D_rho / dfa[:, None],
        W_rho=jnp.diagonal(W_rho * dfa[:, None]),
        D_theta=D_theta,
        W_theta=jnp.diagonal(W_theta),
        D_zeta=D_zeta * eq_data.NFP,
        W_zeta=jnp.diagonal(W_zeta / eq_data.NFP),
    )


@pytest.fixture(scope="session")
def config(eq_meta):
    """AssemblyConfig matching the export."""
    from agnimhd.config import AssemblyConfig

    return AssemblyConfig(gamma=eq_meta["gamma"])


@pytest.fixture(scope="session")
def dense(eq_data, diffmat, config):
    """The assembled dense operator, built once and shared across the suite."""
    from agnimhd.assemble import assemble_dense

    return assemble_dense(eq_data, diffmat, config)


@pytest.fixture(scope="session")
def zernike_reference():
    """NpzFile : the frozen DESC Zernike values."""
    return np.load(_require(ZERNIKE_REFERENCE))


@pytest.fixture(scope="session")
def zernike_cases(zernike_reference):
    """list of dict : one entry per frozen Zernike case, parsed from its tag."""
    cases = []
    tags = sorted(
        {
            k.split("__")[0]
            for k in zernike_reference.files
            if "__" in k and not k.startswith("radial")
        }
    )
    for tag in tags:
        indexing = tag.split("_")[0]
        n_rho, n_theta = (int(v) for v in tag.split("_")[1].split("x"))
        L = int(tag.split("_L")[1].split("_")[0])
        M = int(tag.split("_M")[1])
        cases.append(
            dict(
                tag=tag,
                indexing=indexing,
                n_rho=n_rho,
                n_theta=n_theta,
                L=L,
                M=M,
                L_resolved=2 * (n_rho - 1) if L == -1 else L,
                M_resolved=max((n_theta - 1) // 2, 0) if M == -1 else M,
                # The two cases with an over-resolved basis. The nodal-to-
                # spectral fit is rank-deficient there, which is a regime the
                # reference deliberately covers; see test_zernike.py.
                rank_deficient=(L == 8 and M == 3),
            )
        )
    return cases
