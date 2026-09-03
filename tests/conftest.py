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

import ctypes
import gc
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"

#: The shipped low-resolution Patil QH case, exported from DESC by
#: ``tools/export_fixture.py``. See its ``.json`` sidecar for provenance.
EQ_FIXTURE = DATA / "qh_lowres_24x12x8.npz"
EQ_META = DATA / "qh_lowres_24x12x8.json"

#: The same case at the COARSE radial resolution, for the two-level solve.
#: 16 is the measured coarse radial floor -- see ``docs/resolution.md`` and
#: ``test_pcg_deflated_two_level_matches_dense``. Exported by the same script,
#: from the same DESC equilibrium, differing only in ``--res``.
COARSE_FIXTURE = DATA / "qh_lowres_16x12x8.npz"
COARSE_META = DATA / "qh_lowres_16x12x8.json"

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


def build_diffmat(eq):
    """DiffMat on exactly the nodes ``eq`` was exported on.

    One builder for every level in the suite -- the shipped case, the coarse
    level of the two-level solve, and the one-plane axisymmetric level. They
    must agree on the automorphism kwargs and on the ``NFP`` scaling of the
    toroidal pair, and three hand-written copies is how they stop agreeing.

    ``n_zeta == 1`` is the axisymmetric level: there is no toroidal derivative
    to take across a single node, so ``D_zeta`` is the 1x1 zero matrix and the
    toroidal dependence is carried analytically by ``AssemblyConfig``'s
    ``n_mode_axisym`` instead.
    """
    from agnimhd.backend import jax, jnp
    from agnimhd.basis import DiffMat, fourier_diffmat, legendre_diffmat
    from agnimhd.quadrature import automorphism_staircase1, leggauss_lob

    n_rho, n_theta, n_zeta = eq.resolution
    x_lob, _ = leggauss_lob(n_rho)
    dfa = jax.vmap(
        lambda x: jax.grad(automorphism_staircase1, argnums=0)(x, **AUTO_KW)
    )(x_lob)
    D_rho, W_rho = legendre_diffmat(n_rho)
    D_theta, W_theta = fourier_diffmat(n_theta)
    if n_zeta == 1:
        D_zeta = jnp.zeros((1, 1))
        W_zeta = jnp.asarray([2.0 * jnp.pi / eq.NFP])
    else:
        D_zeta, W_zeta = fourier_diffmat(n_zeta)
        D_zeta = D_zeta * eq.NFP
        W_zeta = jnp.diagonal(W_zeta / eq.NFP)
    return DiffMat(
        D_rho=D_rho / dfa[:, None],
        W_rho=jnp.diagonal(W_rho * dfa[:, None]),
        D_theta=D_theta,
        W_theta=jnp.diagonal(W_theta),
        D_zeta=D_zeta,
        W_zeta=W_zeta,
    )


@pytest.fixture(scope="session")
def diffmat(eq_data):
    """DiffMat on exactly the nodes the shipped fixture was exported on."""
    return build_diffmat(eq_data)


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


# ---------------------------------------------------------------------------
# The complex Hermitian (axisymmetric) level
# ---------------------------------------------------------------------------
#
# `AssemblyConfig(axisym=True)` analyzes a single toroidal Fourier mode, so
# `d/dphi -> i n` and the operator becomes complex Hermitian instead of real
# symmetric. The shipped fixture is 3D and never builds one, and no test
# touched that dtype branch until these -- which is how the solver shipped for
# a while unable to solve it at all. Building the level takes no equilibrium
# code: one zeta plane of the fixture is a valid `EquilibriumData` on its own,
# and it is exactly the level DESC's own axisym test builds, reproducing that
# test's dense eigenvalue of -2.660e-03 to nine digits.


def _zeta_plane(eq, k=0):
    """Return ``eq`` restricted to one toroidal plane, as ``n_zeta = 1``."""
    from agnimhd import EquilibriumData
    from agnimhd.equilibrium import OPTIONAL_ARRAYS, REQUIRED_ARRAYS

    n_rho, n_theta, n_zeta = eq.resolution

    def plane(arr):
        arr = np.asarray(arr)
        tail = arr.shape[1:]
        return arr.reshape(n_rho, n_theta, n_zeta, *tail)[:, :, k].reshape(
            n_rho * n_theta, *tail
        )

    fields = {name: plane(getattr(eq, name)) for name in REQUIRED_ARRAYS}
    for name in OPTIONAL_ARRAYS:
        value = getattr(eq, name, None)
        if value is not None:
            fields[name] = plane(value)
    return EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=1,
        NFP=eq.NFP,
        Psi=eq.Psi,
        a=eq.a,
        **fields,
    )


@pytest.fixture(scope="session")
def axisym_case(eq_data, eq_meta):
    """``(eq, diffmat, config)`` for the complex Hermitian one-plane operator."""
    from agnimhd.config import AssemblyConfig

    eq = _zeta_plane(eq_data)
    config = AssemblyConfig(gamma=eq_meta["gamma"], axisym=True, n_mode_axisym=1)
    return eq, build_diffmat(eq), config


# ---------------------------------------------------------------------------
# The coarse level of the two-level solve
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def coarse_meta():
    """dict : provenance and rho nodes of the coarse level."""
    return json.loads(_require(COARSE_META).read_text())


@pytest.fixture(scope="session")
def coarse_case(coarse_meta, config):
    """``(eq, diffmat, config)`` for the coarse level, at 16x12x8.

    A SECOND EXPORT, not an interpolation of the fine one. The coarse level is
    the same equilibrium re-evaluated at coarser radial nodes, which only an
    equilibrium code can do -- interpolating the fine data would deflate the
    fine solve against modes of a different problem. ``examples/
    matrix_free_solve.py`` says the same thing and is why it stops short of a
    two-level demonstration.

    ``config`` is shared with the fine level deliberately: the two levels must
    discretize the same energy functional, and a differing ``gamma`` or
    ``incompressible`` would make the deflation space meaningless.
    """
    from agnimhd import EquilibriumData

    eq = EquilibriumData.load(_require(COARSE_FIXTURE))
    return eq, build_diffmat(eq), config


# ---------------------------------------------------------------------------
# Giving the memory back between test modules
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _release_memory_after_module():
    """Return what a module's compiled code and its allocator are holding.

    Both halves are load-bearing, and neither is obvious. Measured on the DESC
    build of this solver, which has the same shape of problem:

        after the solver tests            : 6.26 GB
        after gc.collect()                : 6.26 GB   -- no effect at all
        after jax.clear_caches() + gc     : 3.73 GB
        after malloc_trim(0)              : 1.08 GB   (baseline 0.38 GB)

    ``gc.collect()`` alone recovers NOTHING: the memory is held by JAX's cache
    of compiled executables, which is not garbage, and then by glibc's arena,
    which has freed the blocks but not returned them to the kernel. Skipping
    either step leaves several GB resident for every module that follows.

    This matters because ``ci.yml`` runs the entire suite on a runner with a
    fixed memory budget, and every assembly here is a dense operator at the
    shipped resolution. Without this, the modules that run last inherit the
    peak of everything before them.
    """
    yield

    from agnimhd.backend import jax

    jax.clear_caches()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        # Not glibc. `jax.clear_caches()` above is the portable half and has
        # already run; the arena trim is the part that is platform-specific.
        pass
