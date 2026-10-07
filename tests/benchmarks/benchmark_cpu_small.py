"""Runtime benchmarks that fit a CI runner: the shipped 24x12x8 QH case on a CPU.

Not collected by ``pytest tests`` (the file name does not start with ``test_``);
CI runs it on every pull request, on the pull request and on its base, and
comments the difference (``.github/workflows/benchmark.yml``). By hand::

    pytest tests/benchmarks/benchmark_cpu_small.py --benchmark-json=out.json

Each function is timed after its warm-up round, which includes the compile, so
the numbers are the steady state a user sees on the second call.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import build_diffmat, fixture_basis, on_fewer_angles  # noqa: E402

from agnimhd import (  # noqa: E402
    AssemblyConfig,
    EquilibriumData,
    SolverConfig,
    growth_rate,
)
from agnimhd.assemble import assemble_dense, matfree_operator  # noqa: E402
from agnimhd.backend import jax, jnp  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
GAMMA = 5.0 / 3.0
ROUNDS = dict(rounds=3, iterations=1, warmup_rounds=1)


@pytest.fixture(scope="module")
def eq():
    """The shipped QH case, 24x12x8 nodes, one field period."""
    return EquilibriumData.load(DATA / "qh_lowres_24x12x8.npz")


@pytest.fixture(scope="module")
def config():
    """The assembly settings of the export."""
    return AssemblyConfig(gamma=GAMMA)


@pytest.fixture(scope="module")
def diffmat(eq):
    """The derivative matrices on the fixture's nodes."""
    return build_diffmat(eq)


def run(fn):
    """``fn`` with its result on the device finished, so the timer sees the work."""
    return lambda: jax.block_until_ready(fn())


@pytest.mark.benchmark()
def test_build_basis_and_diffmat(benchmark, eq):
    """The radial and angular nodes and all derivative matrices."""
    basis = fixture_basis(eq.resolution)
    benchmark.pedantic(run(lambda: basis.nodes_and_diffmat(eq.NFP)[1].D_rho), **ROUNDS)


@pytest.mark.benchmark()
def test_assemble_dense_matrix(benchmark, eq, diffmat, config):
    """The dense reduced operator, 6720 unknowns."""
    benchmark.pedantic(run(lambda: assemble_dense(eq, diffmat, config)["A"]), **ROUNDS)


@pytest.mark.benchmark()
def test_matrix_free_operator_apply(benchmark, eq, diffmat, config):
    """One application of the matrix-free operator, jitted."""
    op = matfree_operator(eq, diffmat, config)
    apply = jax.jit(op["Ax"])
    x = jnp.ones(op["n_keep"])
    benchmark.pedantic(run(lambda: apply(x)), **ROUNDS)


@pytest.mark.benchmark()
def test_growth_rate_eigsh(benchmark, eq, diffmat, config):
    """The default solver: SciPy ARPACK on the dense matrix."""
    benchmark.pedantic(run(lambda: growth_rate(eq, diffmat, config)), **ROUNDS)


@pytest.mark.benchmark()
def test_growth_rate_dense_lanczos(benchmark, eq, diffmat, config):
    """The one-GPU dense solver (Cholesky, Lanczos), here on a CPU."""
    solver = SolverConfig(eigensolver="jax_lanczos", factor="cholesky", sigma=1e-3)
    benchmark.pedantic(run(lambda: growth_rate(eq, diffmat, config, solver)), **ROUNDS)


@pytest.mark.benchmark()
def test_growth_rate_jacobi_davidson(benchmark, eq, config):
    """Matrix-free JD with its coarse level (every other toroidal node), mpol 3."""
    basis = fixture_basis(eq.resolution, mpol=3, ntor=1)
    diffmat = basis.nodes_and_diffmat(eq.NFP)[1]
    coarse = basis.coarse_level(on_fewer_angles(eq, 1, zeta=slice(None, None, 2)))
    jd = SolverConfig(eigensolver="jd", sigma=3e-4, jd_tol=1e-3, jd_theta_tol=0.0)
    benchmark.pedantic(
        run(lambda: growth_rate(eq, diffmat, config, jd, coarse=coarse)), **ROUNDS
    )
