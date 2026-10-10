"""Toroidal mode families: every toroidal mode number, solved on one field period.

The modes of an ``NFP``-period equilibrium split into ``NFP`` families,
``n = x + k NFP``, and each family is solved on one field period::

    basis = Basis(24, 12, 8)
    eq, diffmat = from_desc("eq.h5", basis)                     # family 0
    for x in basis.families(eq.NFP):                            # 0 ... NFP // 2
        _, diffmat = basis.nodes_and_diffmat(eq.NFP, family=x)
        print(x, growth_rate(eq, diffmat))

The reference here is the full torus: one field period of the QH case at 8x8x3
(``NFP = 4``) repeated four times, so its data is exactly periodic, and solved
on ``NFP * n_zeta`` toroidal nodes.
"""

import numpy as np
import pytest
from conftest import fixture_basis

from agnimhd import (
    AssemblyConfig,
    Basis,
    EquilibriumData,
    SolverConfig,
    growth_rate,
    growth_rate_of,
)
from agnimhd.assemble import assemble_dense, assemble_rows, matfree_operator
from agnimhd.backend import jax, jnp
from agnimhd.basis import fourier_diffmat, fourier_diffmat_truncated
from agnimhd.equilibrium import OPTIONAL_ARRAYS, REQUIRED_ARRAYS

NFP = 4


def full_torus(eq):
    """``eq`` repeated on all ``NFP`` field periods: the full torus, one period long.

    An equilibrium on the full torus is one on a single period of an ``NFP = 1``
    device, so its ``DiffMat`` is ``basis.nodes_and_diffmat(1)``.
    """
    n_rho, n_theta, n_zeta = eq.resolution

    def tile(values):
        values = np.asarray(values)
        grid = values.reshape(n_rho, n_theta, n_zeta, *values.shape[1:])
        reps = (1, 1, eq.NFP) + (1,) * (grid.ndim - 3)
        return np.tile(grid, reps).reshape(-1, *values.shape[1:])

    fields = {
        key: tile(getattr(eq, key))
        for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS
        if getattr(eq, key) is not None
    }
    resolution = dict(n_rho=n_rho, n_theta=n_theta, n_zeta=eq.NFP * n_zeta)
    return EquilibriumData(**resolution, NFP=1, Psi=eq.Psi, a=eq.a, **fields)


def spectrum(eq, diffmat):
    """Every eigenvalue ``lambda = -gamma^2`` of the dense operator, ascending."""
    return np.linalg.eigvalsh(np.asarray(assemble_dense(eq, diffmat)["A"]))


@pytest.mark.parametrize("ntor", [None, 1], ids=["every_mode", "ntor_1"])
def test_full_torus_spectrum_is_the_union_of_the_family_spectra(period_case, ntor):
    """The full torus has exactly the eigenvalues of the ``NFP`` families together.

    ``ntor`` counts field-period harmonics: it keeps ``|n| <= ntor NFP`` in every
    family, and the same on the full torus.
    """
    eq, _ = period_case
    basis = fixture_basis(eq.resolution, ntor=ntor)
    families = [
        spectrum(eq, basis.nodes_and_diffmat(NFP, family=x)[1]) for x in range(NFP)
    ]

    torus = full_torus(eq)
    torus_ntor = None if ntor is None else ntor * NFP
    torus_basis = fixture_basis(torus.resolution, ntor=torus_ntor)
    lam = spectrum(torus, torus_basis.nodes_and_diffmat(1)[1])

    union = np.sort(np.concatenate(families))
    np.testing.assert_allclose(union, lam, rtol=0, atol=1e-13 * np.abs(lam).max())


def test_families_x_and_nfp_minus_x_have_the_same_spectrum(period_case):
    """Family ``NFP - x`` is the complex conjugate of family ``x``, so only
    ``x = 0 ... NFP // 2`` are solved: ``Basis.families``."""
    eq, _ = period_case
    basis = fixture_basis(eq.resolution)
    lam_1, lam_3 = (
        spectrum(eq, basis.nodes_and_diffmat(NFP, family=x)[1]) for x in (1, NFP - 1)
    )
    np.testing.assert_allclose(lam_3, lam_1, rtol=0, atol=1e-13 * np.abs(lam_1).max())
    assert basis.families(NFP) == (0, 1, 2)


@pytest.mark.parametrize("ntor", [None, 1], ids=["every_mode", "ntor_1"])
@pytest.mark.parametrize("n_zeta", [3, 4])
def test_family_zero_is_the_field_period_matrix(n_zeta, ntor):
    """Family 0 (``n = 0, +-NFP, ...``) is what a field-period run solved before
    families, bit for bit: ``NFP`` times the one-period Fourier matrix. It is the
    family-0 block of the full-torus matrix to round-off."""
    D_zeta = Basis(8, 8, n_zeta, ntor=ntor).nodes_and_diffmat(NFP)[1].D_zeta
    one_period = (
        fourier_diffmat(n_zeta)
        if ntor is None
        else fourier_diffmat_truncated(n_zeta, ntor)
    )[0]
    np.testing.assert_array_equal(D_zeta, NFP * one_period)

    torus_ntor = None if ntor is None else ntor * NFP
    D_torus = Basis(8, 8, NFP * n_zeta, ntor=torus_ntor).nodes_and_diffmat(1)[1].D_zeta
    block = sum(D_torus[:n_zeta, p * n_zeta : (p + 1) * n_zeta] for p in range(NFP))
    np.testing.assert_allclose(D_zeta, block, rtol=0, atol=1e-14 * np.abs(block).max())


def test_family_zero_reproduces_the_exported_reference(period_case):
    """Family 0, the default, gives the eigenvalue DESC computed at export."""
    eq, meta = period_case
    diffmat = fixture_basis(eq.resolution).nodes_and_diffmat(NFP)[1]
    gamma2 = float(growth_rate(eq, diffmat))
    assert gamma2 == pytest.approx(-meta["dense_lambda3"], rel=2.8e-5)


def test_a_complex_family_assembles_a_complex_hermitian_operator(period_case):
    """Families ``x != 0, NFP/2`` have a complex ``D_zeta``, and the operator
    follows it: the dense matrix, its rows and the matrix-free operator are
    complex and agree, imaginary part included."""
    eq, _ = period_case
    basis = fixture_basis(eq.resolution)
    diffmats = [basis.nodes_and_diffmat(NFP, family=x)[1] for x in range(NFP)]
    assert [jnp.iscomplexobj(d.D_zeta) for d in diffmats] == [False, True, False, True]

    A = np.asarray(assemble_dense(eq, diffmats[1])["A"])
    assert A.dtype == np.complex128
    v = np.random.default_rng(0).standard_normal((A.shape[0], 2)) @ np.array([1, 1j])
    Av = matfree_operator(eq, diffmats[1])["Ax"](jnp.asarray(v))
    rows = assemble_rows(eq, diffmats[1], rows=np.arange(8))
    for value, expected in ((A.conj().T, A), (rows, A[:8]), (Av, A @ v)):
        atol = 1e-12 * np.abs(expected).max()
        np.testing.assert_allclose(np.asarray(value), expected, rtol=0, atol=atol)


def test_a_complex_family_gradient_matches_finite_differences(period_case):
    """The Hellmann-Feynman gradient of family 1's ``gamma^2`` (complex eigenvector
    from ARPACK) against central differences of the dense lowest eigenvalue
    (measured 3e-6 apart at this step; 2e-5 at ``h = 1e-6 a``)."""
    eq, _ = period_case
    diffmat = fixture_basis(eq.resolution).nodes_and_diffmat(NFP, family=1)[1]

    def equilibrium_map(params):
        return eq.replace(a=params["a"])

    grad = jax.grad(growth_rate_of)({"a": eq.a}, equilibrium_map, diffmat)["a"]
    h = 1e-5 * eq.a
    plus, minus = (-spectrum(eq.replace(a=eq.a + s), diffmat)[0] for s in (h, -h))
    assert float(grad) == pytest.approx((plus - minus) / (2 * h), rel=1e-5)


def test_dense_mg_solves_a_complex_family(period_case, monkeypatch):
    """``dense_mg`` on family 1's complex Hermitian operator, with a Cholesky
    solve standing in for JAXMg (GPUs): the most unstable ``gamma^2`` of the dense
    family-1 spectrum, and a complex eigenvector."""
    from agnimhd import multigpu

    def stand_in(M, B, mesh, tile):
        return jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(M), B)

    monkeypatch.setattr(multigpu, "solve_shifted", stand_in)
    eq, _ = period_case
    diffmat = fixture_basis(eq.resolution).nodes_and_diffmat(NFP, family=1)[1]
    gamma2_ref = -spectrum(eq, diffmat)[0]
    sigma = gamma2_ref + 0.05 * abs(gamma2_ref)  # just above gamma^2: few iterations
    solver = SolverConfig(
        eigensolver="dense_mg", sigma=sigma, mg_tile=1000, mg_iters=20
    )
    v, gamma2 = multigpu.dense_mg(eq, diffmat, AssemblyConfig(), solver)
    assert jnp.iscomplexobj(v)
    assert float(gamma2) == pytest.approx(gamma2_ref, rel=1e-6)
