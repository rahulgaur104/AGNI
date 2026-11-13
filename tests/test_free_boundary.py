"""Free boundary: ``EquilibriumData.phi_matrix`` and the vacuum energy.

The equilibrium is the shipped QH case on fewer angles; ``phi_matrix`` is
synthetic, so nothing here needs DESC (``test_adapters.py`` checks DESC's).
"""

import numpy as np
import pytest
from conftest import build_diffmat, on_fewer_angles

from agnimhd import Basis
from agnimhd.assemble import (
    assemble_dense,
    assemble_rows,
    keep_indices,
    matfree_operator,
)
from agnimhd.backend import jnp
from agnimhd.solvers import build_ring_blocks, make_mode_precond, ring_index_maps


def _phi(n_shell, definite=True, seed=0):
    """A ``phi_matrix``-like operator: positive definite (stabilizing) with a
    10% non-self-adjoint part, or indefinite."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_shell, n_shell)) / np.sqrt(n_shell)
    K = rng.standard_normal((n_shell, n_shell)) / np.sqrt(n_shell)
    sym = G @ G.T if definite else 0.5 * (G + G.T)
    return sym + 0.1 * (K - K.T)


@pytest.fixture(scope="module")
def small(eq_data, config):
    """``(eq, diffmat, config)``: the fixture on 6 x 4 angles, outer node at 1."""
    eq = on_fewer_angles(eq_data, theta_step=2, zeta=slice(None, None, 2))
    return eq, build_diffmat(eq), config


@pytest.fixture(scope="module", params=[True, False], ids=["definite", "indefinite"])
def free(small, request):
    """The small case, free-boundary, with its dense matrix."""
    eq, dm, cfg = small
    n_shell = eq.n_theta * eq.n_zeta
    eqf = eq.replace(phi_matrix=_phi(n_shell, request.param))
    return eqf, dm, cfg, assemble_dense(eqf, dm, cfg)


def test_keep_indices_free_boundary_keeps_the_outer_shell():
    """Free boundary drops xi^rho on the axis shell only."""
    n_rho, n_theta, n_zeta = 5, 4, 3
    n_total, shell = n_rho * n_theta * n_zeta, n_theta * n_zeta
    keep = keep_indices(n_rho, n_theta, n_zeta, free_boundary=True)
    assert keep.size == 3 * n_total - shell
    assert len(set(keep.tolist())) == keep.size
    assert sorted(set(range(3 * n_total)) - set(keep.tolist())) == list(range(shell))


def test_phi_matrix_shape_is_checked(small):
    """A phi_matrix that is not (n_shell, n_shell) is refused."""
    eq = small[0]
    with pytest.raises(ValueError, match="boundary shell"):
        eq.replace(phi_matrix=np.zeros((3, 3))).validate()
    n = eq.n_theta * eq.n_zeta
    assert eq.replace(phi_matrix=np.zeros(n * n)).phi_matrix.shape == (n, n)


@pytest.mark.parametrize("fmt", ["npz", "h5"])
def test_phi_matrix_round_trips(small, tmp_path, fmt):
    """phi_matrix is saved, loaded, replaced and carried through the pytree."""
    import jax

    from agnimhd import EquilibriumData

    eq = small[0]
    P = _phi(eq.n_theta * eq.n_zeta)
    eqf = eq.replace(phi_matrix=P)
    assert eqf.free_boundary and not eq.free_boundary
    if fmt == "npz":
        back = EquilibriumData.load(eqf.save(tmp_path / "f"))
    else:
        pytest.importorskip("h5py")
        back = EquilibriumData.load_hdf5(eqf.save_hdf5(tmp_path / "f.h5"))
    np.testing.assert_array_equal(np.asarray(back.phi_matrix), P)
    leaves, aux = jax.tree_util.tree_flatten(eqf)
    np.testing.assert_array_equal(
        np.asarray(jax.tree_util.tree_unflatten(aux, leaves).phi_matrix), P
    )
    assert not eqf.replace(phi_matrix=None).free_boundary


def test_free_boundary_dense_matches_matfree_and_rows(free):
    """The three definitions of the operator agree when free-boundary."""
    eq, dm, cfg, out = free
    A = np.asarray(out["A"])
    op = matfree_operator(eq, dm, cfg)
    assert op["free_boundary"] and op["n_keep"] == A.shape[0]
    assert op["n_keep"] == keep_indices(*eq.resolution, True).size
    scale = np.max(np.abs(A))
    rng = np.random.default_rng(0)
    for j in rng.choice(op["n_keep"], size=24, replace=False):
        e = jnp.zeros(op["n_keep"]).at[j].set(1.0)
        assert np.max(np.abs(np.asarray(op["Ax"](e)) - A[:, j])) < 1e-13 * scale
    rows = np.asarray(assemble_rows(eq, dm, cfg, np.arange(A.shape[0]), batch=64))
    assert np.max(np.abs(rows - A)) <= 1e-13 * scale
    assert np.max(np.abs(A - A.T)) <= 1e-13 * scale


def test_free_boundary_ring_blocks_are_dense_sub_blocks(free):
    """Each ring block is exactly its sub-block of the free-boundary matrix."""
    eq, dm, cfg, out = free
    A = np.asarray(out["A"])
    res = eq.resolution
    sel, pad, G = ring_index_maps(keep_indices(*res, True), res)
    blocks = np.asarray(build_ring_blocks(eq, dm, cfg, res, sel, pad, 0.0))
    scale = np.max(np.abs(A))
    for b, idx in enumerate(np.asarray(G)):
        ok = idx >= 0
        sub = A[np.ix_(idx[ok], idx[ok])]
        assert np.max(np.abs(blocks[b][np.ix_(ok, ok)] - sub)) <= 1e-13 * scale


def test_matfree_and_dense_vacuum_terms_agree(free):
    """The vacuum energy each solver path actually carries -- the operator with
    phi_matrix minus the operator at phi_matrix = 0, as a Rayleigh quotient --
    is the same matrix-free and dense, on random vectors and the lowest mode."""
    eq, dm, cfg, out = free
    eq0 = eq.replace(phi_matrix=0 * eq.phi_matrix)
    A = np.asarray(out["A"])
    A0 = np.asarray(assemble_dense(eq0, dm, cfg)["A"])
    op, op0 = matfree_operator(eq, dm, cfg), matfree_operator(eq0, dm, cfg)
    assert op0["free_boundary"] and op0["n_keep"] == op["n_keep"]
    scale = np.max(np.abs(A - A0))
    assert scale > 0
    rng = np.random.default_rng(2)
    for v in [*rng.standard_normal((8, A.shape[0])), np.linalg.eigh(A)[1][:, 0]]:
        x = jnp.asarray(v)
        matfree = float(v @ np.asarray(op["Ax"](x) - op0["Ax"](x)) / (v @ v))
        dense = float(v @ (A - A0) @ v / (v @ v))
        assert abs(matfree - dense) <= 1e-10 * max(abs(dense), scale)


def test_mode_precond_uses_the_free_boundary_mask(free):
    """The per-Fourier-mode preconditioner (Zernike JD) works on the operator's
    own reduced space: with free boundary that includes the outer xi^rho shell.
    """
    eq, dm, cfg, out = free
    A = np.asarray(out["A"])
    op = matfree_operator(eq, dm, cfg)
    shift = float(np.linalg.eigvalsh(A)[0]) - 1.0  # A - shift I positive definite
    M, ok = make_mode_precond(op, shift, A.dtype)
    assert bool(ok)
    v = np.random.default_rng(3).standard_normal(op["n_keep"])
    y = np.asarray(M(jnp.asarray(v)))
    assert y.shape == (op["n_keep"],)
    assert np.all(np.isfinite(y)) and np.linalg.norm(y) > 0


def test_fixed_boundary_keeps_the_old_mask(small):
    """No phi_matrix: fixed boundary, xi^rho dropped on the outer shell."""
    eq, dm, cfg = small
    op = matfree_operator(eq, dm, cfg)
    assert not op["free_boundary"]
    assert op["n_keep"] == keep_indices(*eq.resolution).size


@pytest.mark.parametrize("radial", ["zernike", "gauss_radau_jacobi", "lobatto"])
def test_free_boundary_basis_puts_the_outer_node_at_one(radial):
    """Basis(free_boundary=True) rescales rho, D_rho and w_rho affinely."""
    b0 = Basis(6, 7, 5, radial=radial)
    n0, d0 = b0.nodes_and_diffmat(3)
    n1, d1 = Basis(6, 7, 5, radial=radial, free_boundary=True).nodes_and_diffmat(3)
    x0 = float(n0["rho"][-1])
    np.testing.assert_allclose(n1["rho"], np.asarray(n0["rho"]) / x0, rtol=1e-14)
    assert float(n1["rho"][-1]) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(d1.D_rho, np.asarray(d0.D_rho) * x0, rtol=1e-13)
    np.testing.assert_allclose(d1.w_rho, np.asarray(d0.w_rho) / x0, rtol=1e-14)
    coarse = Basis(6, 7, 5, radial=radial, mpol=2, ntor=1, free_boundary=True).coarse()
    assert coarse.free_boundary
