"""General flux coordinates: the energy principle without straight field lines.

Three kinds of evidence, in increasing strength:

1. **The PEST limit.** Fed the PEST fields with ``sqrt(g) B^theta = iota psi'``
   and ``sqrt(g) B^zeta = psi'`` written out, :mod:`agnimhd.general` gives the
   operator of the paper's PEST expansion to round-off, on every kind of
   operator the package builds (real, complex family, single toroidal mode,
   coupled Zernike).
2. **Invariance of the energies.** One smooth displacement has the same
   ``delta W`` and ``delta K`` on PEST nodes with the PEST operator and on nodes
   uniform in angles that are not straight with the general operator, and the
   difference falls spectrally with the resolution. The angle change depends on
   all three coordinates, so ``b^theta`` and ``b^zeta`` both vary on a surface.
3. **Invariance of the eigenvalue.** The most unstable ``gamma^2`` of a single
   toroidal mode is the same in both angle systems, again to spectral accuracy.

The configuration of (2) and (3) is synthetic and analytic: a shaped torus with
a prescribed straight-angle system, profiles and scalar fields, evaluated by
automatic differentiation. It is not an equilibrium and does not need to be:
the energy functional is a scalar for any field that lies in the surfaces, and
that is the property under test. Nothing here imports an equilibrium code.

Every tolerance is a measured number, quoted where it is used.
"""

import jax
import numpy as np
import pytest
from conftest import build_diffmat, fixture_basis, on_fewer_angles

from agnimhd import (
    AssemblyConfig,
    Basis,
    EquilibriumData,
    SolverConfig,
    eigenpair,
    general,
    growth_rate,
)
from agnimhd.assemble import (
    assemble_dense,
    finish_ring_block,
    matfree_operator,
    whitened_vector,
)
from agnimhd.backend import jnp
from agnimhd.plotting import mode_delta_b, mode_displacement


def with_pest_pitch(eq):
    """``eq`` (PEST) with its field written out as general-coordinate fields."""
    return eq.replace(
        B_sup_theta=eq.psi_r * eq.iota / eq.sqrtg, B_sup_zeta=eq.psi_r / eq.sqrtg
    )


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_pest_data_reports_the_pest_pitch(eq_data):
    """Without the two fields the coordinates are PEST and the pitch is (iota, 1)."""
    assert not eq_data.general_coordinates
    b_t, b_z = eq_data.field_pitch()
    np.testing.assert_array_equal(np.asarray(b_t), np.asarray(eq_data.iota))
    np.testing.assert_array_equal(np.asarray(b_z), 1.0)


def test_general_fields_are_validated(eq_data):
    """Both fields or neither; B^zeta of the sign of psi'; the toroidal flux
    integral, which catches Psi used for psi = Psi / (2 pi)."""
    gen = with_pest_pitch(eq_data)
    assert gen.general_coordinates
    gen.validate()
    assert max(gen.flux_mean_errors()) < 1e-12  # PEST: (iota, 1) exactly

    kwargs = {k: getattr(eq_data, k) for k in ("psi_r", "iota", "sqrtg")}
    with pytest.raises(ValueError, match="together or not at all"):
        eq_data.replace(B_sup_theta=kwargs["psi_r"])
    with pytest.raises(ValueError, match="must be positive everywhere"):
        gen.replace(B_sup_zeta=-gen.B_sup_zeta).validate()
    with pytest.raises(ValueError, match="toroidal flux"):
        gen.replace(B_sup_zeta=2 * np.pi * gen.B_sup_zeta).validate()


def test_general_fields_survive_a_round_trip(tmp_path, eq_data):
    gen = with_pest_pitch(eq_data)
    back = EquilibriumData.load(gen.save(tmp_path / "gen"))
    assert back.general_coordinates
    for key in ("B_sup_theta", "B_sup_zeta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(back, key)), np.asarray(getattr(gen, key))
        )
    leaves, aux = jax.tree_util.tree_flatten(gen)
    again = jax.tree_util.tree_unflatten(aux, leaves)
    assert again.general_coordinates and not eq_data.general_coordinates


def test_mass_block_is_the_metric_in_the_unknowns():
    """``mass_blocks`` is ``sqrt(g) M^T g M`` with ``xi = M u``, for ``iota != 0``
    and for a mirror, where the third unknown is the unscaled ``xi^zeta``."""
    rng = np.random.default_rng(1)
    E = rng.standard_normal((5, 3, 3))
    g = np.einsum("nki,nkj->nij", E, E)  # symmetric positive definite
    f = dict(
        g_rr=g[:, 0, 0],
        g_rv=g[:, 0, 1],
        g_rp=g[:, 0, 2],
        g_vv=g[:, 1, 1],
        g_vp=g[:, 1, 2],
        g_pp=g[:, 2, 2],
        psi_r=rng.uniform(0.2, 1.0, 5),
        sqrtg=rng.uniform(0.5, 2.0, 5),
        b_t=rng.uniform(-0.5, 2.0, 5),
        b_z=rng.uniform(0.5, 1.5, 5),
    )
    f = {k: jnp.asarray(v) for k, v in f.items()}
    for mirror, iota in ((False, rng.uniform(0.3, 1.2, 5)), (True, np.zeros(5))):
        f["iota"] = jnp.asarray(iota)
        m = general.mass_blocks(f, jnp.ones(5), mirror)
        s = np.ones(5) if mirror else iota
        b_t, b_z, psi_r = (np.asarray(f[k]) for k in ("b_t", "b_z", "psi_r"))
        M = np.zeros((5, 3, 3))
        M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = psi_r, 1 / b_z, 1 / s
        M[:, 1, 2] = b_t / (b_z * s)
        ref = np.asarray(f["sqrtg"])[:, None, None] * np.einsum(
            "nki,nkl,nlj->nij", M, g, M
        )
        got = np.empty_like(ref)
        for (i, j), key in {
            (0, 0): "rr",
            (0, 1): "ru",
            (1, 1): "uu",
            (0, 2): "rz",
            (1, 2): "uz",
            (2, 2): "zz",
        }.items():
            got[:, i, j] = got[:, j, i] = np.asarray(m[key])
        np.testing.assert_allclose(got, ref, rtol=1e-13)


# ---------------------------------------------------------------------------
# 1. The PEST limit
# ---------------------------------------------------------------------------


def _case(name, request):
    """``(eq, diffmat, config)`` of a named PEST case of the suite."""
    if name == "family":
        eq, _ = request.getfixturevalue("period_case")
        dm = fixture_basis(eq.resolution).nodes_and_diffmat(eq.NFP, family=1)[1]
        return eq, dm, AssemblyConfig()
    if name == "zernike":
        eq = request.getfixturevalue("dshape")
        n_rho, n_theta, _ = eq.resolution
        dm = Basis(n_rho, n_theta, 1, mpol=8, penalty=0.01).nodes_and_diffmat(1)[1]
        return eq, dm, AssemblyConfig(axisym=True, n_mode_axisym=2)
    eq, _ = request.getfixturevalue("period_case")
    return eq, build_diffmat(eq), AssemblyConfig()


@pytest.mark.parametrize("name", ["real", "family", "zernike"])
def test_general_operator_reduces_to_the_pest_operator(name, request):
    """With ``(b^theta, b^zeta) = (iota, 1)`` the general operator is the PEST
    one: the dense matrices, and the matrix-free product against both.

    Measured, as a fraction of ``max |A|`` (3.1e5 on the real case): dense 1.9e-16
    (real), 1.9e-16 (family), 3.3e-16 (Zernike); matrix-free 2.7e-16, 2.9e-16,
    9.0e-16 of ``max |A v|``; the general matrix is Hermitian to 2.3e-17. 1e-12
    leaves room for another BLAS.
    """
    eq, dm, cfg = _case(name, request)
    gen = with_pest_pitch(eq)
    A_pest = np.asarray(assemble_dense(eq, dm, cfg)["A"])
    A_gen = np.asarray(assemble_dense(gen, dm, cfg)["A"])
    scale = np.abs(A_pest).max()
    assert np.abs(A_gen - A_gen.conj().T).max() < 1e-12 * scale
    assert np.abs(A_gen - A_pest).max() < 1e-12 * scale

    op = matfree_operator(gen, dm, cfg)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(op["n_keep"])
    if np.iscomplexobj(A_pest):
        v = v + 1j * rng.standard_normal(op["n_keep"])
    y = np.asarray(op["Ax"](jnp.asarray(v)))
    ref = A_pest @ v
    assert np.abs(y - ref).max() < 1e-12 * np.abs(ref).max()


def test_general_ring_blocks_are_blocks_of_the_dense_matrix(period_case):
    """The ring path of the general operator: the block of every node is the
    whole matrix (as the PEST test of the old permutation tail)."""
    eq, _ = period_case
    gen, dm, cfg = with_pest_pitch(eq), build_diffmat(eq), AssemblyConfig()
    full = np.asarray(assemble_dense(gen, dm, cfg)["A"])
    pre = assemble_dense(gen, dm, cfg, ring_nodes=np.arange(eq.n_nodes))
    blk = finish_ring_block(pre["A"], pre["Linv"], pre["au_diag"], eq.n_nodes)
    keep = matfree_operator(gen, dm, cfg)["keep"]
    blk = np.asarray(blk)[np.ix_(keep, keep)]
    assert np.abs(blk - full).max() < 1e-12 * np.abs(full).max()


@pytest.mark.slow
def test_jd_on_the_general_operator_matches_the_dense_pest_eigenpair(eq_data, config):
    """Matrix-free Jacobi-Davidson through :mod:`agnimhd.general`: the Fourier-mode
    preconditioner is built from the (theta, zeta)-averaged general operator and
    the coarse level from the general fields on fewer angles. The eigenvalue is
    the dense PEST one (the case of ``test_jd_matches_the_dense_eigenpair``,
    family 0, the parity trap). Measured: 6.2e-12 apart, eigen-residual 9.3e-5."""
    gen = with_pest_pitch(eq_data)
    basis = fixture_basis(eq_data.resolution, mpol=5, ntor=1)
    diffmat = basis.nodes_and_diffmat(eq_data.NFP, 0)[1]
    coarse = basis.coarse_level(on_fewer_angles(gen, 1, zeta=slice(None, None, 2)), 0)
    ref = float(growth_rate(eq_data, diffmat, config))
    jd = SolverConfig(
        eigensolver="jd", sigma=1.3 * ref, jd_tol=1e-4, jd_theta_tol=0.0, jd_outer=500
    )
    gamma2, _, resid = eigenpair(gen, diffmat, config, jd, coarse=coarse)
    assert float(gamma2) == pytest.approx(ref, rel=1e-8)
    assert float(resid) <= jd.jd_tol, f"eigen-residual {float(resid):.3e}"


def test_displacement_and_perturbed_field_in_the_pest_limit(period_case):
    """The plotted displacement and ``delta B`` do not change when the PEST
    field is written out as general-coordinate fields."""
    eq, _ = period_case
    gen, dm, cfg = with_pest_pitch(eq), build_diffmat(eq), AssemblyConfig()
    op, op_gen = matfree_operator(eq, dm, cfg), matfree_operator(gen, dm, cfg)
    v = np.random.default_rng(3).standard_normal(op["n_keep"])
    xi, xi_gen = mode_displacement(eq, op, v), mode_displacement(gen, op_gen, v)
    np.testing.assert_allclose(xi_gen, xi, rtol=1e-11, atol=1e-11 * np.abs(xi).max())
    dB = mode_delta_b(eq, dm, op, v, cfg)
    dB_gen = mode_delta_b(gen, dm, op_gen, v, cfg)
    np.testing.assert_allclose(dB_gen, dB, rtol=0, atol=1e-11 * np.abs(dB).max())


# ---------------------------------------------------------------------------
# A synthetic configuration in two angle systems
# ---------------------------------------------------------------------------

PSI_B, R_0, A_MINOR = 0.5, 3.0, 1.0  # B_N = 2 PSI_B / A_MINOR^2 = 1 T


def _position(c):
    """Cartesian position at straight-angle coordinates ``c``: an ellipse of
    elongation 1.3 revolved about the z axis, right-handed in (rho, theta, zeta)."""
    r, t, z = c
    R = R_0 + A_MINOR * r * jnp.cos(t)
    return jnp.array([R * jnp.cos(z), R * jnp.sin(z), -1.3 * A_MINOR * r * jnp.sin(t)])


def _scalars(c, drive=6.0e5):
    """Profiles and scalar fields at straight-angle coordinates ``c``. ``iota``
    stays in (0.41, 0.44): no ``m iota = 1`` surface, so the ``n = 1`` mode the
    drive destabilizes is global and smooth."""
    r, t, _ = c
    return dict(
        J_sup_zeta=1.0e5 * (1 - 0.5 * r**2) * (1 + 0.1 * r * jnp.cos(t)),
        abs_J=3.5e5 * (1 - 0.4 * r**2),
        finite_n_instability_drive=drive * jnp.exp(-(((r - 0.55) / 0.25) ** 2)),
        p=2.0e4 * (1 - r**2) ** 2,
        p_r=-8.0e4 * r * (1 - r**2),
        iota=0.41 + 0.03 * r**2,
        psi_r=2 * PSI_B * r,
        psi_rr=2 * PSI_B + 0 * r,
    )


def _nodes(basis, nfp):
    nodes = basis.nodes_and_diffmat(nfp)[0]
    grids = np.meshgrid(
        *(np.asarray(nodes[k]) for k in ("rho", "theta", "zeta")), indexing="ij"
    )
    return jnp.asarray(np.stack([x.ravel() for x in grids], -1))


def synthetic(basis, lam=None, nfp=1, drive=6.0e5):
    """``EquilibriumData`` of the synthetic configuration on ``basis``' nodes.

    ``lam=None``: the nodes are uniform in the straight angle and the result is
    ordinary PEST data. Otherwise the nodes are uniform in ``theta = theta_s -
    lam(rho, theta, zeta)``, every geometric field is that of ``(rho, theta,
    zeta)``, and the field is given through ``B_sup_theta``, ``B_sup_zeta``.
    """

    def straight(c):
        return c if lam is None else jnp.array([c[0], c[1] + lam(c), c[2]])

    def position(c):
        return _position(straight(c))

    def jacobian(c):
        return jnp.linalg.det(jax.jacfwd(position)(c))

    def node(c):
        e = jax.jacfwd(position)(c)  # columns: covariant basis vectors
        grad = jnp.linalg.inv(e)  # rows: contravariant basis vectors
        g = e.T @ e
        cs = straight(c)
        e_s = jax.jacfwd(_position)(cs)
        s = _scalars(cs, drive)
        B = s["psi_r"] * (s["iota"] * e_s[:, 1] + e_s[:, 2]) / jnp.linalg.det(e_s)
        d_jac = jax.grad(jacobian)(c)
        out = dict(
            g_rr=g[0, 0],
            g_rv=g[0, 1],
            g_rp=g[0, 2],
            g_vv=g[1, 1],
            g_vp=g[1, 2],
            g_pp=g[2, 2],
            g_sup_rr=grad[0] @ grad[0],
            sqrtg=jacobian(c),
            sqrtg_r=d_jac[0],
            sqrtg_v=d_jac[1],
            sqrtg_p=d_jac[2],
            **s,
        )
        if lam is not None:
            out["B_sup_theta"] = grad[1] @ B
            out["B_sup_zeta"] = grad[2] @ B
        return out

    fields = jax.vmap(node)(_nodes(basis, nfp))
    return EquilibriumData(
        n_rho=basis.n_rho,
        n_theta=basis.n_theta,
        n_zeta=basis.n_zeta,
        a=A_MINOR,
        NFP=nfp,
        **fields,
    )


def lam_3d(c):
    """An angle change that depends on all three coordinates, of period 2 pi / 3
    in zeta. Measured on 8 x 16 x 8 nodes: ``b^zeta = 1 + d_theta lam`` runs over
    (0.50, 1.50) and ``b^theta = iota - d_zeta lam`` over (-0.16, 1.04), changing
    sign, with ``iota`` in (0.41, 0.44)."""
    r, t, z = c
    return (
        0.25 * jnp.sin(t)
        + 0.12 * jnp.sin(t - 3 * z)
        + 0.08 * r * jnp.cos(2 * t + 3 * z)
    )


def lam_2d(c):
    """An axisymmetric angle change: ``b^zeta`` runs over (0.70, 1.50) and
    ``b^theta = iota``."""
    return 0.3 * jnp.sin(c[1]) + 0.1 * c[0] * jnp.sin(2 * c[1])


SEPARABLE = dict(radial="lobatto", restrict=False, penalty=0.0)


def test_the_synthetic_angles_are_not_straight():
    """The test is not vacuous: the pitch is far from (iota, 1), ``b^theta``
    changes sign, and both flux integrals hold on the nodes."""
    basis = Basis(8, 16, 8, **SEPARABLE)
    eq = synthetic(basis, lam_3d, nfp=3)
    b_t, b_z = (np.asarray(b) for b in eq.field_pitch())
    assert b_z.min() < 0.6 and b_z.max() > 1.4
    assert b_t.min() < -0.05 and b_t.max() > 0.9
    assert max(eq.flux_mean_errors()) < 1e-11  # measured 2e-16 and 9e-14


# ---------------------------------------------------------------------------
# 2. Invariance of the energies
# ---------------------------------------------------------------------------


def _unknowns(cs):
    """A smooth displacement through its straight-angle unknowns (X, upsilon, Z)."""
    r, t, z = cs
    X = (1 - r**2) * r * (jnp.cos(2 * t - 3 * z) + 0.5 * jnp.sin(t) + 0.3)
    ups = r * (0.7 * jnp.sin(t - 3 * z) + 0.4 * jnp.cos(2 * t)) + 0.2 * r**2
    Z = 0.5 * r * jnp.cos(t + 3 * z) + 0.25 * r**2 * jnp.sin(2 * t - 3 * z) + 0.1
    return jnp.array([X, ups, Z])


def _energies(basis, lam):
    """``(delta W, delta K)`` of the displacement of ``_unknowns`` on ``basis``."""
    nfp = 3
    eq = synthetic(basis, lam, nfp)
    dm = basis.nodes_and_diffmat(nfp)[1]

    def unknowns(c):
        if lam is None:
            return _unknowns(c)
        X, ups, Z = _unknowns(jnp.array([c[0], c[1] + lam(c), c[2]]))
        # e_rho at fixed theta is e_rho + lam_rho e_theta_s of the straight
        # system, so upsilon = upsilon_s - psi' lam_rho X; X and Z are scalars
        # of the two systems (docs/general_coordinates.md). psi' is the
        # normalized one, psi'_SI / (a^2 B_N).
        psi_r = 2 * PSI_B * c[0] / (A_MINOR**2 * 2 * PSI_B / A_MINOR**2)
        return jnp.array([X, ups - psi_r * jax.grad(lam)(c)[0] * X, Z])

    op = matfree_operator(eq, dm, AssemblyConfig())
    v = whitened_vector(op, jax.vmap(unknowns)(_nodes(basis, nfp)))
    return float(jnp.vdot(v, op["Ax"](v)).real), float(jnp.vdot(v, v).real)


def test_energies_do_not_depend_on_the_angles():
    """``delta W`` and ``delta K`` of one displacement, PEST operator on straight
    nodes against the general operator on nodes of non-straight angles.

    Measured relative differences (delta W, delta K):

    ============  =========  =========
    resolution    delta W    delta K
    ============  =========  =========
    8 x 8 x 4     5.3e-04    9.7e-03
    12 x 12 x 6   6.0e-05    2.2e-04
    16 x 16 x 8   5.5e-07    4.8e-07
    20 x 24 x 12  6.1e-10    7.5e-13
    24 x 32 x 16  3.5e-12    < 1e-15
    ============  =========  =========
    """
    errors = []
    for res in ((8, 8, 4), (12, 12, 6), (16, 16, 8)):
        basis = Basis(*res, **SEPARABLE)
        (W, K), (W_gen, K_gen) = _energies(basis, None), _energies(basis, lam_3d)
        errors.append((abs(W_gen / W - 1), abs(K_gen / K - 1)))
    for (dW, dK), bound in zip(errors, (5e-3, 5e-4, 5e-6)):
        assert dW < bound and dK < 20 * bound
    assert errors[2][0] < 1e-2 * errors[0][0]  # spectral, not algebraic


# ---------------------------------------------------------------------------
# 3. Invariance of the eigenvalue
# ---------------------------------------------------------------------------


def _gamma2(eq, basis, n=1):
    dm = basis.nodes_and_diffmat(1)[1]
    A = assemble_dense(eq, dm, AssemblyConfig(axisym=True, n_mode_axisym=n))["A"]
    return -np.linalg.eigvalsh(np.asarray(A))[:2]


@pytest.mark.slow
def test_growth_rate_does_not_depend_on_the_angles():
    """The two most unstable ``gamma^2`` of toroidal mode ``n = 1``, in the
    straight angle and in ``theta = theta_s - lam_2d``.

    Measured (most unstable, second):

    ==========  ==========  ==========  =========  =========
    resolution  straight    general     rel. diff  second
    ==========  ==========  ==========  =========  =========
    16 x 17     0.2656454   0.2469994   -7.0e-02   -3.8e-04
    20 x 25     0.2670181   0.2658280   -4.5e-03   -2.2e-05
    24 x 33     0.2670203   0.2670041   -6.1e-05   -3.3e-07
    ==========  ==========  ==========  =========  =========

    At 20 x 25 the straight-angle value is 8e-6 from its 24 x 33 value and the
    general one 4.5e-3: in angles that are not straight the same mode takes
    more poloidal harmonics, so it needs more poloidal nodes.
    """
    rel = []
    for res in ((20, 25, 1), (24, 33, 1)):
        basis = Basis(*res, **SEPARABLE)
        straight = _gamma2(synthetic(basis), basis)
        gen = _gamma2(synthetic(basis, lam_2d), basis)
        rel.append(np.abs(gen / straight - 1))
    assert rel[0][0] < 1e-2 and rel[0][1] < 1e-4
    assert rel[1][0] < 2e-4 and rel[1][1] < 2e-6
    assert rel[1][0] < 0.05 * rel[0][0]


def test_gradient_through_the_general_operator():
    """Hellmann-Feynman through :mod:`agnimhd.general`: ``d gamma^2 / d(a)`` and
    the derivative with respect to a scale of ``J^zeta``, which enters only
    the general operator's mixed term, against central differences. Measured:
    3.4e-10 and 1.5e-10 apart, with ``d gamma^2 / d(scale)`` = 1.3e-2."""
    basis = Basis(10, 13, 1, **SEPARABLE)
    eq = synthetic(basis, lam_2d)
    dm = basis.nodes_and_diffmat(1)[1]
    cfg = AssemblyConfig(axisym=True, n_mode_axisym=1)
    solver = SolverConfig(eigensolver="eigsh", sigma=1.0)

    def eq_map(p):
        return eq.replace(a=p["a"], J_sup_zeta=p["j"] * eq.J_sup_zeta)

    def value(a, j):
        return float(
            growth_rate(
                {"a": jnp.asarray(a), "j": jnp.asarray(j)},
                dm,
                cfg,
                solver,
                equilibrium_map=eq_map,
            )
        )

    p0 = {"a": jnp.asarray(A_MINOR), "j": jnp.asarray(1.0)}
    grad = jax.grad(growth_rate)(p0, dm, cfg, solver, equilibrium_map=eq_map)
    h = 1e-5
    fd_a = (value(A_MINOR + h, 1.0) - value(A_MINOR - h, 1.0)) / (2 * h)
    fd_j = (value(A_MINOR, 1.0 + h) - value(A_MINOR, 1.0 - h)) / (2 * h)
    assert float(grad["a"]) == pytest.approx(fd_a, rel=1e-5)
    assert float(grad["j"]) == pytest.approx(fd_j, rel=1e-5)
    assert abs(fd_j) > 1e-4  # the general operator is in the derivative
