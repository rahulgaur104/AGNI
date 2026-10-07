"""Anisotropic pressure: Bernstein's double-adiabatic energy principle.

Three independent checks of the terms in ``agnimhd.anisotropy``:

1. a straight periodic cylinder with uniform ``B``, ``p_perp`` and ``p_par``
   on a coupled Zernike-Fourier ``(rho, theta)`` grid, where Bernstein's
   ``dW`` is known in closed form for any displacement (and contains the
   shear Alfven / firehose, fast and slow branches);
2. the isotropic limit on the shipped QH case, where the anisotropic ``dW``
   must exceed the isotropic one by exactly ``(1/3) int p (div xi - 3 s)^2``
   -- Bernstein's remark below his anisotropic ``dW`` -- which also checks the
   direct ``- xi.(j x Q) + div xi  xi.grad p`` against the code's rearranged
   mixed, ``|J|^2`` and drive terms;
3. dense, ring and matrix-free routes agree on a synthetic anisotropic case.
"""

import numpy as np
import pytest

from agnimhd import EquilibriumData
from agnimhd.assemble import assemble_dense, matfree_operator, ring_block
from agnimhd.backend import jnp
from agnimhd.basis import DiffMat, fourier_diffmat, zernike_fourier_diffmat
from agnimhd.config import AssemblyConfig
from agnimhd.equilibrium import ANISOTROPY_ARRAYS
from agnimhd.quadrature import zernike_nodes_weights


def _raw_form(op, xi):
    """``xi^T A xi`` of the unwhitened operator from the whitened matrix-free one.

    ``Ax_full`` acts on ``x`` with ``xi = diagBsqinv * L^{-T} x``, so
    ``x = L^T (xi / diagBsqinv)`` and ``x . Ax_full(x)`` is the raw form.
    """
    n = op["n_total"]
    xi_n = jnp.asarray(xi).reshape(3, n).T / op["diagBsqinv"]
    x = jnp.linalg.solve(op["Linv_DT"], xi_n[..., None])[..., 0]  # L^T xi_n
    x = x.T.reshape(-1)
    return float(jnp.real(jnp.vdot(x, op["Ax_full"](x))))


def _cylinder(n_rho=8, n_theta=12, n_zeta=8, B0=1.0, p_perp=0.1, p_par=0.3, L=5.0):
    """Straight periodic cylinder, radius 1, length ``L``, uniform axial field.

    PEST coordinates ``(rho, theta, zeta)``: ``x = rho cos theta``,
    ``y = rho sin theta``, ``z = L zeta / 2 pi``. ``iota = 0`` (mirror branch).
    Pressures in units of ``B0^2 / mu_0``; the solver's normalization makes
    ``B = 1`` and ``p`` the given numbers. The ``(rho, theta)`` basis is the
    coupled Zernike-Fourier one, on its Gauss-Jacobi nodes, so the axis is
    handled by the basis and nothing is clipped at a small ``rho``.
    """
    from scipy.constants import mu_0

    rho, w_rho, theta, w_theta = zernike_nodes_weights(n_rho, n_theta)
    D_rho, D_theta = zernike_fourier_diffmat(
        rho, theta, L=8, M=4, spectral_indexing="ansi"
    )
    D_zeta, W_zeta = fourier_diffmat(n_zeta)
    dm = DiffMat(
        D_rho=D_rho,
        W_rho=w_rho,
        D_theta=D_theta,
        W_theta=w_theta,
        D_zeta=D_zeta,
        W_zeta=jnp.diagonal(W_zeta),
    )
    config = AssemblyConfig(
        coupled_rt=True, n_rho_coupled=n_rho, n_theta_coupled=n_theta
    )
    ze = 2 * np.pi * np.arange(n_zeta) / n_zeta
    R, T, Z = np.meshgrid(np.asarray(rho), np.asarray(theta), ze, indexing="ij")
    n = R.size
    one, zero = np.ones(n), np.zeros(n)
    Lz = L / (2 * np.pi)
    Psi = np.pi * B0  # a = 1 so B_N = Psi / pi = B0
    fields = dict(
        g_rr=one,
        g_rv=zero,
        g_rp=zero,
        g_vv=(R**2).ravel(),
        g_vp=zero,
        g_pp=Lz**2 * one,
        g_sup_rr=one,
        sqrtg=(R * Lz).ravel(),
        sqrtg_r=Lz * one,
        sqrtg_v=zero,
        sqrtg_p=zero,
        J_sup_zeta=zero,
        abs_J=zero,
        iota=zero,
        psi_r=(Psi * R / np.pi).ravel(),
        psi_rr=Psi / np.pi * one,
        p=p_perp * B0**2 / mu_0 * one,
        p_r=zero,
        finite_n_instability_drive=zero,
        p_perp=p_perp * B0**2 / mu_0 * one,
        p_par=p_par * B0**2 / mu_0 * one,
        grad_p_perp=np.zeros((n, 3)),
        grad_p_par=np.zeros((n, 3)),
        grad_lnB=np.zeros((n, 3)),
        T_b=np.zeros((n, 3, 3)),
        J_sup_rho=zero,
        J_sup_theta=zero,
    )
    eq = EquilibriumData(
        n_rho=n_rho, n_theta=n_theta, n_zeta=n_zeta, Psi=Psi, a=1.0, **fields
    )
    grid = dict(
        X=R * np.cos(T),
        Y=R * np.sin(T),
        Z=Z * Lz,
        W=np.asarray(np.kron(dm.w_rho, np.kron(dm.w_theta, dm.w_zeta))).reshape(R.shape)
        * np.asarray(eq.sqrtg).reshape(R.shape),
    )
    return eq, dm, config, grid


def _homogeneous_dw(p_perp, p_par, dz_perp, grad_perp_z, D, s, Wg):
    """Bernstein's ``dW`` for uniform ``B = z_hat`` (``B = 1``) and pressures.

    ``Q = d_z xi - D z_hat``, ``e~ = d_z xi_perp``, ``v = grad xi_z``,
    ``w = d_z xi``, ``u = 0``, so (``sigma = p_par - p_perp``)

        dW = INT |d_z xi_perp|^2 + (D - s)^2 + (5/3) p_perp D^2
             + (1/3) p_perp (D - 3 s)^2
             + sigma (s D + 2 s^2 - d_z xi_perp . grad_perp xi_z - |d_z xi_perp|^2)

    which reduces to ``(1 - sigma) |d_z xi_perp|^2`` for a shear Alfven
    displacement (firehose beyond ``sigma = 1``), ``(1 + 2 p_perp) D^2`` for a
    fast one and ``3 p_par s^2`` for a slow one.
    """
    sigma = p_par - p_perp
    dzp2 = sum(c**2 for c in dz_perp)
    f = (
        dzp2
        + (D - s) ** 2
        + (5.0 / 3.0) * p_perp * D**2
        + p_perp / 3.0 * (D - 3 * s) ** 2
        + sigma
        * (s * D + 2 * s**2 - sum(a * b for a, b in zip(dz_perp, grad_perp_z)) - dzp2)
    )
    return float(np.sum(Wg * f))


@pytest.mark.parametrize(
    "p_perp, p_par, shear_only",
    [(0.1, 0.3, False), (0.4, 0.05, False), (0.1, 0.3, True), (0.1, 1.4, True)],
)
def test_cylinder_matches_bernstein_closed_form(p_perp, p_par, shear_only):
    """``dW`` on a uniform cylinder equals Bernstein's functional in closed form.

    The displacement is given directly in the solver's variables
    ``(xi^rho, upsilon, xi^zeta)`` as smooth functions on the disc, so the
    Zernike basis represents them exactly:

        xi_perp = (1 + alpha x) h1(z) (x, y) + beta b(y) h2(z) (-y, x),
        xi_z    = (1 + gamma x) h3(z).

    with ``b = y`` in general and ``b = 1`` for ``shear_only``, where only the
    rigid rotation ``h2`` is kept, which is
    a torsional shear Alfven displacement: ``dW = (1 - sigma) INT h2'^2 rho^2``
    and the sign flips at the firehose threshold ``p_par - p_perp = B^2``.
    The same case also checks that the coupled-``(rho, theta)`` dense matrix
    equals the matrix-free operator column by column.
    """
    eq, dm, config, g = _cylinder(p_perp=p_perp, p_par=p_par)
    X, Y = g["X"], g["Y"]
    Lz = 5.0 / (2 * np.pi)
    ze = g["Z"] / Lz  # the periodic coordinate; d/dz = (1/Lz) d/dzeta
    alpha, beta, gamma = (0.0, 0.7, 0.0) if shear_only else (0.3, 0.7, 0.5)
    h1, dh1 = (0 * ze, 0 * ze) if shear_only else (np.sin(ze), np.cos(ze) / Lz)
    h2, dh2 = np.cos(ze), -np.sin(ze) / Lz
    h3, dh3 = (
        (0 * ze, 0 * ze) if shear_only else (np.sin(ze + 0.4), np.cos(ze + 0.4) / Lz)
    )
    a, c = 1 + alpha * X, 1 + gamma * X
    b, b_y = (beta + 0 * Y, 0 * Y) if shear_only else (beta * Y, beta + 0 * Y)
    # solver variables for a mirror: xi^rho = xi~^rho/psi' = a h1 (psi' = B0 rho),
    # upsilon = xi~^theta = b h2, xi^zeta = xi~^zeta = xi_z (2 pi / L)
    xi = np.stack([a * h1, b * h2, c * h3 / Lz])
    dz_perp = (a * X * dh1 - b * Y * dh2, a * Y * dh1 + b * X * dh2)
    D = h1 * (2 * a + alpha * X) + h2 * X * b_y + c * dh3
    s = c * dh3
    grad_perp_z = (gamma * h3, 0 * h3)
    want = _homogeneous_dw(p_perp, p_par, dz_perp, grad_perp_z, D, s, g["W"])
    op = matfree_operator(eq, dm, config)
    got = _raw_form(op, xi)
    assert got == pytest.approx(want, rel=1e-9)
    if shear_only:
        assert (got < 0) == (p_par - p_perp > 1.0)
        A = np.asarray(assemble_dense(eq, dm, config)["A"])
        for j in np.random.default_rng(0).choice(op["n_keep"], 8, replace=False):
            e = jnp.zeros(op["n_keep"]).at[j].set(1.0)
            err = np.max(np.abs(np.asarray(op["Ax"](e)) - A[:, j]))
            assert err < 1e-12 * np.max(np.abs(A[:, j]))


# ---------------------------------------------------------------------------
# Isotropic limit on the shipped case
# ---------------------------------------------------------------------------


def _with_isotropic_anisotropy(eq, diffmat):
    """The shipped case with ``p_perp = p_par = p`` in the anisotropic fields.

    ``J^theta`` comes from the isotropic force balance, ``J^rho = 0``, ``T_b``
    vanishes (it only multiplies ``sigma = 0``) and ``grad ln|B|`` is formed
    spectrally from ``|B|^2 = (psi'/sqrt g)^2 (iota^2 g_vv + 2 iota g_vp + g_pp)``.
    """
    n = eq.n_nodes
    shape = eq.resolution
    B2 = (np.asarray(eq.psi_r) / np.asarray(eq.sqrtg)) ** 2 * (
        np.asarray(eq.iota) ** 2 * np.asarray(eq.g_vv)
        + 2 * np.asarray(eq.iota) * np.asarray(eq.g_vp)
        + np.asarray(eq.g_pp)
    )
    lnB = 0.5 * np.log(B2).reshape(shape)
    D = [
        np.asarray(diffmat.D_rho),
        np.asarray(diffmat.D_theta),
        np.asarray(diffmat.D_zeta),
    ]
    grad = np.stack(
        [
            np.einsum("ij,jkl->ikl", D[0], lnB).ravel(),
            np.einsum("ij,kjl->kil", D[1], lnB).ravel(),
            np.einsum("ij,klj->kli", D[2], lnB).ravel(),
        ],
        -1,
    )
    gp = np.stack([np.asarray(eq.p_r), np.zeros(n), np.zeros(n)], -1)
    # j^theta = iota j^zeta + p'/psi' holds in SI as written: B carries mu_0.
    J_theta = np.asarray(eq.iota) * np.asarray(eq.J_sup_zeta) + np.asarray(
        eq.p_r
    ) / np.asarray(eq.psi_r)
    return eq.replace(
        p_perp=eq.p,
        p_par=eq.p,
        grad_p_perp=gp,
        grad_p_par=gp,
        grad_lnB=grad,
        T_b=np.zeros((n, 3, 3)),
        J_sup_rho=np.zeros(n),
        J_sup_theta=J_theta,
    )


def _smooth_xi(eq, eq_meta, m, n, mu, nu, mz, nz):
    rho = np.asarray(eq_meta["rho_nodes"])
    n_rho, n_theta, n_zeta = eq.resolution
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    ze = 2 * np.pi * np.arange(n_zeta) / (n_zeta * eq.NFP)
    R, T, Z = np.meshgrid(rho, th, ze, indexing="ij")
    xr = R**2 * (1 - R) ** 2 * np.cos(m * T - n * eq.NFP * Z)
    xr[0] = xr[-1] = 0.0
    xu = R * (1 - 0.5 * R) * np.sin(mu * T - nu * eq.NFP * Z)
    xz = R**2 * np.cos(mz * T - nz * eq.NFP * Z + 0.3)
    return np.stack([xr, xu, xz])


def _stretch_term(eq, diffmat, config, xi):
    """``(1/3) int p (div xi - 3 s)^2`` by its own stencils, in solver units.

    ``s = (B . Q)/|B|^2 + xi . grad ln|B| + div xi`` with ``Q`` from paper
    Eq. 22; everything is normalized exactly as ``assemble._normalized_fields``.
    """
    from agnimhd.assemble import _normalized_fields

    f = _normalized_fields(eq, config)
    shape = eq.resolution
    g = {
        k: np.asarray(v).reshape(shape + (() if v.shape[1:] == (1,) else v.shape[1:]))
        for k, v in f.items()
        if getattr(v, "ndim", 0)
    }
    Dr, Dt, Dz = (
        np.asarray(D) for D in (diffmat.D_rho, diffmat.D_theta, diffmat.D_zeta)
    )

    def d_r(u):
        return np.einsum("ij,jkl->ikl", Dr, u)

    def d_t(u):
        return np.einsum("ij,kjl->kil", Dt, u)

    def d_z(u):
        return np.einsum("ij,klj->kli", Dz, u)

    W = np.asarray(
        np.kron(diffmat.w_rho, np.kron(diffmat.w_theta, diffmat.w_zeta))
    ).reshape(shape)
    xr, xu, xz = xi
    iota, psi_r, sqrtg = g["iota"], g["psi_r"], g["sqrtg"]
    Xr, Xt, Xz = psi_r * xr, xu + xz, xz / iota
    Qr = psi_r**2 / sqrtg * (iota * d_t(xr) + d_z(xr))
    Qt = psi_r / sqrtg * (d_z(xu) - d_r(iota * psi_r**2 * xr) / psi_r)
    Qz = -psi_r / sqrtg * (d_t(xu) + d_r(psi_r**2 * xr) / psi_r)
    Div = (
        d_r(sqrtg * Xr) / sqrtg
        + d_t(Xt)
        + Xt * g["sqrtg_v"] / sqrtg
        + d_z(Xz)
        + Xz * g["sqrtg_p"] / sqrtg
    )
    B_sup_t, B_sup_z = iota * psi_r / sqrtg, psi_r / sqrtg
    B_cov_r = g["g_rv"] * B_sup_t + g["g_rp"] * B_sup_z
    B_cov_t = g["g_vv"] * B_sup_t + g["g_vp"] * B_sup_z
    B_cov_z = g["g_vp"] * B_sup_t + g["g_pp"] * B_sup_z
    B2 = B_sup_t * B_cov_t + B_sup_z * B_cov_z
    BQ = B_cov_r * Qr + B_cov_t * Qt + B_cov_z * Qz  # B_i Q^i
    lnB = g["grad_lnB"]
    s = BQ / B2 + Xr * lnB[..., 0] + Xt * lnB[..., 1] + Xz * lnB[..., 2] + Div
    return float(np.sum(W * sqrtg * g["p0"] / 3 * (Div - 3 * s) ** 2))


def test_isotropic_limit_exceeds_isotropic_dw_by_the_stretching_term(
    eq_data, eq_meta, diffmat, config
):
    """With ``p_par = p_perp = p``, ``dW_aniso - dW_iso = (1/3) int p (D - 3 s)^2``.

    The difference also carries the aliasing error of the rearranged isotropic
    terms (mixed Q-J, |J|^2 and drive, which trade derivatives of ``xi`` for
    derivatives of the equilibrium through the force balance). Measured on
    the shipped 24x12x8 case: 1e-7 of the ``|Q|^2`` form for these
    displacements, and 1e-8 at 24x16 angular points.
    """
    eq_an = _with_isotropic_anisotropy(eq_data, diffmat)
    op_iso = matfree_operator(eq_data, diffmat, config)
    op_an = matfree_operator(eq_an, diffmat, config)
    for modes in [(1, 0, 1, 0, 1, 0), (2, 1, 1, 1, 2, 1), (3, 2, 2, 2, 1, 1)]:
        xi = _smooth_xi(eq_data, eq_meta, *modes)
        diff = _raw_form(op_an, xi) - _raw_form(op_iso, xi)
        term = _stretch_term(eq_an, diffmat, config, xi)
        assert term > 0
        assert diff == pytest.approx(term, rel=1e-3, abs=1e-6 * term)


# ---------------------------------------------------------------------------
# The assembly routes agree on an anisotropic case
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eq_aniso(eq_data, diffmat):
    """The shipped case with smooth, nonzero anisotropy fields.

    Only the operator's self-consistency is tested on it, so the fields need
    not come from a real anisotropic equilibrium. ``T_b`` satisfies
    ``B^i T_ik = 0`` by construction (``T_pk = -iota T_vk``).
    """
    eq = _with_isotropic_anisotropy(eq_data, diffmat)
    n_rho, n_theta, n_zeta = eq.resolution
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    ze = 2 * np.pi * np.arange(n_zeta) / (n_zeta * eq.NFP)
    T, Z = np.meshgrid(th, ze, indexing="ij")
    ang = np.tile(np.cos(T - eq.NFP * Z)[None], (n_rho, 1, 1)).ravel()
    p = np.asarray(eq.p)
    gp = np.asarray(eq.grad_p_perp)
    T_b = np.zeros((eq.n_nodes, 3, 3))
    T_b[:, 0, :] = 0.3 * np.stack([ang, 0.5 * ang, 0.2 * ang], -1)
    T_b[:, 1, :] = 0.1 * np.stack([ang, -ang, 0.4 * ang], -1)
    T_b[:, 2, :] = -np.asarray(eq.iota)[:, None] * T_b[:, 1, :]
    return eq.replace(
        p_perp=1.2 * p,
        p_par=0.7 * p * (1 + 0.3 * ang),
        grad_p_perp=1.2 * gp,
        grad_p_par=0.7 * gp * (1 + 0.3 * ang)[:, None],
        T_b=T_b,
        J_sup_rho=0.05 * np.asarray(eq.J_sup_zeta) * ang,
    )


def test_anisotropic_contract(eq_data, eq_aniso, tmp_path):
    """The fields come together, round-trip through ``.npz`` and validate."""
    assert not eq_data.anisotropic and eq_aniso.anisotropic
    eq_aniso.validate()
    path = eq_aniso.save(tmp_path / "an.npz")
    back = EquilibriumData.load(path)
    for key in ANISOTROPY_ARRAYS:
        np.testing.assert_array_equal(
            np.asarray(getattr(back, key)), np.asarray(getattr(eq_aniso, key))
        )
    with pytest.raises(ValueError, match="supplied together"):
        eq_data.replace(p_perp=eq_data.p).validate()
    bad = np.asarray(eq_aniso.T_b).copy()
    bad[:, 2, :] += 1.0
    with pytest.raises(ValueError, match="B\\^i T_ik = 0"):
        eq_aniso.replace(T_b=bad).validate()


def test_anisotropic_dense_matches_matfree(eq_aniso, diffmat, config):
    """Dense, ring and matrix-free routes agree, and the operator is Hermitian."""
    out = assemble_dense(eq_aniso, diffmat, config)
    A = np.asarray(out["A"])
    assert np.all(np.isfinite(A))
    assert np.max(np.abs(A - A.T)) < 1e-9 * np.max(np.abs(A))
    op = matfree_operator(eq_aniso, diffmat, config)
    rng = np.random.default_rng(0)
    for j in rng.choice(op["n_keep"], size=12, replace=False):
        e = jnp.zeros(op["n_keep"]).at[j].set(1.0)
        err = np.max(np.abs(np.asarray(op["Ax"](e)) - A[:, j]))
        assert err < 1e-12 * np.max(np.abs(A[:, j])), f"column {j} disagrees"
    # an interior ring block equals the corresponding sub-block of the matrix
    from agnimhd.solvers import ring_nodes

    n_rho, n_theta, n_zeta = eq_aniso.resolution
    n_total = eq_aniso.n_nodes
    full_to_red = -np.ones(3 * n_total, dtype=np.int64)
    full_to_red[np.asarray(out["keep"])] = np.arange(op["n_keep"])
    nodes = ring_nodes(n_rho, n_theta, n_zeta, n_rho // 2, 1)
    blk = np.asarray(ring_block(eq_aniso, diffmat, config, nodes))
    idx = full_to_red[np.concatenate([nodes, nodes + n_total, nodes + 2 * n_total])]
    sub = A[np.ix_(idx, idx)]
    assert np.max(np.abs(blk - sub)) < 1e-12 * np.max(np.abs(sub))
