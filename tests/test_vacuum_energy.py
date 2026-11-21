"""The vacuum energy against an independent answer: coil fields volume-integrated once.

Reference data (``tests/data/vacuum_coils/``), for coil pairs on the flux surfaces
of ``AGNI_QH_lowres.h5`` (NFP = 4) with opposite currents.

For each pair, the vacuum energy was computed by volume integral:

    W_vol = (1/2mu_0) int_exterior B^2 dV

There are two coil sets.
* ``AGNI_QH_lowres_*``: constant (rho, theta).
* ``AGNI_QH_lowres_n1_*``: helical, theta = theta0 + zeta.

In this test, Bn from the coils is split over the NFP mode families,
and the vacuum energy is computed by summing over all families.

1. ``test_surface_energy_matches_volume_integral``: on the 41 x 25 lattice, finer
   than typical eigensolve resolution, to confirm the correctness of the
   implementation.
2. ``test_family_response_is_the_folded_full_torus_response``: P_x (one field period,
   the Bloch phase in the boundary integral) against the response computed over the
   whole torus and folded with the family phase.
3. ``test_eigensolve_vacuum_energy_matches_verified_matrix`` (families 0 and 1): at
   the lower resolution used by an eigensolve, the surface integral is compared
   to the reference volume integral. Then, the vacuum energy used by the matrix-free
   and dense operators is compared to that verified matrix.

Needs DESC. Missing reference data fails. Set ``AGNI_VACUUM_COILS`` to read the
reference data from another directory.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import mu_0 as MU_0

from agnimhd import AssemblyConfig, Basis, SolverConfig, eigenpair, from_desc, load

DATA = Path(__file__).parent / "data"
DESC_FILE = DATA / "AGNI_QH_lowres.h5"
COILS = Path(os.environ.get("AGNI_VACUUM_COILS", DATA / "vacuum_coils"))
NAME = "AGNI_QH_lowres"
HIGH_RES = (41, 25)
LOW_RES = (8, 17, 11, 6, 4)


def _reference(kind="symmetric"):
    """The volume-integrated energies of the coil sets, and their B_n on the tests'
    lattices: ``kind`` "symmetric" (family 0) or "n1" (every family).
    """
    tag = NAME if kind == "symmetric" else f"{NAME}_n1"
    energy = COILS / f"{tag}_coil_energy.npz"
    if not energy.exists():
        pytest.fail(
            f"vacuum reference data missing from {COILS}: {energy.name}.",
            pytrace=False,
        )
    ref = dict(np.load(energy))
    digest = hashlib.sha256(DESC_FILE.read_bytes()).hexdigest()
    assert (
        str(ref["eq_sha256"]) == digest
    ), "the reference data was made from a different AGNI_QH_lowres.h5"
    if kind == "n1":
        assert not bool(ref["nfp_symmetric"]), "the n1 reference is NFP-symmetric"
    return ref


def reference_normal_field(ref, n_theta, n_zeta):
    """Each coil set's B_n on the ``n_theta x n_zeta`` lattice."""
    key = f"Bn_{n_theta}x{n_zeta}"
    if key not in ref:
        pytest.fail(
            f"the reference has no B_n on the {n_theta} x {n_zeta} lattice",
            pytrace=False,
        )
    return ref[key]


def _eq():
    """The equilibrium used to produce the reference data."""
    return load(str(DESC_FILE)).eq


def pest_to_desc(eq, theta, zeta):
    """Map PEST coordinates to DESC (rho, theta, zeta)."""
    grid = pytest.importorskip("desc.grid").LinearGrid(
        rho=1.0, theta=theta, zeta=zeta, NFP=1
    )
    return eq.map_coordinates(
        grid.meshgrid_reshape(grid.nodes, "rtz").reshape(-1, 3),
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
        tol=1e-12,
        maxiter=50,
    )


def shell_geometry(eq, n_theta, n_zeta, keys=()):
    """Compute surface quantities necessary to compute the vacuum energy."""
    Grid = pytest.importorskip("desc.grid").Grid
    nfp = eq.NFP
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    zeta = 2 * np.pi * np.arange(n_zeta) / (nfp * n_zeta)
    keys = ["g^rr", "sqrt(g)_PEST", *keys]
    data = eq.compute(
        keys, grid=Grid(pest_to_desc(eq, theta, zeta), NFP=nfp, sort=False)
    )
    dA = data["sqrt(g)_PEST"] * np.sqrt(data["g^rr"])
    data["dA"] = dA * 4 * np.pi**2 / (n_theta * n_zeta)
    return data


def family_matrices(eq, n_theta, n_zeta, families):
    """``{x: P_x}`` on the rho = 1 shell of an AGNI basis."""
    from agnimhd.adapters.desc_vacuum import phi_matrix, phi_scaffold

    nfp = eq.NFP
    # only the shell's angular nodes; Basis needs n_rho >= 3, which do not matter here
    nodes, _ = Basis(3, n_theta, n_zeta, free_boundary=True).nodes_and_diffmat(nfp)
    theta, zeta = nodes["theta"], nodes["zeta"]
    scaffold = phi_scaffold(
        eq, theta, zeta, NFP_surf=nfp, stellsym=bool(eq.sym), verbose=False
    )
    rtz = pest_to_desc(eq, theta, zeta)
    return {
        x: phi_matrix(eq, eq.params_dict, scaffold, rtz, family=x) for x in families
    }


def family_split(B_p):
    """Decomposed the magnetic field into NFP mode families."""
    return np.fft.fft(B_p, axis=0) / B_p.shape[0]


def surface_energy(dA, b, P):
    """Sum vacuum energy over NFP mode families."""
    terms = np.array([np.vdot(dA * b[x], P[x] @ b[x]).real for x in range(b.shape[0])])
    return terms.sum() / (2.0 * MU_0), terms / terms.sum()


def _check_pairs(label, rows):
    """Produces a table comparing the volume and surface integrals."""
    lines, bad = [f"{label}: (surface - volume) / volume"], []
    for rho, W_surf, W_vol, tol, note in rows:
        err = W_surf / W_vol - 1
        lines.append(
            f"  rho {rho:.3f}  surface {W_surf:.6e} J  volume {W_vol:.6e} J  "
            f"{err:+.2%}  (tol {tol:.1%}){note}"
            + ("" if abs(err) <= tol else "  <- FAIL")
        )
        bad += [] if abs(err) <= tol else [rho]
    table = "\n".join(lines)
    print(table)
    assert not bad, table


def coil_check(eq, kind, n_theta, n_zeta, tol, label, keys=()):
    """Each coil pair's surface energy compared to the volume integral.
    Returns the  phi matrix for each family and the shell geometry."""
    ref = _reference(kind)
    Bn = reference_normal_field(ref, n_theta, n_zeta)  # (n_sets, periods, n_surf)
    nfp = Bn.shape[1]  # the families x to sum over: NFP for n1, 1 for symmetric
    geo = shell_geometry(eq, n_theta, n_zeta, keys)
    built = family_matrices(eq, n_theta, n_zeta, range(nfp // 2 + 1))
    P = {x: built[x] if x in built else np.conj(built[nfp - x]) for x in range(nfp)}
    W_vol = ref["W_vol"]
    assert Bn.shape[0] == W_vol.shape[0] == ref["rho"].size
    rows = []
    for coil_idx, rho in enumerate(ref["rho"]):
        W_surf, share = surface_energy(geo["dA"], family_split(Bn[coil_idx]), P)
        if kind == "n1":  # else this would test P_0 again
            assert share[0] < 0.1, f"rho = {rho:.2f}: family 0 holds {share[0]:.0%}"
        rows.append(
            (
                rho,
                W_surf,
                W_vol[coil_idx, -1],
                tol,
                f"  shares {np.round(share, 3)}" if kind == "n1" else "",
            )
        )
    _check_pairs(f"{label}, {kind} coils, {n_theta} x {n_zeta}", rows)
    return P, geo


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["symmetric", "n1"])
def test_surface_energy_matches_volume_integral(kind):
    """The surface energy of each coil pair equals the volume integral, at a higher
    resolution and tighter tolerance."""
    pytest.importorskip("desc")
    coil_check(_eq(), kind, *HIGH_RES, 1e-3, "high res")


def test_family_response_is_the_folded_full_torus_response():
    """P_x, the Bloch phase exp(i x zeta) inside the one-period boundary integral,
    against the response computed over the WHOLE torus (one period, NFP_surf = 1) and
    folded with the family phase, sum_p e^{2 pi i x p / NFP} P_{0p}

    For x = NFP/2 no mode window is symmetric about n = 0,
    and the real part of P_x keeps the one edge mode the full torus does not resolve:
    a discretization difference, measured 6.2e-3 at 9 x 7 and 1.8e-3 at 13 x 11.
    Also: P_{NFP-x} = conj(P_x), the vacuum energy is positive, and
    family x's D_zeta differentiates exp(+i x zeta) to i x exp(+i x zeta)."""
    pytest.importorskip("desc")
    from agnimhd.adapters.desc_vacuum import phi_matrix, phi_scaffold

    eq = _eq()
    nfp = eq.NFP
    n_theta, n_zeta = 9, 7
    basis = Basis(4, n_theta, n_zeta, free_boundary=True)
    nodes, _ = basis.nodes_and_diffmat(nfp)
    theta, zeta = nodes["theta"], nodes["zeta"]
    kw = dict(st=6, sz=2, q=3, chunk_size=0, verbose=False, stellsym=bool(eq.sym))

    def shell(zeta_):
        return pest_to_desc(eq, theta, zeta_)

    scaffold = phi_scaffold(eq, theta, zeta, NFP_surf=nfp, **kw)
    P = {
        x: phi_matrix(eq, eq.params_dict, scaffold, shell(zeta), family=x)
        for x in range(nfp)
    }
    zeta_full = np.concatenate([zeta + 2 * np.pi * p / nfp for p in range(nfp)])
    full = phi_scaffold(eq, theta, zeta_full, NFP_surf=1, **kw)
    P_full = phi_matrix(eq, eq.params_dict, full, shell(zeta_full))
    blocks = P_full.reshape(n_theta, nfp * n_zeta, n_theta, nfp, n_zeta)[:, :n_zeta]
    scale = np.max(np.abs(P_full))
    for x in range(nfp):
        phase = np.exp(2j * np.pi * x * np.arange(nfp) / nfp)
        folded = np.tensordot(blocks, phase, axes=(3, 0)).reshape(P[x].shape)
        err = np.max(np.abs(P[x] - folded)) / scale
        assert err < (1e-2 if 2 * x == nfp else 1e-12), f"x = {x}: {err:.2e}"
        assert (2 * x % nfp == 0) == np.isrealobj(P[x]), f"x = {x}: {P[x].dtype}"
    for x in range(1, nfp):
        np.testing.assert_array_equal(P[nfp - x], np.conj(P[x]))
    geo = shell_geometry(eq, n_theta, n_zeta)
    for x in range(nfp):
        H = np.diag(geo["dA"]) @ P[x]
        w = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
        assert w.min() > -1e-10 * w.max(), f"x = {x}: min eigenvalue {w.min():.2e}"
    # the phase convention P_x shares with the operator: family x is exp(+i x zeta)
    for x in range(nfp):
        D = basis.nodes_and_diffmat(nfp, family=x)[1].D_zeta
        f = np.exp(1j * x * zeta)
        np.testing.assert_allclose(D @ f, 1j * x * f, atol=1e-12 * max(x, 1))


@pytest.mark.slow
@pytest.mark.parametrize("family", [0, 1])
def test_eigensolve_vacuum_energy_matches_verified_matrix(family):
    """Checks the low-resolution surface integral against the volume integral,
    and then confirms that the eigensolve computes the same vacuum energy.

    Builds the vacuum energy matrix and tests it against the NFP-symmetric
    coils and the helical coil pairs.

    Then Bn from the eigensolve is computed, and the same matrix is compared
    to the matrix-free operator and to the dense matrix, for the eigenmode
    and for four random vectors."""
    from agnimhd.assemble import (
        _grid_derivs,
        assemble_dense,
        matfree_operator,
        physical_vector,
    )

    pytest.importorskip("desc")
    eq = _eq()
    nfp = eq.NFP
    n_rho, n_theta, n_zeta, mpol, ntor = LOW_RES
    basis = Basis(n_rho, n_theta, n_zeta, mpol=mpol, ntor=ntor, free_boundary=True)
    eqd, dm = from_desc(eq, basis, family=family)
    assert eqd.phi_family == family
    assert (family == 0) == np.isrealobj(eqd.phi_matrix)

    # 1. the high-resolution test's coil check on this shell, and the solve's matrix is
    # the one checked
    kind = "n1" if family else "symmetric"
    Pfam, geo = coil_check(
        eq, kind, n_theta, n_zeta, 3e-2, "low res", keys=("psi_r", "iota", "a")
    )
    np.testing.assert_allclose(
        eqd.phi_matrix, Pfam[family], rtol=0, atol=1e-12 * np.max(np.abs(Pfam[family]))
    )

    # 2. the eigensolve, and the vacuum energy of its mode and of random vectors
    config = AssemblyConfig()
    # dense: one factorization at this size; the default shift-invert eigsh needs
    # minutes here and stops short on this near-marginal case.
    gamma2, v, _ = eigenpair(eqd, dm, config, SolverConfig(eigensolver="dense"))
    assert np.isfinite(gamma2) and np.isfinite(v).all()

    # compute operator with and without vacuum term
    op = matfree_operator(eqd, dm, config)
    A = assemble_dense(eqd, dm, config)["A"]
    eq0 = eqd.replace(phi_matrix=0 * eqd.phi_matrix)
    op_0 = matfree_operator(eq0, dm, config)
    A_0 = assemble_dense(eq0, dm, config)["A"]

    # compute geometry and derivatives
    shape = (n_rho, n_theta, n_zeta)
    _, d_dv, d_dz = _grid_derivs(dm.coupled_rt, *shape)
    B_zeta = geo["psi_r"] / geo["sqrt(g)_PEST"]  # and B^vartheta = iota B^zeta
    grad_rho = np.sqrt(geo["g^rr"])
    a = float(geo["a"])
    B_N = abs(float(eq.Psi)) / (np.pi * a**2)

    def vacuum_energies(u):
        """(W_ref from DESC's fields, W_Ax matrix-free, W_dense) of the vector u."""
        uu = np.vdot(u, u).real
        xi_rho = physical_vector(op, u)[:, 0].reshape(shape)
        d_th = d_dv(dm.D_theta, xi_rho)[-1].ravel()
        d_ze = d_dz(dm.D_zeta, xi_rho)[-1].ravel()
        # the mode is family x alone: its family split is dB_n [T] in row x
        b = np.zeros((len(Pfam), d_th.size), dtype=complex)
        b[family] = B_zeta * (geo["iota"] * d_th + d_ze) / grad_rho

        # reference: the coil check's surface energy, per period, in AGNI's units
        W_surf = surface_energy(geo["dA"], b, Pfam)[0]
        W_ref = 2 * MU_0 * W_surf / nfp / (B_N**2 * a**3) / uu

        # operator and dense
        W_Ax = np.vdot(u, op["Ax"](u) - op_0["Ax"](u)).real
        W_dense = np.vdot(u, (A - A_0) @ u).real
        return W_ref, W_Ax / uu, W_dense / uu

    # The eigenmode, and random vectors: the solvers use only the Hermitian part of
    # the operator, and one vector would test that part in one direction only.
    rng = np.random.default_rng(0)
    vectors = [("eigenmode", v)]
    for k in range(4):
        u = rng.standard_normal(v.size)
        if np.iscomplexobj(v):
            u = u + 1j * rng.standard_normal(v.size)
        vectors.append((f"random {k}", u))
    for name, u in vectors:
        W_ref, W_Ax, W_dense = vacuum_energies(u)
        assert W_ref > 0.0, f"{name}: the vacuum term is not active"
        # measured 1.9e-12 (family 0) and 1.8e-12 (family 1) on the eigenmode: the
        # operator and DESC's fields agree to the coordinate map's tolerance
        assert W_Ax == pytest.approx(W_ref, rel=1e-9), name
        assert W_dense == pytest.approx(W_ref, rel=1e-9), name
