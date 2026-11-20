"""The vacuum energy against an independent answer: coil fields volume-integrated once.

Reference data (``tests/data/vacuum_coils/``, made once on a cluster by
``scripts2/vacuum/make_coil_reference.py``), each for coil pairs on the flux surfaces
of ``AGNI_QH_lowres.h5`` (NFP = 4) with opposite currents (net current zero, so the
exterior field is source-free with a single-valued potential: exactly the exterior
Neumann problem AGNI's vacuum matrix solves), and for each pair

    W_vol = (1/2mu_0) int_exterior B^2 dV

by Biot-Savart over all space outside the plasma, at several radial resolutions.
Nothing of AGNI went into it.

* ``AGNI_QH_lowres_*``: on the (rho, theta) = const curves at theta = pi/2, 3pi/2,
  with the field-period symmetry: B_n is pure toroidal mode family 0 (n = k NFP).
* ``AGNI_QH_lowres_n1_*``: helical, theta = theta0 + zeta (one poloidal turn per
  toroidal turn), WITHOUT the field-period symmetry: B_n is mostly n = +-1, families
  1 and 3. Family 0 alone could never show a wrong response for x != 0.

A toroidal mode family x has boundary data exp(i x zeta) times an NFP-periodic
function, and its own vacuum response P_x (``phi_matrix(..., family=x)``; complex
for 0 < x < NFP/2, P_{NFP-x} = conj(P_x)). A field B_n over the whole torus splits into
families by a DFT over the field periods, b_x = (1/NFP) sum_p e^{-2 pi i x p/NFP} B_p,
and its energy is the sum over families,

    W_V = (1/2mu_0) sum_x Re[(dA b_x)^H P_x b_x]     (dA weighted for the whole torus).

1. ``test_surface_energy_matches_volume_integral``: on the 41 x 25 lattice, finer
   than any eigensolve here, so the physics of the response and not the
   resolution: family 0 with the symmetric pairs (measured, worst pair: 17 x 11
   -3.7 %, 33 x 21 +2.0 %, 41 x 25 -1.3 %, 49 x 31 -0.9 %), and every family with the
   n = 1 pairs.
2. ``test_family_response_is_the_folded_full_torus_response``: P_x (one field period,
   the Bloch phase in the boundary integral) against the response computed over the
   whole torus and folded with the family phase, the construction AGNI's D_zeta uses.
3. ``test_eigensolve_vacuum_energy_matches_validated_matrix`` (families 0 and 1): at
   the eigensolve's own resolution, the very matrices the solve uses checked against
   the coils (family 1 with the n = 1 pairs, every family), then the vacuum energy the
   matrix-free and dense operators carry against the formula above with the
   eigenmode's normal field built from DESC's own fields.

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
# Odd shells (square vacuum solve). Stage A needs only the boundary lattice,
# (n_theta, n_zeta), a fine one; Stage B, a low-resolution eigensolve,
# (n_rho, n_theta, n_zeta, mpol, ntor).
STAGE_A = (41, 25)
STAGE_B = (8, 17, 11, 6, 4)


def _reference(kind="symmetric"):
    """The volume-integrated energies of the coil sets, and their B_n on the tests'
    lattices: ``kind`` "symmetric" (family 0) or "n1" (every family).

    Missing reference data FAILS, as a missing fixture does everywhere in the suite
    (conftest.py): a skip here would leave the vacuum energy unchecked without anyone
    noticing."""
    tag = NAME if kind == "symmetric" else f"{NAME}_n1"
    energy = COILS / f"{tag}_coil_energy.npz"
    if not energy.exists():
        pytest.fail(
            f"vacuum reference data missing from {COILS}: {energy.name}. Make it "
            "with scripts2/vacuum/make_coil_reference.py (sbatch "
            "batch/vacuum/job_coil_reference.sl"
            + (" --coils n1 --rhos ... --theta0s ..." if kind == "n1" else "")
            + ") and copy its files here",
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
    """Each coil set's B_n on the ``n_theta x n_zeta`` lattice, ``(n_sets, periods,
    n_surf)``, as make_coil_reference.py computed it: with the volume integral's own
    coil discretization, so both sides of the comparison see one field."""
    key = f"Bn_{n_theta}x{n_zeta}"
    if key not in ref:
        pytest.fail(
            f"the reference has no B_n on the {n_theta} x {n_zeta} lattice; remake it "
            f"with make_coil_reference.py --bn-lattices ... {n_theta}x{n_zeta}",
            pytrace=False,
        )
    return ref[key]


def _eq():
    """The DESC equilibrium of the reference data."""
    return load(str(DESC_FILE)).eq


def pest_to_desc(eq, theta, zeta):
    """DESC's ``(rho, theta, zeta)`` of the rho = 1 PEST nodes ``theta x zeta``, theta
    outer and zeta fastest (AGNI's order)."""
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
    """DESC's ``g^rr``, ``sqrt(g)_PEST`` and ``keys`` on the rho = 1 PEST lattice of an
    AGNI basis, ``n_theta x n_zeta`` on one field period (theta outer, zeta fastest),
    and ``dA``, the area element in PEST angles times the lattice's full-torus weights,
    4 pi^2 / n_surf."""
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
    """``{x: P_x}`` on the rho = 1 shell of an AGNI basis, built exactly as
    ``from_desc`` builds a free-boundary basis' ``phi_matrix`` for family ``x`` (the
    same scaffold defaults; the refined source grid maps its own nodes), so a test can
    check the matrix the solve uses, not only its construction."""
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


def _tolerance(ref, coil_idx, floor):
    """Reference volume integral's spread across its resolutions."""
    W = ref["W_vol"]
    spread = abs(W[coil_idx, -1] / W[coil_idx, 0] - 1) if W.shape[1] > 1 else 0.0
    return max(floor, 3 * spread), spread


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["symmetric", "n1"])
def test_surface_energy_matches_volume_integral(kind):
    """Stage A: the surface energy of each coil pair equals its exterior
    (1/2mu_0) int B^2 dV at a higher resolution than the eigensolves.

    * "symmetric": family 0, (1/2mu_0) (dA B_n) . P_0 B_n.
    * "n1": every family,  B_n split over the four families."""
    pytest.importorskip("desc")
    ref = _reference(kind)
    eq = _eq()
    n_theta, n_zeta = STAGE_A
    Bn = reference_normal_field(ref, n_theta, n_zeta)  # (n_sets, periods, n_surf)
    nfp = Bn.shape[1]  # the families x to sum over: NFP for n1, 1 for symmetric
    geo = shell_geometry(eq, n_theta, n_zeta)
    built = family_matrices(eq, n_theta, n_zeta, range(nfp // 2 + 1))
    P = {x: built[x] if x in built else np.conj(built[nfp - x]) for x in range(nfp)}
    W_vol = ref["W_vol"]
    assert Bn.shape[0] == W_vol.shape[0] == ref["rho"].size
    for coil_idx, rho in enumerate(ref["rho"]):
        W_surf, share = surface_energy(geo["dA"], family_split(Bn[coil_idx]), P)
        if kind == "n1":  # else this would test P_0 again
            assert share[0] < 0.1, f"rho = {rho:.2f}: family 0 holds {share[0]:.0%}"
        assert W_surf == pytest.approx(W_vol[coil_idx, -1], rel=5e-3), (
            f"{kind}, rho = {rho:.2f}: surface {W_surf:.6e} J (shares "
            f"{np.round(share, 3)}), volume {W_vol[coil_idx, -1]:.6e} J, "
            f"{W_surf / W_vol[coil_idx, -1] - 1:+.2%}"
        )


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
def test_eigensolve_vacuum_energy_matches_validated_matrix(family):
    """Stage B: the vacuum energy an eigensolve carries, for families 0 and 1.

    At the eigensolve's resolution, the matrix the solve uses is checked against the
    coils on its shell: family 0, ``eqd.phi_matrix`` itself, with the symmetric pairs
    (5 %: 17 x 11 was measured -3.7 % for the worst pair); family 1 with the helical
    pairs, whose B_n is split over all four families: x = 1 and 3 are the solve's
    ``eqd.phi_matrix`` and its conjugate, x = 0 and 2 built the same way (2 %, or three
    times the reference's spread: measured -2.6 % at rho = 0.2, whose spread is
    0.95 %, and +1.4 % at rho = 0.4). Then, in physical units on
    rho = 1, from DESC's own fields and NOT from the operator's,

        dB_n = (psi_r / sqrt(g)_PEST) (iota d_vartheta + d_zeta) xi^rho / |grad rho|,
        W = 2 mu_0 W_surf / NFP / (B_N^2 a^3) / (v^H v),   B_N = |Psi| / (pi a^2),

    (straight field lines: B^zeta = psi_r / sqrt(g)_PEST, B^vartheta = iota B^zeta)
    with W_surf the coil check's surface energy, ``surface_energy`` with the same dA
    and P_x, of dB_n, all of it in family x; / NFP for one field period, as the
    operator, and 2 mu_0 / (B_N^2 a^3) to AGNI's units (v^H A v is int |dB|^2 dV),
    with xi^rho from the eigenvector (``physical_vector``) and its angular derivatives
    by AGNI's derivative matrices, tested on their own in test_diffmat and
    test_zernike (on the Zernike basis D_theta is the coupled disk derivative, not a
    Fourier derivative of the shell values). A wrong normal field, area element,
    normalization or matrix scaling in the operator's vacuum term, or the wrong
    family's matrix, shows up here. It must equal the vacuum part of both operators,
    each rebuilt with phi_matrix = 0 for the difference: matrix-free,
    v^H [Ax(v) - Ax_0(v)] / v^H v (the eigenvalue and the Hellmann-Feynman gradient
    use it), and dense, v^H (A - A_0) v / v^H v (``assemble_dense``), for the
    eigenmode and for four random vectors: every solver uses only the operator's
    Hermitian part, which one vector would test in one direction only."""
    from agnimhd.assemble import (
        _grid_derivs,
        assemble_dense,
        matfree_operator,
        physical_vector,
    )

    pytest.importorskip("desc")
    eq = _eq()
    nfp = eq.NFP
    n_rho, n_theta, n_zeta, mpol, ntor = STAGE_B
    basis = Basis(n_rho, n_theta, n_zeta, mpol=mpol, ntor=ntor, free_boundary=True)
    eqd, dm = from_desc(eq, basis, family=family)
    P = eqd.phi_matrix
    assert eqd.phi_family == family
    assert (family == 0) == np.isrealobj(P)

    # 1. the eigensolve's own matrix against the coils, on this shell. Family 1's
    # coils reach every family: the solve's P (and its conjugate, family NFP - 1),
    # the rest built as from_desc builds them.
    Pfam = {family: P, (nfp - family) % nfp: np.conj(P) if family else P}
    rest = [x for x in range(nfp // 2 + 1) if family and x not in Pfam]
    built = family_matrices(eq, n_theta, n_zeta, rest) if rest else {}
    for x in range(nfp) if family else []:
        if x not in Pfam:
            Pfam[x] = built[x] if x in built else np.conj(built[nfp - x])
    kind = "n1" if family else "symmetric"
    ref = _reference(kind)
    Bn = reference_normal_field(ref, n_theta, n_zeta)  # (n_sets, periods, n_surf)
    geo = shell_geometry(eq, n_theta, n_zeta, keys=("psi_r", "iota", "a"))
    W_vol = ref["W_vol"]
    for coil_idx, rho in enumerate(ref["rho"]):
        W_surf, share = surface_energy(geo["dA"], family_split(Bn[coil_idx]), Pfam)
        tol, spread = _tolerance(ref, coil_idx, 2e-2 if family else 5e-2)
        if family:  # else this would test P_0 again
            assert share[0] < 0.1, f"rho = {rho:.2f}: family 0 holds {share[0]:.0%}"
        assert W_surf == pytest.approx(W_vol[coil_idx, -1], rel=tol), (
            f"{kind} coils, rho = {rho:.2f}: surface {W_surf:.6e} J, "
            f"volume {W_vol[coil_idx, -1]:.6e} J (spread {spread:.1e})"
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
