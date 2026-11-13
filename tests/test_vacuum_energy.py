"""The vacuum energy against an independent answer: coil fields volume-integrated once.

Reference data (``tests/data/vacuum_coils/``, made once on a cluster by
``scripts2/vacuum/make_coil_reference.py``): pairs of opposite-current coils on the
flux surfaces rho = 0.1 ... 0.7 of ``AGNI_QH_lowres.h5``, on the exact
(rho, theta) = const curves at theta = pi/2 and 3pi/2, and for each pair

    W_vol = (1/2mu_0) int_exterior B^2 dV,

by Biot-Savart on a quadrature grid over all space outside the plasma, at several
radial resolutions. Nothing of AGNI went into it. Outside the plasma the coils'
field is source-free and, with opposite currents, has a single-valued potential,
so it is exactly the exterior Neumann problem AGNI's vacuum matrix solves.

1. ``test_surface_energy_matches_volume_integral``: AGNI's ``phi_matrix``, as
   ``from_desc`` builds it for a free-boundary basis, gives the same energy as the
   surface integral
       W_V = (1/2mu_0) sum_ij (dA B_n)_i P_ij (B_n)_j,
   with B_n the coils' normal field on the basis' rho = 1 nodes. The lattice is
   41 x 25: the coils' B_n has poloidal harmonics that decay only like
   exp(-m gap / a), and the surface energy converges with the lattice (measured,
   worst pair: 17 x 11 -3.7 %, 33 x 21 +2.0 %, 41 x 25 -1.3 %, 49 x 31 -0.9 %).
2. ``test_eigensolve_vacuum_energy_matches_validated_matrix``: the matrix built
   the same way (the construction validated in 1) at a low-resolution eigensolve's
   basis, in the formula above on the eigenmode's normal field computed from the
   eigenvector and the geometry, equals the vacuum energy the operator carries
   (``agnimhd.assemble.vacuum_energy``): a wrong factor, measure or sign in how the
   operator forms its vacuum term shows up here. That is an identity of the
   discretization, so it holds at any resolution.

Needs DESC. Set ``AGNI_VACUUM_COILS`` to read the reference data from another
directory.
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
# (n_rho, n_theta, n_zeta, mpol, ntor), odd shells (square vacuum solve).
# Stage A needs only the boundary matrix: few radial nodes, a fine lattice.
STAGE_A = (4, 41, 25, 18, 11)
# Stage B: a low-resolution eigensolve.
STAGE_B = (8, 17, 11, 6, 4)


def _reference():
    """The coil sets and their volume-integrated energies, or a skip."""
    coils, energy = COILS / f"{NAME}_coils.h5", COILS / f"{NAME}_coil_energy.npz"
    if not (coils.exists() and energy.exists()):
        pytest.skip(
            f"no reference data in {COILS}: run scripts2/vacuum/make_coil_reference.py "
            "(sbatch batch/vacuum/job_coil_reference.sl) and copy its .h5 and .npz here"
        )
    ref = dict(np.load(energy))
    digest = hashlib.sha256(DESC_FILE.read_bytes()).hexdigest()
    assert (
        str(ref["eq_sha256"]) == digest
    ), "the reference data was made from a different AGNI_QH_lowres.h5"
    return pytest.importorskip("desc.io").load(str(coils)), ref


def build_vacuum_case(n_rho, n_theta, n_zeta, mpol, ntor, **phi):
    """The equilibrium on a free-boundary basis, its phi_matrix, and the rho = 1
    nodes in DESC's coordinates with their position, normal and area.
    ``phi``: ``from_desc``'s ``phi_*`` options."""
    dgrid = pytest.importorskip("desc.grid")
    Grid, LinearGrid = dgrid.Grid, dgrid.LinearGrid

    eq = load(str(DESC_FILE)).eq
    basis = Basis(n_rho, n_theta, n_zeta, mpol=mpol, ntor=ntor, free_boundary=True)
    eqd, dm = from_desc(eq, basis, **phi)
    nfp = int(eq.NFP)

    # The rho = 1 lattice, in AGNI order (theta outer, zeta fastest), mapped to DESC's
    # theta; dA is the area element in PEST angles times the lattice weights.
    g0 = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, NFP=nfp, sym=False)
    pest = np.reshape(np.asarray(g0.meshgrid_reshape(g0.nodes, "rtz")), (-1, 3))
    nodes, _ = basis.nodes_and_diffmat(nfp)
    np.testing.assert_allclose(
        pest[::n_zeta, 1], np.asarray(nodes["theta"]), atol=1e-14
    )
    np.testing.assert_allclose(pest[:n_zeta, 2], np.asarray(nodes["zeta"]), atol=1e-14)
    rtz = np.asarray(
        eq.map_coordinates(
            pest,
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(np.inf, 2 * np.pi, np.inf),
            tol=1e-12,
            maxiter=50,
        )
    )
    d = eq.compute(
        ["x", "n_rho", "g^rr", "sqrt(g)_PEST"], grid=Grid(rtz, NFP=nfp, sort=False)
    )
    weights = np.asarray(g0.meshgrid_reshape(g0.weights, "rtz")).reshape(-1)
    dA = np.asarray(d["sqrt(g)_PEST"]) * np.sqrt(np.asarray(d["g^rr"])) * weights
    return dict(
        eqd=eqd,
        dm=dm,
        P=np.asarray(eqd.phi_matrix),  # metres: -Phi = P @ B_n
        x=np.asarray(d["x"]),
        n=np.asarray(d["n_rho"]),
        dA=dA,
        shape=(n_rho, n_theta, n_zeta),
    )


@pytest.mark.slow
def test_surface_energy_matches_volume_integral():
    """Stage A: (1/2mu_0) (dA B_n) . P B_n of each coil pair equals its exterior
    (1/2mu_0) int B^2 dV, to 2 % or three times the reference's own spread across
    its radial resolutions, whichever is larger."""
    pytest.importorskip("desc")
    LinearGrid = pytest.importorskip("desc.grid").LinearGrid

    coilsets, ref = _reference()
    c = build_vacuum_case(*STAGE_A)
    W_vol = np.asarray(ref["W_vol"])
    assert len(coilsets) == W_vol.shape[0] == ref["rho"].size
    src = LinearGrid(N=int(ref["coil_src"]))
    for k, (coils, rho) in enumerate(zip(coilsets, ref["rho"])):
        B = np.asarray(coils.compute_magnetic_field(c["x"], source_grid=src))
        Bn = np.sum(B * c["n"], axis=-1)
        W_surf = float((c["dA"] * Bn) @ c["P"] @ Bn) / (2.0 * MU_0)
        spread = abs(W_vol[k, -1] / W_vol[k, 0] - 1) if W_vol.shape[1] > 1 else 0.0
        tol = max(2e-2, 3 * spread)
        assert W_surf > 0, f"rho = {rho:.2f}: W_surf = {W_surf:.3e} J"
        assert W_surf == pytest.approx(W_vol[k, -1], rel=tol), (
            f"rho = {rho:.2f}: surface {W_surf:.6e} J, volume {W_vol[k, -1]:.6e} J "
            f"(reference spread {spread:.1e})"
        )


@pytest.mark.slow
def test_eigensolve_vacuum_energy_matches_validated_matrix():
    """Stage B: the vacuum energy an eigensolve carries, against the stage A formula
    with the stage-A-validated matrix on the eigenmode's own normal field.

    In AGNI's normalization (lengths by a, fields by B_N), on rho = 1:
        B_n = psi_r (iota d_theta + d_zeta) xi^rho / (sqrt(g) |grad rho|),
        dA  = w_theta w_zeta sqrt(g) |grad rho|,
        W   = (dA B_n) . (P / a) B_n / (v^H v),
    computed from the eigenvector and the geometry, not through the operator's
    vacuum code, so an error there does not cancel. It is compared with two things:
      * ``vacuum_energy``, the operator's vacuum-energy diagnostic;
      * the vacuum part of the matrix-free operator itself,
        v^H [Ax(v) - Ax_0(v)] / v^H v with Ax_0 the operator rebuilt with
        phi_matrix = 0: the term the eigenvalue (gamma^2 is the Rayleigh quotient of
        Ax for every eigensolver) and the Hellmann-Feynman gradient actually use.
        ``vacuum_energy`` is separate code, so it does not check this."""
    from agnimhd.assemble import (
        _grid_derivs,
        _normalized_fields,
        matfree_operator,
        physical_vector,
        vacuum_energy,
    )

    pytest.importorskip("desc")
    c = build_vacuum_case(*STAGE_B)
    eqd, dm, P, shape = c["eqd"], c["dm"], c["P"], c["shape"]
    config = AssemblyConfig()
    # dense: one factorization at this size; the default shift-invert eigsh needs
    # minutes here and stops short on this near-marginal case. (eigenpair's residual
    # is relative to |gamma^2| = 2.5e-4 and the penalized operator's norm is large,
    # so it is not a useful check of a dense solve.)
    gamma2, v, _ = eigenpair(eqd, dm, config, SolverConfig(eigensolver="dense"))
    assert np.isfinite(float(gamma2)) and np.isfinite(np.asarray(v)).all()
    op = matfree_operator(eqd, dm, config)
    W_eig = float(vacuum_energy(op, v))

    f = _normalized_fields(eqd, config)
    col = {
        k: np.asarray(f[k]).reshape(shape)
        for k in ("psi_r", "sqrtg", "g_sup_rr", "iota")
    }
    _, d_dv, d_dz = _grid_derivs(dm.coupled_rt, *shape)
    xi_rho = np.asarray(physical_vector(op, v))[:, 0].reshape(shape)
    b = (
        col["iota"] * np.asarray(d_dv(dm.D_theta, xi_rho))
        + np.asarray(d_dz(dm.D_zeta, xi_rho))
    )[-1].ravel()
    sqrtg = col["sqrtg"][-1].ravel()
    grad_rho = np.sqrt(col["g_sup_rr"][-1]).ravel()
    Bn = col["psi_r"][-1].ravel() * b / (sqrtg * grad_rho)
    dA = np.kron(np.asarray(dm.w_theta), np.asarray(dm.w_zeta)) * sqrtg * grad_rho
    vv = float(np.real(np.vdot(np.asarray(v), np.asarray(v))))
    W_val = float(np.real(np.vdot(dA * Bn, (P / float(eqd.a)) @ Bn))) / vv

    op_0 = matfree_operator(eqd.replace(phi_matrix=0 * eqd.phi_matrix), dm, config)
    W_Ax = float(
        np.real(np.vdot(np.asarray(v), np.asarray(op["Ax"](v) - op_0["Ax"](v))))
    )
    W_Ax /= vv

    assert W_eig > 0.0  # the mode moves the boundary: the vacuum term is active
    assert W_val == pytest.approx(W_eig, rel=1e-8)
    assert W_val == pytest.approx(W_Ax, rel=1e-8)
