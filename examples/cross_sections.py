#!/usr/bin/env python3
"""DESC-style eigenfunction cross sections for two equilibria.

Draws normalized ``deltaV`` and the three displacement components in the
``(R, Z)`` plane, matching the plotting convention used by DESC's AGNI
implementation. The stellarator is drawn at ``zeta = 0`` and
``zeta = pi / NFP``, the two planes where the shaping differs most. The tokamak
is drawn at ``zeta = 0``, since an axisymmetric equilibrium looks the same at
every angle.

Each case is solved on the basis its export recorded. The stellarator uses the
Legendre-Lobatto grid with a radial clustering map. The tokamak uses the
coupled Zernike-Fourier basis, whose radial nodes are Gauss-Jacobi and take no
clustering map, so its assembly sets ``coupled_rt``.

Both cases use the dense Lanczos-LU eigensolver: the shifted matrix is
factorized once and reused for all Lanczos iterations. No matrix-free solve and
no preconditioner is involved. Each eigenvalue is printed against the value
recorded in the case's sidecar.

Run::

    python examples/cross_sections.py         # both
    python examples/cross_sections.py DSHAPE  # one of them

Writes one PNG per case into ``examples/figures``. Requires matplotlib::

    pip install "agnimhd[plot]"

The input files were produced with ``tools/export_desc_example.py``; the
command for each is in ``examples/data/<case>.json``.
"""

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from agnimhd import AssemblyConfig, EquilibriumData, SolverConfig, eigenpair
from agnimhd.assemble import matfree_operator
from agnimhd.basis import DiffMat, standard_grid, zernike_fourier_diffmat
from agnimhd.plotting import plot_eigenfunction_cross_sections
from agnimhd.quadrature import zernike_nodes_weights

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"

#: One entry per case: the file stem, the toroidal mode number for the
#: axisymmetric ones, and the eigensolver settings that produced the eigenvalue
#: in the sidecar. The shift sits just below the mode being sought, and the
#: Krylov budget is what separates it from its neighbours; a shorter budget
#: converges to a different mode on these cases.
CASES = {
    "LBD-QH": dict(
        stem="qh_modprof_24x12x8",
        n_mode=None,
        solver=SolverConfig(eigensolver="jax_lanczos", sigma=-1e-3, num_matvecs=150),
    ),
    "DSHAPE": dict(
        stem="dshape_imax_zernike_64x48x1",
        n_mode=3,
        solver=SolverConfig(eigensolver="jax_lanczos", sigma=-5e-4, num_matvecs=300),
    ),
}


def basis(eq, meta):
    """Return ``(diffmat, assembly_kwargs)`` for the basis the export recorded.

    The grid parameters come from the sidecar rather than from constants here,
    so they cannot drift away from the nodes the geometry sits on.
    """
    n_rho, n_theta, _ = eq.resolution
    if meta.get("basis") != "zernike":
        _, diffmat = standard_grid(
            *eq.resolution, NFP=eq.NFP, automorphism=meta["automorphism"]
        )
        return diffmat, {}

    rho, w_rho, theta, w_theta = zernike_nodes_weights(n_rho, n_theta)
    D_rho, D_theta = zernike_fourier_diffmat(
        rho, theta, L=meta["zernike_L"], M=meta["zernike_M"], spectral_indexing="ansi"
    )
    diffmat = DiffMat(
        D_rho=D_rho,
        W_rho=jnp.asarray(w_rho),
        D_theta=D_theta,
        W_theta=jnp.asarray(w_theta),
        D_zeta=jnp.zeros((1, 1)),
        W_zeta=jnp.asarray([2.0 * jnp.pi / eq.NFP]),
        zernike_penalty_alpha=meta["zernike_penalty_alpha"],
    )
    return diffmat, dict(coupled_rt=True, n_rho_coupled=n_rho, n_theta_coupled=n_theta)


def recorded(meta, spec):
    """The eigenvalue recorded in the sidecar, for comparison."""
    if spec["n_mode"] is None:
        return meta.get("reference_lambda")
    return meta.get("lambda_by_n", {}).get(str(spec["n_mode"]))


def solve(case):
    """Return ``(eq, op, v, lam, residual, lam_recorded)`` for one case."""
    spec = CASES[case]
    eq = EquilibriumData.load(DATA / f"{spec['stem']}.npz")
    meta = json.loads((DATA / f"{spec['stem']}.json").read_text())
    diffmat, coupled = basis(eq, meta)
    assembly = (
        AssemblyConfig(**coupled)
        if spec["n_mode"] is None
        else AssemblyConfig(axisym=True, n_mode_axisym=spec["n_mode"], **coupled)
    )
    lam, v, residual = eigenpair(eq, diffmat, assembly, spec["solver"])
    op = matfree_operator(eq, diffmat, assembly)
    return eq, op, np.asarray(v), float(lam), float(residual), recorded(meta, spec)


def draw(case):
    """Solve one case and write its cross sections to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq, op, v, lam, residual, lam_recorded = solve(case)
    if lam_recorded is None:
        agreement = ""
    else:
        rel = abs(lam - lam_recorded) / abs(lam_recorded)
        agreement = f"  recorded={lam_recorded:+.6e}  rel={rel:.1e}"
    label = "" if CASES[case]["n_mode"] is None else f"  n={CASES[case]['n_mode']}"
    print(
        f"{case:<10} {eq.resolution}  NFP={eq.NFP}{label}  "
        f"lambda={lam:+.6e}  residual={residual:.2e}  "
        f"{'UNSTABLE' if lam < 0 else 'stable'}{agreement}",
        flush=True,
    )

    geom = np.load(DATA / f"{CASES[case]['stem']}_RZ.npz")
    R, Z = geom["R"], geom["Z"]
    fig, _axes = plot_eigenfunction_cross_sections(eq, op, v, lam, R, Z)

    title = f"lambda = {lam:+.6e}   residual = {residual:.2e}"
    if abs(lam) < 1e-10:
        title += "   NOT RESOLVED: |lambda| is below the noise floor"
    fig.suptitle(title)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"{case.lower()}_cross_section.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{'':<10} wrote {out.relative_to(HERE.parent)}", flush=True)


def main():
    """Draw the requested cases, or both."""
    wanted = sys.argv[1:] or list(CASES)
    for case in wanted:
        if case not in CASES:
            raise SystemExit(f"unknown case {case!r}. Choose from {list(CASES)}.")
        draw(case)


if __name__ == "__main__":
    main()
