#!/usr/bin/env python3
"""DEV-ONLY. Freeze DESC's Zernike values into a committed reference file.

**This script is not part of the package.** It imports DESC, it is never
imported by ``agnimhd`` or by any test, and it runs once in a throwaway
environment that still has DESC.

Why it exists
-------------
``agnimhd.basis.zernike`` implements the Zernike radial polynomials from the
Jacobi recurrence and builds the coupled Zernike-Fourier differentiation
matrices itself, so the package needs no DESC. But the mode ORDERING (both the
``"ansi"`` and ``"fringe"`` conventions) and the conditioning of the
nodal-to-spectral pseudo-inverse are conventions, not theorems -- an independent
implementation can be internally consistent and still disagree with DESC's, and
a DESC-side consumer that mixes the two would then get silently wrong
derivatives.

So the cross-check is preserved by VALUE rather than by import: this script
records what DESC produces, the values are committed to the repository, and the
permanent test compares ``agnimhd`` against the committed file. DESC never
enters the dependency graph.

Usage
-----
    python tools/export_zernike_reference.py --out tests/data/zernike_reference.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Small (n_rho, n_theta) grids to record. Deliberately several shapes: the
#: pseudo-inverse is rank-deficient when the nodes under-determine the basis,
#: and that regime has to be pinned too, not just the well-posed one.
CASES = (
    # (n_rho, n_theta, L, M)
    (4, 6, -1, -1),
    (5, 8, -1, -1),
    (6, 8, 6, 3),
    (4, 6, 8, 3),  # over-resolved basis: pinv is rank-deficient here
)

INDEXINGS = ("ansi", "fringe")


def main():
    """Write the frozen reference archive."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from desc.basis import ZernikePolynomial
    from desc.diffmat_utils import zernike_fourier_diffmat
    from desc.grid import LinearGrid
    from desc.integrals.quad_utils import zernike_nodes_weights
    from desc.transform import Transform

    out = {}
    out["desc_version"] = np.asarray(_desc_version())

    for indexing in INDEXINGS:
        for n_rho, n_theta, L, M in CASES:
            tag = f"{indexing}_{n_rho}x{n_theta}_L{L}_M{M}"
            print(f"[zernike-ref] {tag}", flush=True)

            rho, w_rho, theta, w_theta = zernike_nodes_weights(n_rho, n_theta)
            out[f"{tag}__rho"] = np.asarray(rho)
            out[f"{tag}__theta"] = np.asarray(theta)
            out[f"{tag}__w_rho"] = np.asarray(w_rho)
            out[f"{tag}__w_theta"] = np.asarray(w_theta)

            Lr = 2 * (n_rho - 1) if L == -1 else L
            Mr = max((n_theta - 1) // 2, 0) if M == -1 else M

            basis = ZernikePolynomial(L=Lr, M=Mr, spectral_indexing=indexing)
            out[f"{tag}__modes"] = np.asarray(basis.modes)

            grid = LinearGrid(rho=rho, theta=theta, NFP=1, sym=False)
            out[f"{tag}__grid_nodes"] = np.asarray(grid.nodes)

            tr = Transform(
                grid, basis, derivs=1, build=True, build_pinv=True, method="direct1"
            )
            # Evaluation matrix and its two first derivatives, as DESC orders
            # them. direct1[dr][dt][dz].
            out[f"{tag}__A"] = np.asarray(tr.matrices["direct1"][0][0][0])
            out[f"{tag}__dA_drho"] = np.asarray(tr.matrices["direct1"][1][0][0])
            out[f"{tag}__dA_dtheta"] = np.asarray(tr.matrices["direct1"][0][1][0])
            out[f"{tag}__pinv"] = np.asarray(tr.matrices["pinv"])

            D_rho, D_theta = zernike_fourier_diffmat(
                rho, theta, L=L, M=M, spectral_indexing=indexing
            )
            out[f"{tag}__D_rho"] = np.asarray(D_rho)
            out[f"{tag}__D_theta"] = np.asarray(D_theta)

            # The penalty projector is already DESC-free (pure numpy SVD), but
            # freezing it makes the port's regression test end-to-end.
            from desc.diffmat_utils import zernike_penalty_projector_from_diffmat

            Q, rank = zernike_penalty_projector_from_diffmat(D_rho, D_theta)
            out[f"{tag}__penalty_Q"] = np.asarray(Q)
            out[f"{tag}__penalty_rank"] = np.asarray(rank)

    # Bare radial polynomial values on a fixed rho set, for the recurrence test.
    # Independent of any grid, so a failure here localizes to R_l^m itself
    # rather than to the fit.
    rho_probe = np.array([1e-3, 0.05, 0.2, 0.5, 0.7071067811865476, 0.9, 1.0])
    out["radial__rho"] = rho_probe
    zb = ZernikePolynomial(L=10, M=5, spectral_indexing="ansi")
    modes = np.asarray(zb.modes)
    out["radial__modes"] = modes
    nodes = np.stack(
        [
            np.repeat(rho_probe, 1),
            np.zeros(rho_probe.size),
            np.zeros(rho_probe.size),
        ],
        axis=-1,
    )
    from desc.basis import zernike_radial

    out["radial__R"] = np.asarray(
        zernike_radial(nodes[:, 0:1], modes[:, 0], modes[:, 1], dr=0)
    )
    out["radial__dRdrho"] = np.asarray(
        zernike_radial(nodes[:, 0:1], modes[:, 0], modes[:, 1], dr=1)
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    size = Path(args.out).stat().st_size / 1e3
    print(f"[zernike-ref] wrote {args.out} ({size:.1f} kB, {len(out)} arrays)")
    return 0


def _desc_version():
    """Best-effort DESC version string, recorded for provenance."""
    try:
        import desc

        return getattr(desc, "__version__", "unknown")
    except Exception:  # pragma: no cover - dev script
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
