#!/usr/bin/env python3
"""DEV-ONLY. Export a DESC example equilibrium to an ``EquilibriumData`` file.

This produced the files in ``examples/data``. It imports DESC, is never
imported by ``agnimhd`` or by any test, and exists so the shipped examples can
be regenerated:

    python tools/export_desc_example.py \\
        --eq /path/to/qh_beta1.5_imin1.02_modprof_221410.h5 \\
        --res 24,12,8 --x0 0.6 --m1 2.5 --m2 3.0 \\
        --out examples/data/qh_modprof_24x12x8.npz

    python tools/export_desc_example.py \\
        --eq /path/to/dshape_imax0.98_1608.h5 \\
        --res 64,48,1 --basis zernike \\
        --out examples/data/dshape_imax_zernike_64x48x1.npz

``--case NAME`` uses a DESC example (``DSHAPE``, ``HELIOTRON``, ...) instead of
a file.

``--res`` is ``n_rho,n_theta,n_zeta``. Use ``n_zeta = 1`` for an axisymmetric
equilibrium: the toroidal direction then carries a single mode number, supplied
at solve time through ``AssemblyConfig(axisym=True, n_mode_axisym=n)``.

``tools/export_fixture.py`` is the same export for a DESC equilibrium held in a
file, and additionally computes a reference eigenvalue through DESC itself.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: DESC compute key -> EquilibriumData field. This table is the adapter.
KEY_MAP = {
    "g_rr|PEST": "g_rr",
    "g_rv|PEST": "g_rv",
    "g_rp|PEST": "g_rp",
    "g_vv|PEST": "g_vv",
    "g_vp|PEST": "g_vp",
    "g_pp|PEST": "g_pp",
    "g^rr": "g_sup_rr",
    "sqrt(g)_PEST": "sqrtg",
    "(sqrt(g)_PEST_r)|PEST": "sqrtg_r",
    "(sqrt(g)_PEST_v)|PEST": "sqrtg_v",
    "(sqrt(g)_PEST_p)|PEST": "sqrtg_p",
    "J^zeta": "J_sup_zeta",
    "|J|": "abs_J",
    "finite-n instability drive": "finite_n_instability_drive",
    "iota": "iota",
    "psi_r": "psi_r",
    "psi_rr": "psi_rr",
    "p": "p",
    "p_r": "p_r",
}

#: Radial node clustering. The solve must use the same values, or the geometry
#: and the differentiation matrices sit on different nodes.
AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)


def main():
    """Export one DESC example and report what was written."""
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="DESC example name, e.g. DSHAPE")
    src.add_argument("--eq", help="path to a DESC .h5 equilibrium")
    ap.add_argument("--res", required=True, help="n_rho,n_theta,n_zeta")
    ap.add_argument("--x0", type=float, default=0.65, help="clustering target radius")
    ap.add_argument("--m1", type=float, default=2.0, help="inner clustering exponent")
    ap.add_argument("--m2", type=float, default=3.0, help="outer clustering exponent")
    ap.add_argument(
        "--basis",
        default="legendre",
        choices=("legendre", "zernike"),
        help=(
            "radial node family. 'legendre' is Legendre-Lobatto through the "
            "clustering map. 'zernike' is the Gauss-Jacobi radial nodes of the "
            "coupled Zernike-Fourier basis, which take no clustering map and "
            "keep the magnetic axis off the grid by construction."
        ),
    )
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    n_rho, n_theta, n_zeta = (int(v) for v in args.res.split(","))

    from desc.equilibrium import Equilibrium
    from desc.examples import get
    from desc.grid import Grid
    from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob

    from agnimhd.equilibrium import EquilibriumData

    if args.case:
        eq, name = get(args.case), args.case
    else:
        eq, name = Equilibrium.load(args.eq), Path(args.eq).stem
    print(f"[export] {name}: NFP={eq.NFP} L={eq.L} M={eq.M} N={eq.N}", flush=True)

    if args.basis == "zernike":
        from agnimhd.quadrature import zernike_nodes_weights

        auto_kw = None
        rho_j, _, theta_j, _ = zernike_nodes_weights(n_rho, n_theta)
        rho, theta = np.asarray(rho_j), np.asarray(theta_j)
    else:
        auto_kw = dict(AUTOMORPHISM, x_0=args.x0, m_1=args.m1, m_2=args.m2)
        x_lob, _ = leggauss_lob(n_rho)
        rho = np.asarray(automorphism_staircase1(x_lob, **auto_kw))
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / eq.NFP, n_zeta, endpoint=False)
    R, T, Z = np.meshgrid(rho, theta, zeta, indexing="ij")  # rho-major
    pest_nodes = np.column_stack([R.ravel(), T.ravel(), Z.ravel()])

    # theta_PEST is not DESC's theta. This root find is not optional, and its
    # tolerance propagates into every geometric coefficient below.
    print(f"[export] map_coordinates on {pest_nodes.shape[0]} nodes", flush=True)
    desc_nodes = eq.map_coordinates(
        pest_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2.0 * np.pi, np.inf),
        tol=1e-12,
        maxiter=60,
    )

    # R and Z come along for plotting only. They are not part of the
    # EquilibriumData specification, which carries the metric and not the
    # real-space positions, so they go in a companion file that the examples
    # load when they draw a cross section.
    data = eq.compute(list(KEY_MAP) + ["a", "R", "Z"], grid=Grid(desc_nodes))
    flat = lambda key: np.asarray(data[key]).reshape(-1)  # noqa: E731

    eqd = EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        NFP=int(eq.NFP),
        Psi=float(np.asarray(eq.params_dict["Psi"]).reshape(-1)[0]),
        a=float(flat("a")[0]),
        **{dst: flat(src) for src, dst in KEY_MAP.items()},
    )
    eqd.save(args.out)
    print(
        f"[export] wrote {args.out}: a={eqd.a:.6f}  Psi={eqd.Psi:.6f}  "
        f"basis={args.basis}",
        flush=True,
    )

    shape = (n_rho, n_theta, n_zeta)
    rz_path = str(Path(args.out).with_suffix("")) + "_RZ.npz"
    np.savez_compressed(
        rz_path,
        R=flat("R").reshape(shape),
        Z=flat("Z").reshape(shape),
        rho=rho,
        theta=theta,
        zeta=zeta,
        NFP=np.asarray(int(eq.NFP)),
    )
    print(f"[export] wrote {rz_path} for plotting", flush=True)

    meta = {
        "source": name,
        "resolution": [n_rho, n_theta, n_zeta],
        "NFP": int(eq.NFP),
        "basis": args.basis,
        "automorphism": (
            None if auto_kw is None else {k: float(v) for k, v in auto_kw.items()}
        ),
        "Psi": float(eqd.Psi),
        "a": float(eqd.a),
    }
    meta_path = str(Path(args.out).with_suffix(".json"))
    Path(meta_path).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[export] wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
