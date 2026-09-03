#!/usr/bin/env python3
"""DEV-ONLY. Export a DESC equilibrium to the agnimhd EquilibriumData format.

**This script is not part of the package.** It imports DESC, it is never
imported by ``agnimhd`` or by any test, and it exists only to produce the
version-controlled test fixture in ``tests/data/``. Run it once, in an
environment that still has DESC, commit the output, then forget it.

That separation is the whole point of the design: after this runs, every test in
the suite reads a serialized ``EquilibriumData`` file and the entire package is
testable with no equilibrium code installed anywhere.

It also serves as the reference implementation of a DESC adapter -- the mapping
from DESC compute keys to contract field names below is exactly what a DESC-side
wrapper has to do. See ``docs/adapters.md`` and ``examples/desc_adapter.py``.

Usage
-----
    python tools/export_fixture.py --eq /path/to/AGNI_QH_lowres.h5 \\
        --res 24,12,8 --out tests/data/qh_lowres_24x12x8.npz
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# The package itself, for the writer. agnimhd does not import DESC; this script
# imports both, which is legal precisely because it is not part of the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: DESC compute key -> EquilibriumData field name. This table IS the adapter.
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

#: The two vector fields the drive is built from, exported so the test suite can
#: check that both routes to `drive` agree.
VECTOR_KEY_MAP = {
    "J x grad(rho)": "J_cross_grad_rho",
    "(B*grad) grad(rho)": "B_dot_grad_grad_rho",
}


def load_equilibrium(path):
    """Load a DESC equilibrium saved before newer DESC attributes existed."""
    from desc.equilibrium import Equilibrium

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*The object attribute .* was not loaded from the file\.",
            category=RuntimeWarning,
        )
        return Equilibrium.load(path)


def build_pest_level(eq, n_rho, n_theta, n_zeta):
    """Build the PEST grid and DiffMat used by the reference dense solve.

    Legendre-Lobatto radial nodes pushed through the staircase automorphism,
    Fourier in theta and zeta. This reproduces the discretization in DESC's
    ``tests/test_AGNI.py``, which is where the reference eigenvalue comes from.
    """
    from desc.backend import jax, jnp
    from desc.diffmat_utils import DiffMat, fourier_diffmat, legendre_diffmat
    from desc.grid import LinearGrid
    from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob

    auto_kw = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
    x_lob, _ = leggauss_lob(n_rho)
    rho = automorphism_staircase1(x_lob, **auto_kw)
    dfa = jax.vmap(
        lambda x: jax.grad(automorphism_staircase1, argnums=0)(x, **auto_kw)
    )(x_lob)

    d_rho_raw, w_rho_raw = legendre_diffmat(n_rho)
    d_rho = d_rho_raw / dfa[:, None]
    w_rho = w_rho_raw * dfa[:, None]

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    d_theta, w_theta = fourier_diffmat(n_theta)

    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / eq.NFP, n_zeta, endpoint=False)
    d_zeta, w_zeta = fourier_diffmat(n_zeta)
    d_zeta = d_zeta * eq.NFP
    w_zeta = w_zeta / eq.NFP

    diffmat = DiffMat(
        D_rho=d_rho,
        W_rho=jnp.diagonal(w_rho),
        D_theta=d_theta,
        W_theta=jnp.diagonal(w_theta),
        D_zeta=d_zeta,
        W_zeta=jnp.diagonal(w_zeta),
    )
    pest_grid = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)
    return pest_grid, diffmat, np.asarray(rho)


def _scalar(value):
    """Return ``value`` as a Python float, whatever shape DESC hands back.

    ``eq.params_dict["Psi"]`` used to be a 0-d array and is now a length-1
    array, and ``data["a"]`` is grid-resolved. A bare ``float(np.asarray(...))``
    raised "only 0-dimensional arrays can be converted to Python scalars" on
    current DESC, which meant this script could not regenerate its own fixture.
    """
    flat = np.asarray(value).reshape(-1)
    assert flat.size >= 1, "empty scalar"
    return float(flat[0])


def main():
    """Export the fixture and print the measured reference eigenvalue."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eq", required=True, help="path to the DESC .h5 equilibrium")
    ap.add_argument("--res", default="24,12,8", help="n_rho,n_theta,n_zeta")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument(
        "--meta", default=None, help="optional .json path for the reference values"
    )
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    n_rho, n_theta, n_zeta = (int(v) for v in args.res.split(","))

    from desc.backend import jnp
    from desc.grid import Grid

    from agnimhd.equilibrium import EquilibriumData

    print(f"[export] loading {args.eq}", flush=True)
    eq = load_equilibrium(args.eq)
    print(f"[export] NFP={eq.NFP} L={eq.L} M={eq.M} N={eq.N}", flush=True)

    pest_grid, diffmat, rho1d = build_pest_level(eq, n_rho, n_theta, n_zeta)

    # PEST -> DESC coordinate map. rho-major (r, t, z) node ordering, which is
    # the ordering the EquilibriumData contract requires.
    pest_nodes = jnp.reshape(
        pest_grid.meshgrid_reshape(pest_grid.nodes, order="rtz"), (-1, 3)
    )
    print("[export] map_coordinates PEST -> DESC ...", flush=True)
    rtz_nodes = eq.map_coordinates(
        pest_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    grid = Grid(rtz_nodes)

    n_total = n_rho * n_theta * n_zeta
    n_shell = n_theta * n_zeta
    n_keep = 3 * n_total - 2 * n_shell

    keys = ["finite-n lambda3", "a"]
    keys += list(KEY_MAP) + list(VECTOR_KEY_MAP)
    print(f"[export] eq.compute on {n_total} nodes (n_keep={n_keep}) ...", flush=True)
    data = eq.compute(
        keys,
        grid=grid,
        diffmat=diffmat,
        incompressible=False,
        gamma=args.gamma,
        v_guess=np.ones(n_keep),
    )

    lam_dense = float(np.asarray(data["finite-n lambda3"]).reshape(-1)[0])
    print(f"[export] MEASURED dense finite-n lambda3 = {lam_dense:.9e}", flush=True)

    fields = {dst: np.asarray(data[src]).reshape(-1) for src, dst in KEY_MAP.items()}
    for src, dst in VECTOR_KEY_MAP.items():
        fields[dst] = np.asarray(data[src]).reshape(n_total, 3)

    eqd = EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        NFP=int(eq.NFP),
        Psi=_scalar(eq.params_dict["Psi"]),
        a=_scalar(data["a"]),
        **fields,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = eqd.save(out)
    print(f"[export] wrote {written} ({Path(written).stat().st_size/1e3:.1f} kB)")

    meta = {
        "source_equilibrium": str(Path(args.eq).name),
        "resolution": [n_rho, n_theta, n_zeta],
        "NFP": int(eq.NFP),
        "gamma": args.gamma,
        "radial_basis": "legendre_lobatto + automorphism_staircase1"
        "(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)",
        "poloidal_basis": "fourier",
        "toroidal_basis": "fourier (scaled by NFP)",
        "rho_nodes": rho1d.tolist(),
        "dense_lambda3": lam_dense,
        "Psi": _scalar(eq.params_dict["Psi"]),
        "a": _scalar(data["a"]),
        "a_definition": "QuadratureGrid cross-section area integral, a=sqrt(A/pi)",
        "desc_version": _desc_version(),
    }
    meta_path = Path(args.meta) if args.meta else out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[export] wrote {meta_path}")
    return 0


def _desc_version():
    """Best-effort DESC version string, for provenance in the fixture metadata."""
    try:
        import desc

        return getattr(desc, "__version__", "unknown")
    except Exception:  # pragma: no cover - dev script
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
