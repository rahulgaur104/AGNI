#!/usr/bin/env python3
"""Zernike, Gauss-Radau-Jacobi and Lobatto: dense, JD and eigsh against each other.

For every case (the DSHAPE tokamak, the QH stellarator), grid and radial basis,
three solves of the same problem:

- ``dense``: restricted on the separable bases; on Zernike at ``PENALTY``, with
  ``dense_mg`` where ``A`` does not fit one GPU;
- ``jd``: matrix-free Jacobi-Davidson at ``PENALTY`` on every basis;
- ``eigsh``: SciPy's ARPACK on the host, restricted or at ``PENALTY`` as ``dense``.

``--report`` prints one Markdown table per case and grid: a column per basis, a
row per solver, each cell gamma^2 with its wall time, the device it ran on and its
relative difference from that basis's ``dense``; a cell above ``RTOL_SOLVER`` is
marked ``!``.

A diagnostic, not a test: too large for CI (one 80 GB GPU per solve, a 4-GPU node
for ``dense_mg``, host memory for ``eigsh``). Each solve appends one JSON line to
``OUTPUT``, so runs split over GPUs and jobs are compared by ``--report``::

    python benchmarks/bases_jd_vs_dense.py --case dshape --bases lobatto
    python benchmarks/bases_jd_vs_dense.py --case qh --bases zernike --solvers jd
    python benchmarks/bases_jd_vs_dense.py --report

Zernike's ``dense_mg`` on QH runs one process per GPU: write the equilibrium
once, then solve under ``srun``::

    python benchmarks/bases_jd_vs_dense.py --case qh --bases zernike --grid 0 \\
        --saved qh.npz --solvers
    srun -n 4 --gpus-per-node 4 python benchmarks/bases_jd_vs_dense.py --case qh \\
        --bases zernike --grid 0 --saved qh.npz --solvers dense --distributed
"""

from desc import set_device

set_device("gpu")  # before JAX starts: DESC otherwise selects the CPU

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from agnimhd import (  # noqa: E402
    AssemblyConfig,
    Basis,
    EquilibriumData,
    SolverConfig,
    eigenpair,
    from_desc,
)
from agnimhd.assemble import matfree_operator  # noqa: E402
from agnimhd.basis import AUTOMORPHISM  # noqa: E402
from agnimhd.multigpu import dense_mg  # noqa: E402

# ---- settings ----------------------------------------------------------------
DATA = Path(__file__).resolve().parents[1] / "tests" / "data"
CASES = {
    "dshape": dict(
        file=DATA / "dshape_imax0.98_1608.h5",
        grids=[(96, 96, 1), (128, 128, 1)],  # the paper's, and a finer one
        basis=dict(mpol=8),
        assembly=dict(axisym=True, n_mode_axisym=2),
        families=[0],
        zernike_dense="dense",  # one GPU holds the 96x96 complex A
        zernike_eigsh=True,
    ),
    "qh": dict(
        file=DATA / "AGNI_QH_lowres.h5",
        grids=[(48, 48, 16)],
        basis=dict(mpol=8, ntor=2),
        assembly=dict(),
        families=[0],
        zernike_dense="dense_mg",  # A does not fit one GPU
        zernike_eigsh=False,  # A and its LU on the host: ~600 GB at 48x48x16
    ),
}
BASES = ["zernike", "gauss_radau_jacobi", "lobatto"]
SOLVERS = ["dense", "jd", "eigsh"]
EPS = 1e-3  # inner edge of the separable bases (DSHAPE n = 2: gamma^2 3 % low at 1e-2)
PENALTY = 1e3  # every penalized solve
SIGMAS = [1e-4, 1e-3, 1e-2]  # dense and jd: tried in turn until finite
SIGMA_EIGSH = (
    1e-1  # eigsh returns the eigenvalue nearest the shift: above every gamma^2
)
RTOL_SOLVER = 1e-4  # jd and eigsh against dense, per basis
MG = dict(mg_tile=256, mg_iters=12)
OUTPUT = Path("bases_jd_vs_dense.jsonl")
# ------------------------------------------------------------------------------


def options():
    """The settings above, overridden from the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", choices=sorted(CASES), default="qh")
    p.add_argument("--bases", nargs="+", choices=BASES, default=BASES)
    p.add_argument("--grid", type=int, nargs="*")  # indices into the case's grids
    p.add_argument("--families", type=int, nargs="+")
    p.add_argument("--solvers", nargs="*", choices=SOLVERS, default=SOLVERS)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--saved", type=Path)  # equilibrium .npz: read, else written
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--report", action="store_true")
    return p.parse_args()


def setup(case, radial, solver):
    """``(eigensolver, restrict)`` of one solve; None where it is not run."""
    c = CASES[case]
    if radial == "zernike":
        if solver == "eigsh" and not c["zernike_eigsh"]:
            return None
        return (c["zernike_dense"] if solver == "dense" else solver), None
    return solver, (None if solver == "jd" else True)


def basis_of(case, radial, grid, restrict=None):
    """The basis of one solve; the separable ones on the default map, inner edge
    at ``EPS``."""
    kw = dict(CASES[case]["basis"], radial=radial, penalty=PENALTY)
    if radial != "zernike":
        kw.update(automorphism=dict(AUTOMORPHISM, eps=EPS), restrict=restrict)
    return Basis(*grid, **kw)


_EQ = {}  # equilibria by (case, nodes): the solvers of one basis and grid share it


def equilibrium(o, case, basis):
    """The equilibrium on ``basis``'s nodes: read from ``--saved``, else from DESC
    (and written to ``--saved``)."""
    if o.saved and o.saved.exists():
        return EquilibriumData.load(o.saved)
    key = (case, basis.radial, basis.n_rho, basis.n_theta, basis.n_zeta)
    if key not in _EQ:
        _EQ[key] = from_desc(str(CASES[case]["file"]), basis)[0]
        if o.saved:
            _EQ[key].save(o.saved)
    return _EQ[key]


def solve(o, case, radial, grid, family, kind, restrict):
    """One solve; returns its JSON record."""
    c = CASES[case]
    basis = basis_of(case, radial, grid, restrict)
    eq = equilibrium(o, case, basis)
    diffmat = basis.nodes_and_diffmat(eq.NFP, family)[1]
    assembly = AssemblyConfig(gamma=5.0 / 3.0, **c["assembly"])
    extra = {}
    if kind == "jd":
        eq_c = equilibrium(argparse.Namespace(saved=None), case, basis.coarse())
        extra["coarse"] = basis.coarse_level(eq_c, family)
    t = time.time()
    for sigma in [SIGMA_EIGSH] if kind == "eigsh" else SIGMAS:
        cfg = SolverConfig(eigensolver=kind, sigma=sigma, **MG)
        if o.distributed:  # dense_mg, one process per GPU, not under jit
            v, gamma2 = dense_mg(eq, diffmat, assembly, cfg)
            r = matfree_operator(eq, diffmat, assembly)["Ax"](v) + gamma2 * v
            resid = np.linalg.norm(r) / (abs(gamma2) * np.linalg.norm(v))
        else:
            gamma2, _, resid = eigenpair(eq, diffmat, assembly, cfg, **extra)
        gamma2 = float(jax.block_until_ready(gamma2))
        if np.isfinite(gamma2):
            break
    stats = jax.local_devices()[0].memory_stats() or {}
    dev = jax.devices()[0]
    if kind == "eigsh" or dev.platform != "gpu":  # eigsh: ARPACK on the host
        device = "CPU"
    else:
        n = jax.device_count() if kind == "dense_mg" else 1
        device = f"{n} {dev.device_kind.replace('NVIDIA ', '').split('-')[0]}"
    return dict(
        case=case,
        radial=radial,
        grid=list(grid),
        family=family,
        solver="dense" if kind == "dense_mg" else kind,
        eigensolver=kind,
        restricted=bool(restrict),
        penalty=None if restrict else PENALTY,
        sigma=sigma,
        device=device,
        gamma2=gamma2,
        residual=float(resid),
        seconds=time.time() - t,
        peak_gb=stats.get("peak_bytes_in_use", 0) / 1e9,
    )


def rel(a, b):
    """Relative difference of ``a`` from ``b``; NaN if either is missing."""
    return abs(a - b) / abs(b) if np.isfinite(a) and np.isfinite(b) else np.nan


NAMES = {
    "zernike": "Zernike",
    "gauss_radau_jacobi": "Gauss-Radau-Jacobi",
    "lobatto": "Lobatto",
}


def report(path):
    """One Markdown table per case and grid; returns the number of marked cells."""
    g, how = {}, {}  # (case, grid) -> {(solver, radial): gamma2, time}
    for line in path.read_text().splitlines():  # the latest run wins
        r = json.loads(line)
        if tuple(r["grid"]) not in CASES[r["case"]]["grids"]:
            continue
        key = (r["case"], tuple(r["grid"]))
        g.setdefault(key, {})[r["solver"], r["radial"]] = r["gamma2"]
        t = f"{r['seconds']:.0f} s, {r.get('device', '?')}"
        how.setdefault(key, {})[r["solver"], r["radial"]] = t
    bad = 0
    for case, grid in sorted(g):
        cell = g[case, grid]
        print(f"\n**{case}, {'x'.join(map(str, grid))}**\n")
        print("| | " + " | ".join(NAMES[b] for b in BASES) + " |")
        print("|---" * (len(BASES) + 1) + "|")
        for s in SOLVERS:
            row = []
            for b in BASES:
                x = cell.get((s, b), np.nan)
                if setup(case, b, s) is None:  # not run: see CASES
                    row.append("OOM (host memory)")
                    continue
                if not np.isfinite(x):
                    row.append("-")
                    continue
                txt = f"{x:.6e} ({how[case, grid][s, b]}"
                d = rel(x, cell.get(("dense", b), np.nan))
                if s != "dense" and np.isfinite(d):
                    txt += f", {d:.0e}"
                    if d > RTOL_SOLVER:
                        txt += " !"
                        bad += 1
                row.append(txt + ")")
            print(f"| {s} | " + " | ".join(row) + " |")
    return bad


def main():
    """Run the solves of the command line, or print the tables (``--report``)."""
    o = options()
    if o.report:
        raise SystemExit(1 if report(o.output) else 0)
    if o.distributed:
        local = int(os.environ.get("SLURM_LOCALID", "0"))
        jax.distributed.initialize(local_device_ids=[local])
    c = CASES[o.case]
    grids = [c["grids"][i] for i in o.grid] if o.grid else c["grids"]
    for radial in o.bases:
        for grid in grids:
            if not o.solvers:  # --saved only: write the equilibrium
                equilibrium(o, o.case, basis_of(o.case, radial, grid))
            for family in o.families or c["families"]:
                for solver in o.solvers:
                    how = setup(o.case, radial, solver)
                    if how is None:
                        continue
                    r = solve(o, o.case, radial, grid, family, *how)
                    if jax.process_index() == 0:
                        print(json.dumps(r), flush=True)
                        with open(o.output, "a") as f:
                            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
