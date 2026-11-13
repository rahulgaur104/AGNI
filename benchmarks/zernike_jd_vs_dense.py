#!/usr/bin/env python3
"""Zernike basis: matrix-free JD against the dense solve, tokamak and stellarator.

The coupled Zernike basis at high resolution over the penalty range 1e-2 ... 1e4:
the DSHAPE tokamak (``n = 2``, complex operator) at 96x96 and the QH stellarator
of ``tests/data`` at 48x48x16 (families 0 and 1, either stellarator parity). For
every case ``eigensolver="jd"`` and the dense solve (``"dense"`` on one GPU for
the tokamak, ``"dense_mg"`` on all the node's GPUs for the stellarator; ``--dense``
and ``--grid`` choose others) give ``gamma^2``; the run fails when they differ by
more than ``RTOL`` or a solve returns NaN. Too large for CI: one 80 GB GPU per
solve, a 4 x 80 GB node for ``dense_mg``. Needs DESC to load the equilibria.

Run on a GPU node::

    python benchmarks/zernike_jd_vs_dense.py                 # the settings below
    python benchmarks/zernike_jd_vs_dense.py --solvers jd --penalties 1e4
    python benchmarks/zernike_jd_vs_dense.py --report        # compare OUTPUT

Each solve appends one JSON line to ``OUTPUT`` (``gamma^2``, residual, shift,
wall time, peak GPU memory of the process so far), so runs split over GPUs and
jobs are compared by ``--report``. The command-line options override the
settings below.

With ``jaxmg`` 1.x the dense solve runs one process per GPU, possibly on several
nodes (``--distributed``, dense solves only; process 0 writes the results)::

    python benchmarks/zernike_jd_vs_dense.py --saved qh.npz --penalties
    srun -N 2 --ntasks-per-node 4 --gpus-per-node 4 \\
        python benchmarks/zernike_jd_vs_dense.py --distributed --saved qh.npz \\
        --solvers dense

The first line writes the equilibrium (no solve: no penalties); the processes
read it, since DESC builds its transforms on the CPU device of process 0.
"""

from desc import set_device

set_device("gpu")  # before JAX starts: DESC otherwise selects the CPU

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
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
from agnimhd.multigpu import dense_mg  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"
CASES = {
    "tokamak": dict(
        file=DATA / "dshape_imax0.98_1608.h5",
        grid=(96, 96, 1),
        basis=dict(mpol=8),  # MPOL = 4 n
        assembly=dict(axisym=True, n_mode_axisym=2),
        dense="dense",  # one GPU: 12.1 GB complex A
    ),
    "stellarator": dict(
        file=DATA / "AGNI_QH_lowres.h5",
        grid=(48, 48, 16),
        basis=dict(mpol=8, ntor=2),  # as in the AGNI paper
        assembly=dict(),
        dense="dense_mg",
    ),
}
GEOMETRY = "stellarator"
PENALTIES = [1e-2, 1.0, 1e2, 1e3, 1e4]
FAMILIES = [0, 1]  # the stellarator's toroidal families; the tokamak has 0 only
PARITIES = [None]  # None, "even", "odd" (stellarator only)
SOLVERS = ["jd", "dense"]
SIGMAS = [1e-4, 1e-3, 1e-2, 1e-1]  # tried in turn until the solve is finite
RTOL = 1e-4  # JD vs dense, relative in gamma^2
MG_ITERS = 12  # dense_mg inverse iterations (its 6 leave 1e-4 at 48x48x16)
MG_TILE = 1024  # dense_mg JAXMg tile width
OUTPUT = Path("zernike_jd_vs_dense.jsonl")


def options():
    """The settings above, overridden from the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--geometry", choices=sorted(CASES), default=GEOMETRY)
    p.add_argument("--grid", type=int, nargs=3)
    p.add_argument("--penalties", type=float, nargs="*", default=PENALTIES)
    p.add_argument("--families", type=int, nargs="+", default=FAMILIES)
    p.add_argument("--parities", nargs="+", default=PARITIES)
    p.add_argument("--solvers", choices=["jd", "dense"], nargs="+", default=SOLVERS)
    p.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--dense", choices=["dense", "dense_mg"])
    p.add_argument("--saved", type=Path)  # the equilibrium as .npz: read, else written
    p.add_argument("--report", action="store_true")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--mpol", type=int)  # default: the case's
    p.add_argument("--ntor", type=int)
    p.add_argument("--mg-tile", type=int, default=MG_TILE)
    p.add_argument("--mg-iters", type=int, default=MG_ITERS)
    o = p.parse_args()
    o.parities = [None if x in (None, "None", "none") else x for x in o.parities]
    return o


def peak_gb():
    """Peak GPU memory of this process so far on its first device, GB."""
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", 0) / 1e9


def dense_untraced(eq, diffmat, assembly, cfg):
    """``(gamma^2, residual)`` of ``dense_mg`` called directly, as several
    processes need (``eigenpair`` compiles the whole solve as one program),
    with ``gamma^2`` printed after every iteration."""
    t0 = time.time()

    def log(it, gamma2, res, v):
        if jax.process_index() == 0:
            g = float(gamma2)
            print(f"  it {it}: gamma2 {g:+.10e} ({time.time() - t0:.0f} s)", flush=True)
        return False

    v, gamma2 = dense_mg(eq, diffmat, assembly, cfg, log=log)
    r = matfree_operator(eq, diffmat, assembly)["Ax"](v) + gamma2 * v
    return gamma2, np.linalg.norm(r) / (abs(gamma2) * np.linalg.norm(v))


def key(r):
    """The case a result belongs to."""
    return (r["geometry"], tuple(r["grid"]), r["family"], r["parity"], r["penalty"])


def report(path):
    """Print JD vs dense per case from ``path``; return the number of failures."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    cases = {}
    for r in rows:
        cases.setdefault(key(r), {})[r["solver"]] = r  # the latest run wins
    bad = 0
    print(
        f"{'case':44s} {'dense gamma^2':>14s} {'JD gamma^2':>14s} {'rel. diff':>9s} "
        f"{'JD s':>6s} {'JD GB':>6s} {'dense GB':>8s}  verdict"
    )
    for k in sorted(cases, key=str):
        c = cases[k]
        d, j = c.get("dense", {}), c.get("jd", {})
        gd, gj = d.get("gamma2", np.nan), j.get("gamma2", np.nan)
        rel = abs(gj - gd) / abs(gd) if d and j else np.nan
        ok = bool(np.isfinite(rel) and rel <= RTOL)
        verdict = "ok" if ok else ("missing" if not (d and j) else "FAIL")
        bad += verdict == "FAIL"
        g, grid, fam, par, pen = k
        name = f"{g} {'x'.join(map(str, grid))} f{fam} {par or 'both'} a {pen:.0e}"
        print(
            f"{name:44s} {gd:+14.6e} {gj:+14.6e} {rel:9.1e} "
            f"{j.get('seconds', np.nan):6.0f} {j.get('peak_gb', np.nan):6.2f} "
            f"{d.get('peak_gb', np.nan):8.2f}  {verdict}"
        )
    return bad


def main():
    """Run the cases of the settings, or compare them (``--report``)."""
    o = options()
    if o.report:
        sys.exit(1 if report(o.output) else 0)
    if o.distributed:
        if o.solvers != ["dense"] or not (o.saved and o.saved.exists()):
            sys.exit("--distributed: --solvers dense and an existing --saved file")
        local = int(os.environ.get("SLURM_LOCALID", "0"))
        jax.distributed.initialize(local_device_ids=[local])
    main_process = jax.process_index() == 0
    case = dict(CASES[o.geometry])
    case["basis"] = dict(case["basis"])
    for name in ("mpol", "ntor"):
        if getattr(o, name) is not None:
            case["basis"][name] = getattr(o, name)
    grid = tuple(o.grid or case["grid"])
    families = o.families if grid[2] > 1 else [0]
    t0 = time.time()
    base = Basis(*grid, **case["basis"])
    if o.saved and o.saved.exists():  # DESC's transforms need one process
        eq = EquilibriumData.load(o.saved)
    else:
        eq = from_desc(str(case["file"]), base)[0]
        if o.saved:
            eq.save(o.saved)
    if "jd" in o.solvers:
        eq_coarse = from_desc(str(case["file"]), base.coarse())[0]
    if main_process:
        print(f"{o.geometry} {grid}: equilibrium {time.time() - t0:.0f} s", flush=True)
    for parity in o.parities:
        assembly = AssemblyConfig(gamma=5.0 / 3.0, parity=parity, **case["assembly"])
        for family in families:
            for penalty in o.penalties:
                basis = Basis(*grid, penalty=penalty, **case["basis"])
                diffmat = basis.nodes_and_diffmat(eq.NFP, family)[1]
                coarse = (
                    basis.coarse_level(eq_coarse, family) if "jd" in o.solvers else None
                )
                for solver in o.solvers:
                    kind = "jd" if solver == "jd" else (o.dense or case["dense"])
                    for sigma in o.sigmas:
                        t = time.time()
                        cfg = SolverConfig(
                            eigensolver=kind,
                            sigma=sigma,
                            mg_tile=o.mg_tile,
                            mg_iters=o.mg_iters,
                        )
                        kw = dict(coarse=coarse) if kind == "jd" else {}
                        if o.distributed:  # not under jit: one process per GPU
                            gamma2, resid = dense_untraced(eq, diffmat, assembly, cfg)
                        else:
                            gamma2, _, resid = eigenpair(
                                eq, diffmat, assembly, cfg, **kw
                            )
                        gamma2 = float(jax.block_until_ready(gamma2))
                        if np.isfinite(gamma2):
                            break
                    r = dict(
                        geometry=o.geometry,
                        grid=list(grid),
                        **case["basis"],
                        family=family,
                        parity=parity,
                        penalty=penalty,
                        solver=solver,
                        eigensolver=kind,
                        sigma=sigma,
                        gamma2=gamma2,
                        residual=float(resid),
                        seconds=time.time() - t,
                        peak_gb=peak_gb(),
                    )
                    if kind == "dense_mg":
                        r.update(mg_tile=o.mg_tile, mg_iters=o.mg_iters)
                    if main_process:
                        print(json.dumps(r), flush=True)
                        with open(o.output, "a") as f:
                            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
