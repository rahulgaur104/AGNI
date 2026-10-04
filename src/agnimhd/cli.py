"""Command line entry point: ``agnimhd``.

Deliberately thin. Everything it does is a call into the library, because a
capability that only exists behind the CLI is a capability no consumer can use
and no test can reach.

Subcommands
-----------
``validate``
    Check a saved equilibrium against the contract and print what it contains.
    This is the first thing to run when a newly written adapter produces a file
    the solver rejects.
``info``
    Print the contract itself: which arrays are required, which are optional,
    and what the scalars mean.
``solve``
    Assemble and report the growth rate of every toroidal mode family (or of
    one, ``--family``) for a saved equilibrium.
"""

import argparse
import sys

__all__ = ["main"]


def _load(args):
    """Load an agnimhd ``.npz``/``.h5``, or a DESC ``.h5`` on the flags' Basis.

    Returns ``(EquilibriumData, Basis)``: the flags' basis at the file's
    resolution (a DESC file is evaluated on its nodes).
    """
    from .adapters.desc import from_desc, is_desc_file
    from .equilibrium import EquilibriumData

    if is_desc_file(args.path):
        if args.res is None:
            raise SystemExit("a DESC file needs --res n_rho,n_theta,n_zeta")
        basis = _basis(args, tuple(int(v) for v in args.res.split(",")))
        return from_desc(str(args.path), basis)[0], basis
    if str(args.path).endswith((".h5", ".hdf5")):
        eq = EquilibriumData.load_hdf5(args.path)
    else:
        eq = EquilibriumData.load(args.path)
    return eq, _basis(args, eq.resolution)


def _basis(args, resolution):
    """The :class:`~agnimhd.Basis` of the flags at ``resolution``."""
    import json

    from .basis import Basis

    knobs = dict(radial=args.radial, mpol=args.mpol, ntor=args.ntor)
    if args.automorphism is not None:
        knobs["automorphism"] = json.loads(args.automorphism)
    return Basis(*resolution, **knobs)


def _cmd_info(_args):
    """Print the interface contract."""
    from .equilibrium import OPTIONAL_ARRAYS, REQUIRED_ARRAYS, REQUIRED_SCALARS

    print("agnimhd equilibrium contract")
    print()
    print("Node ordering is rho-major: the flat index of (i, j, k) is")
    print("    (i * n_theta + j) * n_zeta + k")
    print("and component c lives at c * n_total + that. Coordinates are PEST")
    print("straight-field-line (rho, theta, zeta). Units are SI.")
    print()
    n_req = len(REQUIRED_ARRAYS)
    print(f"required arrays, each of length n_rho*n_theta*n_zeta ({n_req}):")
    for key in REQUIRED_ARRAYS:
        print(f"    {key}")
    print()
    print("required scalars:")
    for key in REQUIRED_SCALARS:
        print(f"    {key}")
    print()
    print("optional arrays:")
    for key in OPTIONAL_ARRAYS:
        print(f"    {key}")
    print()
    print("Supply `finite_n_instability_drive` directly, or both")
    print("`J_cross_grad_rho` and `B_dot_grad_grad_rho` and agnimhd will form it.")
    print()
    print("Two things that are easy to get wrong and hard to notice:")
    print("  * `a` is the minor radius, and the eigenvalue is hypersensitive")
    print("    to it. Two defensible definitions were measured to differ by")
    print("    3.76%. Record which one you used.")
    print("  * `p` is PRESSURE in pascals. A kinetic energy density or a")
    print("    temperature in eV produces NaN, not a wrong answer.")
    return 0


def _cmd_validate(args):
    """Load an equilibrium and check it against the contract."""
    import numpy as np

    from .equilibrium import OPTIONAL_ARRAYS, REQUIRED_ARRAYS

    try:
        eq, _ = _load(args)
    except ValueError as err:
        print(f"INVALID: {err}", file=sys.stderr)
        return 1

    print(f"{args.path}")
    print(f"  resolution   {eq.resolution}  ({eq.n_nodes} nodes)")
    print(f"  NFP          {eq.NFP}")
    print(f"  Psi          {float(eq.Psi):+.9e} Wb")
    print(f"  a            {float(eq.a):+.9e} m")
    if args.verbose:
        print("  arrays:")
        for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
            val = getattr(eq, key)
            if val is None:
                print(f"    {key:<22} (absent)")
                continue
            arr = np.asarray(val)
            print(f"    {key:<22} min {arr.min():+.6e}  max {arr.max():+.6e}")
    drive = np.asarray(eq.instability_drive())
    print(
        f"  drive        min {drive.min():+.6e}  max {drive.max():+.6e}"
        f"  ({'supplied' if eq.finite_n_instability_drive is not None else 'derived'})"
    )
    print("VALID")
    return 0


def _cmd_solve(args):
    """Assemble and report the growth rate of each toroidal mode family."""
    import numpy as np

    from .config import AssemblyConfig, SolverConfig
    from .objective import eigenpair

    eq, basis = _load(args)
    assembly = AssemblyConfig(gamma=args.gamma)
    solver = SolverConfig(eigensolver=args.eigensolver, sigma=args.sigma)
    families = basis.families(eq.NFP) if args.family is None else (args.family,)
    gamma2 = {}
    for x in families:
        _, diffmat = basis.nodes_and_diffmat(eq.NFP, family=x)
        try:
            value, _, resid = eigenpair(eq, diffmat, assembly, solver)
        except RuntimeError as err:  # e.g. ARPACK, when nothing lies below round-off
            reason = str(err).strip().splitlines()[-1]
            print(f"family {x}  no converged eigenpair ({reason})")
            gamma2[x] = np.nan
            continue
        gamma2[x] = float(value)
        print(f"family {x}  gamma^2 {gamma2[x]:+.10e}  residual {float(resid):.3e}")
    found = {x: g for x, g in gamma2.items() if np.isfinite(g)}
    if not found:
        return 1
    worst = max(found, key=found.get)
    print(f"most unstable: family {worst}")
    print(f"verdict  {'UNSTABLE' if found[worst] > 0 else 'stable'}")
    return 0 if np.isfinite(list(gamma2.values())).all() else 1


def main(argv=None):
    """Entry point for the ``agnimhd`` console script.

    Parameters
    ----------
    argv : list of str, optional
        Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="agnimhd",
        description="Finite-n ideal MHD stability.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print the equilibrium contract")
    p_info.set_defaults(func=_cmd_info)

    basis_flags = argparse.ArgumentParser(add_help=False)  # agnimhd.Basis
    basis_flags.add_argument(
        "--res", help="n_rho,n_theta,n_zeta, n_zeta per field period (DESC file)"
    )
    basis_flags.add_argument(
        "--radial",
        default="gauss_radau_jacobi",
        choices=("gauss_radau_jacobi", "lobatto"),
    )
    basis_flags.add_argument("--mpol", type=int, help="highest poloidal mode kept")
    basis_flags.add_argument("--ntor", type=int, help="highest toroidal mode kept")
    basis_flags.add_argument(
        "--automorphism",
        help=(
            "JSON kwargs of the staircase radial map (default: Basis's), null "
            "for none. For an agnimhd file it MUST match the export, or the "
            "matrices are built on other nodes than the geometry."
        ),
    )

    p_val = sub.add_parser(
        "validate", help="check a saved equilibrium", parents=[basis_flags]
    )
    p_val.add_argument("path", help=".npz or .h5 written by EquilibriumData.save")
    p_val.add_argument(
        "-v", "--verbose", action="store_true", help="print per-array ranges"
    )
    p_val.set_defaults(func=_cmd_validate)

    p_solve = sub.add_parser(
        "solve", help="report the growth rate", parents=[basis_flags]
    )
    p_solve.add_argument("path", help="agnimhd .npz/.h5, or a DESC .h5 with --res")
    p_solve.add_argument(
        "--family",
        type=int,
        help="toroidal mode family x (n = x + k NFP); default: every one needed",
    )
    p_solve.add_argument("--gamma", type=float, default=5.0 / 3.0)
    p_solve.add_argument(
        "--sigma",
        type=float,
        default=1e-1,
        help=(
            "shift-invert shift. Must be above the largest gamma^2, and for "
            "--eigensolver jax_lanczos not far above it either: the default is "
            "safe for ARPACK, which iterates to a tolerance, but a fixed-budget "
            "Lanczos at a far shift can return the wrong mode. Watch the "
            "printed residual."
        ),
    )
    p_solve.add_argument(
        "--eigensolver", default="eigsh", choices=("eigsh", "jax_lanczos")
    )
    p_solve.set_defaults(func=_cmd_solve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
