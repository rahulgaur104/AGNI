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
    Assemble and report the growth rate for a saved equilibrium.
"""

import argparse
import sys

__all__ = ["main"]


def _load(path, res=None, automorphism=None):
    """Load ``.npz``/``.h5`` written by agnimhd, or a DESC ``.h5`` (needs ``res``).

    Returns ``(EquilibriumData, DiffMat or None)``; the DiffMat comes with a
    DESC file because its nodes are built in the same call.
    """
    from .adapters.desc import AUTOMORPHISM, from_desc, is_desc_file
    from .equilibrium import EquilibriumData

    if is_desc_file(path):
        if res is None:
            raise SystemExit("a DESC file needs --res n_rho,n_theta,n_zeta")
        auto = AUTOMORPHISM if automorphism is None else automorphism
        return from_desc(str(path), *res, automorphism=auto)
    if str(path).endswith((".h5", ".hdf5")):
        return EquilibriumData.load_hdf5(path), None
    return EquilibriumData.load(path), None


def _res(args):
    """``--res`` as a tuple of three ints, or None."""
    return tuple(int(v) for v in args.res.split(",")) if args.res else None


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
        eq, _ = _load(args.path, _res(args))
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
    """Assemble and report the growth rate."""
    import json

    import numpy as np

    from .basis import standard_grid
    from .config import AssemblyConfig, SolverConfig
    from .objective import eigenpair

    auto_kw = json.loads(args.automorphism) if args.automorphism else None
    eq, diffmat = _load(args.path, _res(args), auto_kw)
    if diffmat is None:
        n_rho, n_theta, n_zeta = eq.resolution
        _, diffmat = standard_grid(
            n_rho, n_theta, n_zeta, NFP=eq.NFP, automorphism=auto_kw
        )

    gamma2, _, resid = eigenpair(
        eq,
        diffmat,
        AssemblyConfig(gamma=args.gamma),
        SolverConfig(eigensolver=args.eigensolver, sigma=args.sigma),
    )
    gamma2 = float(gamma2)
    print(f"gamma^2  {gamma2:+.10e}")
    print(f"residual {float(resid):.3e}")
    print(f"verdict  {'UNSTABLE' if gamma2 > 0 else 'stable'}")
    if not np.isfinite(gamma2):
        return 1
    return 0


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

    p_val = sub.add_parser("validate", help="check a saved equilibrium")
    p_val.add_argument("path", help=".npz or .h5 written by EquilibriumData.save")
    p_val.add_argument(
        "-v", "--verbose", action="store_true", help="print per-array ranges"
    )
    p_val.add_argument("--res", default=None, help="n_rho,n_theta,n_zeta (DESC file)")
    p_val.set_defaults(func=_cmd_validate)

    p_solve = sub.add_parser("solve", help="report the growth rate")
    p_solve.add_argument("path", help="agnimhd .npz/.h5, or a DESC .h5 with --res")
    p_solve.add_argument(
        "--res", default=None, help="n_rho,n_theta,n_zeta; required for a DESC file"
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
    p_solve.add_argument(
        "--automorphism",
        default=None,
        help=(
            "JSON kwargs for the staircase radial automorphism the nodes were "
            'built with, e.g. \'{"eps": 0.01, "x_0": 0.65, "m_1": 2.0, '
            '"m_2": 3.0}\'. Omit for unclustered Lobatto nodes. This MUST '
            "match the export, or the differentiation matrices are built on "
            "different nodes than the geometry."
        ),
    )
    p_solve.set_defaults(func=_cmd_solve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
