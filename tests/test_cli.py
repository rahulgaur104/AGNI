"""The command line entry point.

The CLI is deliberately thin -- every subcommand is a call into the library --
so what is worth testing is the *contract with a shell*: exit status, what is
printed, and that a bad input is reported as bad instead of crashing or, worse,
being accepted. Anything that could only be reached through the CLI would be a
capability no consumer could use.

``solve`` is exercised at the shipped resolution. It is the slowest test in the
suite by a wide margin and it stays that way: reducing the resolution to make it
cheap would be testing a different problem than the one that ships.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from agnimhd.cli import main

# The same files conftest uses. Named here rather than imported, because
# `tests/` is not a package and `from .conftest import ...` does not work.
DATA = Path(__file__).parent / "data"
EQ_FIXTURE = DATA / "qh_lowres_24x12x8.npz"
EQ_META = DATA / "qh_lowres_24x12x8.json"
PERIOD_FIXTURE = DATA / "qh_lowres_8x8x3.npz"
#: The radial nodes of every fixture: Lobatto through this staircase map.
FIXTURE_NODES = [
    "--radial",
    "lobatto",
    "--automorphism",
    json.dumps(dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)),
]


def _run(capsys, argv, expect=0):
    """Run the CLI, assert the exit status, return (stdout, stderr)."""
    code = main(argv)
    out, err = capsys.readouterr()
    assert code == expect, f"agnimhd {' '.join(argv)} exited {code}\n{out}\n{err}"
    return out, err


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def test_info_prints_the_contract(capsys):
    """``info`` documents the interface without needing a file.

    It is what someone runs when an adapter's output is rejected, so it has to
    name every required field, not summarize them.
    """
    from agnimhd.equilibrium import OPTIONAL_ARRAYS, REQUIRED_ARRAYS, REQUIRED_SCALARS

    out, _ = _run(capsys, ["info"])
    for key in REQUIRED_ARRAYS + REQUIRED_SCALARS + OPTIONAL_ARRAYS:
        assert key in out, f"`info` does not mention {key}"


def test_info_states_the_node_ordering_and_both_traps(capsys):
    """The three things an adapter gets wrong, in the one place it will look."""
    out, _ = _run(capsys, ["info"])
    assert "(i * n_theta + j) * n_zeta + k" in out, "node ordering is not stated"
    assert "3.76" in out, "the `a`-definition trap is not stated"
    assert (
        "PRESSURE" in out or "pressure" in out.lower()
    ), "the pressure-not-kinetic-energy trap is not stated"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_accepts_the_shipped_fixture(capsys, eq_data, eq_meta):
    """The file the whole suite is built on passes its own validator."""
    out, _ = _run(capsys, ["validate", str(EQ_FIXTURE)])
    assert "VALID" in out
    assert str(eq_data.NFP) in out
    # The printed resolution is the real one, not a default.
    assert str(eq_meta["resolution"][0]) in out


def test_validate_verbose_prints_every_array(capsys):
    """``-v`` is what turns "invalid" into "which field"."""
    from agnimhd.equilibrium import REQUIRED_ARRAYS

    out, _ = _run(capsys, ["validate", str(EQ_FIXTURE), "-v"])
    for key in REQUIRED_ARRAYS:
        assert key in out, f"-v does not report {key}"


def test_validate_reports_a_bad_file_as_a_failure(capsys, tmp_path, eq_data):
    """A corrupt equilibrium exits nonzero and says so on stderr.

    Exit status is the whole point: a script that loops over exported files has
    nothing else to branch on.
    """
    bad = tmp_path / "bad.npz"
    with np.load(str(EQ_FIXTURE)) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["g_rr"] = np.full_like(arrays["g_rr"], np.nan)
    np.savez(bad, **arrays)

    _, err = _run(capsys, ["validate", str(bad)], expect=1)
    assert "INVALID" in err


def test_validate_reports_the_drive_and_where_it_came_from(capsys):
    """Supplied or derived -- the two routes are not interchangeable in a log."""
    out, _ = _run(capsys, ["validate", str(EQ_FIXTURE)])
    assert "drive" in out
    assert ("supplied" in out) or ("derived" in out)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_solve_reports_the_reference_eigenvalue_and_the_verdict(capsys):
    """End to end through the shell interface, against the sidecar number.

    The fixture was exported on Lobatto nodes through a staircase map other
    than the default: ``--radial`` and ``--automorphism`` must match the export,
    or the operators are built on a different grid than the geometry lives on.
    The reference is the periodic family's, ``--family 0``.
    """
    meta = json.loads(EQ_META.read_text())

    out, _ = _run(capsys, ["solve", str(EQ_FIXTURE), *FIXTURE_NODES, "--family", "0"])

    assert "UNSTABLE" in out, "the shipped case is unstable; the CLI says otherwise"
    gamma2 = float(out.split("gamma^2")[1].split()[0])
    ref = -float(meta["dense_lambda3"])  # gamma^2 = -lambda
    assert np.sign(gamma2) == np.sign(ref)
    assert (
        abs(gamma2 - ref) / abs(ref) < 2.8e-5
    ), f"CLI gamma^2 {gamma2:+.9e} vs reference {ref:+.9e}"


@pytest.mark.parametrize(
    "family, solved, most_unstable",
    [([], [0, 1, 2], 2), (["--family", "1"], [1], 1)],
    ids=["every_family", "one_family"],
)
def test_solve_reports_each_family_and_the_most_unstable(
    capsys, family, solved, most_unstable
):
    """Without ``--family`` every family ``x = 0 ... NFP // 2`` is solved and the
    most unstable one named (family 2 on this NFP = 4 case); ``--family x``
    solves only ``x``."""
    out, _ = _run(capsys, ["solve", str(PERIOD_FIXTURE), *FIXTURE_NODES, *family])
    lines = out.splitlines()
    assert [int(ln.split()[1]) for ln in lines if ln.startswith("family")] == solved
    assert f"most unstable: family {most_unstable}" in out
    assert "UNSTABLE" in out


def test_solve_goes_on_past_a_family_without_a_converged_mode(capsys, monkeypatch):
    """A family whose eigensolve fails is reported and the others are solved; the
    exit status is 1. ARPACK fails like this when no eigenvalue lies below
    round-off (measured: family 0 of this case at 8x8x5)."""
    from agnimhd import objective

    solve, calls = objective.eigenpair, []

    def first_solve_fails(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("ARPACK error -1: No convergence")
        return solve(*args)

    monkeypatch.setattr(objective, "eigenpair", first_solve_fails)
    out, _ = _run(capsys, ["solve", str(PERIOD_FIXTURE), *FIXTURE_NODES], expect=1)
    assert "family 0  no converged eigenpair" in out
    assert "most unstable: family 2" in out


def test_solve_rejects_a_negative_shift(capsys):
    """``--sigma -0.1`` is refused rather than quietly solving the wrong problem.

    A shift below the largest ``gamma^2`` converges to the wrong mode and makes
    ``A + sigma I`` indefinite, so ``SolverConfig`` refuses it. The CLI must
    surface that rather than swallow it.
    """
    with pytest.raises(ValueError, match="sigma"):
        main(["solve", str(EQ_FIXTURE), "--sigma", "-0.1"])


def test_unknown_subcommand_is_a_usage_error():
    """argparse exits 2; that is the shell contract, not an exception."""
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2


def test_version_is_the_package_version(capsys):
    """``--version`` reports what ``agnimhd.__version__`` says."""
    import agnimhd

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out, _ = capsys.readouterr()
    assert agnimhd.__version__ in out
