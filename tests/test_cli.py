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

    The automorphism has to be passed: the fixture was exported on clustered
    radial nodes, and omitting it silently builds the operators on a different
    grid than the geometry lives on. That is exactly the failure the flag's
    help text warns about, so the test that pins the value also pins the need
    to pass it.
    """
    meta = json.loads(EQ_META.read_text())
    auto = json.dumps(dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0))

    out, _ = _run(
        capsys,
        ["solve", str(EQ_FIXTURE), "--automorphism", auto],
    )

    assert "UNSTABLE" in out, "the shipped case is unstable; the CLI says otherwise"
    lam = float(out.split("lambda")[1].split()[0])
    ref = float(meta["dense_lambda3"])
    assert np.sign(lam) == np.sign(ref)
    assert (
        abs(lam - ref) / abs(ref) < 2.8e-5
    ), f"CLI lambda {lam:+.9e} vs reference {ref:+.9e}"


def test_solve_rejects_a_positive_shift(capsys):
    """``--sigma 0.1`` is refused rather than quietly solving the wrong problem.

    A shift above the spectrum converges to the wrong mode and makes
    ``A - sigma I`` indefinite, so ``SolverConfig`` refuses it. The CLI must
    surface that rather than swallow it.
    """
    with pytest.raises(ValueError, match="sigma"):
        main(["solve", str(EQ_FIXTURE), "--sigma", "0.1"])


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
