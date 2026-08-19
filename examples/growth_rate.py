#!/usr/bin/env python3
"""Solve for the growth rate of the shipped QH equilibrium.

The smallest complete thing you can do with ``agnimhd``: load an equilibrium,
build the matching grid operators, solve, and read the sign.

Run::

    python examples/growth_rate.py

Everything it needs is in this repository. No equilibrium code is involved --
the fixture is a serialized ``EquilibriumData``, which is the whole point of the
interface.
"""

from pathlib import Path

from agnimhd import AssemblyConfig, EquilibriumData, SolverConfig, eigenpair
from agnimhd.basis import standard_grid

# The automorphism the fixture was EXPORTED with. It is recorded in the sidecar
# `.json` next to the `.npz`; it is not guessable, and using different values
# here would build the operators on different nodes than the geometry lives on.
AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)

FIXTURE = Path(__file__).resolve().parents[1] / "tests/data/qh_lowres_24x12x8.npz"


def main():
    """Load, solve, report."""
    eq = EquilibriumData.load(FIXTURE)
    print(f"loaded {FIXTURE.name}: {eq.resolution} nodes, NFP={eq.NFP}")

    _, diffmat = standard_grid(*eq.resolution, NFP=eq.NFP, automorphism=AUTOMORPHISM)

    lam, v, resid = eigenpair(
        eq,
        diffmat,
        AssemblyConfig(gamma=5.0 / 3.0),
        SolverConfig(eigensolver="eigsh"),
    )

    lam = float(lam)
    print(f"lambda           {lam:+.10e}")
    print(f"Rayleigh residual {float(resid):.3e}")
    print(f"eigenvector       {v.shape[0]} retained degrees of freedom")
    print()
    # The sign is the physics answer. lambda is the energy quotient, so negative
    # is unstable -- the opposite of the convention in the AGNI paper. An
    # optimizer must RAISE this number.
    print("verdict:", "UNSTABLE" if lam < 0 else "stable")
    # And the magnitude is only meaningful well above the noise floor: the
    # absolute floor is ~1e-10 and the relative floor is 2.8e-5.
    print(f"         |lambda| / 1e-10 = {abs(lam) / 1e-10:.3g} (needs to be >> 1)")


if __name__ == "__main__":
    main()
