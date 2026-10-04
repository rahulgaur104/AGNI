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

from agnimhd import AssemblyConfig, Basis, EquilibriumData, SolverConfig, eigenpair

# The radial nodes the fixture was EXPORTED on: Lobatto through this staircase
# map, not the default basis. Both are recorded in the sidecar `.json` next to
# the `.npz`; they are not guessable, and other values here would build the
# operators on different nodes than the geometry lives on.
AUTOMORPHISM = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)

FIXTURE = Path(__file__).resolve().parents[1] / "tests/data/qh_lowres_24x12x8.npz"


def main():
    """Load, solve, report."""
    eq = EquilibriumData.load(FIXTURE)
    print(f"loaded {FIXTURE.name}: {eq.resolution} nodes, NFP={eq.NFP}")

    basis = Basis(
        *eq.resolution,
        radial="lobatto",
        automorphism=AUTOMORPHISM,
    )
    _, diffmat = basis.nodes_and_diffmat(eq.NFP)

    gamma2, v, resid = eigenpair(
        eq,
        diffmat,
        AssemblyConfig(gamma=5.0 / 3.0),
        SolverConfig(eigensolver="eigsh"),
    )

    gamma2 = float(gamma2)
    print(f"gamma^2           {gamma2:+.10e}")
    print(f"Rayleigh residual {float(resid):.3e}")
    print(f"eigenvector       {v.shape[0]} retained degrees of freedom")
    print()
    # The sign is the physics answer. gamma^2 = -lambda is the squared growth
    # rate, so positive is unstable, as in the AGNI paper. An optimizer must
    # LOWER this number.
    print("verdict:", "UNSTABLE" if gamma2 > 0 else "stable")
    # And the magnitude is only meaningful well above the noise floor: the
    # absolute floor is ~1e-10 and the relative floor is 2.8e-5.
    print(f"         |gamma^2| / 1e-10 = {abs(gamma2) / 1e-10:.3g} (needs to be >> 1)")


if __name__ == "__main__":
    main()
