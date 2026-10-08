"""AGNI -- Analysis of Global Normal-modes in Ideal MHD.

A differentiable finite-n ideal MHD stability solver.

AGNI solves ideal MHD stability from a **variational principle**: it discretizes
the energy functional rather than the force operator. That gives a generalized
symmetric eigenvalue problem ``A x = lambda B x`` with ``B`` (the kinetic/mass
matrix) symmetric positive definite. ``B`` is Cholesky-factored to reduce this to
a standard symmetric eigenvalue problem. Everything agnimhd returns is the
squared growth rate ``gamma^2 = -lambda`` of the most negative eigenvalue
``lambda``: **positive means unstable**.

The package depends on ``jax``, ``numpy``, ``scipy`` and ``matfree``.
:func:`from_desc` imports DESC when it is called; nothing else does. Other
equilibrium codes fill an :class:`EquilibriumData` directly; see
``docs/interface.md``.
"""

from .adapters import from_desc
from .basis import Basis, DiffMat
from .config import AssemblyConfig, SolverConfig
from .equilibrium import FORMAT_VERSION, EquilibriumData
from .objective import (
    eigenpair,
    growth_rate,
    growth_rate_and_grad,
    growth_rate_of,
    solve,
)
from .sources import load

__all__ = [
    "AssemblyConfig",
    "Basis",
    "DiffMat",
    "EquilibriumData",
    "FORMAT_VERSION",
    "SolverConfig",
    "eigenpair",
    "from_desc",
    "growth_rate",
    "growth_rate_and_grad",
    "growth_rate_of",
    "load",
    "solve",
    "__version__",
]

try:  # written by setuptools_scm at build or install time
    from ._version import __version__
except ImportError:  # a bare checkout on sys.path
    __version__ = "0+unknown"
