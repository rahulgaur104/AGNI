"""AGNI -- Analysis of Global Normal-modes in Ideal MHD.

A differentiable finite-n ideal MHD stability solver.

AGNI solves ideal MHD stability from a **variational principle**: it discretizes
the energy functional rather than the force operator. That gives a generalized
symmetric eigenvalue problem ``A x = lambda B x`` with ``B`` (the kinetic/mass
matrix) symmetric positive definite. ``B`` is Cholesky-factored to reduce this to
a standard symmetric eigenvalue problem. The most negative eigenvalue ``lambda``
is the squared growth rate, and **the sign of lambda decides whether the
equilibrium is stable**.

The package depends on ``jax``, ``numpy``, ``scipy`` and ``matfree``, and
nothing else. In particular it does not depend on DESC, in any form -- the
dependency runs the other way: DESC (or any other equilibrium code) installs
``agnimhd``, converts its own equilibrium into an :class:`EquilibriumData`, and
wraps :func:`growth_rate` as an objective. See ``docs/adapters.md``.
"""

from .basis import DiffMat
from .config import AssemblyConfig, SolverConfig
from .equilibrium import FORMAT_VERSION, EquilibriumData
from .objective import eigenpair, growth_rate, growth_rate_and_grad, growth_rate_of

__all__ = [
    "AssemblyConfig",
    "DiffMat",
    "EquilibriumData",
    "FORMAT_VERSION",
    "SolverConfig",
    "eigenpair",
    "growth_rate",
    "growth_rate_and_grad",
    "growth_rate_of",
    "__version__",
]

__version__ = "0.1.0.dev0"
