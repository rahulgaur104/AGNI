"""Adapters from equilibrium codes to :class:`agnimhd.EquilibriumData`.

Each adapter imports its equilibrium code lazily, inside the function, so the
package itself stays free of those dependencies: ``agnimhd`` installs and runs
without DESC; ``from_desc`` only works when DESC is importable.
"""

from .desc import from_desc

__all__ = ["from_desc"]
