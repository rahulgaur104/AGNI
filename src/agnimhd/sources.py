"""Equilibrium sources: :func:`load` and the two kinds it returns.

A source evaluates an equilibrium on the nodes of a :class:`~agnimhd.Basis`,
``load(x).evaluate(basis) -> EquilibriumData``. A DESC equilibrium is evaluated
on any basis; a saved ``EquilibriumData`` holds the nodes of one basis only.
"""

from pathlib import Path

from .backend import errorif
from .equilibrium import EquilibriumData

__all__ = ["DescSource", "SavedSource", "load"]


def load(path_or_object):
    """The equilibrium source of a file or an object.

    Parameters
    ----------
    path_or_object : str, Path, EquilibriumData or desc.equilibrium.Equilibrium
        A DESC ``.h5`` file (its last equilibrium) or ``Equilibrium``, or an
        agnimhd ``.npz`` / ``.h5`` file or ``EquilibriumData``.

    Returns
    -------
    DescSource or SavedSource
    """
    from .adapters.desc import is_desc_file

    x = path_or_object
    if isinstance(x, EquilibriumData):
        return SavedSource(x)
    if not isinstance(x, (str, Path)) or is_desc_file(x):
        return DescSource(x if not isinstance(x, Path) else str(x))
    if str(x).endswith((".h5", ".hdf5")):
        return SavedSource(EquilibriumData.load_hdf5(x))
    return SavedSource(EquilibriumData.load(x))


class DescSource:
    """A DESC equilibrium, evaluated on any basis by :func:`~agnimhd.from_desc`."""

    def __init__(self, eq):
        from .adapters.desc import desc_equilibrium

        self.eq = desc_equilibrium(eq)

    def evaluate(self, basis, density=False):
        """``EquilibriumData`` on the nodes of ``basis``, weighted if ``density``."""
        from .adapters.desc import from_desc

        return from_desc(self.eq, basis, density=density)[0]


class SavedSource:
    """A saved ``EquilibriumData``: its own nodes, its own density (or none)."""

    def __init__(self, eq):
        self.eq = eq

    def evaluate(self, basis, density=False):
        """The saved data, on a ``basis`` of its resolution.

        The radial nodes and map of ``basis`` must be the ones it was exported
        on; only the resolution can be checked.
        """
        resolution = (basis.n_rho, basis.n_theta, basis.n_zeta)
        errorif(
            resolution != self.eq.resolution,
            ValueError,
            f"a saved EquilibriumData holds the nodes of one basis, resolution "
            f"{self.eq.resolution}, not {resolution}; other nodes need the "
            "equilibrium code, e.g. a DESC file.",
        )
        errorif(
            density and self.eq.density is None,
            ValueError,
            "density=True, but the saved EquilibriumData has no density.",
        )
        return self.eq
