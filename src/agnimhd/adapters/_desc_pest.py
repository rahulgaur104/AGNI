"""TEMPORARY: DESC's ``Transform(method="pest")`` for DESC versions without it.

``from_desc`` evaluates the equilibrium on the PEST nodes mapped to DESC's
coordinates. Those nodes are uniform in zeta (the map leaves zeta alone) but not a
tensor product in (rho, theta), so DESC's own choice is ``direct1``: the full
Fourier-Zernike basis at every node, ``num_nodes x num_modes`` per derivative, which
is the time and memory hog at production resolution. The ``pest`` method evaluates
the (l, m) factor at every node and the toroidal factor only at the unique zeta
values, at machine precision.

``pest`` is DESC PR ``ma/pest_transform``; DESC master and the 0.17.x releases do
not have it, and their ``get_transforms`` replaces any method but ``"jitable"`` by
``"auto"``. Until the PR is merged and released, this module supplies the
forward transform -- all ``eq.compute`` needs -- as a ``Transform`` subclass, with
the PR's own construction (``_check_inputs_pest``, ``build``, ``transform``).
``fit`` and ``project`` are not supported (``from_desc`` never calls them).

REMOVE this module, and the fallback in ``desc._pest_transforms``, once the DESC
that agnimhd supports has ``method="pest"`` (:func:`desc_has_pest` is then always
True).
"""

from functools import lru_cache

import numpy as np


def desc_has_pest():
    """Whether the installed DESC's ``Transform`` has ``method="pest"``."""
    from desc.transform import Transform

    return hasattr(Transform, "_check_inputs_pest")


@lru_cache(maxsize=None)
def _pest_transform_class():
    """The fallback ``Transform`` subclass (built lazily: DESC is optional)."""
    from desc.backend import jnp, put
    from desc.grid import Grid
    from desc.transform import Transform

    class PestTransform(Transform):
        """``Transform`` with DESC PR ``ma/pest_transform``'s ``pest`` method.

        Forward transform only. Constructed as ``direct1`` (which builds nothing),
        then switched to ``pest`` by setting the method's backing field: DESC's
        ``method`` setter does not know ``"pest"``.
        """

        def __init__(self, grid, basis, derivs=0):
            super().__init__(grid, basis, derivs=derivs, build=False, method="direct1")
            self._setup_pest()

        def _setup_pest(self):
            """``Transform._check_inputs_pest`` of the PR, without the fallbacks:
            the caller only builds one for a basis with ``fft_toroidal``."""
            grid, basis = self.grid, self.basis
            self._method = "pest"
            self._built = False
            if hasattr(self, "_matrices"):
                del self._matrices
            self.lm_modes = basis.modes[basis.unique_LM_idx, :2]
            self.n_modes = basis.modes[basis.unique_N_idx, 2]
            self.zeta_nodes = grid.nodes[grid.unique_zeta_idx, 2]
            self.num_lm_modes = self.lm_modes.shape[0]
            self.num_n_modes = self.n_modes.size
            row = np.where(
                (basis.modes[:, None, :2] == self.lm_modes[None, :, :]).all(axis=-1)
            )[1]
            col = np.where(
                basis.modes[None, :, 2] == basis.modes[basis.unique_N_idx, None, 2]
            )[0]
            self.fft_index = np.atleast_1d(np.squeeze(self.num_n_modes * row + col))
            dft_nodes = np.hstack(
                [np.zeros((self.zeta_nodes.size, 2)), self.zeta_nodes[:, None]]
            )
            self.dft_grid = Grid(dft_nodes, sort=False, jitable=True, axis_shift=0)
            self.pest_zeta_idx = grid.inverse_zeta_idx

        @property
        def method(self):
            """``"pest"``."""
            return self.__dict__.setdefault("_method", "direct1")

        @method.setter
        def method(self, method):
            # Only DESC's own __init__ sets it ("direct1"), before _setup_pest.
            Transform.method.fset(self, method)

        @property
        def matrices(self):
            if not hasattr(self, "_matrices"):
                ndi, ndj, ndk = self.derivatives.max(axis=0)
                self._matrices = {
                    "pest": {
                        i: {j: {} for j in range(ndj + 1)} for i in range(ndi + 1)
                    },
                    "pest_toroidal": {k: {} for k in range(ndk + 1)},
                }
            return self._matrices

        def build(self):
            """Radial/poloidal factor at every node, toroidal at the unique zeta."""
            if self.built:
                return
            if self.basis.num_modes == 0:
                self._built = True
                return
            if hasattr(self, "_matrices"):  # derivatives may have been added
                del self._matrices
            d = self.derivatives
            temp_d = np.hstack([d[:, :2], np.zeros((len(d), 1))]).astype(int)
            temp_modes = np.hstack([self.lm_modes, np.zeros((self.num_lm_modes, 1))])
            for dd in temp_d:
                self.matrices["pest"][dd[0]][dd[1]] = self.basis.evaluate(
                    self.grid, dd, modes=temp_modes
                )
            temp_d = np.hstack([np.zeros((len(d), 2)), d[:, 2:]]).astype(int)
            temp_modes = np.hstack(
                [np.zeros((self.num_n_modes, 2)), self.n_modes[:, None]]
            )
            for dd in temp_d:
                self.matrices["pest_toroidal"][dd[2]] = self.basis.evaluate(
                    self.dft_grid, dd, modes=temp_modes
                )
            self._built = True

        def transform(self, c, dr=0, dt=0, dz=0):
            """Spectral -> physical, as the PR's ``pest`` branch."""
            if not self.built:
                raise RuntimeError(
                    "Transform must be precomputed with transform.build() before "
                    "being used"
                )
            if self.basis.num_modes == 0:
                return jnp.zeros(self.grid.num_nodes)
            A = self.matrices["pest"].get(dr, {}).get(dt, {})
            B = self.matrices["pest_toroidal"].get(dz, {})
            if isinstance(A, dict) or isinstance(B, dict):
                raise ValueError("Derivative orders are out of initialized bounds")
            c_mtrx = jnp.zeros((self.num_lm_modes * self.num_n_modes,))
            c_mtrx = put(c_mtrx, self.fft_index, c).reshape((-1, self.num_n_modes))
            cc = A @ c_mtrx  # (num_nodes, num_n_modes)
            return jnp.einsum("kn,kn->k", cc, B[self.pest_zeta_idx])

        def fit(self, x):  # noqa: D102
            raise NotImplementedError("agnimhd's pest fallback is forward-only")

        def project(self, y):  # noqa: D102
            raise NotImplementedError("agnimhd's pest fallback is forward-only")

    return PestTransform


def pest_transforms_fallback(eq, keys, grid):
    """``get_transforms(keys, eq, grid, method="pest")`` on a DESC without ``pest``.

    Mirrors DESC's ``get_transforms`` for an equilibrium -- ``get_derivs`` for the
    derivative orders, one transform per ``<c>_basis``, shared between equivalent
    bases -- but builds ``pest`` transforms. It does not call ``get_transforms``:
    that builds every transform before returning, i.e. exactly the ``direct1``
    matrices this avoids. Anything else (a quantity without an ``<c>_basis``, e.g.
    Boozer's, or a basis ``pest`` cannot take) falls back to DESC's
    ``get_transforms``, i.e. ``direct1`` as before this patch.
    """
    from desc.compute import get_transforms
    from desc.compute.utils import get_derivs

    PestTransform = _pest_transform_class()
    derivs = get_derivs(keys, eq, has_axis=bool(grid.axis.size))
    transforms, made = {"grid": grid}, []
    for c, d in derivs.items():
        if c == "grid" or not len(d):  # get_derivs lists the grid, with no orders
            continue
        d = np.atleast_2d(np.asarray(d, dtype=int))
        basis = getattr(eq, c + "_basis", None)
        if basis is None or not getattr(basis, "fft_toroidal", False):
            return get_transforms(keys, obj=eq, grid=grid)
        for t in made:  # as get_transforms: one transform per equivalent basis
            if basis.equiv(t.basis):
                ders = np.unique(np.vstack([d, t.derivatives]), axis=0).astype(int)
                t.change_derivatives(ders, build=False)
                transforms[c] = t
                break
        else:
            transforms[c] = PestTransform(grid, basis, derivs=d)
            made.append(transforms[c])
    for t in made:
        t.build()
    return transforms
