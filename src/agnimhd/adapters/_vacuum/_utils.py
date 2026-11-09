"""``apply`` and ``safenorm`` as on DESC's ``ma/AGNI_ext`` branch.

Copied from DESC (https://github.com/PlasmaControl/DESC), file ``desc/utils.py`` on
branch ``ma/AGNI_ext`` at commit 1583a921ae4d36171f7d970b2a56b81ae8132bcd. The branch
changed both, and the copied boundary-integral code depends on the change:

* ``apply`` skips ``subset`` keys missing from the dict (master raises ``KeyError``;
  ``_prune_data`` asks for a kernel's optional ``eval_keys``);
* ``safenorm`` keeps the zero mask's shape under ``keepdims`` (master squeezes every
  singleton axis of it before broadcasting back).

DESC is MIT licensed:

    Copyright (c) 2020 Daniel Dudt, Rory Conlin, Dario Panici, Kaya Unalmis,
    Yigit Gunsur Elmacioglu, Rahul Gaur, Patrick Kim, Egemen Kolemen
"""

from desc.backend import jnp
from desc.utils import identity


def apply(d, fun=identity, subset=None, exclude=None):
    """Applies ``fun`` to ``d``.

    Parameters
    ----------
    d : dict
        Dictionary to map.
    fun : callable
        Function to apply to values in dictionary.
        Default is the identity.
    subset : list or set or tuple
        Subset of keys in ``d`` to consider.
        Default is all keys in ``d``.
    exclude : collection
        Stuff in subset to exclude.

    Returns
    -------
    d : dict
        New dictionary with ``fun`` mapped over values with keys in ``subset``
        and keys not in ``exclude``.

    """
    if d is None:
        return {}
    if subset is None:
        subset = d.keys()
    elif isinstance(subset, str):
        subset = (subset,)
    exclude = () if (exclude is None) else exclude
    return {k: fun(d[k]) for k in subset if (k in d and k not in exclude)}


def safenorm(x, ord=None, axis=None, fill=0, threshold=0, keepdims=False):
    """Like jnp.linalg.norm, but without nan gradient at x=0.

    Parameters
    ----------
    x : ndarray
        Vector or array to norm.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of norm.
    axis : {None, int, 2-tuple of ints}, optional
        Axis to take norm along.
    fill : float, ndarray, optional
        Value to return where x is zero.
    threshold : float >= 0
        How small is x allowed to be.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the result
        as dimensions with size one. With this option the result will broadcast
        correctly against the original x.

    """
    is_zero = (jnp.abs(x) <= threshold).all(axis=axis, keepdims=True)
    y = jnp.where(is_zero, jnp.ones_like(x), x)
    if not keepdims:
        is_zero = is_zero.squeeze(axis=axis)
    n = jnp.linalg.norm(y, ord=ord, axis=axis, keepdims=keepdims)
    n = jnp.where(is_zero, fill, n)
    return n


def dfs_evaluate(basis, grid, t=None, z=None):
    """``DoubleFourierSeries.evaluate`` as on the branch, with its ``t``/``z`` override.

    The branch lets ``evaluate`` take the unique theta and zeta values as ``t``, ``z``
    in place of the grid's (``_singular_part`` evaluates the Phi basis at the polar
    nodes this way); master's ``evaluate`` has no such keywords. Values only (no
    derivatives), the full basis, a ``Grid`` argument: all that the vacuum pipeline
    needs.
    """
    import numpy as np
    from desc.basis import fourier

    try:
        zidx, zoutidx = grid.unique_zeta_idx, grid.inverse_zeta_idx
    except AttributeError:
        zidx = zoutidx = np.arange(grid.num_nodes)
    try:
        tidx, toutidx = grid.unique_poloidal_idx, grid.inverse_poloidal_idx
    except AttributeError:
        tidx = toutidx = np.arange(grid.num_nodes)
    _, tg, zg = grid.nodes.T
    _, m, n = basis.modes.T
    t = tg[tidx] if t is None else t
    z = zg[zidx] if z is None else z
    m, n = m[basis.unique_M_idx], n[basis.unique_N_idx]
    poloidal = fourier(t[:, np.newaxis], m, 1, 0)[:, basis.inverse_M_idx]
    toroidal = fourier(z[:, np.newaxis], n, basis.NFP, 0)[:, basis.inverse_N_idx]
    if grid.is_meshgrid and grid.num_rho == 1:
        return (poloidal[:, np.newaxis] * toroidal[np.newaxis]).reshape(
            -1, basis.num_modes, order="F"
        )
    return poloidal[toutidx] * toroidal[zoutidx]
