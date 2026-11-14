"""``apply`` and ``safenorm`` as on DESC branch ``ku/laplace``.

Copied from DESC (https://github.com/PlasmaControl/DESC), ``desc/utils.py``, unchanged
from branch ``ku/laplace`` (identical on branch ``ma/AGNI_ext``). Both differ from
master, and the copied code depends on the change:

* ``apply`` skips ``subset`` keys missing from the dict (master raises ``KeyError``;
  ``_prune_data`` asks for a kernel's optional ``eval_keys``);
* ``safenorm`` keeps the zero mask's shape under ``keepdims`` (master squeezes every
  singleton axis of it, which broadcasts a single source point to the wrong shape).

Branches of PlasmaControl/DESC quoted above:

    ku/laplace   f3b0c0de876bb5df75d7a24c2ca3d0276106e328  (Kaya Unalmis)
    ma/AGNI_ext  1583a921ae4d36171f7d970b2a56b81ae8132bcd  (Maya Avida)

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
