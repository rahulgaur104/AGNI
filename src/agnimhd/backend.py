"""JAX aliasing and small validation helpers.

This module is the package's single point of contact with JAX. It exists so
that no other module has to decide how to import ``jax``, and so the float64
requirement is stated once, loudly, at import time.

``agnimhd`` requires 64-bit floats. The eigenvalues it computes are small
(``|lambda| ~ 1e-4`` on the shipped case, with an absolute noise floor around
``1e-10``), and the assembled operator is formed by differencing large terms, so
float32 does not produce a less accurate growth rate -- it produces a different
mode. Rather than let that happen silently, enabling ``jax_enable_x64`` is done
here at import.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

__all__ = [
    "check_posint",
    "errorif",
    "jax",
    "jit",
    "jnp",
    "warnif",
]

jit = jax.jit


def errorif(cond, err=ValueError, msg=""):
    """Raise ``err(msg)`` if ``cond`` is true.

    Reimplemented rather than imported from DESC: ``agnimhd`` must not depend on
    DESC in any direction. See ``docs/adapters.md``.

    Parameters
    ----------
    cond : bool
        Condition to test. Must be a concrete Python bool, not a traced array.
    err : type
        Exception class to raise.
    msg : str
        Message for the exception.

    Raises
    ------
    err
        If ``cond`` is true.
    """
    if cond:
        raise err(msg)


def warnif(cond, err=UserWarning, msg=""):
    """Emit ``err(msg)`` as a warning if ``cond`` is true.

    Parameters
    ----------
    cond : bool
        Condition to test.
    err : type
        Warning class.
    msg : str
        Message for the warning.
    """
    if cond:
        import warnings

        warnings.warn(msg, err)


def check_posint(x, name="", allow_none=True):
    """Return ``x`` as a positive ``int``, or raise.

    Parameters
    ----------
    x : int or None
        Value to check.
    name : str
        Name used in the error message.
    allow_none : bool
        Whether ``None`` is an acceptable value.

    Returns
    -------
    int or None
        ``int(x)``, or ``None`` when ``x`` is ``None`` and ``allow_none``.

    Raises
    ------
    TypeError
        If ``x`` is not an integer.
    ValueError
        If ``x`` is not positive, or is ``None`` when ``allow_none`` is false.
    """
    if x is None:
        if allow_none:
            return x
        raise ValueError(f"{name} cannot be None.")
    if int(x) != x:
        raise TypeError(f"{name} should be an integer, got {x}.")
    if x <= 0:
        raise ValueError(f"{name} should be positive, got {x}.")
    return int(x)
