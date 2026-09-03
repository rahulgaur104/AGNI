"""Turning an eigenvector into something you can look at.

The solver returns a vector in **whitened, reduced, component-major**
coordinates: not a displacement, not on a grid, and with the Dirichlet degrees
of freedom removed. Everything here exists to undo that, in two stages that are
worth keeping separate:

* :func:`mode_components` undoes the whitening and the reduction and returns the
  three solver components on the ``(n_rho, n_theta, n_zeta)`` grid. This is an
  exact inverse -- ``agnimhd.solvers.from_phys`` takes it back.
* :func:`mode_displacement` and :func:`mode_speed` additionally undo the
  near-axis rescaling of the displacement, which is the step that needs the
  equilibrium and where a factor is easy to lose. Read their notes.

``matplotlib`` is **not** a dependency of this package -- the dependency set is
jax, numpy, scipy, matfree and nothing else. The array-returning functions here
therefore need nothing extra, and only the ``plot_*`` functions import
matplotlib, lazily, raising a named ``ImportError`` if it is absent::

    pip install "agnimhd[plot]"
"""

import numpy as np

from .backend import errorif
from .solvers import level_meta, to_phys

__all__ = [
    "mode_components",
    "mode_displacement",
    "mode_speed",
    "plot_mode_cross_section",
    "plot_radial_profile",
    "plot_spectrum",
]


def _matplotlib():
    """Import matplotlib, or explain what to install."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib, which agnimhd does not depend on. "
            "The package's dependency set is jax, numpy, scipy and matfree. "
            'Install it with `pip install "agnimhd[plot]"`. The array-returning '
            "functions in agnimhd.plotting work without it."
        ) from err
    return plt


def mode_components(op, v):
    """Undo the whitening and the reduction: solver vector -> grid components.

    Parameters
    ----------
    op : dict
        What :func:`agnimhd.assemble.matfree_operator` returns.
    v : array-like, shape (n_keep,)
        An eigenvector as returned by :func:`agnimhd.eigenpair`.

    Returns
    -------
    ndarray, shape (n_rho, n_theta, n_zeta, 3)
        The three solver components at each node, in the order
        ``(xi^rho_s, upsilon, xi^zeta)`` with ``upsilon = xi^theta - xi^zeta``.
        The Dirichlet-constrained ``xi^rho`` entries on the innermost and
        outermost radial shells come back as exact zeros, which is what they
        are.

    Notes
    -----
    These are the *solver's* variables, not the physical displacement. The
    change of variables that regularizes the magnetic axis has not been undone
    here; :func:`mode_displacement` does that.
    """
    meta = level_meta(op)
    v = np.asarray(v).reshape(-1)
    errorif(
        v.size != meta["n_keep"],
        ValueError,
        f"v has {v.size} entries but this operator has {meta['n_keep']} "
        "retained degrees of freedom. An eigenvector from a different "
        "resolution will reshape without complaining and mean nothing.",
    )
    return np.asarray(to_phys(meta, v))


def mode_displacement(eq, op, v):
    """The physical displacement components ``(xi^rho, xi^theta, xi^zeta)``.

    Undoes the near-axis rescaling as well as the whitening.

    Parameters
    ----------
    eq : EquilibriumData
    op : dict
    v : array-like, shape (n_keep,)

    Returns
    -------
    ndarray, shape (n_rho, n_theta, n_zeta, 3)

    Notes
    -----
    The solver works in rescaled variables to keep the near-axis terms finite:

    .. math::

        \\xi^\\rho = \\tilde\\xi^\\rho / \\psi',\\quad
        \\xi^\\theta = \\tilde\\xi^\\theta,\\quad
        \\xi^\\zeta = \\iota\\, \\tilde\\xi^\\zeta

    and carries ``upsilon = xi^theta - xi^zeta`` in place of ``xi^theta``. So

    .. math::

        \\tilde\\xi^\\rho = \\psi'\\, \\xi^{\\rho_s},\\quad
        \\tilde\\xi^\\theta = \\upsilon + \\xi^\\zeta,\\quad
        \\tilde\\xi^\\zeta = \\xi^\\zeta / \\iota

    which is the same grouping the kinetic-energy integral is written in, so it
    can be checked against that rather than taken on faith.

    **The overall scale is not physical.** An eigenvector has arbitrary
    normalization, and the operator is built from normalized fields, so treat
    the returned components as a shape. :func:`mode_speed` normalizes to its own
    maximum for exactly this reason.
    """
    comp = mode_components(op, v)
    shape = (eq.n_rho, eq.n_theta, eq.n_zeta)
    psi_r = np.asarray(eq.psi_r).reshape(shape)
    iota = np.asarray(eq.iota).reshape(shape)

    xi_rho = psi_r * comp[..., 0]
    xi_zeta = comp[..., 2] / iota
    xi_theta = comp[..., 1] + comp[..., 2]
    return np.stack([xi_rho, xi_theta, xi_zeta], axis=-1)


def mode_speed(eq, op, v, lam):
    """Normalized perturbed plasma speed ``|delta V| / max|delta V|``.

    The quantity plotted in the AGNI paper's eigenfunction comparison
    against NIMSTELL (its Eq. 58, Fig. 4):

    .. math::

        |\\delta V| = \\sqrt{|\\lambda|\\,
            g_{ab}\\, \\tilde\\xi^a \\tilde\\xi^b}

    contracting the displacement with the **covariant** PEST metric.

    Parameters
    ----------
    eq : EquilibriumData
    op : dict
    v : array-like, shape (n_keep,)
    lam : float
        The eigenvalue. Only ``|lam|`` is used.

    Returns
    -------
    ndarray, shape (n_rho, n_theta, n_zeta)
        Non-negative, with maximum exactly 1.

    Notes
    -----
    Returned **normalized to its own maximum**, because the absolute scale
    carries the eigenvector's arbitrary normalization and the operator's
    internal field normalization. Comparisons against another code -- which is
    what this is for -- are comparisons of shape.
    """
    xi = mode_displacement(eq, op, v)
    shape = (eq.n_rho, eq.n_theta, eq.n_zeta)

    def g(name):
        return np.asarray(getattr(eq, name)).reshape(shape)

    r, t, z = xi[..., 0], xi[..., 1], xi[..., 2]
    q = (
        g("g_rr") * r**2
        + g("g_vv") * t**2
        + g("g_pp") * z**2
        + 2.0 * (g("g_rv") * r * t + g("g_vp") * t * z + g("g_rp") * z * r)
    )
    speed = np.sqrt(np.abs(lam) * np.maximum(q, 0.0))
    peak = speed.max()
    return speed / peak if peak > 0 else speed


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_mode_cross_section(eq, op, v, lam, k=0, ax=None, **kwargs):
    """Filled contour of the normalized mode speed on one toroidal plane.

    Parameters
    ----------
    eq : EquilibriumData
    op : dict
    v : array-like
    lam : float
    k : int, optional
        Toroidal plane index. Default 0.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ``contourf``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Plotted in ``(theta, rho)``, not in ``(R, Z)``: the real-space positions of
    the nodes are not part of the ``EquilibriumData`` contract -- only the metric
    is. A caller that has ``R`` and ``Z`` can pass this array straight to its own
    plotter.
    """
    plt = _matplotlib()
    speed = mode_speed(eq, op, v, lam)[:, :, k]
    ax = ax or plt.gca()
    theta = np.linspace(0.0, 2.0 * np.pi, eq.n_theta, endpoint=False)
    rho = np.arange(eq.n_rho)
    kwargs.setdefault("levels", 21)
    cs = ax.contourf(theta, rho, speed, **kwargs)
    ax.set_xlabel("theta_PEST")
    ax.set_ylabel("radial node index")
    ax.set_title(f"|dV| / max, zeta plane {k},  lambda = {float(lam):+.4e}")
    ax.figure.colorbar(cs, ax=ax)
    return ax


def plot_radial_profile(eq, op, v, lam, ax=None, **kwargs):
    """Radial profile of the mode amplitude, maximized over both angles.

    The quickest check that a mode is what it claims to be: an interchange mode
    peaks at a resonant surface, and a mode that peaks at the boundary or at the
    innermost surface is usually telling you about the grid rather than the
    plasma.

    Parameters
    ----------
    eq, op, v, lam : see :func:`plot_mode_cross_section`
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ``plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _matplotlib()
    speed = mode_speed(eq, op, v, lam)
    ax = ax or plt.gca()
    ax.plot(np.arange(eq.n_rho), speed.max(axis=(1, 2)), **kwargs)
    ax.set_xlabel("radial node index")
    ax.set_ylabel("max |dV| / max")
    return ax


def plot_spectrum(eigenvalues, noise_floor=1e-10, ax=None, **kwargs):
    """Eigenvalues against the finite-precision floor.

    Parameters
    ----------
    eigenvalues : array-like
        Whatever a full or partial eigensolve returned.
    noise_floor : float, optional
        Absolute floor to draw, default 1e-10 -- ``eps * ||A_hat||_2`` for a
        typical stellarator, where ``||A_hat||_2 ~ 1e6``.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ``scatter``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    The band is the point of the plot. An eigenvalue inside it is not a small
    growth rate, it is roundoff: with the instability drive switched off, every
    eigenvalue of the paper's benchmark case falls below 4e-10.
    """
    plt = _matplotlib()
    lam = np.sort(np.asarray(eigenvalues).reshape(-1))
    ax = ax or plt.gca()
    ax.scatter(np.arange(lam.size), lam, s=8, **kwargs)
    ax.axhspan(-noise_floor, noise_floor, alpha=0.25, color="grey")
    ax.axhline(0.0, lw=0.8, color="k")
    ax.set_yscale("symlog", linthresh=noise_floor)
    ax.set_xlabel("index (ascending)")
    ax.set_ylabel("lambda   (negative = unstable)")
    return ax
