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
    "cross_section_planes",
    "mode_components",
    "mode_delta_v",
    "mode_displacement",
    "mode_plot_displacement",
    "mode_speed",
    "plot_eigenfunction_cross_sections",
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


def mode_plot_displacement(eq, op, v):
    """Real displacement components for plotting, with DESC's phase convention.

    Axisymmetric finite-``n`` solves are complex Hermitian: the eigenvector is a
    complex amplitude for one toroidal harmonic. Matplotlib will accept such an
    array and silently cast it to real, which makes a plot that looks polished
    and may be physically meaningless. DESC's plotting script fixes the
    arbitrary complex phase first and then plots one real phase of the mode; this
    helper does the same.

    Parameters
    ----------
    eq : EquilibriumData
    op : dict
    v : array-like

    Returns
    -------
    ndarray, shape (n_rho, n_theta, n_zeta, 3)
        Real-valued displacement components.
    """
    xi = mode_displacement(eq, op, v)
    if not np.iscomplexobj(xi):
        return np.real(xi)

    xi_ref = xi[..., 0].reshape(-1)
    rot = np.exp(1j * np.arctan2(np.mean(xi_ref.real), np.mean(xi_ref.imag)))
    return np.asarray((xi * rot).imag, dtype=float)


def mode_delta_v(eq, op, v, lam, *, normalize=False):
    """Perturbed plasma speed ``|delta V|`` using the plotted real phase.

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
        Real, non-negative perturbed speed. With ``normalize=True`` the maximum
        is exactly 1 unless the field is identically zero.

    Notes
    -----
    The complex phase projection follows DESC's plotting implementation before
    the metric contraction. This prevents a complex mode from being handed to
    matplotlib and cast implicitly.
    """
    xi = mode_plot_displacement(eq, op, v)
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
    speed = np.asarray(speed, dtype=float)
    if not normalize:
        return speed
    peak = speed.max()
    return speed / peak if peak > 0 else speed


def mode_speed(eq, op, v, lam):
    """Normalized perturbed plasma speed ``|delta V| / max|delta V|``.

    This is the DESC-style plotted field returned by :func:`mode_delta_v` with
    ``normalize=True``. The result is always real-valued; complex axisymmetric
    modes are phase-projected explicitly before the metric contraction.
    """
    return mode_delta_v(eq, op, v, lam, normalize=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _field_shape(eq):
    """Return the tensor-product node shape."""
    return (eq.n_rho, eq.n_theta, eq.n_zeta)


def _as_node_field(eq, name, value):
    """Return ``value`` as ``(n_rho, n_theta, n_zeta)`` or raise clearly."""
    arr = np.asarray(value)
    shape = _field_shape(eq)
    if arr.shape == shape:
        return arr
    if arr.size == eq.n_nodes:
        return arr.reshape(shape)
    raise ValueError(
        f"{name} has shape {arr.shape}, but expected {shape} or "
        f"{eq.n_nodes} flattened node values."
    )


def _wrap_poloidal(field):
    """Close the poloidal contour at theta = 2*pi."""
    return np.concatenate([field, field[:, :1]], axis=1)


def _normalize_signed(field):
    """Normalize a real field by max(abs(field)), preserving sign."""
    field = np.asarray(np.real(field), dtype=float)
    peak = np.nanmax(np.abs(field))
    return field / (peak + 1e-300)


def cross_section_planes(eq):
    """Toroidal planes used by DESC-style cross-section plots.

    Axisymmetric cases have one physical plane, ``zeta = 0``. Three-dimensional
    cases plot the two most useful cross-sections, ``zeta = 0`` and
    ``zeta = pi / NFP``; on the tensor grid the latter is the halfway toroidal
    index.

    Returns
    -------
    list of tuple
        ``[(plane_index, label), ...]``.
    """
    if eq.n_zeta == 1:
        return [(0, "zeta = 0")]
    return [(0, "zeta = 0"), (eq.n_zeta // 2, "zeta = pi / NFP")]


def plot_mode_cross_section(eq, op, v, lam, R, Z, k=0, ax=None, **kwargs):
    """Filled contour of normalized ``deltaV`` on one real-space cross-section.

    Parameters
    ----------
    eq : EquilibriumData
    op : dict
    v : array-like
    lam : float
    R, Z : array-like, shape (n_rho, n_theta, n_zeta)
        Real-space coordinates for the same PEST nodes as ``eq``.
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
    This follows DESC's eigenfunction plotting convention: close the poloidal
    contour, plot against ``(R, Z)``, and only pass real-valued fields to
    matplotlib.
    """
    plt = _matplotlib()
    R = _as_node_field(eq, "R", R)
    Z = _as_node_field(eq, "Z", Z)
    speed = mode_speed(eq, op, v, lam)[:, :, k]
    ax = ax or plt.gca()
    kwargs.setdefault("levels", 60)
    kwargs.setdefault("cmap", "RdBu_r")
    cs = ax.contourf(
        _wrap_poloidal(R[:, :, k]),
        _wrap_poloidal(Z[:, :, k]),
        _wrap_poloidal(speed),
        **kwargs,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(f"deltaV, zeta plane {k}, lambda={float(lam):+.3e}")
    ax.figure.colorbar(cs, ax=ax)
    return ax


def plot_eigenfunction_cross_sections(
    eq, op, v, lam, R, Z, planes=None, *, figsize=None, **kwargs
):
    """Plot DESC-style eigenfunction cross-sections.

    The figure contains normalized ``deltaV`` and the three normalized
    displacement components ``xi^rho``, ``xi^theta`` and ``xi^zeta``. Axisymmetric
    cases default to ``zeta = 0``. Three-dimensional cases default to
    ``zeta = 0`` and ``zeta = pi / NFP``.

    Parameters
    ----------
    eq, op, v, lam : see :func:`plot_mode_cross_section`
    R, Z : array-like
        Real-space coordinates on the same nodes as ``eq``.
    planes : sequence, optional
        Plane indices, or ``(index, label)`` pairs. Defaults to
        :func:`cross_section_planes`.
    figsize : tuple, optional
        Matplotlib figure size.
    **kwargs
        Passed to ``contourf``.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """
    plt = _matplotlib()
    R = _as_node_field(eq, "R", R)
    Z = _as_node_field(eq, "Z", Z)
    if planes is None:
        planes = cross_section_planes(eq)
    else:
        planes = [(p, f"zeta plane {p}") if np.isscalar(p) else p for p in planes]

    delta_v = mode_speed(eq, op, v, lam)
    xi = mode_plot_displacement(eq, op, v)
    fields = [
        ("deltaV", delta_v),
        (r"$\xi^\rho$", xi[..., 0]),
        (r"$\xi^\theta$", xi[..., 1]),
        (r"$\xi^\zeta$", xi[..., 2]),
    ]
    nrows = len(planes)
    ncols = len(fields)
    figsize = figsize or (4.6 * ncols, 4.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    kwargs.setdefault("levels", 60)
    kwargs.setdefault("cmap", "RdBu_r")

    for row, (k, label) in enumerate(planes):
        for col, (title, field) in enumerate(fields):
            ax = axes[row, col]
            plot_field = _normalize_signed(field[:, :, k])
            cs = ax.contourf(
                _wrap_poloidal(R[:, :, k]),
                _wrap_poloidal(Z[:, :, k]),
                _wrap_poloidal(plot_field),
                **kwargs,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            ax.set_title(f"{title}, {label}")
            fig.colorbar(cs, ax=ax)

    fig.suptitle(f"lambda = {float(lam):+.6e}")
    fig.tight_layout()
    return fig, axes


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
