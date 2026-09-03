"""Getting from the solver's vector back to something physical.

The eigenvector comes out whitened, reduced and component-major. Every step
back to a displacement is a place to lose a factor, and a lost factor produces a
picture that still looks like a mode. So the transforms are tested as
mathematics -- exact round-trip, exact zeros where the boundary condition says
so, invariance under the eigenvector's arbitrary scale -- rather than by
checking that a plot was produced.

matplotlib is not a dependency. Everything here that needs it is skipped when it
is absent; nothing that needs it is correctness-critical, and the array-level
functions, which are, run unconditionally.
"""

import numpy as np
import pytest

from agnimhd import eigenpair
from agnimhd.assemble import matfree_operator
from agnimhd.backend import jnp
from agnimhd.plotting import (
    cross_section_planes,
    mode_components,
    mode_delta_v,
    mode_displacement,
    mode_plot_displacement,
    mode_speed,
    plot_eigenfunction_cross_sections,
    plot_mode_cross_section,
    plot_radial_profile,
    plot_spectrum,
)
from agnimhd.solvers import from_phys, level_meta


def _mpl():
    """Return pyplot, or skip -- matplotlib is an optional extra, not a dep."""
    plt = pytest.importorskip(
        "matplotlib.pyplot",
        reason="matplotlib is an optional extra (agnimhd[plot]); the "
        "array-returning functions are tested without it",
    )
    return plt


@pytest.fixture(scope="module")
def solved(eq_data, diffmat, config):
    """(op, lam, v) for the shipped case. One eigensolve for the module."""
    op = matfree_operator(eq_data, diffmat, config)
    lam, v, _ = eigenpair(eq_data, diffmat, config)
    return op, float(lam), np.asarray(v)


@pytest.fixture(scope="module")
def axisym_solved(axisym_case):
    """(eq, op, lam, v) for an actual complex axisymmetric eigenfunction."""
    eq, diffmat, config = axisym_case
    op = matfree_operator(eq, diffmat, config)
    lam, v, _ = eigenpair(eq, diffmat, config)
    return eq, op, float(lam), np.asarray(v)


def _toy_rz(eq):
    """Simple real-space geometry with the same tensor shape as ``eq``."""
    rho = np.linspace(0.05, 1.0, eq.n_rho)
    theta = np.linspace(0.0, 2.0 * np.pi, eq.n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / eq.NFP, eq.n_zeta, endpoint=False)
    rr, tt, zz = np.meshgrid(rho, theta, zeta, indexing="ij")
    shaping = 1.0 + 0.08 * np.cos(eq.NFP * zz)
    R = 10.0 + shaping * rr * np.cos(tt)
    Z = (1.0 - 0.06 * np.cos(eq.NFP * zz)) * rr * np.sin(tt)
    return R, Z


# ---------------------------------------------------------------------------
# The transforms
# ---------------------------------------------------------------------------


def test_components_round_trip_exactly(eq_data, solved):
    """``mode_components`` is invertible, and ``from_phys`` is the inverse.

    Not "close to" invertible: the whitening is a per-node 3x3 triangular solve
    whose inverse is formed once, so the round-trip should return the same
    vector to roundoff on the vector's own scale.
    """
    op, _, v = solved
    comp = mode_components(op, v)
    assert comp.shape == (*eq_data.resolution, 3)
    back = np.asarray(from_phys(level_meta(op), jnp.asarray(comp)))
    err = np.max(np.abs(back - v)) / np.max(np.abs(v))
    assert err < 1e-12, f"round trip lost {err:.3e} relative"


def test_the_dirichlet_shells_come_back_as_exact_zeros(eq_data, solved):
    """``xi^rho`` is constrained to zero on the first and last radial shells.

    Exact zeros, not small numbers: those degrees of freedom are not in the
    solve at all, so anything nonzero there would mean the scatter is writing to
    the wrong indices -- which would otherwise show up only as a mode that looks
    slightly wrong near the axis.
    """
    op, _, v = solved
    comp = mode_components(op, v)
    assert np.all(comp[0, :, :, 0] == 0.0)
    assert np.all(comp[-1, :, :, 0] == 0.0)
    # The two tangential components are free everywhere, including there.
    assert np.max(np.abs(comp[0, :, :, 1:])) > 0.0


def test_a_wrong_length_vector_is_refused(eq_data, solved):
    """An eigenvector from another resolution would reshape and mean nothing."""
    op, _, v = solved
    with pytest.raises(ValueError, match="retained degrees of freedom"):
        mode_components(op, v[:-1])


def test_displacement_undoes_the_axis_rescaling(eq_data, solved):
    """``xi_tilde`` is the solver's components with Eq. (21) undone.

    Checked against the definition rather than against a stored array, because
    the definition is the thing that is easy to get wrong: the solver carries
    ``upsilon = xi^theta - xi^zeta``, so recovering ``xi^theta`` is an addition,
    not a copy.
    """
    op, _, v = solved
    comp = mode_components(op, v)
    xi = mode_displacement(eq_data, op, v)
    shape = eq_data.resolution
    psi_r = np.asarray(eq_data.psi_r).reshape(shape)
    iota = np.asarray(eq_data.iota).reshape(shape)

    assert np.allclose(xi[..., 0], psi_r * comp[..., 0], rtol=0, atol=0)
    assert np.allclose(xi[..., 1], comp[..., 1] + comp[..., 2], rtol=0, atol=0)
    assert np.allclose(xi[..., 2], comp[..., 2] / iota, rtol=0, atol=0)


def test_speed_is_normalized_nonnegative_and_scale_invariant(eq_data, solved):
    """The plotted quantity is a shape, so it must not depend on ``|v|``.

    An eigenvector's normalization is arbitrary. If doubling it changed the
    picture, every comparison against another code would be meaningless.
    """
    op, lam, v = solved
    s1 = mode_speed(eq_data, op, v, lam)
    s2 = mode_speed(eq_data, op, -3.7 * v, lam)

    assert s1.shape == eq_data.resolution
    assert np.all(s1 >= 0.0)
    assert np.isclose(s1.max(), 1.0)
    assert np.allclose(
        s1, s2, rtol=1e-12, atol=1e-12
    ), "the normalized mode speed depends on the eigenvector's scale"


def test_axisymmetric_plot_fields_are_real(axisym_solved):
    """Complex axisymmetric modes are phase-projected before plotting."""
    eq, op, lam, v = axisym_solved
    xi = mode_plot_displacement(eq, op, v)
    delta_v = mode_delta_v(eq, op, v, lam)
    speed = mode_speed(eq, op, v, lam)

    assert not np.iscomplexobj(xi)
    assert not np.iscomplexobj(delta_v)
    assert not np.iscomplexobj(speed)
    assert np.all(np.isfinite(speed))
    assert np.isclose(speed.max(), 1.0)


def test_speed_is_the_metric_contraction(eq_data, solved):
    """``|dV|`` really is ``sqrt(|lambda| g_ab xi^a xi^b)``, up to the norm.

    Recomputed here from the covariant metric the long way. The cross terms
    carry a factor 2 and dropping them is the classic error -- it leaves a
    quantity that is still positive, still peaked in the right place, and wrong.
    """
    op, lam, v = solved
    xi = mode_displacement(eq_data, op, v)
    shape = eq_data.resolution
    G = {
        k: np.asarray(getattr(eq_data, k)).reshape(shape)
        for k in ("g_rr", "g_vv", "g_pp", "g_rv", "g_vp", "g_rp")
    }
    r, t, z = xi[..., 0], xi[..., 1], xi[..., 2]
    q = (
        G["g_rr"] * r**2
        + G["g_vv"] * t**2
        + G["g_pp"] * z**2
        + 2 * (G["g_rv"] * r * t + G["g_vp"] * t * z + G["g_rp"] * z * r)
    )
    want = np.sqrt(abs(lam) * np.maximum(q, 0.0))
    want = want / want.max()
    assert np.allclose(mode_speed(eq_data, op, v, lam), want, rtol=1e-12)


def test_the_metric_quadratic_form_is_positive(eq_data, solved):
    """The covariant metric contraction is a norm, so it cannot be negative.

    If it were, either the metric components are not a consistent covariant set
    or the components have been paired with the wrong ones -- both of which the
    clamp inside ``mode_speed`` would otherwise hide.
    """
    op, lam, v = solved
    xi = mode_displacement(eq_data, op, v)
    shape = eq_data.resolution
    G = {
        k: np.asarray(getattr(eq_data, k)).reshape(shape)
        for k in ("g_rr", "g_vv", "g_pp", "g_rv", "g_vp", "g_rp")
    }
    r, t, z = xi[..., 0], xi[..., 1], xi[..., 2]
    q = (
        G["g_rr"] * r**2
        + G["g_vv"] * t**2
        + G["g_pp"] * z**2
        + 2 * (G["g_rv"] * r * t + G["g_vp"] * t * z + G["g_rp"] * z * r)
    )
    assert q.min() >= -1e-12 * q.max(), (
        f"the metric quadratic form went negative: min {q.min():.3e} against "
        f"max {q.max():.3e}"
    )


# ---------------------------------------------------------------------------
# The plots
# ---------------------------------------------------------------------------


def test_plots_draw_something(eq_data, solved):
    """Each plot function returns an axes carrying data. Smoke, deliberately."""
    plt = _mpl()
    op, lam, v = solved
    R, Z = _toy_rz(eq_data)
    fig, axes = plt.subplots(1, 3)

    ax = plot_mode_cross_section(eq_data, op, v, lam, R, Z, k=0, ax=axes[0])
    assert ax.collections, "cross-section drew nothing"

    ax = plot_radial_profile(eq_data, op, v, lam, ax=axes[1])
    assert ax.lines, "radial profile drew nothing"
    ydata = ax.lines[0].get_ydata()
    assert np.isclose(ydata.max(), 1.0), "the profile is not the normalized speed"

    ax = plot_spectrum([lam, -1e-12, 1e-3], ax=axes[2])
    assert ax.collections, "spectrum drew nothing"
    assert ax.get_yscale() == "symlog"
    plt.close(fig)


def test_cross_section_plane_defaults(eq_data, axisym_solved):
    """3D draws zeta=0 and pi/NFP; axisym draws only zeta=0."""
    axisym_eq, *_ = axisym_solved
    assert cross_section_planes(axisym_eq) == [(0, "zeta = 0")]
    assert cross_section_planes(eq_data) == [
        (0, "zeta = 0"),
        (eq_data.n_zeta // 2, "zeta = pi / NFP"),
    ]


def test_eigenfunction_cross_sections_draw_requested_planes(eq_data, solved):
    """The DESC-style plot has one row per requested 3D cross-section."""
    plt = _mpl()
    op, lam, v = solved
    R, Z = _toy_rz(eq_data)
    fig, axes = plot_eigenfunction_cross_sections(eq_data, op, v, lam, R, Z)

    assert axes.shape == (2, 4)
    for ax in axes.reshape(-1):
        assert ax.collections, "eigenfunction cross-section drew nothing"
    plt.close(fig)


def test_axisymmetric_eigenfunction_cross_section_draws_zeta_zero(axisym_solved):
    """The axisymmetric DESC-style plot has only the zeta=0 row."""
    plt = _mpl()
    eq, op, lam, v = axisym_solved
    R, Z = _toy_rz(eq)
    fig, axes = plot_eigenfunction_cross_sections(eq, op, v, lam, R, Z)

    assert axes.shape == (1, 4)
    for ax in axes.reshape(-1):
        assert ax.collections, "axisymmetric cross-section drew nothing"
    plt.close(fig)


def test_plotting_import_does_not_require_matplotlib():
    """Importing the module must work in the four-dependency environment.

    The array functions are usable without matplotlib, so a module-level import
    of it would make an optional extra mandatory for everyone.
    """
    import importlib
    import sys

    mod = importlib.reload(sys.modules["agnimhd.plotting"])
    assert hasattr(mod, "mode_speed")
