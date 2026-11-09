"""AGNI's copy of DESC's boundary-integral machinery (``agnimhd.adapters._vacuum``).

Ported from the ``TestSingularities`` cases of the DESC branch ``ma/AGNI_ext``
(``tests/test_integrals.py``) that cover the copied code: the FFT and DFT interpolators,
their selection, ``jitable``, and the double-layer kernel through ``singular_integral``.
Needs DESC (master is enough); skipped where it is absent. DESC is reached through
``pytest.importorskip`` only: the test suite may not ``import desc`` (see
``tools/check_no_desc.sh``).
"""

import warnings
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

dgrid = pytest.importorskip("desc.grid")
Grid, LinearGrid = dgrid.Grid, dgrid.LinearGrid

from agnimhd.adapters._vacuum.polar_quadrature import (  # noqa: E402
    _best_params,
    _best_ratio,
    _get_polar_quadrature,
)
from agnimhd.adapters._vacuum.singularities import (  # noqa: E402
    DFTInterpolator,
    FFTInterpolator,
    get_interpolator,
    singular_integral,
)


def _c_2d(x, y):
    """Non-separable test function (DESC's ``tests/test_interp_utils.py``)."""
    return (
        jnp.cos(3 * x) * jnp.sin(2 * x + y)
        + jnp.sin(5 * y) * jnp.cos(x + 3 * y)
        - 33.2
        + jnp.cos(x)
        + jnp.cos(y)
    )


@pytest.mark.parametrize(
    "interpolator, settings",
    product(
        [FFTInterpolator, DFTInterpolator],
        [
            (_c_2d, 5, 8, 1),  # the Nyquist frequencies of _c_2d
            (lambda x, y: np.cos(6 * x) ** 2 + np.sin(90 * y) + 1, 12, 3, 30),
        ],
    ),
)
def test_interpolators_reproduce_a_resolved_function(interpolator, settings):
    """At every polar node, the interpolant equals the function itself."""
    f, M, N, NFP = settings
    s_grid = LinearGrid(M=M, N=N, NFP=NFP)
    h_t = 2 * np.pi / s_grid.num_theta
    h_z = 2 * np.pi / s_grid.num_zeta / s_grid.NFP
    st, sz, q = 3, 5, 4
    r, w, _, _ = _get_polar_quadrature(q)
    dt = st / 2 * h_t * r * np.sin(w)
    dz = sz / 2 * h_z * r * np.cos(w)
    e_grid = LinearGrid(
        theta=s_grid.num_theta // 2 + 1, zeta=s_grid.num_zeta // 2 + 1, NFP=NFP
    )
    interp = interpolator(e_grid, s_grid, st, sz, q)
    theta, zeta = e_grid.nodes[:, 1], e_grid.nodes[:, 2]
    _f = f(s_grid.nodes[:, 1], s_grid.nodes[:, 2])
    for i in range(dt.size):
        np.testing.assert_allclose(interp(_f, i), f(theta + dt[i], zeta + dz[i]))


def _tensor_grid(nt, nz, nfp, nodes=None, rho=1.0):
    """A jitable copy of a uniform (theta, zeta) grid, with its index arrays."""
    g = LinearGrid(rho=rho, theta=nt, zeta=nz, NFP=nfp, sym=False)
    n = nt * nz
    return Grid(
        nodes=g.nodes if nodes is None else nodes,
        jitable=True,
        NFP=nfp,
        _unique_rho_idx=np.array([0]),
        _unique_poloidal_idx=np.asarray(g.unique_poloidal_idx),
        _unique_zeta_idx=np.asarray(g.unique_zeta_idx),
        _inverse_rho_idx=np.zeros(n, dtype=int),
        _inverse_poloidal_idx=np.asarray(g.inverse_poloidal_idx),
        _inverse_zeta_idx=np.asarray(g.inverse_zeta_idx),
    )


@pytest.mark.parametrize("cls", [FFTInterpolator, DFTInterpolator])
def test_interpolators_jitable(cls):
    """``jitable=True`` builds on traced grid nodes; the default refuses them."""
    nfp = 3
    source = LinearGrid(rho=1.0, theta=24, zeta=16, NFP=nfp, sym=False)
    nodes = jnp.asarray(_tensor_grid(12, 8, nfp).nodes)

    def build(x, **kw):
        return cls(_tensor_grid(12, 8, nfp, nodes=x), source, 12, 7, 9, **kw).ht

    ht = jax.jit(lambda x: build(x, jitable=True))(nodes)
    np.testing.assert_allclose(ht, 2 * np.pi / source.num_theta)
    with pytest.raises(Exception):
        jax.jit(build)(nodes)


def test_interpolators_refuse_a_different_surface():
    """Grids on two different surfaces are refused, unless the caller vouches."""
    source = LinearGrid(rho=1.0, theta=24, zeta=16, NFP=3, sym=False)
    other = LinearGrid(rho=0.5, theta=24, zeta=16, NFP=3, sym=False)
    with pytest.raises(AssertionError):
        DFTInterpolator(other, source, 12, 7, 9)
    DFTInterpolator(other, source, 12, 7, 9, jitable=True)


def test_get_interpolator_selection():
    """FFT for a uniform tensor eval grid, else DFT; no auto-selection when jitable."""
    nfp = 3
    source = LinearGrid(rho=1.0, theta=24, zeta=16, NFP=nfp, sym=False)
    separable = LinearGrid(rho=1.0, theta=12, zeta=8, NFP=nfp, sym=False)
    data = {}  # st, sz, q are given, so the heuristic needs no geometry
    f = get_interpolator(separable, source, data, st=12, sz=7, q=9, warn_fft=False)
    assert isinstance(f, FFTInterpolator)
    # theta shifted differently on every zeta plane: not a tensor product
    nodes = np.asarray(separable.nodes).copy()
    nodes[:, 1] = (nodes[:, 1] + 0.05 * np.sin(nfp * nodes[:, 2])) % (2 * np.pi)
    skewed = Grid(nodes, NFP=nfp, sort=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = get_interpolator(skewed, source, data, st=12, sz=7, q=9)
    assert isinstance(f, DFTInterpolator)
    with pytest.raises(ValueError, match="cannot auto-select"):
        get_interpolator(separable, source, data, st=12, sz=7, q=9, jitable=True)


@pytest.mark.slow
def test_double_layer_of_a_constant_is_minus_one_half():
    """Green's identity: on the surface, the double layer of Phi = 1 is -1/2.

    DESC's ``test_singular_integral_greens_id`` checks ``int n.(x-y)/|x-y|^3 da =
    -2 pi`` with its ``nr_over_r3`` kernel; ``_kernel_dipole`` is that kernel over
    ``4 pi`` times Phi, so the same cases and errors (over ``4 pi``) apply. The errors
    are Dhairya Malhotra's convergence table, quoted in the DESC test.
    """
    Equilibrium = pytest.importorskip("desc.equilibrium").Equilibrium
    eq = Equilibrium()  # circular-cross-section torus
    keys = ["R", "phi", "Z", "e_theta", "e_zeta"]  # all on DESC master
    for Nu, Nv, s, q, err in [
        (13, 30, 13, 10, 0.4),
        (13, 45, 13, 12, 2e-2),
        (13, 60, 13, 12, 3e-3),
        (19, 90, 19, 16, 5e-5),
    ]:
        grid = LinearGrid(M=Nu // 2, N=Nv // 2, NFP=eq.NFP)
        data = dict(eq.compute(keys, grid=grid))
        data["e_theta x e_zeta"] = jnp.cross(data["e_theta"], data["e_zeta"])
        data["|e_theta x e_zeta|"] = jnp.linalg.norm(data["e_theta x e_zeta"], axis=-1)
        data["Phi (periodic)"] = jnp.ones(grid.num_nodes)
        interp = FFTInterpolator(grid, grid, s, s, q)
        D1 = singular_integral(data, data, interp, kernel="dipole", chunk_size=50)
        np.testing.assert_array_less(np.abs(0.5 + np.ravel(D1)), err / (4 * np.pi))
    # the support heuristic on the same surface also gets there
    st, sz, q = _best_params(grid, _best_ratio(data))
    assert min(st, sz, q) > 0
