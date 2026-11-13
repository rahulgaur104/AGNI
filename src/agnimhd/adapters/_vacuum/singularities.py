"""High order method for singular surface integrals, from Malhotra 2019.

Copied from DESC (https://github.com/PlasmaControl/DESC),
``desc/integrals/singularities.py``, where it is not on master: master's module has
no kernels and its integrals take one density column, not ``N_modes`` at once. What
master does have, and gives the same result for here, is imported from it: the
interpolator base class ``_BIESTInterpolator``, ``_chi``, ``_eta`` and
``_get_quadrature_nodes``. By source:

* unchanged from branch ``ku/laplace``: ``_prune_data``, ``_nonsingular_part``,
  ``_add_reduce``, ``singular_integral``, ``_dx``, ``_G``, ``_grad_G``, and
  ``nfp_loop`` (a nested function of master's ``_nonsingular_part``, a module
  function on the branch);
* as modified on branch ``ma/AGNI_ext``: ``_kernel_monopole`` taking several
  ``B0*n`` columns at once; the kernels' ``Phi_tilde`` key renamed
  ``Phi (periodic)``;
* changed here: ``FFTInterpolator`` subclasses master's with ``ku/laplace``'s
  interpolation (see its docstring); ``known_map`` removed from ``singular_integral``
  and ``_singular_part``, where ``ku/laplace`` uses it to evaluate a known function
  at the polar nodes rather than interpolate it (the only user here is the Phi
  basis, which the interpolation reproduces exactly); trimmed to what
  ``phi_matrix_pest`` uses.

Branches of PlasmaControl/DESC quoted above:

    ku/laplace   f3b0c0de876bb5df75d7a24c2ca3d0276106e328  (Kaya Unalmis)
    ma/AGNI_ext  1583a921ae4d36171f7d970b2a56b81ae8132bcd  (Maya Avida)

DESC is MIT licensed:

    Copyright (c) 2020 Daniel Dudt, Rory Conlin, Dario Panici, Kaya Unalmis,
    Yigit Gunsur Elmacioglu, Rahul Gaur, Patrick Kim, Egemen Kolemen
"""

from desc.backend import fori_loop, jnp
from desc.batching import batch_map, vmap_chunked
from desc.integrals.singularities import FFTInterpolator as _FFTInterpolatorMaster
from desc.integrals.singularities import _BIESTInterpolator
from desc.integrals.singularities import _chi as chi
from desc.integrals.singularities import _eta as eta
from desc.integrals.singularities import _get_quadrature_nodes
from desc.utils import (
    dot,
    errorif,
    parse_argname_change,
    rpz2xyz,
    rpz2xyz_vec,
    safediv,
    setdefault,
    warnif,
    xyz2rpz_vec,
)
from interpax_fft import rfft_interp2d

from ._utils import apply, safenorm


class FFTInterpolator(_FFTInterpolatorMaster):
    """DESC master's ``FFTInterpolator``, interpolating with the source's full spectrum.

    Master's ``__call__`` goes through ``fft_interp2d``, which truncates the source
    spectrum to the eval grid's: with a source grid finer than the eval grid (the
    point of refining the quadrature) that changes ``phi_matrix`` by 2-3 % on the
    QH test case. This one, as on branch ``ku/laplace``, interpolates through
    ``rfft_interp2d`` with the source's full spectrum, and adds the ``eval_grid`` /
    ``source_grid`` properties the integrals read. Everything else, including the
    checks in the constructor, is master's ``_BIESTInterpolator``.

    Parameters
    ----------
    eval_grid, source_grid : Grid
        Evaluation and source points for the integral transform.
        Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
    st, sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.
    q : int
        Order of quadrature in polar domain.
    warn_fft : bool
        Whether to warn if the interpolation will be lossy. Default is ``True``.

    """

    def __init__(self, eval_grid, source_grid, st, sz, q, *, warn_fft=True):
        assert eval_grid.can_fft2, "Got False for eval_grid.can_fft2."
        warnif(
            warn_fft and eval_grid.num_theta < (source_grid.num_theta // 2 + 1),
            msg="Frequency spectrum of FFT interpolation will be truncated.\n"
            f"Got eval_grid.num_theta = {eval_grid.num_theta} < "
            f"{source_grid.num_theta // 2 + 1} = source_grid.num_theta // 2 + 1.",
        )
        warnif(
            warn_fft and eval_grid.num_zeta < (source_grid.num_zeta // 2 + 1),
            msg="Frequency spectrum of FFT interpolation will be truncated.\n"
            f"Got eval_grid.num_zeta = {eval_grid.num_zeta} < "
            f"{source_grid.num_zeta // 2 + 1} = source_grid.num_zeta // 2 + 1.",
        )
        # master's base constructor; skips master FFTInterpolator's own warnings,
        # which describe its truncating interpolation
        _BIESTInterpolator.__init__(self, eval_grid, source_grid, st, sz, q)

    @property
    def eval_grid(self):
        """Grid : Evaluation points."""
        return self._eval_grid

    @property
    def source_grid(self):
        """Grid : Source (quadrature) points."""
        return self._source_grid

    def fourier(self, f):
        """Return ``f`` on the source grid as ``__call__`` expects it."""
        return self._source_grid.meshgrid_reshape(f, "rtz")[0]

    def __call__(self, f, i, *, is_fourier=False, vander=None):
        """Interpolate ``f`` to polar node ``i`` around evaluation grid.

        Parameters
        ----------
        f : ndarray
            Data at source grid points to interpolate.
        i : int
            Index of polar node.
        is_fourier : bool
            Whether ``f`` is already ``self.fourier(f)``. Default is false.

        Returns
        -------
        fi : ndarray
            Source data interpolated to ith polar node.

        """
        if not is_fourier:
            f = self.fourier(f)
        return rfft_interp2d(
            f,
            n1=self._eval_grid.num_theta,
            n2=self._eval_grid.num_zeta,
            sx=self._shift_t[i],
            sy=self._shift_z[i],
            dx=self._ht,
            dy=self._hz,
        ).reshape(self._eval_grid.num_nodes, *f.shape[2:], order="F")


def nfp_loop(source_grid, func, init_val):
    """Calculate effects from source points on a single field period.

    The integral is computed on the full domain because the kernels of interest
    have toroidal variation and are not NFP periodic. To that end, the integral
    is computed on every field period and summed. The ``source_grid`` is the
    first field period because DESC truncates the computational domain to
    ζ ∈ [0, 2π/grid.NFP) and changes variables to the spectrally condensed
    ζ* = basis.NFP ζ. The domain is shifted to the next field period by
    incrementing the toroidal coordinate of the grid by 2π/NFP.

    For an axisymmetric configuration, it is most efficient for ``source_grid`` to
    be a single toroidal cross-section. To capture toroidal effects of the kernels
    on those grids for axisymmetric configurations, we set a dummy value for NFP to
    an integer larger than 1 so that the toroidal increment can move to a new spot.

    Parameters
    ----------
    source_grid : _Grid
        Grid with points ζ ∈ [0, 2π/grid.NFP).
    func : callable
        Should accept argument ``zeta_j`` denoting toroidal coordinates of
        field period ``j``.
    init_val : jnp.ndarray
        Initial loop carry value.

    Returns
    -------
    result : jnp.ndarray
        Shape is ``init_val.shape``.

    """
    errorif(
        source_grid.num_zeta == 1 and source_grid.NFP == 1,
        msg="Source grid cannot compute toroidal effects.\n"
        "Increase NFP of source grid to e.g. 64.",
    )
    zeta = source_grid.nodes[:, 2]
    NFP = source_grid.NFP

    def body(j, f):
        return f + func(zeta + j * 2 * jnp.pi / NFP)

    return fori_loop(0, NFP, body, init_val)


def _prune_data(eval_data, eval_grid, source_data, source_grid, kernel):
    """Returns new dictionaries with only required data."""
    keys = ["R", "phi", "Z", "theta", "zeta"]
    if hasattr(kernel, "eval_keys"):
        keys = keys + kernel.eval_keys

    eval_data = apply(eval_data, jnp.asarray, keys)
    if eval_grid is not None:
        # Casting to JAX arrays reduces memory usage.
        if "theta" not in eval_data:
            eval_data["theta"] = jnp.asarray(eval_grid.nodes[:, 1])
        if "zeta" not in eval_data:
            eval_data["zeta"] = jnp.asarray(eval_grid.nodes[:, 2])

    # Can't prune ω because ω is need to interpolate ϕ in _singular_part.
    keys = kernel.keys + ["omega", "theta", "zeta"]
    source_data = apply(source_data, jnp.asarray, keys)
    # to avoid adding keys to dictionary during iteration
    if "theta" not in source_data:
        source_data["theta"] = jnp.asarray(source_grid.nodes[:, 1])
    if "zeta" not in source_data:
        source_data["zeta"] = jnp.asarray(source_grid.nodes[:, 2])

    return eval_data, source_data


def _nonsingular_part(
    eval_data,
    eval_grid,
    source_data,
    source_grid,
    st,
    sz,
    kernel,
    *,
    ndim=None,
    chunk_size=None,
):
    """Integrate kernel over non-singular points.

    Generally follows sec 3.2.1 of [2].
    If ``eval_grid`` is ``None``, then takes η = 0.
    """
    assert source_grid.can_fft2
    ht = 2 * jnp.pi / source_grid.num_theta
    hz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP

    ndim = setdefault(ndim, kernel.ndim)

    source_zeta = source_data["zeta"]
    source_phi = source_data["phi"]

    def func(zeta_j):
        source_data["zeta"] = zeta_j
        source_data["phi"] = zeta_j  # TODO (#465)

        # nest this def and let JAX figure it out
        def eval_pt(eval_data_i):
            _eta = (
                0
                if eval_grid is None
                else eta(
                    source_data["theta"],
                    source_data["zeta"],
                    eval_data_i["theta"][:, jnp.newaxis],
                    eval_data_i["zeta"][:, jnp.newaxis],
                    ht,
                    hz,
                    st,
                    sz,
                )
            )
            # absorbing (1 - eta) into ds to reduce number of flops by factor of ndim
            return (
                kernel(eval_data_i, source_data, (ht * hz) * (1 - _eta))
                .reshape(-1, source_grid.num_nodes, ndim)
                .sum(-2)
            )

        return batch_map(eval_pt, eval_data, chunk_size).reshape(-1, ndim)

    f = nfp_loop(source_grid, func, jnp.zeros((eval_data["phi"].size, ndim)))
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    # undo rotation of ζ and ϕ
    source_data["zeta"] = source_zeta
    source_data["phi"] = source_phi

    return f


def _singular_part(eval_data, source_data, interpolator, kernel, *, chunk_size=None):
    """Integrate singular point by interpolating to polar grid.

    Generally follows sec 3.2.2 of [2], with the following differences:

    - hyperparameter M replaced by ``st`` and ``sz``.
    - density sigma / function f is absorbed into kernel.

    TODO (#465): For nonzero ω, the quadrature may not be symmetric about the
       singular point. Hence the quadrature may not converge for Cauchy
       principal values. Prove otherwise or remove singularity.

    """
    eval_grid = interpolator.eval_grid

    r, w, dr, dw = _get_quadrature_nodes(interpolator.q)
    r = jnp.abs(r)
    # integrand of eq 38 in [2] except stuff that needs to be interpolated
    v = interpolator.ht * interpolator.hz * interpolator.st * interpolator.sz / 4
    v = v * (chi(r) * r * dr * dw)

    keys = set(kernel.keys)
    if "phi" in keys:
        keys.remove("phi")  # ϕ is not a periodic map of θ, ζ.
        keys.add("omega")
    # It is necessary to take the Fourier transforms of the
    # vector components of the orthonormal polar basis vectors R̂, ϕ̂, Ẑ.
    # Vector components of the Cartesian basis are not NFP periodic.
    fsource = [
        (key, interpolator.fourier(source_data[key]))
        for key in keys
        if key in source_data
    ]

    def polar_pt(i):
        """See sec 3.2.2 of [2].

        Evaluate the effect from a single polar node around each eval point
        on that eval point. Polar grids from other singularities have no effect,
        so only the diagonal term of the kernel is needed.
        """
        vander = interpolator.vander_polar(i)
        source_data_polar = {
            key: interpolator(val, i, is_fourier=True, vander=vander)
            for key, val in fsource
        }
        # coordinates of the polar nodes around the evaluation point
        source_data_polar["theta"] = eval_data["theta"] + interpolator.shift_t[i]
        source_data_polar["zeta"] = eval_data["zeta"] + interpolator.shift_z[i]
        if "omega" in keys:
            source_data_polar["phi"] = (
                source_data_polar["zeta"] + source_data_polar["omega"]
            )
        return kernel(eval_data, source_data_polar, v[i], diag=True)

    f = vmap_chunked(
        polar_pt,
        chunk_size=chunk_size,
        reduction=jnp.add,
        chunk_reduction=_add_reduce,
    )(jnp.arange(v.size)).reshape(eval_grid.num_nodes, -1)

    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f


def _add_reduce(x):
    return x.sum(0)


def singular_integral(
    eval_data,
    source_data,
    interpolator,
    kernel,
    *,
    ndim=None,
    chunk_size=None,
    **kwargs,
):
    """Evaluate a singular integral transform on a surface.

    eg f(θ, ζ) = ∫ ∫ K(θ, ζ, θ', ζ') g(θ', ζ') dθ' dζ'

    Where K(θ, ζ, θ', ζ') is the (singular) kernel and g(θ', ζ') is the metric on the
    surface. See eq. 3.7 in [1]_, but we have absorbed the density σ into K

    Uses method by Malhotra et al. [1]_ [2]_

    Parameters
    ----------
    eval_data : dict
        Dictionary of data at evaluation points (``interpolator.eval_grid``).
        Should store (R, ϕ, Z) coordinates to evaluate field and any keys
        in ``kernel.eval_keys``.
        Vector data should be in rpz basis.
    source_data : dict
        Dictionary of data at source points (``interpolator.source_grid``). Keys
        should be those required by kernel as ``kernel.keys``.
        Vector data should be in rpz basis.
    interpolator : FFTInterpolator
        Function to interpolate from rectangular source grid to polar
        source grid around each singular point. See ``FFTInterpolator``.
    kernel : str or callable
        Kernel function to evaluate. If str, one of the following:
            '1_over_r'        : 1 / |𝐫 − 𝐫'| dS
            'nr_over_r3'      : 𝐧'⋅(𝐫 − 𝐫') / |𝐫 − 𝐫'|³ dS
            'biot_savart'     : μ₀/4π 𝐊'×(𝐫 − 𝐫') / |𝐫 − 𝐫'|³ dS
            'biot_savart_A'   : μ₀/4π 𝐊' / |𝐫 − 𝐫'| dS
        If callable, should take 4 arguments:
            eval_data   : dict of data at evaluation points (primed)
            source_data : dict of data at source points (unprimed)
            ds          : Surface area element (not weighted by ‖e_θ × e_ζ‖ Jacobian).
                          Broadcasts with shape
                          (eval_grid.num_nodes, source_grid.num_nodes).
            diag        : boolean, whether to evaluate full cross interactions
                          or just diagonal
        If a callable, should also have the attributes ``ndim`` and ``keys`` defined.
        ``ndim`` is an integer representing the dimensionality of the output function f,
        1 if f is scalar, 3 if f is a vector, etc.
        ``keys`` is a list of strings of what data is required to evaluate the kernel.
        The kernel will be called with dictionaries containing this data at source and
        evaluation points.
        If vector valued, the input to the kernel function will be in rpz and output
        should be in xyz.
    ndim : int
        Default is kernel.ndim.
        In some applications it, may be useful to supply other values for batching.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    f : ndarray, shape(eval_grid.num_nodes, kernel.ndim)
        Integral transform evaluated at eval_grid. Vectors are in rpz basis.

    References
    ----------
    .. [1] Malhotra, Dhairya, et al. "Efficient high-order singular quadrature schemes
       in magnetic fusion." Plasma Physics and Controlled Fusion 62.2 (2019): 024004.
    .. [2] Malhotra, Dhairya, et al. "Taylor states in stellarators: A fast high-order
       boundary integral solver." Journal of Computational Physics 397 (2019): 108791.

    """
    chunk_size = parse_argname_change(chunk_size, kwargs, "loop", "chunk_size")
    chunk_size = None if (chunk_size == 0) else chunk_size

    if isinstance(kernel, str):
        kernel = kernels[kernel]

    eval_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid
    if kwargs.get("_prune_data", True):
        eval_data, source_data = _prune_data(
            eval_data,
            eval_grid,
            source_data,
            source_grid,
            kernel,
        )
    out1 = _singular_part(
        eval_data,
        source_data,
        interpolator,
        kernel,
        chunk_size=chunk_size,
    )
    out2 = _nonsingular_part(
        eval_data,
        eval_grid,
        source_data,
        source_grid,
        interpolator.st,
        interpolator.sz,
        kernel,
        ndim=ndim,
        chunk_size=chunk_size,
    )
    return out1 + out2


def _dx(eval_data, source_data, diag=False):
    """Compute distance vector between eval and source points.

    Parameters
    ----------
    eval_data : dict[str, jnp.ndarray]
        x data evaluated on eval grid.
    source_data : dict[str, jnp.ndarray]
        y data evaluated on source grid.
    diag : bool
        Set to True to bypass outer product.

    Returns
    -------
    dx : jnp.ndarray
        The vector x-y where y is a source point and x is eval point,
        in Cartesian coordinates.
        Shape (num eval, num source, 3).

    """
    source_x = rpz2xyz(
        jnp.column_stack([source_data["R"], source_data["phi"], source_data["Z"]])
    )
    eval_x = rpz2xyz(
        jnp.column_stack([eval_data["R"], eval_data["phi"], eval_data["Z"]])
    )
    if not diag:
        eval_x = eval_x[:, jnp.newaxis]
    return eval_x - source_x


_dx.keys = ["R", "phi", "Z"]


def _G(dx, keepdims=False):
    """Fundamental solution to the Laplacian in ℝ³.

    Parameters
    ----------
    dx : jnp.ndarray
        The vector x-y where y is a source point and x is eval point,
        in Cartesian coordinates.
        Shape (num eval, num source, 3).

    Returns
    -------
    G : jnp.ndarray
        G(x-y) = -1/(4π ‖x−y‖).
        Shape (num eval, num source).

    """
    return safediv(-1, 4 * jnp.pi * safenorm(dx, axis=-1, keepdims=keepdims))


def _grad_G(dx):
    """∇_x G(x−y) where G is the fundamental solution to the Laplacian in ℝ³.

    Parameters
    ----------
    dx : jnp.ndarray
        The vector x-y where y is a source point and x is eval point,
        in Cartesian coordinates.
        Shape (num eval, num source, 3).

    Returns
    -------
    grad_G : jnp.ndarray
        ∇_x G(x−y) = (4π)⁻¹ ‖x−y‖⁻³ (x-y).
        Shape (num eval, num source).

    """
    return safediv(dx, 4 * jnp.pi * safenorm(dx, axis=-1, keepdims=True) ** 3)


def _kernel_monopole(eval_data, source_data, ds, diag=False):
    """Kernel of single layer operator S[B0*n]: (B0*n)(y) G(x-y) da(y)."""
    out = (
        ds * (source_data["|e_theta x e_zeta|"]) * _G(_dx(eval_data, source_data, diag))
    )
    if source_data["B0*n"].ndim > 1:
        out = out[..., jnp.newaxis]
    return out * source_data["B0*n"]


_kernel_monopole.ndim = 1
_kernel_monopole.keys = _dx.keys + ["B0*n", "|e_theta x e_zeta|"]


def _kernel_dipole(eval_data, source_data, ds, diag=False):
    """Kernel of double layer operator D[Φ]: Φ(y)〈∇_x G(x−y),n(y)〉da(y)."""
    out = ds * dot(
        rpz2xyz_vec(source_data["e_theta x e_zeta"], phi=source_data["phi"]),
        _grad_G(_dx(eval_data, source_data, diag)),
    )
    if source_data["Phi (periodic)"].ndim > 1:
        out = out[..., jnp.newaxis]
    # Do operation with Φ at the end, so that the following
    # outer product plus reduction is more likely to be fused.
    return source_data["Phi (periodic)"] * out


_kernel_dipole.ndim = 1
_kernel_dipole.keys = _dx.keys + ["e_theta x e_zeta", "Phi (periodic)"]


def _kernel_dipole_plus_half(eval_data, source_data, ds, diag=False):
    """Kernel of operator (D[Φ] + Φ/2)(x)."""
    eval_Phi = eval_data["Phi(x) (periodic)"]
    if not diag:
        eval_Phi = eval_Phi[:, jnp.newaxis]
    out = ds * dot(
        rpz2xyz_vec(source_data["e_theta x e_zeta"], phi=source_data["phi"]),
        _grad_G(_dx(eval_data, source_data, diag)),
    )
    if source_data["Phi (periodic)"].ndim > 1:
        out = out[..., jnp.newaxis]
    # Do operation with Φ at the end, so that the following
    # outer product plus reduction is more likely to be fused.
    return (source_data["Phi (periodic)"] - eval_Phi) * out


_kernel_dipole_plus_half.ndim = 1
_kernel_dipole_plus_half.keys = _dx.keys + ["e_theta x e_zeta", "Phi (periodic)"]
_kernel_dipole_plus_half.eval_keys = ["Phi(x) (periodic)"]


# The kernels `singular_integral` accepts by name (DESC's table, trimmed to the copy).
kernels = {
    "monopole": _kernel_monopole,
    "dipole": _kernel_dipole,
    "dipole_plus_half": _kernel_dipole_plus_half,
}
