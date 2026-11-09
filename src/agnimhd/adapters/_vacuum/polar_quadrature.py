"""Polar quadrature and singular-grid heuristics for BIEST.

Copied from DESC (https://github.com/PlasmaControl/DESC), file
``desc/integrals/quad_utils.py`` on branch ``ma/AGNI_ext`` at commit
1583a921ae4d36171f7d970b2a56b81ae8132bcd, where this code is not on DESC master, so that
AGNI's free-boundary term runs against DESC master. Trimmed to what the PEST
``phi_matrix_pest`` chain uses; see the DESC branch for history.

DESC is MIT licensed:

    Copyright (c) 2020 Daniel Dudt, Rory Conlin, Dario Panici, Kaya Unalmis,
    Yigit Gunsur Elmacioglu, Rahul Gaur, Patrick Kim, Egemen Kolemen
"""

import scipy
from desc.backend import fori_loop, jnp
from desc.utils import errorif


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


def chi(r):
    """Partition of unity function in polar coordinates. Eq 39 in [2].

    Parameters
    ----------
    r : jnp.ndarray
        Absolute value of radial coordinate in polar domain.

    """
    return jnp.exp(-36 * jnp.abs(r) ** 8)


def eta(theta, zeta, theta0, zeta0, ht, hz, st, sz):
    """Partition of unity function in rectangular coordinates.

    Consider the mapping from
    (θ,ζ) ∈ [-π, π) × [-π/NFP, π/NFP) to (ρ,ω) ∈ [−1, 1] × [0, 2π)
    defined by
    θ − θ₀ = h₁ s₁/2 ρ sin ω
    ζ − ζ₀ = h₂ s₂/2 ρ cos ω
    with Jacobian determinant norm h₁h₂ s₁s₂/4 |ρ|.

    In general in dimensions higher than one, the mapping that determines a
    change of variable for integration must be bijective. This is satisfied
    only if s₁ = 2π/h₁ and s₂ = (2π/NFP)/h₂. In the particular case the
    integrand is nonzero in a subset of the domain, then the change of variable
    need only be a bijective map where the function does not vanish, more
    precisely, its set of compact support.

    The functions we integrate are proportional to η₀(θ,ζ) = χ₀(r) far from the
    singularity at (θ₀,ζ₀). Therefore, the support matches χ₀(r)'s, assuming
    this region is sufficiently large compared to the singular region.
    Here χ₀(r) has support where the argument r lies in [0, 1]. The map r
    defines a coordinate mapping between the toroidal domain and a polar domain
    such that the integration region in the polar domain (ρ,ω) ∈ [−1, 1] × [0, 2π)
    equals the compact support, and furthermore is a circular region around the
    singular point in (θ,ζ) geometry when s₁ × s₂ denote the number of grid points
    on a uniformly discretized toroidal domain (θ,ζ) ∈ [0, 2π)².
      χ₀ : r ↦ exp(−36r⁸)

      r : ρ, ω ↦ |ρ|

      r : θ, ζ ↦ 2 [ (θ−θ₀)²/(h₁s₁)² + (ζ−ζ₀)²/(h₂s₂)² ]⁰ᐧ⁵

    Hence, r ≥ 1 (r ≤ 1) outside (inside) the integration domain.

    The choice for the size of the support is determined by s₁ and s₂.
    The optimal choice is dependent on the nature of the singularity e.g. if the
    integrand decays quickly then the elliptical grid determined by s₁ and s₂
    can be made smaller and the integration will have higher resolution for the
    same number of quadrature points.

    With the above definitions the support lies on an s₁ × s₂ subset
    of a field period which has ``num_theta`` × ``num_zeta`` nodes total.
    Since kernels are 2π periodic, the choice for s₂ should be multiplied by NFP.
    Then the support lies on an s₁ × s₂ subset of the full domain. For large NFP
    devices such as Heliotron or tokamaks it is typical that s₁ ≪ s₂.

    Parameters
    ----------
    theta, zeta : jnp.ndarray
        Coordinates of points to evaluate partition function η₀(θ,ζ).
    theta0, zeta0 : jnp.ndarray
        Origin (θ₀,ζ₀) where the partition η₀ is unity.
    ht, hz : float
        Grid step size in θ and ζ.
    st, sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.

    """
    dt = jnp.abs(theta - theta0)
    dz = jnp.abs(zeta - zeta0)
    # The distance spans (dθ,dζ) ∈ [0, π]², independent of NFP.
    dt = jnp.minimum(dt, 2 * jnp.pi - dt)
    dz = jnp.minimum(dz, 2 * jnp.pi - dz)
    r = 2 * jnp.hypot(dt / (ht * st), dz / (hz * sz))
    return chi(r)


def _get_polar_quadrature(q):
    """Polar nodes for quadrature around singular point.

    Parameters
    ----------
    q : int
        Order of quadrature in radial and azimuthal directions.

    Returns
    -------
    r, w : ndarray
        Radial and azimuthal coordinates.
    dr, dw : ndarray
        Radial and azimuthal spacing and quadrature weights.

    """
    Nr = Nw = q
    r, dr = scipy.special.roots_legendre(Nr)
    # integrate separately over [-1,0] and [0,1]
    r1 = 1 / 2 * r - 1 / 2
    r2 = 1 / 2 * r + 1 / 2
    r = jnp.concatenate([r1, r2])
    dr = jnp.concatenate([dr, dr]) / 2
    w = jnp.linspace(0, jnp.pi, Nw, endpoint=False)
    dw = jnp.ones_like(w) * jnp.pi / Nw
    r, w = jnp.meshgrid(r, w)
    r = r.ravel()
    w = w.ravel()
    dr, dw = jnp.meshgrid(dr, dw)
    dr = dr.ravel()
    dw = dw.ravel()
    return r, w, dr, dw


def _vanilla_params(grid):
    """Parameters for support size and quadrature resolution.

    These parameters do not account for grid anisotropy.

    Parameters
    ----------
    grid : LinearGrid
        Grid that can fft2.

    Returns
    -------
    st : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    q : int
        Order of quadrature in radial and azimuthal directions.

    """
    Nt = grid.num_theta
    Nz = grid.num_zeta * grid.NFP
    q = int(jnp.sqrt(grid.num_nodes) // 2)
    s = min(q, Nt, Nz)
    return s, s, q


def _best_params(grid, ratio):
    """Parameters for heuristic support size and quadrature resolution.

    These parameters account for global grid anisotropy which ensures
    more robust convergence across a wider aspect ratio range.

    Parameters
    ----------
    grid : LinearGrid
        Grid that can fft2.
    ratio : float or jnp.ndarray
        Best ratio.

    Returns
    -------
    st : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    q : int
        Order of quadrature in radial and azimuthal directions.

    """
    assert grid.can_fft2
    Nt = grid.num_theta
    Nz = grid.num_zeta * grid.NFP
    q = int(jnp.sqrt(grid.num_nodes if (grid.num_zeta > 1) else (Nt * Nz)) // 2)
    s = min(q, Nt, Nz)
    # Size of singular region in real space = s * h * |e_.|
    # For it to be a circle, choose radius ~ equal
    # s_t * h_t * |e_t| = s_z * h_z * |e_z|
    # s_z / s_t = h_t / h_z  |e_t| / |e_z| = Nz*NFP/Nt |e_t| / |e_z|
    # Denote ratio = < |e_z| / |e_t| > and
    #      s_ratio = s_z / s_t = Nz*NFP/Nt / ratio
    # Also want sqrt(s_z*s_t) ~ s = q.
    s_ratio = jnp.sqrt(Nz / Nt / ratio)
    st = jnp.clip(jnp.ceil(s / s_ratio).astype(int), None, Nt)
    sz = jnp.clip(jnp.ceil(s * s_ratio).astype(int), None, Nz)
    if s_ratio.size == 1:
        st = int(st)
        sz = int(sz)
    return st, sz, q


def _best_ratio(data):
    """Ratio to make singular integration partition ~circle in real space.

    Parameters
    ----------
    data : dict[str, jnp.ndarray]
        Dictionary of data evaluated on single flux surface grid that ``can_fft2``
        with keys ``|e_theta x e_zeta|``, ``e_theta``, and ``e_zeta``.

    """
    scale = jnp.linalg.norm(data["e_zeta"], axis=-1) / jnp.linalg.norm(
        data["e_theta"], axis=-1
    )
    return jnp.mean(scale * data["|e_theta x e_zeta|"]) / jnp.mean(
        data["|e_theta x e_zeta|"]
    )
