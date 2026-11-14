"""The exterior Neumann problem on the plasma boundary: ``phi_matrix_pest``.

Copied from DESC (https://github.com/PlasmaControl/DESC), ``desc/compute/_laplace.py``,
which is not on DESC master. By source:

* ``_D_plus_half``: branch ``ku/laplace``, with branch ``ma/AGNI_ext``'s
  ``pest_coords`` argument and ``Phi (periodic)`` key;
* ``_interp_to_potential_grid``, ``_compute_single_layer_matrix`` and
  ``_lsmr_compute_phi_matrix``: written on branch ``ma/AGNI_ext``;
* ``pest_source_data``, ``potential_data_pest`` and ``phi_matrix_pest``: the bodies
  of that branch's compute functions ``_potential_grid_position_pest`` and
  ``_phi_matrix_pest_compute``, as plain functions outside DESC's compute registry.

Changed here: the exterior Neumann problem only (the ``problem`` option and its
interior problems are gone); ``-Phi = phi_matrix @ B_n`` (the branch returns
``Phi``), so the vacuum energy is ``+int B_n (phi_matrix @ B_n) dS / 2``; the
least-squares solve by QR (``_lstsq_qr``) for a smooth derivative; the source and
eval Phi transforms passed in, built once per resolution; no ``known_map`` (see
``singularities``) and no ``_midpoint_quad`` debugging path.

Branches of PlasmaControl/DESC quoted above:

    ku/laplace   f3b0c0de876bb5df75d7a24c2ca3d0276106e328  (Kaya Unalmis, Dario Panici)
    ma/AGNI_ext  1583a921ae4d36171f7d970b2a56b81ae8132bcd  (Maya Avida)

DESC is MIT licensed:

    Copyright (c) 2020 Daniel Dudt, Rory Conlin, Dario Panici, Kaya Unalmis,
    Yigit Gunsur Elmacioglu, Rahul Gaur, Patrick Kim, Egemen Kolemen
"""

from desc.backend import jnp
from interpax_fft import rfft_interp2d
from jax.scipy.linalg import solve_triangular

from ._utils import apply
from .singularities import (
    _kernel_dipole,
    _kernel_dipole_plus_half,
    _kernel_monopole,
    _prune_data,
    singular_integral,
)


def _interp_to_potential_grid(data, source_grid, potential_grid, keys):
    """FFT-interpolate periodic surface data from the source grid to the eval grid.

    Only quantities that are PERIODIC in (theta, zeta) may go through here -- R, Z
    and omega. phi may not: it is secular in zeta (it advances by 2*pi per toroidal
    transit), so the callers take its secular part from the target grid's own nodes.
    """
    dt = 2 * jnp.pi / source_grid.num_theta
    dz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP

    def fun(x):
        return rfft_interp2d(
            source_grid.meshgrid_reshape(x, "rtz")[0],
            potential_grid.num_theta,
            potential_grid.num_zeta,
            dx=dt,
            dy=dz,
        ).ravel(order="F")

    return apply(data, fun, keys)


def _D_plus_half(
    eval_data,
    source_data,
    interpolator,
    basis=None,
    chunk_size=None,
    prune_data=True,
    pest_coords=False,
    _D_quad=False,
):
    """Compute (D[Φ] + Φ/2)(x).

    D[Φ](x) = ∫_y Φ(y)〈∇_x G(x−y),ds(y)〉.

    The integral runs over the N_source quadrature points (``source_data``,
    ``interpolator.source_grid``) and is evaluated at the N_eval points
    (``eval_data``, ``interpolator.eval_grid``). The two need not match.

    Parameters
    ----------
    eval_data : dict
        Data at the N_eval evaluation points. With ``basis`` supplied,
        ``eval_data["Phi(x) (periodic)"]`` is the eval-grid Vandermonde,
        shape (N_eval, N_modes).
    source_data : dict
        Data at the N_source quadrature points. With ``basis`` supplied,
        ``source_data["Phi (periodic)"]`` is the source-grid Vandermonde,
        shape (N_source, N_modes).
    basis : DoubleFourierSeries
        If not supplied, then computes (D[Φ] + Φ/2)(x), shape (N_eval,).
        If supplied, then constructs the operator which
        acts on the spectral coefficients of Φ in the supplied + secular basis,
        shape (N_eval, N_modes).
    prune_data : bool
        Whether the data should be pruned. Default is True.
    _D_quad : bool
        Set to ``True`` to perform double layer potential quadrature without removing
        singularities. Default is ``False``. This is intended for developer use.
    pest_coords : bool
        Set to True to do the entire computation in PEST coordinates. Interpolator
        grid must be in PEST coordinates, and fft2able.
    """
    # With a basis, Phi (periodic) is the source-grid Vandermonde: band-limited and
    # resolved by the source grid, so the polar interpolation reproduces it exactly.
    ndim = 1 if basis is None else basis.num_modes

    kernel = _kernel_dipole if _D_quad else _kernel_dipole_plus_half

    result = singular_integral(
        eval_data,
        source_data,
        interpolator,
        kernel,
        ndim=ndim,
        chunk_size=chunk_size,
        _prune_data=prune_data,
    )
    if ndim == 1:
        result = result.squeeze(-1)

    if _D_quad:
        result += eval_data["Phi(x) (periodic)"] / 2

    return result


def _compute_single_layer_matrix(
    eval_data,
    source_data,
    interpolator,
    chunk_size=None,
    ndim=None,
):
    """Compute the single-layer operator as a matrix M_S where S[B0*n] = M_S @ B_n.

    Parameters
    ----------
    eval_data : dict
        Data at the N_eval evaluation points (R, phi, Z and any eval_keys needed
        by the kernel).
    source_data : dict
        Data at the N_source quadrature points (must include |e_theta x e_zeta|
        and geometry). Does not need to include B0*n; the caller supplies it
        per-column. N_source need not equal N_eval.
    interpolator : FFTInterpolator
    chunk_size : int or None
        Chunk size for eval-point batching *inside* each ``singular_integral`` call.
    ndim : int or None
        Number of columns carried through the integral, i.e. N_modes when
        ``source_data["B0*n"]`` is a Vandermonde matrix.

    Returns
    -------
    M_S : jnp.ndarray, shape (N_eval, N_modes)
        Matrix satisfying S[B0*n] = M_S @ B_n.
    """
    source_grid = interpolator.source_grid

    eval_data_p, source_data_p = _prune_data(
        eval_data, interpolator.eval_grid, source_data, source_grid, _kernel_monopole
    )
    # B0*n is not in source_data_p (no specific B_n to prune); supply it per-column.
    spectral_matrix = singular_integral(
        eval_data_p,
        source_data_p,
        interpolator,
        _kernel_monopole,
        chunk_size=chunk_size,
        _prune_data=False,
        ndim=ndim,
    )
    return spectral_matrix


def _lstsq_qr(D, B):
    """``jnp.linalg.lstsq(D, B)[0]`` through a reduced QR, so its derivative is smooth.

    lstsq goes through an SVD: its derivative is unstable for near-degenerate
    singular values and not smooth at the rcond cut. QR's is well defined at full
    rank. A wide ``D`` (fewer eval nodes than modes -- every even ``n_theta`` or
    ``n_zeta`` shell, whose Nyquist mode vanishes on the nodes) gets lstsq's
    minimum-norm solution, ``D^H (D D^H)^-1 B = Q R^-H B`` with ``D^H = Q R``; a tall
    one the least-squares solution, ``R^-1 Q^H B`` with ``D = Q R``. ``D`` is real
    (real Fourier basis and kernels), so ``^H`` is ``^T``; the ``.conj()`` keeps it
    right regardless.
    """
    if D.shape[0] < D.shape[1]:
        Q, R = jnp.linalg.qr(D.conj().T)
        return Q @ solve_triangular(R.conj().T, B, lower=True)
    Q, R = jnp.linalg.qr(D)
    return solve_triangular(R, Q.conj().T @ B, lower=False)


def _lsmr_compute_phi_matrix(
    eval_data,
    source_data,
    interpolator,
    source_transform,
    eval_transform,
    chunk_size=None,
    pest_coords=False,
    _D_quad=False,
):
    """Compute matrix A where Phi (periodic) = -A @ B_n.

    Constructs the double-layer operator D and single-layer matrix M_S, then
    solves D @ A_mn = M_S in one batch, returning E @ A_mn where E is the
    Vandermonde matrix (so the output maps directly to nodal potential values).

    Two grids, and they need not have the same size:

    * the SOURCE grid (``interpolator.source_grid``) holds the N_source
      quadrature points that discretize the singular integrals;
    * the EVAL grid (``interpolator.eval_grid``) holds the N_eval points where
      those integrals are evaluated, i.e. where B_n is supplied and Phi is
      wanted.

    Refining only the source grid resolves the singular integral -- which is
    what makes the discrete operator self-adjoint in the surface measure --
    without changing the size of the returned matrix or the number of modes.

    Parameters
    ----------
    eval_data : dict
        Data at the N_eval evaluation points (geometry, not B0*n).
    source_data : dict
        Data at the N_source quadrature points (geometry including
        |e_theta x e_zeta|).
    interpolator : FFTInterpolator
        Carries both grids; nothing in it requires N_source == N_eval.
    source_transform : Transform
        The Phi basis on the SOURCE grid, ``direct1``: its Vandermonde is what the
        quadrature integrates against.
    eval_transform : Transform
        The same basis on the EVAL grid, ``direct1`` with its pseudoinverse: B_n is
        nodal there. ``source_transform`` itself when the two grids coincide. Both
        grids are uniform PEST grids fixed by the resolution, so both transforms are
        built once (``desc_vacuum.phi_scaffold``) and reused on every call.
    chunk_size : int or None
        Inner chunk size for eval-point batching inside each ``singular_integral`` call.
    _D_quad : bool

    Returns
    -------
    A_mn : jnp.ndarray, shape (N_modes, N_eval)
        Nodal B_n on the eval grid -> spectral coefficients of Phi (periodic).
    A : jnp.ndarray, shape (N_eval, N_eval)
        Matrix satisfying -Phi (periodic) = A @ B_n, both nodal on the eval grid,
        B_n along the outward normal. With this sign the vacuum energy is
        +(1/2mu_0) int B_n (A B_n) dS >= 0.
    """
    eval_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    basis = source_transform.basis
    assert eval_transform.basis is basis
    assert eval_transform.grid.num_nodes == interpolator.eval_grid.num_nodes
    assert source_transform.grid.num_nodes == interpolator.source_grid.num_nodes
    if pest_coords:
        assert (
            source_grid.can_fft2
        ), f"pest_grid must have can_fft2=True, got {source_grid}"
        assert (
            eval_grid.can_fft2
        ), f"potential pest_grid must have can_fft2=True, got {eval_grid}"
    # The mode count is bounded by the EVAL grid, never by the source grid, so
    # refining the quadrature alone never changes the number of unknowns.
    assert basis.M <= eval_grid.M
    assert basis.N <= eval_grid.N

    # Vandermonde matrices: `matrices["direct1"][0][0][0]` is basis.evaluate(grid).
    Phi_src = source_transform.matrices["direct1"][0][0][0]  # (N_source, N_modes)
    Phi_eval = eval_transform.matrices["direct1"][0][0][0]  # (N_eval, N_modes)
    # B_n is supplied nodally ON THE EVAL GRID, so the fit to spectral
    # coefficients is the eval-grid pseudoinverse.
    pinv_eval = eval_transform.matrices["pinv"]  # (N_modes, N_eval)

    # Double-layer operator D, shape (N_eval, N_modes). Prune into a separate
    # copy so the original dicts are still available for M_S below.
    eval_data_d, source_data_d = _prune_data(
        eval_data,
        eval_grid,
        source_data,
        source_grid,
        _kernel_dipole_plus_half,
    )
    eval_data_d["Phi(x) (periodic)"] = Phi_eval  # (N_eval, N_modes)
    source_data_d["Phi (periodic)"] = Phi_src  # (N_source, N_modes)

    # Expand B0*n in the same basis as Phi. This lives on the SOURCE grid: it is
    # what the quadrature integrates against.
    source_data["B0*n"] = Phi_src  # (N_source, N_modes)

    D = _D_plus_half(
        eval_data_d,
        source_data_d,
        interpolator,
        basis,
        chunk_size,
        prune_data=False,
        _D_quad=_D_quad,
    )
    assert D.shape == (eval_grid.num_nodes, basis.num_modes)
    # D + 1/2 -> D - 1/2: the exterior Neumann problem
    D -= Phi_eval

    # Single-layer matrix M_S, shape (N_eval, N_modes). Uses the original
    # (unpruned) data so that |e_theta x e_zeta| is available.
    M_S = _compute_single_layer_matrix(
        eval_data, source_data, interpolator, chunk_size, ndim=basis.num_modes
    )
    # Solve D @ A_mn = M_S for all N_modes right-hand sides simultaneously.
    # A_mn is (N_modes, N_modes) here; @ pinv_eval makes it (N_modes, N_eval).
    if eval_grid.num_nodes == basis.num_modes:
        A_mn = jnp.linalg.solve(D, M_S)
    else:
        A_mn = _lstsq_qr(D, M_S)
    A_mn = A_mn @ pinv_eval  # (N_modes, N_eval)

    # -Phi (periodic) = Phi_eval @ A_mn @ B_n, shape (N_eval, N_eval), B_n along the
    # outward normal: the sign that makes the vacuum energy +int B_n (A B_n) dS / 2.
    return A_mn, Phi_eval @ A_mn


# ---------------------------------------------------------------------------------
# The PEST compute functions of the DESC branch (`potential data pest`,
# `phi_matrix_pest`), as plain functions: AGNI evaluates the boundary geometry itself
# and passes it in, so nothing goes through DESC's compute registry. Same bodies and
# relabelling as the branch's `_potential_grid_position_pest` and
# `_phi_matrix_pest_compute`; the interpolator is built by `desc_vacuum.phi_scaffold`.
# ---------------------------------------------------------------------------------


def pest_source_data(data):
    """Relabel PEST basis vectors to the names the kernels and interpolators read.

    ``data`` holds ``e_theta_PEST``, ``e_phi|r,v``, ``e_theta_PEST x e_phi|r,v`` and
    ``|e_theta_PEST x e_phi|r,v|`` (any subset); the standard names are added beside
    them.
    """
    out = dict(data)
    for pest, std in (
        ("e_theta_PEST", "e_theta"),
        ("e_phi|r,v", "e_zeta"),
        ("e_theta_PEST x e_phi|r,v", "e_theta x e_zeta"),
        ("|e_theta_PEST x e_phi|r,v|", "|e_theta x e_zeta|"),
    ):
        if pest in out:
            out[std] = out[pest]
    return out


def potential_data_pest(data, pest_grid, potential_grid=None):
    """R, phi, Z (and theta, zeta) on the eval grid; branch ``potential data pest``.

    ``data`` is the source-grid geometry. With a separate eval grid, R and Z are
    FFT-interpolated in the PEST angles and phi is the eval nodes' zeta (DESC zeta is
    PEST phi).
    """
    potential_grid = pest_grid if potential_grid is None else potential_grid
    if potential_grid == pest_grid:
        out = apply(data, subset=("R", "phi", "Z"))
    else:
        out = _interp_to_potential_grid(data, pest_grid, potential_grid, ("R", "Z"))
        out["phi"] = potential_grid.nodes[:, 2]
    out["theta"] = potential_grid.nodes[:, 1]
    out["zeta"] = potential_grid.nodes[:, 2]
    return out


def phi_matrix_pest(
    data,
    interpolator,
    source_transform,
    eval_transform,
    chunk_size=None,
    _D_quad=False,
):
    """``(A_mn, phi_matrix)`` with ``-Phi (periodic) = phi_matrix @ B_n``; branch
    ``phi_matrix_pest``.

    Parameters
    ----------
    data : dict
        Source-grid geometry, BIEST order: ``R``, ``phi``, ``Z``,
        ``e_theta_PEST x e_phi|r,v``, ``|e_theta_PEST x e_phi|r,v|`` (and ``zeta``,
        ``omega``).
    interpolator : FFTInterpolator
        Its grids are the source and eval grids.
    source_transform, eval_transform : desc.transform.Transform
        The Phi basis on the source and on the eval grid; see
        :func:`_lsmr_compute_phi_matrix`.
    """
    source_data = pest_source_data(data)
    eval_data = potential_data_pest(
        source_data, interpolator.source_grid, interpolator.eval_grid
    )
    return _lsmr_compute_phi_matrix(
        eval_data=eval_data,
        source_data=source_data,
        interpolator=interpolator,
        source_transform=source_transform,
        eval_transform=eval_transform,
        chunk_size=chunk_size,
        pest_coords=True,
        _D_quad=_D_quad,
    )
