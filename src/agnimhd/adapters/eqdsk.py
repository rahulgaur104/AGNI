"""EFIT g-file -> fixed-boundary DESC ``Equilibrium``.

``desc_from_eqdsk`` reads a G-EQDSK file, takes the ``psi_N = psi_n_edge`` flux
surface as the boundary and the g-file's own pressure and q profiles, and solves
the equilibrium with DESC at fixed iota. :func:`~agnimhd.from_desc` then
evaluates it on any basis. DESC is imported inside the function only.

This is a port of ``convert_EFIT_to_DESC`` from efit2desc:

    D. Panici, *efit2desc: a Python tool for converting an EFIT g-file to a
    (fixed-boundary) DESC Equilibrium*, https://github.com/dpanici/efit2desc
    (2025). Copyright (c) 2025 Dario Panici, released under the MIT license,
    the terms of AGNI's own LICENSE.

efit2desc reads the file and traces the flux surfaces with OMFIT; here the
reader and the boundary are numpy and scipy only.
"""

import re

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from ..backend import errorif

__all__ = ["desc_from_eqdsk", "eqdsk_profiles", "read_eqdsk"]

#: A Fortran real; ``5e16.9`` fields can touch (``1.0E+00-2.0E+00``).
_REAL = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[EeDd][+-]?\d+)?")


def read_eqdsk(path):
    """The grid, scalars and profiles of a G-EQDSK file, as a dict.

    Keys are the format's names in lower case: ``nw, nh, rdim, zdim, rcentr,
    rleft, zmid, rmaxis, zmaxis, simag, sibry, bcentr, current`` and the arrays
    ``fpol, pres, ffprim, pprime, qpsi`` (``nw`` values, uniform in psi from the
    axis to the boundary) and ``psirz`` (shape ``(nh, nw)``). The boundary and
    limiter points are not read.
    """
    with open(path) as f:
        nw, nh = (int(n) for n in re.findall(r"\d+", f.readline())[-2:])
        reals = [float(x.upper().replace("D", "E")) for x in _REAL.findall(f.read())]
    size = 20 + 5 * nw + nw * nh
    errorif(
        len(reals) < size,
        ValueError,
        f"{path} has {len(reals)} numbers, a {nw}x{nh} g-file needs {size}.",
    )
    keys = "rdim zdim rcentr rleft zmid rmaxis zmaxis simag sibry bcentr current"
    g = dict(zip(keys.split(), reals), nw=nw, nh=nh)
    arrays = np.array(reals[20:size])
    g["fpol"], g["pres"], g["ffprim"], g["pprime"] = arrays[: 4 * nw].reshape(4, nw)
    g["psirz"] = arrays[4 * nw : 4 * nw + nw * nh].reshape(nh, nw)
    g["qpsi"] = arrays[4 * nw + nw * nh :]
    return g


def eqdsk_profiles(g, psi_n_edge=0.99, n_rho=129, n_angle=256):
    """Boundary and profiles of a g-file up to ``psi_N = psi_n_edge``.

    ``q`` and ``p`` are the g-file's ``qpsi`` and ``pres``. ``rho = sqrt(psi_T /
    psi_T(edge))`` with the toroidal flux ``psi_T = int q dpsi``, psi in Wb/rad
    (COCOS 1-8). The boundary is found on ``n_angle`` rays from the magnetic
    axis, the minimum of a bicubic spline of ``psirz`` near the file's axis.

    Parameters
    ----------
    g : dict or str
        :func:`read_eqdsk`'s dict, or the path of a g-file.
    psi_n_edge : float
        Normalized poloidal flux of the boundary, below 1 (the separatrix).

    Returns
    -------
    dict
        ``rho, psi_n, p, q`` at ``n_rho`` values of ``psi_N`` from 0 to
        ``psi_n_edge``; ``Psi``, the toroidal flux at the boundary in Wb, signed
        as ``bcentr``; ``R, Z`` of the boundary at the polar angles ``theta``
        about the axis ``R0, Z0``.
    """
    g = read_eqdsk(g) if isinstance(g, str) else g
    flux = np.linspace(0, 1, g["nw"])
    psi_n = psi_n_edge * np.linspace(0, 1, n_rho)
    q = CubicSpline(flux, g["qpsi"])
    psi_T = q.antiderivative()(psi_n) * abs(g["sibry"] - g["simag"])

    R = g["rleft"] + np.linspace(0, g["rdim"], g["nw"])
    Z = g["zmid"] + np.linspace(-0.5, 0.5, g["nh"]) * g["zdim"]
    spline = RectBivariateSpline(
        R, Z, (g["psirz"].T - g["simag"]) / (g["sibry"] - g["simag"])
    )
    x = np.array([g["rmaxis"], g["zmaxis"]])
    for _ in range(10):  # Newton on grad psi_N = 0
        d = {
            (i, j): spline.ev(x[:1], x[1:], dx=i, dy=j)[0]  # 0-d or (1,) per scipy
            for i, j in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
        }
        hess = [[d[2, 0], d[1, 1]], [d[1, 1], d[0, 2]]]
        x = x - np.linalg.solve(hess, [d[1, 0], d[0, 1]])
    R0, Z0 = x

    # first crossing of psi_n_edge on each ray inside the grid, then bisection
    theta = 2 * np.pi * np.arange(n_angle) / n_angle
    c, s = np.cos(theta), np.sin(theta)
    with np.errstate(divide="ignore"):  # distance to the grid's edge along each ray
        reach = np.minimum(
            np.where(c > 0, R[-1] - R0, R0 - R[0]) / abs(c),
            np.where(s > 0, Z[-1] - Z0, Z0 - Z[0]) / abs(s),
        )
    r = np.linspace(0, 1, 2000)[:, None] * reach
    first = np.argmax(spline.ev(R0 + r * c, Z0 + r * s) >= psi_n_edge, axis=0)
    errorif(
        np.any(first == 0),
        ValueError,
        f"psi_N = {psi_n_edge} is not a closed surface inside the g-file's grid.",
    )
    ray = np.arange(n_angle)
    lo, hi = r[first - 1, ray], r[first, ray]
    for _ in range(50):
        mid = (lo + hi) / 2
        inside = spline.ev(R0 + mid * c, Z0 + mid * s) < psi_n_edge
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    return {
        "rho": np.sqrt(psi_T / psi_T[-1]),
        "psi_n": psi_n,
        "p": CubicSpline(flux, g["pres"])(psi_n),
        "q": q(psi_n),
        "Psi": np.sign(g["bcentr"]) * 2 * np.pi * abs(psi_T[-1]),
        "R": R0 + lo * c,
        "Z": Z0 + lo * s,
        "theta": theta,
        "R0": R0,
        "Z0": Z0,
    }


def desc_from_eqdsk(
    path, psi_n_edge=0.99, L=24, M=24, profile_L=24, solve=True, solve_options=None
):
    """A fixed-boundary DESC ``Equilibrium`` from a g-file, solved at fixed iota.

    The boundary is the ``psi_N = psi_n_edge`` surface of
    :func:`eqdsk_profiles`, fit with ``M`` Fourier modes in the polar angle about
    the axis. The g-file's pressure and ``iota = 1/q`` are fit as even power
    series of order ``profile_L`` in ``rho`` (splines if ``profile_L`` is None).
    Up-down asymmetric (``sym=False``).

    Parameters
    ----------
    path : str
        The g-file.
    psi_n_edge : float
        Normalized poloidal flux of the boundary, below 1.
    L, M : int
        DESC's radial and poloidal resolution.
    profile_L : int or None
        Order of the power-series profiles; None for splines.
    solve : bool
        Solve the equilibrium (``eq.solve``) before returning it.
    solve_options : dict, optional
        Passed to ``eq.solve``; defaults ``ftol=1e-8, xtol=0, gtol=0,
        maxiter=100`` with a ``ForceBalance`` objective.

    Returns
    -------
    desc.equilibrium.Equilibrium
    """
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZCurve, FourierRZToroidalSurface
    from desc.grid import QuadratureGrid
    from desc.objectives import ForceBalance, ObjectiveFunction
    from desc.profiles import PowerSeriesProfile, SplineProfile

    f = eqdsk_profiles(path, psi_n_edge)
    rho = f["rho"]
    profiles = {
        name: (
            SplineProfile(knots=rho, values=v)
            if profile_L is None
            else PowerSeriesProfile.from_values(rho, v, order=profile_L, sym="even")
        )
        for name, v in (("pressure", f["p"]), ("iota", 1 / f["q"]))
    }
    surface = FourierRZToroidalSurface.from_values(
        coords=np.stack([f["R"], np.zeros_like(f["R"]), f["Z"]], axis=-1),
        theta=-f["theta"],  # clockwise in (R, Z): DESC's right-handed theta
        M=M,
        N=0,
        NFP=1,
        sym=False,
    )
    axis = FourierRZCurve(R_n=f["R0"], Z_n=f["Z0"], modes_R=[0], modes_Z=[0], sym=False)
    eq = Equilibrium(
        surface=surface, axis=axis, Psi=f["Psi"], L=L, M=M, sym=False, **profiles
    )
    if solve:
        options = dict(ftol=1e-8, xtol=0, gtol=0, maxiter=100, verbose=0)
        options.update(solve_options or {})
        options.setdefault(
            "objective",
            ObjectiveFunction(
                ForceBalance(eq, grid=QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=0))
            ),
        )
        eq.solve(**options)
    return eq
