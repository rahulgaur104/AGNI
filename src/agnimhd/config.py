"""Static configuration objects.

Everything here is **static**: frozen, hashable dataclasses passed as
non-traced arguments. That is deliberate and load-bearing. Resolution, basis
choice and solver selection drive Python branches and array *shapes*, neither of
which can be derived from a traced value, and holding them as ordinary pytree
leaves means ``jit`` retraces on every call.

Configuration resolves **keyword argument first, environment variable second,
default last**. Every option is a documented keyword argument of the public API.
Environment variables exist only as operational fallbacks for job scripts; none
of them is the only way to reach a code path, and a value passed by a caller
always wins over an exported one.
"""

import os
from dataclasses import dataclass, replace

from .backend import errorif

__all__ = [
    "AssemblyConfig",
    "SolverConfig",
    "resolve_flag",
    "resolve_option",
]


def resolve_option(value, env, default, cast=None):
    """Resolve one option: **keyword first**, then environment, then default.

    Parameters
    ----------
    value : object or None
        The caller's value. ``None`` means "not set".
    env : str
        Environment variable consulted when ``value`` is ``None``.
    default : object
        Used when neither is set.
    cast : callable, optional
        Applied to the result.

    Returns
    -------
    object

    Notes
    -----
    The keyword wins. This inverts a pattern that is easy to write by accident::

        os.environ.get("AGNI_NUM_MATVECS", str(kwargs.get("num_matvecs", 50)))

    which uses the keyword only as the *environment's* default, so an exported
    variable silently discards an explicit argument. A caller that passes a
    value must get that value.
    """
    if value is None:
        value = os.environ.get(env, default)
    return cast(value) if cast is not None else value


def resolve_flag(value, env, default=False):
    """Boolean option, keyword first. Accepts bools or the usual strings.

    Parameters
    ----------
    value : bool, str, or None
    env : str
    default : bool

    Returns
    -------
    bool
    """
    if value is None:
        value = os.environ.get(env, "1" if default else "0")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


@dataclass(frozen=True)
class AssemblyConfig:
    """Static settings for building the AGNI operator.

    Parameters
    ----------
    gamma : float
        Adiabatic index used by the compressibility term. The term is purely
        stabilizing and does not move marginal stability, so a large ``gamma``
        is an alternative way to impose incompressibility. Default ``5/3``.
    incompressible : bool
        Impose incompressibility directly. Default False.
    axisym : bool
        Treat the equilibrium as axisymmetric, so each displacement component is
        a single toroidal Fourier mode and the operator becomes complex.
        Default False.
    n_mode_axisym : int
        Toroidal mode number analyzed when ``axisym`` is set. Default 1.
    coupled_rt : bool
        ``D_rho`` and ``D_theta`` are the full non-separable ``(n_rho*n_theta)``
        coupled Zernike-Fourier operators rather than per-direction matrices.
        Requires ``n_rho_coupled`` and ``n_theta_coupled``. Default False.
    n_rho_coupled, n_theta_coupled : int, optional
        Per-direction node counts in coupled mode. They cannot be inferred:
        in coupled mode ``D_rho.shape[0]`` is the product, not either factor.

    Raises
    ------
    ValueError
        If ``coupled_rt`` is set without both node counts, or if the counts are
        inconsistent.
    """

    gamma: float = 5.0 / 3.0
    incompressible: bool = False
    axisym: bool = False
    n_mode_axisym: int = 1
    coupled_rt: bool = False
    n_rho_coupled: int = None
    n_theta_coupled: int = None

    def __post_init__(self):
        """Validate the coupled-mode node counts."""
        errorif(
            self.coupled_rt
            and (self.n_rho_coupled is None or self.n_theta_coupled is None),
            ValueError,
            "coupled_rt=True requires both n_rho_coupled and n_theta_coupled. "
            "In coupled mode D_rho is the full (n_rho * n_theta) operator, so "
            "the per-direction counts cannot be recovered from its shape.",
        )
        errorif(
            self.axisym and self.n_mode_axisym == 0 and self.incompressible,
            NotImplementedError,
            "axisym with n_mode_axisym=0 and incompressible=True is not "
            "implemented.",
        )

    def replace(self, **changes):
        """Return a copy with fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class SolverConfig:
    """Static settings for the eigensolve.

    Parameters
    ----------
    eigensolver : {"eigsh", "jax_lanczos", "pcg_deflated"}
        Which eigensolve to run.

        ``"eigsh"`` assembles the dense matrix and calls SciPy ARPACK. Measured
        **1.53x faster than the hand-rolled JAX Lanczos on CPU**, so it is the
        default wherever the dense matrix fits.

        ``"jax_lanczos"`` assembles in JAX and runs matfree Lanczos with an
        exact dense LU shift-invert. Stays on the accelerator.

        ``"pcg_deflated"`` never forms a dense matrix: it applies the operator
        matrix-free, preconditions with the ring (block-Jacobi) blocks, and
        deflates with a coarse space. This is the path for resolutions where the
        dense matrix does not fit.
    sigma : float
        Shift for the shift-invert. The constraint is **two-sided**, and only
        one side of it is obvious.

        *Not above the spectrum.* Above the smallest eigenvalue the solve
        converges to the wrong mode, and for the deflated path ``H = A - sigma
        I`` stops being positive definite, so the preconditioned CG is not a
        legal Krylov method at all.

        *Not arbitrarily far below it either*, for any solver that stops at a
        fixed matvec count rather than at a tolerance -- which means
        ``"jax_lanczos"`` and ``"pcg_deflated"``, but not ``"eigsh"``.
        Shift-invert maps ``lambda`` to ``mu = 1/(lambda - sigma)``, and Lanczos
        separates two modes at a rate set by the *ratio* of their ``mu``. As
        ``sigma`` recedes, every ``mu`` collapses onto ``-1/sigma`` and the
        ratio goes to one. Measured on the shipped 24x12x8 case, whose spectrum
        starts ``-1.34e-4, -6.25e-5``, then a cluster of numerically null modes
        at ``1e-11``:

        =========  ==============================  ================================
        ``sigma``  ``mu[0]/mu[1]``                 ``jax_lanczos``, 50 matvecs
        =========  ==============================  ================================
        ``-1e-1``  1.0007                          wrong mode, ``lambda > 0``
        ``-1e-2``  1.0075                          ``-1.337435e-04`` (1.4e-5 off)
        ``-1e-3``  1.0823                          ``-1.337627e-04`` (exact)
        =========  ==============================  ================================

        The default is ``-1e-1``, which is safe for the default ``"eigsh"``
        because ARPACK iterates to ``eigsh_tol`` instead of stopping at a fixed
        count, and it is deliberately conservative about the side that has no
        recovery. On the shipped case that same shift makes a 50-matvec
        ``jax_lanczos`` return ``+1.598e-04`` -- the wrong sign, and therefore
        the wrong physics answer. **It is not silent**: the Rayleigh residual
        from :func:`agnimhd.objective.eigenpair` is 4.6e+04 for that vector
        against 1.6e-04 for the converged one. Check it. Raising
        ``num_matvecs`` to 200 also recovers the right mode at the far shift, at
        four times the cost of moving ``sigma``.

        ``sigma_mode="adapt"`` does **not** rescue a shift this far out. Its
        first pass returns a positive ``lambda``, the ``sigma2 < 0`` guard
        rejects it, and the second pass repeats the first at the same bad shift.
        Re-shifting to ``-sigma_factor * |lambda|`` instead is worse, not
        better: from a wrong first pass it chases the numerically null cluster
        down to ``sigma = -1e-10`` and converges there.
    num_matvecs : int
        Lanczos matvec count for the fine solve. Default 50.

        This value is **fixed and untuned** -- it was never swept. It is a
        plausible knob for further speedup, not a converged choice.
    coarse_num_matvecs : int
        Lanczos steps for the coarse generalized solve. Deliberately separate
        from ``num_matvecs``: the two levels were never tied together.
    cg_tol : float
        Relative-residual tolerance for the inner PCG.

        **Do not read the returned residual as a quality proxy.** On this
        operator it is anti-correlated with accuracy: a run with relative
        residual 1.42 gave an answer 0.10% from truth while one at 0.91 gave
        7.9%. Neither converged in the residual sense.
    cg_maxiter : int
        Inner PCG iteration cap. Hitting the cap is not an error.
    cg_maxiter_cold : int, optional
        Budget for a genuinely cold solve -- no deflation vectors and no seed,
        i.e. the ring preconditioner alone. Defaults to ``6 * cg_maxiter``.
    k_defl : int
        Deflation rank. Default 50.
    rr_refine : bool
        Rayleigh-Ritz re-extraction of the eigenvector against ``A`` itself,
        rather than taking the Lanczos tridiagonal's eigenvector.

        This closes cases no plain budget increase could: the Krylov *space* is
        orthonormal to machine precision even when CG's residual has corrupted
        the *selection within it*, so projecting ``A`` onto the space and
        solving the small symmetric problem recovers the variational optimum.

        **The ``trusted`` flag is meaningless when this is on** and is not
        reported; see :func:`agnimhd.objective.growth_rate`.
    factor : {"lu", "cholesky"}
        Dense factorization behind the ``jax_lanczos`` shift-invert. ``H = A -
        sigma I`` is positive definite whenever ``sigma`` sits below the
        spectrum, so Cholesky is legal there and costs half the flops -- but it
        returns NaN rather than raising on an indefinite input, so the guard is
        mandatory. Default ``"lu"``.
    sigma_mode : {"fixed", "adapt"}
        ``"adapt"`` runs a cheap first pass, then re-shifts to
        ``sigma_factor * lambda`` and solves again.

        Measured ranking over a deterministic comparison: ``adapt`` first,
        ``fixed`` second. A third mode, ``track``, which re-based the shift on
        the previous step's eigenvalue, is **not implemented**: it degrades as
        lambda approaches zero and a tracked excursion can end worse than it
        started.
    sigma_factor : float
        Shift multiplier for ``sigma_mode="adapt"``. Default 2.5.
    seed : int
        Seed for the Lanczos start vector, so a run is reproducible. The AGNI
        solve is deterministic; repeated runs are reproducibility checks, not
        statistical samples.

    Raises
    ------
    ValueError
        For an unknown ``eigensolver``, ``factor`` or ``sigma_mode``.
    """

    eigensolver: str = "eigsh"
    sigma: float = -1e-1
    num_matvecs: int = 50
    coarse_num_matvecs: int = 100
    cg_tol: float = 1e-10
    cg_maxiter: int = 8000
    cg_maxiter_cold: int = None
    k_defl: int = 50
    rr_refine: bool = False
    factor: str = "lu"
    sigma_mode: str = "fixed"
    sigma_factor: float = 2.5
    eigsh_tol: float = 1e-8
    seed: int = 0

    _VALID_EIGENSOLVERS = ("eigsh", "jax_lanczos", "pcg_deflated")
    _VALID_FACTORS = ("lu", "cholesky")
    _VALID_SIGMA_MODES = ("fixed", "adapt")

    def __post_init__(self):
        """Validate the string options against their allowed values."""
        errorif(
            self.eigensolver not in self._VALID_EIGENSOLVERS,
            ValueError,
            f"eigensolver must be one of {self._VALID_EIGENSOLVERS}, got "
            f"{self.eigensolver!r}.",
        )
        errorif(
            self.factor not in self._VALID_FACTORS,
            ValueError,
            f"factor must be one of {self._VALID_FACTORS}, got {self.factor!r}.",
        )
        errorif(
            self.sigma_mode not in self._VALID_SIGMA_MODES,
            ValueError,
            f"sigma_mode must be one of {self._VALID_SIGMA_MODES}, got "
            f"{self.sigma_mode!r}. 'track' is deliberately not implemented: it "
            "degrades as lambda approaches zero, and a tracked excursion can "
            "end worse than it started.",
        )
        errorif(
            self.sigma >= 0.0,
            ValueError,
            f"sigma must be negative, got {self.sigma}. The shift has to sit "
            "below the whole spectrum: above it, shift-invert converges to the "
            "wrong mode and H = A - sigma I stops being positive definite, so "
            "the preconditioned CG is not a legal Krylov method.",
        )

    @property
    def adapt(self):
        """bool : whether the two-pass adaptive shift is active."""
        return self.sigma_mode == "adapt"

    @property
    def cold_budget(self):
        """int : CG budget for a genuinely cold solve."""
        return (
            6 * self.cg_maxiter
            if self.cg_maxiter_cold is None
            else self.cg_maxiter_cold
        )

    def replace(self, **changes):
        """Return a copy with fields replaced."""
        return replace(self, **changes)
