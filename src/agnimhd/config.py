"""Static configuration objects.

Everything here is **static**: frozen, hashable dataclasses passed as
non-traced arguments. That is deliberate and load-bearing. Resolution, basis
choice and solver selection drive Python branches and array *shapes*, neither of
which can be derived from a traced value, and holding them as ordinary pytree
leaves means ``jit`` retraces on every call.
"""

from dataclasses import dataclass, replace

from .backend import errorif

__all__ = [
    "AssemblyConfig",
    "SolverConfig",
]


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
    eigensolver : {"eigsh", "dense", "jd", "dense_mg"}
        Which eigensolve to run.

        ``"eigsh"`` assembles the dense matrix and calls SciPy ARPACK. Measured
        **1.53x faster than the hand-rolled JAX Lanczos on CPU**, so it is the
        default wherever the dense matrix fits.

        ``"dense"`` assembles in JAX and runs Lanczos with an exact dense
        shift-invert (Cholesky, ``factor``). Stays on the accelerator: the
        one-GPU dense solver.

        ``"jd"`` never forms the fine dense matrix: matrix-free Jacobi-Davidson
        (:func:`agnimhd.solvers.jacobi_davidson`), preconditioned by the ring
        (block-Jacobi) blocks of ``A + sigma I`` and deflated by the softest
        modes of its coarse level, which it requires (``coarse=`` of
        :func:`agnimhd.objective.growth_rate`, from
        :meth:`agnimhd.Basis.coarse_level`): the same radial nodes and Fourier
        truncation on fewer angular nodes. The path for resolutions where the
        dense matrix does not fit; ``sigma`` should sit just above ``gamma^2``
        (DESC used ``1.3 * gamma^2``).

        ``"dense_mg"`` splits the dense matrix over all visible GPUs and runs
        block inverse iteration with JAXMg's Cholesky solve and Rayleigh-Ritz on
        the exact operator (:mod:`agnimhd.multigpu`). ``sigma`` must lie above
        the largest ``gamma^2`` and close to it. Needs ``jaxmg``.
    sigma : float
        Shift for the shift-invert, above the largest ``gamma^2 = -lambda``
        (e.g. ``1.05 * gamma2_estimate``); the solvers shift ``A`` by ``-sigma``
        (:attr:`shift`). The constraint is **two-sided**, and only one side of
        it is obvious.

        *Not below the largest gamma^2.* There the solve converges to the wrong
        mode, and for the deflated path ``H = A + sigma I`` stops being
        positive definite, so the preconditioned CG is not a legal Krylov
        method at all.

        *Not arbitrarily far above it either*, for any solver that stops at a
        fixed matvec count rather than at a tolerance -- which means
        ``"dense"`` and ``"jd"``, but not ``"eigsh"``.
        Shift-invert maps ``gamma^2`` to ``mu = 1/(sigma - gamma^2)``, and
        Lanczos separates two modes at a rate set by the *ratio* of their
        ``mu``. As ``sigma`` grows, every ``mu`` collapses onto ``1/sigma`` and
        the ratio goes to one. Measured on the shipped 24x12x8 case, whose
        ``gamma^2`` starts ``1.34e-4, 6.25e-5``, then a cluster of numerically
        null modes at ``-1e-11``:

        =========  ==============================  ================================
        ``sigma``  ``mu[0]/mu[1]``                 ``dense``, 50 matvecs
        =========  ==============================  ================================
        ``1e-1``   1.0007                          wrong mode, ``gamma^2 < 0``
        ``1e-2``   1.0075                          ``1.222757e-04``, residual 1.6e4
        ``1e-3``   1.0823                          ``1.337627e-04`` (exact)
        =========  ==============================  ================================

        The default is ``1e-1``, which is safe for the default ``"eigsh"``
        because ARPACK iterates to ``eigsh_tol`` instead of stopping at a fixed
        count, and it is deliberately conservative about the side that has no
        recovery. On the shipped case that same shift makes a 50-matvec
        ``dense`` return ``-1.598e-04`` -- the wrong sign, and therefore
        the wrong physics answer. **It is not silent**: the Rayleigh residual
        from :func:`agnimhd.objective.eigenpair` is 4.6e+04 for that vector
        against 1.6e-04 for the converged one. Check it. Raising
        ``num_matvecs`` to 200 also recovers the right mode at the far shift, at
        four times the cost of moving ``sigma``.

        ``sigma_mode="adapt"`` does **not** rescue a shift this far out. Its
        first pass returns a negative ``gamma^2``, the guard rejects it, and the
        second pass repeats the first at the same bad shift. Re-shifting to
        ``sigma_factor * |gamma^2|`` instead is worse, not better: from a wrong
        first pass it chases the numerically null cluster down to
        ``sigma = 1e-10`` and converges there.
    num_matvecs : int
        Lanczos matvec count for the fine solve. Default 50.

        This value is **fixed and untuned** -- it was never swept. It is a
        plausible knob for further speedup, not a converged choice.
    coarse_num_matvecs : int
        Lanczos steps for the coarse generalized solve. Deliberately separate
        from ``num_matvecs``: the two levels were never tied together.
    k_defl : int
        Deflation rank. Default 50.
    jd_outer, jd_inner, jd_maxdim, jd_keep, jd_tol, jd_theta_tol
        ``"jd"`` only: outer iterations (200), projected PCG steps per
        correction (100), basis size at restart (60), vectors kept (10), stop
        when the returned vector's eigen-residual ``||A v - theta v|| / |theta|``
        is at most ``jd_tol`` (0 = off) or at relative Ritz-value change
        ``jd_theta_tol`` (1e-8). DESC's defaults. For gradients use the
        residual stop: the Ritz-value stop left residuals of 0.5 to 1.7 and
        gradients up to 3.3e-2 off in DESC.
    ring_batch : int
        Rings of the ring preconditioner assembled at once, on both JD levels
        (24, as the production drivers). Lower it if the build runs out of
        memory.
    factor : {"lu", "cholesky"}
        Dense factorization behind the ``dense`` shift-invert. ``H = A +
        sigma I`` is positive definite whenever ``sigma`` lies above the
        largest ``gamma^2``, so Cholesky is legal there and costs half the
        flops -- but it returns NaN rather than raising on an indefinite input,
        so the guard is mandatory. Default ``"cholesky"``.
    sigma_mode : {"fixed", "adapt"}
        ``"adapt"`` runs a cheap first pass, then re-shifts to
        ``sigma = sigma_factor * gamma^2`` and solves again.

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
    mg_tile, mg_block, mg_iters, mg_tol
        ``"dense_mg"`` only: JAXMg tile width (at most 1024; the matrix is
        padded to a multiple of it times the GPU count), block size (16),
        maximum iterations (6) and residual stop (1e-6).

    Raises
    ------
    ValueError
        For an unknown ``eigensolver``, ``factor`` or ``sigma_mode``.
    """

    eigensolver: str = "eigsh"
    sigma: float = 1e-1
    num_matvecs: int = 50
    coarse_num_matvecs: int = 100
    k_defl: int = 50
    jd_outer: int = 200
    jd_inner: int = 100
    jd_maxdim: int = 60
    jd_keep: int = 10
    jd_tol: float = 0.0
    jd_theta_tol: float = 1e-8
    ring_batch: int = 24
    factor: str = "cholesky"
    sigma_mode: str = "fixed"
    sigma_factor: float = 2.5
    eigsh_tol: float = 1e-8
    seed: int = 0
    mg_tile: int = 1024
    mg_block: int = 16
    mg_iters: int = 6
    mg_tol: float = 1e-6

    _VALID_EIGENSOLVERS = ("eigsh", "dense", "jd", "dense_mg")
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
            self.sigma <= 0.0,
            ValueError,
            f"sigma must be positive, got {self.sigma}. It has to lie above the "
            "largest gamma^2 = -lambda: below it, shift-invert converges to the "
            "wrong mode and H = A + sigma I stops being positive definite, so "
            "the preconditioned CG is not a legal Krylov method.",
        )

    @property
    def adapt(self):
        """bool : whether the two-pass adaptive shift is active."""
        return self.sigma_mode == "adapt"

    @property
    def shift(self):
        """float : ``-sigma``, the shift of ``A`` inside the solvers."""
        return -self.sigma

    def replace(self, **changes):
        """Return a copy with fields replaced."""
        return replace(self, **changes)
