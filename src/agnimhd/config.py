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
        stabilizing and does not move marginal stability. A large ``gamma``
        penalizes compression; it does not impose incompressibility. Default
        ``5/3``.
    incompressible : bool
        Exact incompressibility is not implemented: ``True`` raises
        ``NotImplementedError``. Default False.
    axisym : bool
        Treat the equilibrium as axisymmetric, so each displacement component is
        a single toroidal Fourier mode and the operator becomes complex.
        Default False.
    n_mode_axisym : int
        Toroidal mode number analyzed when ``axisym`` is set. Default 1.
    parity : {None, "even", "odd"}
        On a stellarator-symmetric equilibrium, solve one parity block of the
        operator under the reflection ``(rho, theta, zeta) -> (rho, -theta,
        -zeta)`` (:mod:`agnimhd.symmetry`): about half the unknowns, an eighth
        of the dense factorization. The most unstable mode is in one of the
        two blocks; solve both and take the larger ``gamma^2``. Real toroidal
        families only (0 and ``NFP / 2``), not with ``axisym``; the equilibrium
        is checked at every solve. Default None: the full operator.

    Raises
    ------
    ValueError
        For an unknown ``parity`` or ``parity`` with ``axisym``.
    NotImplementedError
        For ``incompressible=True``.
    """

    gamma: float = 5.0 / 3.0
    incompressible: bool = False
    axisym: bool = False
    n_mode_axisym: int = 1
    parity: str = None

    def __post_init__(self):
        """Validate the parity and the axisymmetric options."""
        errorif(
            self.parity not in (None, "even", "odd"),
            ValueError,
            f"parity must be None, 'even' or 'odd', got {self.parity!r}.",
        )
        errorif(
            self.parity is not None and self.axisym,
            ValueError,
            "parity needs the real operator of a toroidal mode family; "
            "axisym=True builds a complex one.",
        )
        errorif(
            self.incompressible,
            NotImplementedError,
            "incompressible=True is not implemented; no assembly reads it. "
            "A large gamma only penalizes compression.",
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

        *Not far above it either*, for a solver with a fixed step count.
        Shift-invert maps ``gamma^2`` to ``mu = 1/(sigma - gamma^2)``, and
        Lanczos separates two modes at a rate set by the *ratio* of their
        ``mu``. As ``sigma`` grows, every ``mu`` collapses onto ``1/sigma`` and
        the ratio goes to one. Measured on the shipped 24x12x8 case, whose
        ``gamma^2`` starts ``1.34e-4, 6.25e-5``, then a cluster of numerically
        null modes at ``-1e-11``:

        =========  ==============================  ================================
        ``sigma``  ``mu[0]/mu[1]``                 ``dense``, 50 steps
        =========  ==============================  ================================
        ``1e-1``   1.0007                          wrong mode, ``gamma^2 < 0``
        ``1e-2``   1.0075                          ``1.222757e-04``, residual 1.6e4
        ``1e-3``   1.0823                          ``1.337627e-04`` (exact)
        =========  ==============================  ================================

        ``"eigsh"`` and ``"dense"`` iterate to a tolerance (``eigsh_tol``,
        ``dense_tol``), so a far shift costs them steps, not the answer. The
        default ``1e-1`` is deliberately conservative about the side that has
        no recovery, and the default ``num_matvecs`` is sized for it. A
        ``"dense"`` solve that reaches ``num_matvecs`` before ``dense_tol``
        returns its last Ritz pair, which at a far shift can be the wrong
        mode. **It is not silent**: the Rayleigh residual from
        :func:`agnimhd.objective.eigenpair` was 4.6e+04 for the 50-step vector
        at ``1e-1`` against 1.6e-04 for the converged one. Check it, and move
        ``sigma`` nearer or raise ``num_matvecs``. A smaller ``gamma^2`` or a
        tighter gap to the next mode needs more steps at the same shift.
        ``"jd"`` wants ``sigma`` just above ``gamma^2`` (see above).

        ``sigma_mode="adapt"`` needs a right first pass. With 50 steps at
        ``1e-1`` the first pass returned a negative ``gamma^2``, the guard
        rejected it, and the second pass repeated the first. Re-shifting to
        ``sigma_factor * |gamma^2|`` instead chased the numerically null cluster
        down to ``sigma = 1e-10`` and converged there.
    num_matvecs : int
        ``"dense"``: the most Lanczos steps of the fine solve, which stops
        earlier at ``dense_tol``. Capped at the matrix size ``n``. Default 500,
        sized for the default ``sigma``. On the shipped case the two largest
        ``mu`` there differ by 7e-4 of their size, and the Chebyshev bound for
        that separation brings a random start to a residual of ``1e-8`` within
        about 440 steps. A fixed 50 steps returned the wrong mode, 200 the right
        eigenvalue.

        Memory: the Krylov basis is kept whole for the reorthogonalization,
        ``num_matvecs x n`` entries. That is 4 kB per unknown in float64 at the
        default (8 kB complex), the fraction ``num_matvecs / n`` of the dense
        matrix: 7% beside the shipped case's 6,720 unknowns, about 1% at
        40,000. Every step also projects on the whole basis, at about twice
        that fraction of the flops of one solve with the factor.
    dense_tol : float
        ``"dense"``: stop when the residual of the shift-inverted problem,
        ``||(A + sigma I)^-1 y - mu y||`` for the Ritz pair ``(mu, y)``, is at
        most ``dense_tol * |mu|``. ARPACK's test, as ``eigsh_tol``. Checked
        every 10 steps. ``0`` runs all ``num_matvecs`` steps. Default 1e-8.
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
    mg_tile, mg_block, mg_iters, mg_tol, mg_theta_tol
        ``"dense_mg"`` only: JAXMg tile width (at most 1024; the matrix is
        padded to a multiple of it times the GPU count), block size (16),
        maximum iterations (6), residual stop (1e-6) and stop at relative
        ``gamma^2`` change (1e-8, 0 = off).

    Raises
    ------
    ValueError
        For an unknown ``eigensolver``, ``factor`` or ``sigma_mode``.
    """

    eigensolver: str = "eigsh"
    sigma: float = 1e-1
    num_matvecs: int = 500
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
    dense_tol: float = 1e-8
    seed: int = 0
    mg_tile: int = 1024
    mg_block: int = 16
    mg_iters: int = 6
    mg_tol: float = 1e-6
    mg_theta_tol: float = 1e-8

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
