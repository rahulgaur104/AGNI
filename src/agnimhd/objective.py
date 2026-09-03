"""Solve mode and optimize mode.

The two modes are separate entry points and the separation is enforced.

**Solve mode** -- :func:`growth_rate`, :func:`eigenpair` -- returns the
stability of a stored :class:`~agnimhd.EquilibriumData` and requires no
equilibrium code. It is not differentiable: ``dlambda/d(EquilibriumData)`` is a
sensitivity to the metric, Jacobian, current and profiles as sampled on the
grid, which are not free parameters and are not independent, since they satisfy
force balance because an equilibrium solve made them do so. A step along that
derivative gives arrays that are not in force balance.

**Optimize mode** -- :func:`growth_rate_of` -- takes the design parameters and
a differentiable map from them to an equilibrium, and differentiates
``dlambda/dp = dlambda/d(eq) x d(eq)/dp``. The left factor is computed here;
the right factor comes from the map, which is an equilibrium solve. DESC
supplies that map and the outer optimizer; see ``docs/adapters.md``.

How the derivative works
------------------------

The quantity returned is the Rayleigh quotient
:math:`\\lambda = v^T A(q) v / v^T v` at the eigenvector ``v``. By
Hellmann-Feynman, at an eigenvector the eigenvalue's derivative is the
derivative of the quotient **holding the vector fixed**, so no derivative of
the eigensolve is needed -- only one operator application per cotangent. The
eigensolve is therefore wrapped in a ``jax.custom_vjp`` with a **zero**
backward rule: ``v`` reaches the quotient as a constant, and ordinary autodiff
of ``v^T A(q) v`` is exactly the contraction. Two consequences, both
load-bearing: the eigensolve need not be differentiable (ARPACK behind a
``pure_callback`` is fine), and ``v`` is still recomputed at every call -- a
fixed-vector gradient, not a stale-vector one. This also removes the
eigenvector-selection ``argmax``, which has no useful derivative.

Accuracy
--------

Validated against central finite differences at **0.45% agreement**, and only
at ``h = 1e-7``: larger steps are dominated by the quotient's curvature,
smaller ones by the eigenvalue's **relative** noise floor of 2.8e-5. A step
outside that window disagrees with a correct gradient and looks like a bug in
it. In optimize mode there is a second requirement AGNI cannot enforce -- the
equilibrium must be converged at **both** points, or the difference measures a
solver residual.
"""

import numpy as np

from .assemble import assemble_dense, matfree_operator, operator_dtype
from .backend import errorif, jax, jnp
from .config import AssemblyConfig, SolverConfig

__all__ = ["growth_rate", "eigenpair", "growth_rate_of", "growth_rate_and_grad"]


# ---------------------------------------------------------------------------
# Primal eigensolves
# ---------------------------------------------------------------------------


def _eigsh_host(A, sigma, tol, seed):
    """Shift-invert ARPACK on the dense matrix. Runs on the host.

    Measured **1.53x faster than the hand-rolled JAX Lanczos on CPU**, which is
    why it is the default wherever the dense matrix fits. It is not
    differentiable, and does not need to be: the derivative rule discards it.

    ``v0`` is supplied from ``seed`` rather than left to ARPACK's own random
    start. The AGNI solve is deterministic and repeated runs are
    reproducibility checks, not statistical samples -- a random start would
    make two calls at the same equilibrium differ at the eigensolve tolerance
    and quietly turn every exact comparison into an approximate one.
    """
    from scipy.sparse.linalg import eigsh

    A_np = np.asarray(A)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(A_np.shape[0])
    if np.iscomplexobj(A_np):
        # A complex Hermitian A (axisym=True) needs a complex start. A real v0
        # is not merely a worse guess: ARPACK dispatches on the dtype pair, and
        # the real-symmetric driver on a complex matrix is the wrong algorithm.
        # The real branch draws the same first n normals, so the real case's
        # measured eigenvalues are unchanged.
        v0 = v0 + 1j * rng.standard_normal(A_np.shape[0])
    w, v = eigsh(
        A_np,
        k=1,
        sigma=sigma,
        which="LM",
        tol=tol,
        v0=v0 / np.linalg.norm(v0),
        return_eigenvectors=True,
    )
    return (
        np.asarray(v[:, 0], dtype=A_np.dtype),
        np.asarray(w[0], dtype=A_np.dtype),
    )


def _lanczos_at(A, sigma, config):
    """One exact-factorization shift-invert Lanczos solve at a fixed shift.

    Returns ``(v, lam, ok)``. ``ok`` is a traced boolean: with
    ``factor="cholesky"``, ``potrf`` returns NaN rather than raising on an
    indefinite ``A - sigma I``, so the guard is mandatory rather than
    defensive. The check is applied to the outputs as well as the factor,
    because a poisoned factor poisons everything downstream of it.
    """
    from matfree import decomp, eig

    n = A.shape[0]
    H = A.at[jnp.diag_indices(n)].add(-sigma)

    if config.factor == "cholesky":
        # `cho_factor` is `potrf`: half the flops of `getrf`, legal because
        # H is SPD whenever sigma sits below the whole spectrum. `lower=True`
        # is not cosmetic -- JAX's `lower=False` path transposes the matrix
        # twice, which at production size is another two full-size temporaries.
        fac = jax.scipy.linalg.cho_factor(H, lower=True)
        ok = jnp.isfinite(fac[0]).all()
        opinv = lambda b: jax.scipy.linalg.cho_solve(fac, b)  # noqa: E731
    else:
        fac = jax.scipy.linalg.lu_factor(H)
        ok = jnp.array(True)
        opinv = lambda b: jax.scipy.linalg.lu_solve(fac, b)  # noqa: E731

    tri = decomp.tridiag_sym(config.num_matvecs, reortho="full", materialize=True)
    alg = eig.eigh_partial(tri)
    v0 = jnp.asarray(
        np.random.default_rng(config.seed).standard_normal(n), dtype=A.dtype
    )
    v0 = v0 / jnp.linalg.norm(v0)
    mu, vecs = alg(opinv, v0)

    # Largest |mu| is the eigenvalue closest to the shift, which is the softest
    # mode when sigma sits below the spectrum.
    idx = jnp.argmax(jnp.abs(mu))
    v = vecs[idx]
    mu_i = mu[idx]
    lam = sigma + 1.0 / jnp.where(mu_i == 0, jnp.inf, mu_i)
    ok = ok & jnp.isfinite(lam) & jnp.isfinite(v).all()
    return v, lam, ok


def _lanczos(A, config):
    """Shift-invert Lanczos, with the optional adaptive second pass.

    In ``sigma_mode="adapt"`` a cheap first pass supplies a better shift,
    ``sigma_factor * lambda``, and the solve is repeated. The second shift comes
    from an *estimate*, so it can land above the smallest eigenvalue and make
    ``A - sigma I`` indefinite; the result is selected back to the first pass
    with ``jnp.where``, which is a select and therefore safe with NaN on the
    discarded branch and fixed-shape under ``jit``.
    """
    v, lam, _ = _lanczos_at(A, config.sigma, config)
    if not config.adapt:
        return v, lam

    sigma2 = config.sigma_factor * jax.lax.stop_gradient(lam)
    sigma2 = jnp.where(jnp.isfinite(sigma2) & (sigma2 < 0), sigma2, config.sigma)
    v2, lam2, ok2 = _lanczos_at(A, sigma2, config)
    return jnp.where(ok2, v2, v), jnp.where(ok2, lam2, lam)


def _primal(eq, diffmat, assembly, solver, n_keep):
    """Return ``(v, lam)`` at the current point. Not differentiated.

    ``eigsh`` goes through ``jax.pure_callback``, which is what lets a host
    ARPACK call sit inside an otherwise jitted, traceable function.
    """
    if solver.eigensolver == "jax_lanczos":
        A = assemble_dense(eq, diffmat, assembly)["A"]
        return _lanczos(A, solver)

    if solver.eigensolver == "eigsh":
        # BOTH pytrees are flattened and passed through the callback as
        # arguments. Closing over `diffmat` instead would work eagerly and then
        # fail under `jit` with an UnexpectedTracerError: under trace `diffmat`
        # is a pytree of tracers, and a tracer captured by a host callback has
        # escaped its transformation. `jit` from outside the package is a
        # requirement, so a closure that only works eagerly is not an option.
        eq_leaves, eq_def = jax.tree_util.tree_flatten(eq)
        dm_leaves, dm_def = jax.tree_util.tree_flatten(diffmat)
        n_eq = len(eq_leaves)

        def _host(leaves):
            from jax.tree_util import tree_unflatten

            eq_h = tree_unflatten(eq_def, list(leaves[:n_eq]))
            dm_h = tree_unflatten(dm_def, list(leaves[n_eq:]))
            A = assemble_dense(eq_h, dm_h, assembly)["A"]
            return _eigsh_host(A, solver.sigma, solver.eigsh_tol, solver.seed)

        # NOT the default float dtype: `axisym=True` assembles a complex
        # Hermitian operator, and `pure_callback` casts the host result to
        # whatever is declared here rather than checking it.
        dtype = operator_dtype(assembly)
        return jax.pure_callback(
            _host,
            (
                jax.ShapeDtypeStruct((n_keep,), dtype),
                jax.ShapeDtypeStruct((), dtype),
            ),
            tuple(eq_leaves) + tuple(dm_leaves),
        )

    raise NotImplementedError(
        f"eigensolver {solver.eigensolver!r} is not wired into growth_rate yet. "
        "The two-level 'pcg_deflated' path is exercised through "
        "agnimhd.solvers; see docs/resolution.md for its coarse-level floor."
    )


# ---------------------------------------------------------------------------
# The Hellmann-Feynman quotient: the inner factor of the chain rule
# ---------------------------------------------------------------------------


def _lambda_hf(eq, diffmat, assembly, solver):
    """``lambda`` at ``eq``, differentiable in ``eq`` by Hellmann-Feynman.

    The inner factor of the chain rule, kept private because it is not a
    derivative with respect to any design variable. The public route is
    :func:`growth_rate_of`, which requires the outer factor.
    """
    op = matfree_operator(eq, diffmat, assembly)
    n_keep = op["n_keep"]

    @jax.custom_vjp
    def _v_of(eq_d):
        """The eigenvector at the current point, with a zero derivative rule."""
        v, _ = _primal(eq_d, diffmat, assembly, solver, n_keep)
        return v

    def _v_fwd(eq_d):
        return _v_of(eq_d), eq_d

    def _v_bwd(res, _g):
        """Zero cotangent: at an eigenvector the eigensolve's own derivative is
        exactly the term that must not be included. Not an approximation."""
        return (jax.tree_util.tree_map(jnp.zeros_like, res),)

    _v_of.defvjp(_v_fwd, _v_bwd)

    v = _v_of(eq)
    # `Ax` is differentiable in `eq`; `v` is not. Autodiff of this expression is
    # therefore exactly v^T (dA/dq) v / v^T v.
    return jnp.real(jnp.vdot(v, op["Ax"](v)) / jnp.vdot(v, v))


# ---------------------------------------------------------------------------
# Solve mode
# ---------------------------------------------------------------------------


def _check_configs(assembly, solver):
    errorif(
        not isinstance(assembly, AssemblyConfig),
        TypeError,
        "assembly must be an AssemblyConfig. It is static, hashable "
        "configuration -- passing a dict would retrace on every call.",
    )
    errorif(
        not isinstance(solver, SolverConfig),
        TypeError,
        "solver must be a SolverConfig.",
    )


_NO_GRAD = """\
{name} is solve mode and is not differentiable.

d(lambda)/d(EquilibriumData) is a sensitivity to grid samples. They are not
free parameters and are not independent: they satisfy force balance because an
equilibrium solve made them do so, and a step along this derivative gives
arrays that are not in force balance. Supply the map from the design
parameters instead:

    def equilibrium_map(params):            # must be differentiable
        return to_equilibrium_data(solve_equilibrium(params))

    g = jax.grad(agnimhd.growth_rate_of)(params, equilibrium_map, diffmat)

DESC supplies that map and the outer optimizer; see docs/adapters.md.
agnimhd.{name} remains correct for the stability of one stored equilibrium."""


def _forbid_gradient(name, fn, *args):
    """Run ``fn(*args)``; raise if anything tries to differentiate it.

    A raising ``custom_vjp`` rather than a ``stop_gradient``: a zero gradient
    is indistinguishable from an optimization that has converged. The error is
    raised when ``jax.grad`` builds the backward pass.
    """

    @jax.custom_vjp
    def _guarded(*a):
        return fn(*a)

    def _fwd(*a):
        return _guarded(*a), None

    def _bwd(_res, _g):
        raise TypeError(_NO_GRAD.format(name=name))

    _guarded.defvjp(_fwd, _bwd)
    return _guarded(*args)


def eigenpair(eq, diffmat, assembly=None, solver=None):
    """Solve mode: ``(lambda, v, residual)`` for one stored equilibrium.

    Not differentiable; see the module docstring and :func:`growth_rate_of`.

    Parameters
    ----------
    eq : EquilibriumData
    diffmat : DiffMat
    assembly : AssemblyConfig, optional
    solver : SolverConfig, optional

    Returns
    -------
    lam : jax.Array
        The Rayleigh quotient at the computed eigenvector. **Its sign is the
        physics answer**: negative means unstable.
    v : jax.Array, shape (n_keep,)
    residual : jax.Array
        ``||A v - lam v|| / (|lam| ||v||)``. A genuine quality measure, unlike
        the inner CG's relative residual.

    Notes
    -----
    ``lam`` is the Rayleigh quotient, not the eigensolver's reported
    eigenvalue -- they agree to the eigensolve tolerance, and the quotient is
    the quantity the gradient differentiates. Reporting a different number than
    the one being differentiated is how a gradient check ends up chasing a
    discrepancy that is not there.
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver

    def _run(eq_d, dm_d):
        op = matfree_operator(eq_d, dm_d, assembly)
        v, _ = _primal(eq_d, dm_d, assembly, solver, op["n_keep"])
        Av = op["Ax"](v)
        vv = jnp.vdot(v, v)
        lam = jnp.real(jnp.vdot(v, Av) / vv)
        resid = jnp.linalg.norm(Av - lam * v) / (
            jnp.abs(lam) * jnp.sqrt(jnp.real(vv)) + 1e-300
        )
        return lam, v, resid

    return _forbid_gradient("eigenpair", _run, eq, diffmat)


def growth_rate(eq, diffmat, assembly=None, solver=None):
    """Solve mode: squared growth rate of the most unstable finite-n mode.

    One stored equilibrium in, one stability answer out. ``jax.jit`` may be
    applied from outside the package, with the two configs static; ``jax.grad``
    raises. Use :func:`growth_rate_of` for a derivative.

    Parameters
    ----------
    eq : EquilibriumData
        The equilibrium, already solved by somebody else and loaded from disk.
    diffmat : DiffMat
        Differentiation and quadrature operators on the same nodes ``eq`` was
        evaluated on.
    assembly : AssemblyConfig, optional
        Static. Defaults to :class:`~agnimhd.config.AssemblyConfig`.
    solver : SolverConfig, optional
        Static. Defaults to :class:`~agnimhd.config.SolverConfig`.

    Returns
    -------
    jax.Array
        Scalar. **Negative means unstable**, and the magnitude is the squared
        growth rate. Minimizing it is the wrong direction; an optimizer should
        *raise* it toward zero.

    See Also
    --------
    eigenpair : the same solve, plus the eigenvector and a residual.
    growth_rate_of : optimize mode -- the same lambda, over parameters.
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver
    _check_configs(assembly, solver)
    return _forbid_gradient(
        "growth_rate",
        lambda eq_d, dm_d: _lambda_hf(eq_d, dm_d, assembly, solver),
        eq,
        diffmat,
    )


# ---------------------------------------------------------------------------
# Optimize mode
# ---------------------------------------------------------------------------


def _check_map(params, equilibrium_map):
    """Reject the two ways of calling optimize mode that are really solve mode."""
    from .equilibrium import EquilibriumData

    errorif(
        isinstance(params, EquilibriumData),
        TypeError,
        "params is an EquilibriumData, so equilibrium_map has nothing to do "
        "and the derivative would be with respect to grid samples again. "
        "params must be what you control -- boundary or profile coefficients, "
        "coil currents -- and equilibrium_map the differentiable map from them "
        "to an equilibrium, which is an equilibrium solve. See docs/adapters.md.",
    )
    errorif(
        not callable(equilibrium_map),
        TypeError,
        "equilibrium_map must be a callable params -> EquilibriumData, not "
        f"{type(equilibrium_map).__name__}. If you have an EquilibriumData and "
        "only want its stability, that is solve mode: growth_rate(eq, diffmat).",
    )


def growth_rate_of(params, equilibrium_map, diffmat, assembly=None, solver=None):
    """Optimize mode: the growth rate as a function of *your* parameters.

    ``jax.grad`` returns ``dlambda/d(params)``, a pytree shaped like ``params``
    rather than like an ``EquilibriumData``. An optimizer can step along it,
    because every point in parameter space maps through ``equilibrium_map`` to
    an equilibrium.

    Parameters
    ----------
    params : pytree
        Whatever your equilibrium solver is parameterized by: boundary Fourier
        coefficients, profile coefficients, coil currents. Differentiable.
    equilibrium_map : callable
        ``params -> EquilibriumData``, **differentiable in JAX**: the
        equilibrium solve plus your adapter. A Python callable, so it is static
        under ``jax.jit``.
    diffmat : DiffMat
        Operators on the nodes ``equilibrium_map`` evaluates on. Fixed across
        the optimization -- the grid is not a parameter.
    assembly : AssemblyConfig, optional
    solver : SolverConfig, optional

    Returns
    -------
    jax.Array
        Scalar ``lambda``. **Negative means unstable**; an optimizer raises it
        toward zero, so a minimizer wants ``-growth_rate_of(...)``.

    Notes
    -----
    Whether ``equilibrium_map`` solves force balance is not verified here and
    is the caller's responsibility. If the equilibrium code is not
    differentiable the chain rule does not close, and only solve mode is
    available.

    See Also
    --------
    growth_rate : solve mode, for a single stored equilibrium.
    growth_rate_and_grad : value and gradient from one eigensolve.
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver
    _check_configs(assembly, solver)
    _check_map(params, equilibrium_map)
    eq = equilibrium_map(params)
    # The chain closes here and nowhere else: `eq` carries `params`' tracers, so
    # ordinary autodiff of the Hellmann-Feynman quotient in `eq` continues back
    # through `equilibrium_map` to `params`.
    return _lambda_hf(eq, diffmat, assembly, solver)


def growth_rate_and_grad(params, equilibrium_map, diffmat, assembly=None, solver=None):
    """Optimize mode: value and ``dlambda/d(params)`` from a single eigensolve.

    Returns
    -------
    lam : jax.Array
        Scalar growth rate.
    grad : pytree
        Same structure as ``params``, holding ``dlambda/d(each leaf)``.
    """
    return jax.value_and_grad(growth_rate_of)(
        params, equilibrium_map, diffmat, assembly, solver
    )
