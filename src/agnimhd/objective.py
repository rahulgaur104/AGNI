"""Solve mode and optimize mode.

The two modes are separate entry points and the separation is enforced.

**Solve mode** -- :func:`growth_rate`, :func:`eigenpair` -- returns the
stability of a stored :class:`~agnimhd.EquilibriumData` and requires no
equilibrium code. It is not differentiable: ``dlambda/d(EquilibriumData)`` is a
sensitivity to the metric, Jacobian, current and profiles as sampled on the
grid, which are not free parameters and are not independent, since they satisfy
force balance because an equilibrium solve made them do so. A step along that
derivative gives arrays that are not in force balance.

**Optimize mode** -- :func:`growth_rate_of` -- takes the equilibrium's
parameters and a differentiable map from them to an ``EquilibriumData``, and
returns ``d(gamma^2)/dp = d(gamma^2)/d(eq) x d(eq)/dp``. The map evaluates geometry
and profiles and contains no equilibrium solve, so this is a partial derivative
at a fixed force balance residual. Force balance is a constraint on the
optimization and is enforced by the optimizer, which in DESC is
``ProximalProjection``: the equilibrium is perturbed and re-solved onto the
constraint after each step, and the reduced derivative
``dg/dc = @g/@c - (@g/@x)(@F/@x)^-1 (@F/@c)``, ``g = gamma^2``, is assembled
from the force balance residual ``F``. See ``docs/index.md``.

How the derivative works
------------------------

The quantity returned is the squared growth rate :math:`\\gamma^2 = -\\lambda`,
minus the Rayleigh quotient :math:`\\lambda = v^T A(q) v / v^T v` at the
eigenvector ``v``: positive means unstable. By
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

from functools import partial

import numpy as np

from .assemble import (
    assemble_dense,
    assemble_rows,
    keep_indices,
    matfree_operator,
    operator_dtype,
)
from .backend import errorif, jax, jit, jnp
from .config import AssemblyConfig, SolverConfig
from .solvers import (
    build_ring_blocks,
    coarse_seed_and_deflation,
    factor_ring_blocks_traced,
    jacobi_davidson,
    lanczos_shift_invert,
    level_meta,
    make_block_precond,
    ring_index_maps,
)

__all__ = [
    "growth_rate",
    "eigenpair",
    "growth_rate_of",
    "growth_rate_and_grad",
    "coarse_space",
]


# ---------------------------------------------------------------------------
# Primal eigensolves
# ---------------------------------------------------------------------------


def _eigsh_host(A, sigma, tol, seed, v0=None):
    """Shift-invert ARPACK on the dense matrix. Runs on the host.

    Measured **1.53x faster than the hand-rolled JAX Lanczos on CPU**, which is
    why it is the default wherever the dense matrix fits. It is not
    differentiable, and does not need to be: the derivative rule discards it.

    ``v0`` is the caller's warm start, or else supplied from ``seed`` rather
    than left to ARPACK's own random start. The AGNI solve is deterministic
    and repeated runs are reproducibility checks, not statistical samples -- a
    random start would make two calls at the same equilibrium differ at the
    eigensolve tolerance and turn every exact comparison into an approximate one.
    """
    from scipy.sparse.linalg import eigsh

    A_np = np.asarray(A)
    if v0 is None:
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(A_np.shape[0])
        if np.iscomplexobj(A_np):
            # A complex Hermitian A (axisym=True) needs a complex start. A real
            # v0 is not merely a worse guess: ARPACK dispatches on the dtype
            # pair, and the real-symmetric driver on a complex matrix is the
            # wrong algorithm. The real branch draws the same first n normals,
            # so the real case's measured eigenvalues are unchanged.
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


def _lanczos_at(A, sigma, config, v0=None):
    """One exact-factorization shift-invert Lanczos solve at a fixed shift.

    Returns ``(v, lam, ok)``. ``ok`` is a traced boolean: with
    ``factor="cholesky"``, ``potrf`` returns NaN rather than raising on an
    indefinite ``A - sigma I``, so the guard is mandatory rather than
    defensive. The check is applied to the outputs as well as the factor,
    because a poisoned factor poisons everything downstream of it.
    """
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

    v, lam = lanczos_shift_invert(
        opinv, n, A.dtype, sigma, config.num_matvecs, config.seed, v0
    )
    ok = ok & jnp.isfinite(lam) & jnp.isfinite(v).all()
    return v, lam, ok


def _lanczos(A, config, v0=None):
    """Shift-invert Lanczos, with the optional adaptive second pass.

    In ``sigma_mode="adapt"`` a cheap first pass supplies a better shift,
    ``sigma_factor * lambda``, and the solve is repeated. The second shift comes
    from an *estimate*, so it can land above the smallest eigenvalue and make
    ``A - sigma I`` indefinite; the result is selected back to the first pass
    with ``jnp.where``, which is a select and therefore safe with NaN on the
    discarded branch and fixed-shape under ``jit``.
    """
    v, lam, _ = _lanczos_at(A, config.shift, config, v0)
    if not config.adapt:
        return v, lam

    sigma2 = config.sigma_factor * jax.lax.stop_gradient(lam)
    sigma2 = jnp.where(jnp.isfinite(sigma2) & (sigma2 < 0), sigma2, config.shift)
    v2, lam2, ok2 = _lanczos_at(A, sigma2, config, v0)
    return jnp.where(ok2, v2, v), jnp.where(ok2, lam2, lam)


class _ShiftedSolver:
    """``solver`` with ``sigma`` replaced, e.g. by a traced scalar; everything
    else is the solver's own setting. The solvers shift ``A`` by ``-sigma``."""

    def __init__(self, solver, sigma):
        self._solver = solver
        self.sigma = sigma
        self.shift = -sigma

    def __getattr__(self, name):
        return getattr(self._solver, name)


#: A solve whose gamma^2 comes out above ``sigma / SHIFT_MARGIN`` was too close to
#: the shift (a shift below the largest gamma^2 gives a wrong mode): it is redone.
SHIFT_MARGIN = 1.3


def _ring_blocks(eq, diffmat, assembly, solver, op):
    """Ring blocks of ``A - sigma I``, ``solver.ring_batch`` rings at a time, and
    their group index map ``G``."""
    res = (op["n_rho"], op["n_theta"], op["n_zeta"])
    sel, pad, G = ring_index_maps(keep_indices(*res), res)
    blocks = build_ring_blocks(
        eq, diffmat, assembly, res, sel, pad, solver.shift, batch=solver.ring_batch
    )
    return blocks, G


def _jd(eq, diffmat, assembly, solver, v0, Z):
    """Matrix-free Jacobi-Davidson: ring preconditioner at ``sigma``, deflated
    by ``Z``, started from ``v0``."""
    op = matfree_operator(eq, diffmat, assembly)
    blocks, G = _ring_blocks(eq, diffmat, assembly, solver, op)
    M = make_block_precond(factor_ring_blocks_traced(blocks)[0], G, op["n_keep"])
    v0 = jnp.asarray(v0, dtype=operator_dtype(assembly, diffmat))
    kw = ("outer", "inner", "maxdim", "keep", "tol", "theta_tol")
    kw = {k: getattr(solver, "jd_" + k) for k in kw}
    theta, v, _ = jacobi_davidson(op["Ax"], M, v0, Z, sigma=solver.shift, **kw)
    return v, theta


_COARSE = (
    "coarse must be basis.coarse_level(eq_c, family), eq_c on the nodes of "
    "basis.coarse(n_theta, n_zeta) (the fine radial nodes, mpol and ntor), or "
    "the (v0, Z) that coarse_space returned for this level."
)


def _coarse_space(coarse, diffmat, assembly, solver, op_f):
    """``(v0, Z)``: ``Z`` the ``k_defl`` softest modes of the coarse pencil
    ``(A_c - sigma I, M_ring,c)``, interpolated to the fine angular nodes, ``v0``
    the sum of the ``jd_keep`` softest, normalized. Both pencil matrices are
    assembled in chunks: ``assemble_rows``' row batches and
    ``solver.ring_batch`` rings at a time. A solver aid: no derivative flows
    through it."""
    eq_c, dm_c, (theta, zeta) = jax.lax.stop_gradient(tuple(coarse))
    if assembly.coupled_rt:  # the coupled (rho, theta) block is the coarse level's
        assembly = assembly.replace(
            n_rho_coupled=eq_c.n_rho, n_theta_coupled=eq_c.n_theta
        )
    op_c = matfree_operator(eq_c, dm_c, assembly)
    n_c, res_c = op_c["n_keep"], (op_c["n_rho"], op_c["n_theta"], op_c["n_zeta"])
    errorif(
        res_c[0] != op_f["n_rho"]
        or theta.shape != (op_f["n_theta"], res_c[1])
        or zeta.shape != (op_f["n_zeta"], res_c[2])
        or operator_dtype(assembly, dm_c) != operator_dtype(assembly, diffmat),
        ValueError,
        _COARSE,
    )
    blocks, G = _ring_blocks(eq_c, dm_c, assembly, solver, op_c)
    Hc = assemble_rows(eq_c, dm_c, assembly, shift=solver.shift)
    P = jnp.eye(res_c[0]), theta, zeta
    k = min(solver.k_defl, n_c - 1), min(solver.coarse_num_matvecs, n_c - 1)
    meta = level_meta(op_c), level_meta(op_f)
    _, Z, _ = coarse_seed_and_deflation(Hc, blocks, G, *meta, *P, *k)
    # The start: the jd_keep softest modes, normalized and summed. Under the
    # stellarator reflection every mode is even or odd, and JD keeps the parity
    # of its start; the softest coarse mode alone can be even while the softest
    # fine mode is odd, and JD then returns the softest even mode.
    modes = Z[:, : solver.jd_keep]
    v0 = jnp.sum(modes / jnp.linalg.norm(modes, axis=0), axis=1)
    return jax.lax.stop_gradient((v0, Z))


def coarse_space(eq, diffmat, coarse, assembly=None, solver=None):
    """The start vector and deflation space ``(v0, Z)`` of ``eigensolver="jd"``.

    ``coarse=(v0, Z)`` in a later solve at the same equilibrium skips the
    coarse level, which can then be built elsewhere, e.g. on a CPU node.

    Parameters
    ----------
    eq, diffmat : EquilibriumData, DiffMat
        The fine level.
    coarse : tuple
        From :meth:`agnimhd.Basis.coarse_level`.
    assembly, solver : AssemblyConfig, SolverConfig, optional

    Returns
    -------
    v0 : jax.Array, shape (n_keep,)
    Z : jax.Array, shape (n_keep, k_defl)
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver
    op = matfree_operator(eq, diffmat, assembly)
    return _coarse_space(coarse, diffmat, assembly, solver, op)


def _start(op, diffmat, assembly, solver, v_guess, coarse):
    """``(v0, Z)``: the warm start (``v_guess`` beats the coarse seed) and the
    deflation space (None without a coarse level, which ``"jd"`` requires)."""
    errorif(
        coarse is None and solver.eigensolver == "jd",
        ValueError,
        'eigensolver="jd" needs its coarse level (without it JD stalled on the '
        "test case): coarse=basis.coarse_level(eq_c, family), eq_c the "
        "equilibrium on the nodes of basis.coarse(n_theta, n_zeta), e.g. "
        "eq, diffmat, coarse = from_desc(eq, basis, family, coarse=basis.coarse()).",
    )
    valid = None
    if isinstance(v_guess, tuple):  # (vector, valid): the vector counts only if valid
        v_guess, valid = v_guess
        errorif(
            coarse is None, ValueError, "v_guess=(vector, valid) needs a coarse level"
        )
    v0 = None if v_guess is None else _as_reduced(v_guess, op, "v_guess")
    if coarse is None:
        return v0, None
    errorif(len(coarse) not in (2, 3), ValueError, _COARSE)
    if len(coarse) == 3:
        seed, Z = _coarse_space(coarse, diffmat, assembly, solver, op)
    else:
        seed, Z = _as_reduced(coarse[0], op, "coarse v0"), jnp.asarray(coarse[1])
        errorif(Z.shape[0] != op["n_keep"], ValueError, _COARSE)
    if v0 is None:
        return seed, Z
    return (v0 if valid is None else jnp.where(valid, v0, seed)), Z


def _primal(eq, diffmat, assembly, solver, n_keep, v0=None, Z=None):
    """``(v, value)`` at the current point; callers keep only ``v``. Not differentiated.

    ``eigsh`` goes through ``jax.pure_callback``, which is what lets a host
    ARPACK call sit inside an otherwise jitted, traceable function.
    """
    if solver.eigensolver == "jd":
        return _jd(eq, diffmat, assembly, solver, v0, Z)

    if solver.eigensolver == "dense_mg":
        from .multigpu import dense_mg

        return dense_mg(eq, diffmat, assembly, solver, v0)

    if solver.eigensolver == "jax_lanczos":
        A = assemble_dense(eq, diffmat, assembly)["A"]
        return _lanczos(A, solver, v0)

    if solver.eigensolver == "eigsh":
        # BOTH pytrees and the warm start are passed through the callback as
        # arguments. Closing over `diffmat` instead would work eagerly and then
        # fail under `jit` with an UnexpectedTracerError: under trace `diffmat`
        # is a pytree of tracers, and a tracer captured by a host callback has
        # escaped its transformation. `jit` from outside the package is a
        # requirement, so a closure that only works eagerly is not an option.
        eq_leaves, eq_def = jax.tree_util.tree_flatten(eq)
        dm_leaves, dm_def = jax.tree_util.tree_flatten(diffmat)
        n_eq, n_dm = len(eq_leaves), len(dm_leaves)

        def _host(leaves):
            from jax.tree_util import tree_unflatten

            eq_h = tree_unflatten(eq_def, list(leaves[:n_eq]))
            dm_h = tree_unflatten(dm_def, list(leaves[n_eq : n_eq + n_dm]))
            A = assemble_dense(eq_h, dm_h, assembly)["A"]
            v0_h = leaves[n_eq + n_dm] if len(leaves) > n_eq + n_dm else None
            return _eigsh_host(A, solver.shift, solver.eigsh_tol, solver.seed, v0_h)

        # NOT the default float dtype: `axisym=True` and complex toroidal
        # families assemble a complex Hermitian operator, and `pure_callback`
        # casts the host result to whatever is declared here without checking.
        dtype = operator_dtype(assembly, diffmat)
        return jax.pure_callback(
            _host,
            (
                jax.ShapeDtypeStruct((n_keep,), dtype),
                jax.ShapeDtypeStruct((), dtype),
            ),
            tuple(eq_leaves) + tuple(dm_leaves) + (() if v0 is None else (v0,)),
        )

    raise NotImplementedError(solver.eigensolver)


# ---------------------------------------------------------------------------
# The Hellmann-Feynman quotient: the inner factor of the chain rule
# ---------------------------------------------------------------------------


def _as_reduced(v, op, name):
    """``v`` on the kept DOFs; a full ``3 * n_total`` vector is restricted."""
    v = jnp.asarray(v)
    if v.shape == (3 * op["n_total"],):
        v = v[op["keep"]]
    msg = f"{name}: shape {v.shape}, expected ({op['n_keep']},) or (3 * n_total,)."
    errorif(v.shape != (op["n_keep"],), ValueError, msg)
    return v


def _squared_growth_rate(v, Av):
    """``gamma^2 = -v^H A v / v^H v``: the one place a returned value is negated."""
    return -jnp.real(jnp.vdot(v, Av) / jnp.vdot(v, v))


def _lambda_hf(
    eq,
    diffmat,
    assembly,
    solver,
    v_fixed=None,
    v_guess=None,
    coarse=None,
    on_vector=None,
    sigma=None,
):
    """``gamma^2 = -lambda`` at ``eq``, differentiable in ``eq`` by Hellmann-Feynman.

    The inner factor of the chain rule, kept private because it is not a
    derivative with respect to any design variable. The public route is
    :func:`growth_rate_of`, which requires the outer factor. ``v_fixed``,
    ``v_guess`` and ``coarse`` are documented on :func:`growth_rate`; ``sigma``
    and ``on_vector`` on :func:`growth_rate_of`.
    """
    op = matfree_operator(eq, diffmat, assembly)
    n_keep = op["n_keep"]
    shifted = sigma is not None

    # Every array goes in as an argument. A closed-over `diffmat` is a tracer
    # when a caller differentiates a jitted value (DESC's Jacobian does), and
    # `custom_vjp` keeps it as a jaxpr constant that cannot be lowered.
    @jax.custom_vjp
    def _v_of(eq_d, diffmat_d, v0, Z, sig):
        """The eigenvector at the current point, with a zero derivative rule."""
        if not shifted:
            return _primal(eq_d, diffmat_d, assembly, solver, n_keep, v0, Z)[0]
        v, theta = _primal(
            eq_d, diffmat_d, assembly, _ShiftedSolver(solver, sig), n_keep, v0, Z
        )
        too_close = ~jnp.isfinite(theta) | (-theta > sig / SHIFT_MARGIN)
        # Too close to the shift: solve again at the configured shift, which
        # sits far above (``SolverConfig.sigma``).
        return jax.lax.cond(
            too_close,
            lambda: _primal(eq_d, diffmat_d, assembly, solver, n_keep, v0, Z)[0],
            lambda: v,
        )

    def _v_fwd(eq_d, diffmat_d, v0, Z, sig):
        """Forward rule: the eigenvector, and the inputs for zero cotangents."""
        return _v_of(eq_d, diffmat_d, v0, Z, sig), (eq_d, diffmat_d, v0, Z, sig)

    def _v_bwd(res, _g):
        """Zero cotangent: at an eigenvector the eigensolve's own derivative is
        exactly the term that must not be included. Not an approximation."""
        return jax.tree_util.tree_map(jnp.zeros_like, res)

    _v_of.defvjp(_v_fwd, _v_bwd)

    if v_fixed is None:
        start = _start(
            op,
            diffmat,
            assembly,
            _ShiftedSolver(solver, sigma) if shifted else solver,
            v_guess,
            coarse,
        )
        v = _v_of(eq, diffmat, *start, jnp.zeros(()) if sigma is None else sigma)
    else:
        v = jax.lax.stop_gradient(_as_reduced(v_fixed, op, "v_fixed"))
    # `Ax` is differentiable in `eq`; `v` is not. Autodiff of this expression is
    # therefore exactly -v^T (dA/dq) v / v^T v.
    gamma2 = _squared_growth_rate(v, op["Ax"](v))
    if on_vector is not None:  # e.g. to keep them for the next call
        on_vector(jax.lax.stop_gradient(v), jax.lax.stop_gradient(gamma2))
    return gamma2


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
equilibrium solve produced them, and a step along this derivative gives arrays
that are not in force balance. Supply the map from the equilibrium's parameters
instead:

    def equilibrium_map(params):            # geometry and profiles, no solve
        return to_equilibrium_data(evaluate_on_pest_grid(params))

    g = jax.grad(agnimhd.growth_rate_of)(params, equilibrium_map, diffmat)

That derivative is a partial one at fixed force balance residual. Enforcing
force balance is the optimizer's task; see docs/index.md. agnimhd.{name}
remains correct for the stability of one stored equilibrium."""


@partial(jit, static_argnames=("assembly", "solver"))
def _eigenpair_jit(eq, diffmat, assembly, solver, v_guess, coarse):
    """The whole solve of :func:`eigenpair` as one compiled program: the coarse
    level's temporaries are freed as soon as they are dead, and a second call
    with the same shapes and settings reuses the compiled code."""
    op = matfree_operator(eq, diffmat, assembly)
    v0, Z = _start(op, diffmat, assembly, solver, v_guess, coarse)
    v, _ = _primal(eq, diffmat, assembly, solver, op["n_keep"], v0, Z)
    Av = op["Ax"](v)
    gamma2 = _squared_growth_rate(v, Av)
    resid = jnp.linalg.norm(Av + gamma2 * v) / (
        jnp.abs(gamma2) * jnp.linalg.norm(v) + 1e-300
    )
    return gamma2, v, resid


@partial(jit, static_argnames=("assembly", "solver"))
def _growth_rate_jit(eq, diffmat, assembly, solver, v_fixed, v_guess, coarse):
    """:func:`growth_rate`'s solve as one compiled program (see
    :func:`_eigenpair_jit`)."""
    return _lambda_hf(eq, diffmat, assembly, solver, v_fixed, v_guess, coarse)


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


def eigenpair(eq, diffmat, assembly=None, solver=None, v_guess=None, coarse=None):
    """Solve mode: ``(gamma2, v, residual)`` for one stored equilibrium.

    Not differentiable; see the module docstring and :func:`growth_rate_of`.

    Parameters
    ----------
    eq : EquilibriumData
    diffmat : DiffMat
    assembly : AssemblyConfig, optional
    solver : SolverConfig, optional
    v_guess : ndarray, optional
        Warm start (``eigsh``'s ``v0`` / the Lanczos start), e.g. ``v`` from a
        previous call; reduced length or full ``3 * n_total``.
    coarse : tuple, optional
        The coarse level; see :func:`growth_rate`.

    Returns
    -------
    gamma2 : jax.Array
        ``gamma^2 = -lambda``, minus the Rayleigh quotient at the computed
        eigenvector. **Its sign is the physics answer**: positive is unstable.
    v : jax.Array, shape (n_keep,)
    residual : jax.Array
        ``||A v + gamma2 v|| / (|gamma2| ||v||)``. A genuine quality measure,
        unlike the inner CG's relative residual.

    Notes
    -----
    ``gamma2`` comes from the Rayleigh quotient, not the eigensolver's reported
    eigenvalue -- they agree to the eigensolve tolerance, and the quotient is
    the quantity the gradient differentiates. Reporting a different number than
    the one being differentiated is how a gradient check ends up chasing a
    discrepancy that is not there.
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver

    return _forbid_gradient(
        "eigenpair",
        lambda e, d, vg, c: _eigenpair_jit(e, d, assembly, solver, vg, c),
        eq,
        diffmat,
        v_guess,
        coarse,
    )


def growth_rate(
    eq, diffmat, assembly=None, solver=None, v_fixed=None, v_guess=None, coarse=None
):
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
    v_fixed : ndarray, optional
        Skip the eigensolve: the Rayleigh quotient of this vector held constant,
        so the value (and, through :func:`growth_rate_of`, the Hellmann-Feynman
        gradient) costs one operator application. Valid ONLY for a vector from
        :func:`eigenpair` at this exact ``eq``; after the equilibrium moves it
        is silently wrong (DESC: a 7e-5 relative mesh shift flipped the sign).
        Never hand it to an optimizer.
    v_guess : ndarray, optional
        Warm start for the eigensolve; see :func:`eigenpair`.
    coarse : tuple, optional
        Required by ``eigensolver="jd"``: :meth:`agnimhd.Basis.coarse_level`,
        the same equilibrium on fewer angular nodes, or the ``(v0, Z)`` that
        :func:`coarse_space` returned for it. Its softest modes seed and deflate
        the JD solve; other solvers use only the seed. A solver aid: no
        gradient flows through it.

    Returns
    -------
    jax.Array
        Scalar, the squared growth rate ``gamma^2 = -lambda``. **Positive means
        unstable**; an optimizer seeking stability lowers it toward zero.

    See Also
    --------
    eigenpair : the same solve, plus the eigenvector and a residual.
    growth_rate_of : optimize mode -- the same gamma^2, over parameters.
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver
    _check_configs(assembly, solver)
    return _forbid_gradient(
        "growth_rate",
        lambda e, d, vf, vg, c: _growth_rate_jit(e, d, assembly, solver, vf, vg, c),
        eq,
        diffmat,
        v_fixed,
        v_guess,
        coarse,
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
        "to an equilibrium, which is an equilibrium solve. See docs/index.md.",
    )
    errorif(
        not callable(equilibrium_map),
        TypeError,
        "equilibrium_map must be a callable params -> EquilibriumData, not "
        f"{type(equilibrium_map).__name__}. If you have an EquilibriumData and "
        "only want its stability, that is solve mode: growth_rate(eq, diffmat).",
    )


def growth_rate_of(
    params,
    equilibrium_map,
    diffmat,
    assembly=None,
    solver=None,
    v_fixed=None,
    v_guess=None,
    coarse=None,
    on_vector=None,
    sigma=None,
):
    """Optimize mode: the growth rate as a function of *your* parameters.

    ``jax.grad`` returns ``d(gamma^2)/d(params)``, a pytree shaped like ``params``
    rather than like an ``EquilibriumData``. It is a partial derivative at a
    fixed force balance residual. Keeping the iterate in force balance is the
    optimizer's task, not this function's.

    Parameters
    ----------
    params : pytree
        The equilibrium's parameters: spectral coefficients, profile
        coefficients, and the free parameters derived from them.
        Differentiable.
    equilibrium_map : callable
        ``params -> EquilibriumData``, differentiable in JAX. It evaluates
        geometry and profiles and packs the result, and contains no equilibrium
        solve. A Python callable, so it is static under ``jax.jit``.
    diffmat : DiffMat
        Operators on the nodes ``equilibrium_map`` evaluates on. Fixed across
        the optimization -- the grid is not a parameter.
    assembly : AssemblyConfig, optional
    solver : SolverConfig, optional
    v_fixed, v_guess, coarse : optional
        As for :func:`growth_rate`. ``v_guess`` may also be ``(vector, valid)``
        with a traced boolean ``valid``: the vector starts the solve if valid,
        else the coarse level's seed does (needs ``coarse``).
    on_vector : callable, optional
        Called with the converged eigenvector and its ``gamma^2`` (constants for
        the derivative), e.g. to keep them for the next call.
    sigma : traced scalar, optional
        ``eigensolver="jd"`` only: the shift of this solve in place of
        ``solver.sigma``, e.g. a multiple of the last ``gamma^2``. Must lie above
        the largest ``gamma^2``; a solve that ends within a factor
        ``SHIFT_MARGIN`` of it is redone at ``solver.sigma``.

    Returns
    -------
    jax.Array
        Scalar ``gamma^2 = -lambda``. **Positive means unstable**; a minimizer
        lowers it toward zero as it is.

    Notes
    -----
    This function does not enforce force balance and cannot. If the adapter is
    not differentiable the chain rule does not close and only solve mode is
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
    errorif(
        sigma is not None and solver.eigensolver != "jd",
        ValueError,
        'sigma= (an adaptive shift) is for eigensolver="jd"',
    )
    return _lambda_hf(
        eq, diffmat, assembly, solver, v_fixed, v_guess, coarse, on_vector, sigma
    )


def growth_rate_and_grad(
    params,
    equilibrium_map,
    diffmat,
    assembly=None,
    solver=None,
    v_fixed=None,
    v_guess=None,
    coarse=None,
):
    """Optimize mode: value and ``d(gamma^2)/d(params)`` from a single eigensolve.

    Arguments as for :func:`growth_rate_of`.

    Returns
    -------
    gamma2 : jax.Array
        Scalar squared growth rate ``gamma^2 = -lambda``, positive when unstable.
    grad : pytree
        Same structure as ``params``, holding ``d(gamma^2)/d(each leaf)``.
    """
    return jax.value_and_grad(growth_rate_of)(
        params, equilibrium_map, diffmat, assembly, solver, v_fixed, v_guess, coarse
    )
