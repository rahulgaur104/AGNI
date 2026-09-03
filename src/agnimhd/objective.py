"""The differentiable growth rate.

:func:`growth_rate` is AGNI's public entry point and the whole reason the
package is a package: it is a pure JAX function of an
:class:`~agnimhd.EquilibriumData`, it survives ``jax.jit`` and ``jax.grad``
applied **from outside**, and its derivative is analytic.

How the derivative works
------------------------

The quantity returned is the Rayleigh quotient

.. math::  \\lambda = \\frac{v^T A(q)\\, v}{v^T v}

evaluated at the eigenvector ``v`` of the current operator. By the
Hellmann-Feynman theorem, at an eigenvector the derivative of the eigenvalue
with respect to any parameter is the derivative of the quotient **holding the
vector fixed**:

.. math::  \\frac{d\\lambda}{dq} = \\frac{v^T (dA/dq)\\, v}{v^T v}.

That identity is what makes the gradient cheap: no derivative of the
eigensolve is needed, only one operator application per cotangent. It is also
what dictates the implementation. The eigensolve is wrapped in a
``jax.custom_vjp`` whose backward rule returns **zero** cotangents, so
differentiation cannot flow through the eigensolve or through the eigenvector
*selection* (an ``argmax``, which has no useful derivative anyway). ``v``
arrives at the quotient as a constant, and ordinary autodiff of ``v^T A(q) v``
then produces exactly the Hellmann-Feynman contraction.

Two consequences follow, and both matter:

* The eigensolve does not need to be differentiable. ARPACK behind a
  ``pure_callback`` would be fine.
* ``v`` is still recomputed at the current point on every call. The gradient is
  a fixed-vector gradient, but it is not a stale-vector gradient.

Accuracy
--------

Validated against central finite differences at **0.45% agreement**, and the
step size is not incidental: only ``h = 1e-7`` converged. Larger steps are
dominated by the quotient's curvature, smaller ones by the eigenvalue noise
floor, which is 2.8e-5 **relative** -- so a finite-difference check that uses a
step outside a narrow window will disagree with a correct gradient and look
like a bug in the gradient.
"""

import numpy as np

from .assemble import assemble_dense, matfree_operator, operator_dtype
from .backend import errorif, jax, jnp
from .config import AssemblyConfig, SolverConfig

__all__ = ["growth_rate", "growth_rate_and_grad", "eigenpair"]


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
# Public API
# ---------------------------------------------------------------------------


def eigenpair(eq, diffmat, assembly=None, solver=None):
    """Return ``(lambda, v, residual)`` without any derivative machinery.

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
    eigenvalue. The two agree to the eigensolve tolerance; the quotient is the
    one returned because it is the quantity the gradient differentiates, and
    reporting a different number than the one being differentiated is how a
    gradient check ends up chasing a discrepancy that is not there.
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver
    op = matfree_operator(eq, diffmat, assembly)
    v, _ = _primal(eq, diffmat, assembly, solver, op["n_keep"])
    Av = op["Ax"](v)
    vv = jnp.vdot(v, v)
    lam = jnp.real(jnp.vdot(v, Av) / vv)
    resid = jnp.linalg.norm(Av - lam * v) / (
        jnp.abs(lam) * jnp.sqrt(jnp.real(vv)) + 1e-300
    )
    return lam, v, resid


def growth_rate(eq, diffmat, assembly=None, solver=None):
    """Squared growth rate of the most unstable finite-n ideal MHD mode.

    A pure JAX function: ``jax.jit`` and ``jax.grad`` may be applied to it from
    outside the package, and ``jax.grad`` returns the analytic
    Hellmann-Feynman derivative with respect to every array and scalar in
    ``eq``.

    Parameters
    ----------
    eq : EquilibriumData
        The equilibrium. Differentiable: every array leaf and both scalars
        (``Psi``, ``a``) receive a gradient.
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

    Warnings
    --------
    The eigenvalue's **absolute** noise floor is 1e-10 and its **relative**
    floor is 2.8e-5. A finite-difference check of this gradient agrees to 0.45%
    only at ``h = 1e-7``; at other steps the disagreement is the finite
    difference's, not the gradient's.

    See Also
    --------
    eigenpair : the same solve, plus the eigenvector and a residual.
    growth_rate_and_grad : value and gradient in one pass.

    Examples
    --------
    >>> import jax                                        # doctest: +SKIP
    >>> lam = growth_rate(eq, diffmat)                    # doctest: +SKIP
    >>> g = jax.grad(growth_rate)(eq, diffmat)            # doctest: +SKIP
    >>> float(g.a)                                        # doctest: +SKIP
    """
    assembly = AssemblyConfig() if assembly is None else assembly
    solver = SolverConfig() if solver is None else solver
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
        """Zero cotangent. This is what enforces Hellmann-Feynman.

        Not an approximation and not a shortcut: at an eigenvector the
        eigenvalue's derivative *is* the fixed-vector derivative of the
        Rayleigh quotient, so the eigensolve's own derivative is exactly the
        term that must not be included. Returning zero here also removes the
        eigenvector-selection ``argmax``, which has no useful derivative.
        """
        return (jax.tree_util.tree_map(jnp.zeros_like, res),)

    _v_of.defvjp(_v_fwd, _v_bwd)

    v = _v_of(eq)
    # `Ax` is differentiable in `eq`; `v` is not. Autodiff of this expression is
    # therefore exactly v^T (dA/dq) v / v^T v.
    return jnp.real(jnp.vdot(v, op["Ax"](v)) / jnp.vdot(v, v))


def growth_rate_and_grad(eq, diffmat, assembly=None, solver=None):
    """Value and gradient in a single eigensolve.

    Parameters
    ----------
    eq : EquilibriumData
    diffmat : DiffMat
    assembly : AssemblyConfig, optional
    solver : SolverConfig, optional

    Returns
    -------
    lam : jax.Array
        Scalar growth rate.
    grad : EquilibriumData
        A pytree of the same structure as ``eq``, holding
        ``dlambda/d(each array)``.
    """
    return jax.value_and_grad(growth_rate)(eq, diffmat, assembly, solver)
