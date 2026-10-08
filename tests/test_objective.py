"""The public entry point: the growth rate and its analytic gradient.

Two things are being asserted here, and they are different things.

**That the value is right.** The sign comes first -- it is the physics answer,
and a solver that reports the wrong sign reports a stable equilibrium as
unstable or the reverse. Then the magnitude, against the number in the
fixture's sidecar.

**That the gradient is right.** Analytic (Hellmann-Feynman), which means it is
not obviously wrong the way a hand-differentiated expression is: it will
happily return a smooth, plausible, incorrect number. The only real check is a
finite difference, and that check has its own trap -- the eigenvalue's relative
noise floor is 2.8e-5, so a step chosen for a well-conditioned function
disagrees with a correct gradient. See
``test_gradient_matches_finite_differences``.

**And that the mode boundary holds.** Solve mode is not differentiable; see
``test_the_mode_boundary_is_enforced``.

Every test runs on CPU against the shipped fixture. Nothing here reaches
outside the repository.
"""

import numpy as np
import pytest
from conftest import build_diffmat, fixture_basis, on_fewer_angles

import agnimhd
from agnimhd import (
    AssemblyConfig,
    EquilibriumData,
    SolverConfig,
    eigenpair,
    growth_rate,
    growth_rate_and_grad,
    growth_rate_of,
)
from agnimhd.assemble import keep_indices
from agnimhd.backend import jax, jnp
from agnimhd.objective import _lambda_hf


def a_map(eq):
    """A stand-in ``params -> EquilibriumData``: ``{"a": v} -> eq.replace(a=v)``.

    Not a physical parameterization, and a real optimization must not use one
    like it. It is sufficient here because these tests check the derivative
    machinery, for which the map need only be differentiable and move something
    the operator depends on. ``a`` is used because it is one scalar the whole
    operator is normalized by, so a finite difference costs one extra pair of
    solves rather than one per node, and the eigenvalue is most sensitive to
    it.
    """
    return lambda p: eq.replace(a=p["a"])


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------


def test_an_unstable_equilibrium_has_a_positive_squared_growth_rate(eq_data, diffmat):
    """The sign convention: agnimhd returns ``gamma^2 = -lambda_min``, positive
    when unstable. The shipped QH equilibrium is unstable."""
    gamma2 = agnimhd.growth_rate(eq_data, diffmat)
    assert float(gamma2) > 0.0


def test_growth_rate_reproduces_the_reference(eq_data, diffmat, config, eq_meta):
    """The value matches the number recorded when the fixture was exported.

    The reference is read from the sidecar, not typed from a document. It is
    DESC's eigenvalue ``lambda`` of ``A``; agnimhd returns ``-lambda``.

    The tolerance is the eigenvalue's **relative** noise floor, 2.8e-5. That
    is a measured property of this operator: below it, two correct runs are
    allowed to disagree.
    """
    ref = -float(eq_meta["dense_lambda3"])
    lam = float(growth_rate(eq_data, diffmat, config))
    assert np.sign(lam) == np.sign(ref), "sign disagrees with the reference"
    rel = abs(lam - ref) / abs(ref)
    assert rel < 2.8e-5, f"lambda is {rel:.3e} from the reference"


def test_eigenpair_returns_a_converged_mode(eq_data, diffmat, config):
    """The eigenvector really is an eigenvector of the matrix-free operator.

    The Rayleigh residual is a genuine quality measure. The inner CG's relative
    residual is not -- on this operator it is anti-correlated with accuracy --
    so this is the number worth asserting on.
    """
    gamma2, v, resid = eigenpair(eq_data, diffmat, config)
    assert float(gamma2) > 0.0
    assert np.all(np.isfinite(np.asarray(v)))
    assert float(resid) < 1e-4, f"Rayleigh residual {float(resid):.3e}"


def test_rayleigh_quotient_is_what_is_returned(eq_data, diffmat, config):
    """``growth_rate`` and ``eigenpair`` report the same number.

    They must: the quotient is what the gradient differentiates, and reporting
    the eigensolver's own eigenvalue instead would leave a small discrepancy
    for a gradient check to chase.
    """
    lam_ep, _, _ = eigenpair(eq_data, diffmat, config)
    lam_gr = growth_rate(eq_data, diffmat, config)
    assert float(lam_ep) == float(lam_gr)


def test_dense_agrees_with_eigsh(eq_data, diffmat, config):
    """The two dense eigensolvers find the same mode.

    They share nothing but the matrix: one is host ARPACK behind a callback,
    the other (``"dense"``) is Lanczos on an exact JAX Cholesky factor. ARPACK is
    1.53x faster on CPU and is the default; this keeps the GPU solver honest.
    ``test_dshape.py`` checks the same on the complex Hermitian operator.

    The shift is ``1e-3`` rather than the default ``1e-1``, and that is not a
    tolerance being nudged to make a test pass -- see
    ``test_a_far_shift_selects_the_wrong_mode_and_the_residual_says_so`` for the
    measurement, and ``SolverConfig.sigma`` for why the default is where it is.
    A fixed-matvec Lanczos needs a shift that is above the largest ``gamma^2``
    *and* near it; ARPACK, which iterates to a tolerance, does not.
    """
    lam_a = float(growth_rate(eq_data, diffmat, config))
    lam_b = float(
        growth_rate(
            eq_data,
            diffmat,
            config,
            SolverConfig(eigensolver="dense", sigma=1e-3),
        )
    )
    assert np.sign(lam_a) == np.sign(lam_b), "the two eigensolvers disagree on sign"
    rel = abs(lam_a - lam_b) / abs(lam_a)
    assert rel < 2.8e-5, f"eigensolvers differ by {rel:.3e}, above the noise floor"


def test_a_far_shift_selects_the_wrong_mode_and_the_residual_says_so(
    eq_data, diffmat, config
):
    """A shift far above the largest ``gamma^2`` breaks a fixed-budget Lanczos.

    Shift-invert maps ``gamma^2`` to ``1/(sigma - gamma^2)``, and Lanczos
    separates modes at a rate set by the ratio of those. As ``sigma`` grows
    the ratio goes to one: on this case it is 1.0007 at ``sigma = 1e-1``
    against 1.0823 at ``1e-3``. So the default shift -- chosen conservatively,
    because a shift *below* the largest ``gamma^2`` has no recovery at all --
    makes a 50-matvec ``dense`` return the wrong mode, with the wrong
    sign.

    This is pinned rather than fixed because both halves are load-bearing. The
    failure is real, it is a property of the method and not of this
    implementation, and it is **detectable**: the Rayleigh residual is eight
    orders of magnitude apart between the two shifts. That is the check a
    caller running the matrix-free path has to make, so it is worth a test that
    demonstrates it discriminates.

    ARPACK at the same shift is unaffected -- it iterates to ``eigsh_tol``
    instead of stopping at a fixed count -- which is why the default has never
    caused trouble on the default path.
    """
    lam_ref, _, resid_ref = eigenpair(eq_data, diffmat, config)
    lam_near, _, resid_near = eigenpair(
        eq_data, diffmat, config, SolverConfig(eigensolver="dense", sigma=1e-3)
    )
    lam_far, _, resid_far = eigenpair(
        eq_data, diffmat, config, SolverConfig(eigensolver="dense", sigma=1e-1)
    )

    # The near shift is converged; the far one is not the same mode at all.
    #
    # The bound on resid_near is loose ON PURPOSE. Measured locally it is
    # 1.6e-4; on a CI runner with different BLAS/LAPACK it came back 1.63e-3 --
    # ten times worse, still a converged mode, and enough to fail a bound of
    # 1e-3 that had essentially no margin around a number measured exactly
    # once. Real mode-correctness is asserted next, against the reference, to
    # the eigenvalue's actual noise floor; this bound exists only to catch a
    # genuinely wrong near-shift solve, which the sigma table two lines down
    # puts at resid ~ 1e2-1e4 -- orders of magnitude above anything sane
    # numerical noise would produce here.
    assert float(resid_near) < 1e-1
    assert abs(float(lam_near) - float(lam_ref)) / abs(float(lam_ref)) < 2.8e-5
    assert float(lam_far) < 0.0, (
        "the far shift is expected to select the wrong mode on this case; if it "
        f"no longer does ({float(lam_far):+.6e}), the spectrum or the Lanczos "
        "budget moved and SolverConfig.sigma's table needs remeasuring"
    )

    # And the residual is what tells them apart. Nothing else does: the far
    # answer is a perfectly finite number of the right magnitude.
    assert float(resid_far) > 1e2 * float(resid_near), (
        f"the Rayleigh residual stopped discriminating: {float(resid_far):.3e} "
        f"for the wrong mode against {float(resid_near):.3e} for the right one"
    )


# ---------------------------------------------------------------------------
# The complex Hermitian operator
# ---------------------------------------------------------------------------


def _dense_reference(eq, diffmat, config):
    """Smallest eigenvalue of the assembled matrix, by dense LAPACK."""
    from agnimhd.assemble import assemble_dense

    A = np.asarray(assemble_dense(eq, diffmat, config)["A"])
    return A, float(np.linalg.eigvalsh(A)[0])


@pytest.mark.parametrize("eigensolver", ["eigsh", "dense"])
def test_the_equilibrium_density_weights_every_solver(period_case, eigensolver):
    """``EquilibriumData.density`` is the mass weighting with no new argument:
    the solvers give the dense ``gamma^2`` of ``assemble_dense(..., density=w)``,
    which differs from the unweighted one, and the matrix-free operator (JD's
    and ``dense_mg``'s) applies the same weighted matrix."""
    from agnimhd.assemble import assemble_dense, matfree_operator

    eq, meta = period_case
    diffmat = build_diffmat(eq)
    config = AssemblyConfig(gamma=meta["gamma"])
    rho = np.asarray(meta["rho_nodes"])
    w = np.repeat(1 - 0.8 * rho**2, eq.n_theta * eq.n_zeta)  # falls off to the edge
    weighted = eq.replace(density=w)

    A = np.asarray(assemble_dense(eq, diffmat, config, density=w)["A"])
    lam = float(np.linalg.eigvalsh(A)[0])
    assert abs(lam - _dense_reference(eq, diffmat, config)[1]) > 1e-3 * abs(lam)
    solver = SolverConfig(eigensolver=eigensolver, sigma=-1.3 * lam, num_matvecs=100)
    assert float(growth_rate(weighted, diffmat, config, solver)) == pytest.approx(
        -lam, rel=1e-8
    )
    x = np.random.default_rng(0).standard_normal(A.shape[0])
    Ax = matfree_operator(weighted, diffmat, config)["Ax"](x)
    np.testing.assert_allclose(Ax, A @ x, atol=1e-10 * np.max(np.abs(A @ x)))


def test_the_axisym_operator_is_complex_hermitian(axisym_case):
    """``axisym=True`` builds a complex Hermitian matrix, not a real one.

    Asserted separately because every downstream test in this section is
    vacuous if the dtype branch was not taken -- a real matrix would pass them
    all while testing nothing.
    """
    eq, diffmat, config = axisym_case
    A, _ = _dense_reference(eq, diffmat, config)
    assert np.iscomplexobj(A), "axisym=True did not produce a complex operator"
    scale = np.max(np.abs(A))
    herm = np.max(np.abs(A - A.conj().T)) / scale
    symm = np.max(np.abs(A - A.T)) / scale
    assert herm < 1e-12, f"operator is not Hermitian: {herm:.3e}"
    assert symm > 1e-6, (
        "operator is Hermitian AND symmetric, so it is effectively real and "
        "the complex path is untested by everything below"
    )


@pytest.mark.parametrize("eigensolver", ["eigsh", "dense"])
def test_both_eigensolvers_match_dense_on_the_complex_operator(
    axisym_case, eigensolver
):
    """Both eigensolvers find the dense mode of the complex Hermitian operator.

    This is the check that pins ``matfree>=0.6.2``. Before matfree PR #288 the
    Lanczos recurrence orthonormalized with ``Q.T @ Q`` rather than
    ``Q.conj().T @ Q``, which is the same thing on a real symmetric operator
    and a different thing here. The failure is silent: the returned Ritz VALUE
    stayed at -2.776e-03, close enough to the truth to look converged, while
    the Ritz VECTOR was wrong and the Rayleigh quotient computed from it came
    back +9.713e-02 -- an unstable equilibrium reported as stable -- with a
    residual of 1.14e+03.

    ``eigsh`` fails differently and for its own reason: ARPACK's output shape
    and dtype are declared to ``jax.pure_callback``, which casts rather than
    checks, so a real declaration on a complex operator is a silent truncation.
    See ``assemble.operator_dtype``.

    The shift is placed just above the dense ``gamma^2``. A fixed-matvec Lanczos
    needs a shift that is above the largest ``gamma^2`` *and* near it; see
    ``test_a_far_shift_selects_the_wrong_mode_and_the_residual_says_so``.
    """
    eq, diffmat, config = axisym_case
    _, lam_dense = _dense_reference(eq, diffmat, config)

    solver = SolverConfig(
        eigensolver=eigensolver, sigma=-1.3 * lam_dense, num_matvecs=100
    )
    gamma2, v, resid = eigenpair(eq, diffmat, config, solver)
    gamma2 = float(gamma2)

    assert v.dtype == np.complex128, "the eigenvector came back real"
    # A real eigenvector would satisfy the assertions below for the wrong
    # reason: it would mean the solve collapsed onto the real subspace.
    v = np.asarray(v)
    assert np.linalg.norm(v.imag) / np.linalg.norm(v) > 1e-3

    assert np.sign(gamma2) == np.sign(-lam_dense), (
        f"{eigensolver} flipped the sign of the growth rate: {gamma2:.6e} vs dense "
        f"{-lam_dense:.6e} -- a stable/unstable misclassification"
    )
    assert float(resid) < 1e-3, f"eigenvector not converged: residual {resid:.3e}"
    np.testing.assert_allclose(gamma2, -lam_dense, rtol=1e-6)


def test_the_growth_rate_is_real_on_the_complex_operator(axisym_case):
    """``growth_rate`` returns a real scalar, and differentiates to a real one.

    The Rayleigh quotient of a Hermitian operator is real by construction, but
    only if it is formed with the conjugating inner product. A ``v @ A @ v``
    written for the real case returns a complex number here, and a complex
    objective is not something ``jax.grad`` will accept -- so this catches the
    slip at the package boundary rather than inside an optimizer.
    """
    eq, diffmat, config = axisym_case
    _, lam_dense = _dense_reference(eq, diffmat, config)
    solver = SolverConfig(eigensolver="eigsh", sigma=-1.3 * lam_dense)

    lam = growth_rate(eq, diffmat, config, solver)
    assert lam.dtype == jnp.zeros(()).dtype, f"growth_rate returned {lam.dtype}"

    g = jax.grad(growth_rate_of)({"a": eq.a}, a_map(eq), diffmat, config, solver)["a"]
    assert np.isrealobj(np.asarray(g)), "the gradient came back complex"
    assert np.isfinite(float(g))
    assert float(g) != 0.0


@pytest.mark.parametrize("bad", [{"assembly": {}}, {"solver": {}}])
def test_config_must_be_a_config_object(eq_data, diffmat, config, bad):
    """A dict would retrace on every call, so it is refused, not accepted."""
    kwargs = dict(assembly=config)
    kwargs.update(bad)
    with pytest.raises(TypeError):
        growth_rate(eq_data, diffmat, **kwargs)


# ---------------------------------------------------------------------------
# The gradient
# ---------------------------------------------------------------------------


def test_the_mode_boundary_is_enforced(eq_data, diffmat, config):
    """Solve mode refuses to differentiate; optimize mode refuses a missing map.

    ``dlambda/d(EquilibriumData)`` is a sensitivity to grid samples: not free
    parameters, in force balance only because a solve put them there. Returning
    it would be indistinguishable from a usable gradient, and an optimizer
    would step along it into arrays that are not in force balance. A
    ``stop_gradient`` was rejected as the implementation because a zero
    gradient cannot be told apart from an optimization that has converged. The
    two refused optimize-mode calls are the same error in the other
    signature.
    """
    for fn in (
        lambda e: growth_rate(e, diffmat, config),
        lambda e: eigenpair(e, diffmat, config)[0],
    ):
        with pytest.raises(TypeError, match="not differentiable"):
            jax.grad(fn)(eq_data)
    with pytest.raises(TypeError, match="params is an EquilibriumData"):
        growth_rate_of(eq_data, lambda p: p, diffmat, config)
    with pytest.raises(TypeError, match="must be a callable"):
        growth_rate_of({"a": eq_data.a}, eq_data, diffmat, config)


def test_grad_of_optimize_mode_works_from_outside_the_package(eq_data, diffmat, config):
    """``jax.grad`` applied by a caller, on the public optimize-mode function.

    The interface contract: a consumer supplies the map from its own parameters
    and differentiates. The gradient comes back shaped like ``params``, not
    like an ``EquilibriumData``, since ``params`` is what the optimizer
    steps.
    """
    params = {"a": eq_data.a}
    g = jax.grad(growth_rate_of)(params, a_map(eq_data), diffmat, config)
    assert set(g) == {"a"}, "the gradient is not shaped like params"
    assert not isinstance(g, EquilibriumData)
    assert np.isfinite(float(g["a"]))
    assert abs(float(g["a"])) > 0.0, "no gradient with respect to the minor radius"


def test_the_inner_factor_reaches_every_leaf(eq_data, diffmat, config):
    """``dlambda/d(EquilibriumData)`` is finite and nonzero on every leaf.

    The chain rule's *private* inner factor, tested directly because nothing
    public exposes it. It has to reach every leaf: an array the assembly
    silently drops shows up here as a zero and nowhere else -- and the custom
    VJP returns zero cotangents for the eigensolve deliberately, so a leak into
    the Rayleigh quotient would zero the whole gradient while an optimizer sat
    still reporting success.
    """
    solver = SolverConfig()
    g = jax.grad(lambda e: _lambda_hf(e, diffmat, config, solver))(eq_data)
    assert isinstance(g, EquilibriumData)
    assert np.isfinite(float(g.a)) and abs(float(g.a)) > 0.0
    assert np.isfinite(float(g.Psi))
    for key in ("g_rr", "sqrtg", "iota", "p_r", "finite_n_instability_drive"):
        arr = np.asarray(getattr(g, key))
        assert arr.shape == (eq_data.n_nodes,), f"{key} gradient has the wrong shape"
        assert np.all(np.isfinite(arr)), f"{key} gradient is not finite"
    assert np.max(np.abs(np.asarray(g.finite_n_instability_drive))) > 0.0


def test_jit_from_outside_the_package(eq_data, diffmat, config):
    """``jax.jit`` applied by a caller, on both the value and the gradient."""
    f = jax.jit(growth_rate, static_argnums=(2, 3))
    lam_j = float(f(eq_data, diffmat, config, SolverConfig()))
    lam_e = float(growth_rate(eq_data, diffmat, config))
    assert np.sign(lam_j) == np.sign(lam_e)
    assert abs(lam_j - lam_e) / abs(lam_e) < 2.8e-5

    # Optimize mode too. `equilibrium_map` is a Python callable, so it is
    # static: argument 1 joins the two configs.
    g = jax.jit(jax.grad(growth_rate_of), static_argnums=(1, 3, 4))(
        {"a": eq_data.a}, a_map(eq_data), diffmat, config, SolverConfig()
    )
    assert np.isfinite(float(g["a"])) and abs(float(g["a"])) > 0.0


def test_a_jitted_value_can_be_differentiated(period_case):
    """``jax.grad`` of a jitted ``growth_rate_of``: the order DESC's optimizer uses.

    ``diffmat`` is traced inside the caller's ``jit``. The eigensolve's custom
    VJP once closed over it and the derivative failed to lower ("No constant
    handler for type DynamicJaxprTracer").
    """
    eq, _ = period_case
    diffmat = build_diffmat(eq)
    params = {"a": eq.a}
    value = jax.jit(growth_rate_of, static_argnums=1)
    grad = jax.grad(value)(params, a_map(eq), diffmat)["a"]
    expected = jax.grad(growth_rate_of)(params, a_map(eq), diffmat)["a"]
    assert float(grad) == pytest.approx(float(expected), rel=1e-9)


def test_value_and_grad_agrees_with_the_two_calls(eq_data, diffmat, config):
    """One pass returns the same value and gradient as two separate ones."""
    params = {"a": eq_data.a}
    emap = a_map(eq_data)
    lam, g = growth_rate_and_grad(params, emap, diffmat, config)
    # Solve mode on the same equilibrium must agree -- the two modes are the
    # same eigensolve reached two ways, not two solvers -- but NOT bit-exactly.
    # The two paths build different jaxprs (optimize mode traces through
    # `equilibrium_map` and `value_and_grad`), so the assembly sums in a
    # different order and the eigensolve starts from a different rounding.
    # Measured spread is ~2e-12 relative; the eigenvalue's own relative noise
    # floor is 2.8e-5, so this bound is strict, and `==` was simply wrong.
    assert np.isclose(
        float(lam), float(growth_rate(eq_data, diffmat, config)), rtol=1e-9
    )
    g2 = jax.grad(growth_rate_of)(params, emap, diffmat, config)
    assert np.isclose(float(g["a"]), float(g2["a"]), rtol=1e-12)


def test_gradient_matches_finite_differences(eq_data, diffmat, config):
    """The analytic gradient against a central difference, in ``a``.

    ``a`` is chosen because it is a single scalar the whole operator is
    normalized by, so the finite difference is one extra pair of solves rather
    than one pair per node -- and because ``a`` is the input the eigenvalue is
    most sensitive to, which is precisely why getting its gradient right
    matters.

    **The step size is not free.** Recorded agreement is 0.45%, and only at
    ``h = 1e-7``. Larger steps are dominated by the quotient's curvature;
    smaller ones fall into the eigenvalue's relative noise floor of 2.8e-5, at
    which point the difference quotient is measuring noise divided by a small
    number. A disagreement at some other step is the finite difference's
    problem, not the gradient's -- do not "fix" the gradient to match one.
    """
    h = 1e-7
    a0 = float(eq_data.a)

    def lam_at(a):
        return float(growth_rate(eq_data.replace(a=a), diffmat, config))

    fd = (lam_at(a0 * (1 + h)) - lam_at(a0 * (1 - h))) / (2 * h * a0)
    analytic = float(
        jax.grad(growth_rate_of)({"a": a0}, a_map(eq_data), diffmat, config)["a"]
    )

    rel = abs(analytic - fd) / abs(fd)
    assert np.sign(analytic) == np.sign(
        fd
    ), f"gradient sign disagrees: analytic {analytic:+.6e}, fd {fd:+.6e}"
    assert rel < 0.02, (
        f"gradient disagrees with the h={h:g} central difference by {rel:.2%} "
        f"(analytic {analytic:+.6e}, fd {fd:+.6e}). Recorded agreement is "
        "0.45%. Before adjusting anything, check that the eigensolve is "
        "converging at both perturbed points -- a mode swap between them looks "
        "exactly like a wrong gradient. In a real optimization there is a "
        "second requirement this test does not exercise: the equilibrium "
        "itself must be converged at both points, or the difference measures a "
        "solver residual."
    )


def test_a_descent_step_moves_lambda_the_right_way(eq_data, diffmat, config):
    """One gradient step in ``a`` lowers ``gamma^2`` toward zero.

    Instability is ``gamma^2 > 0``, so a minimizer steps along ``-grad``. This
    is the end-to-end statement that the sign convention holds all the way from
    the operator to something a caller would write, and it is the check that
    catches a globally flipped gradient -- which every finiteness and
    magnitude test above would pass.
    """
    a0 = float(eq_data.a)
    gamma2_0, g = growth_rate_and_grad({"a": a0}, a_map(eq_data), diffmat, config)
    gamma2_0 = float(gamma2_0)
    assert gamma2_0 > 0.0

    dgamma2_da = float(g["a"])
    # Descent: step along -grad, sized to a small relative change in a.
    a1 = a0 - 1e-4 * a0 * np.sign(dgamma2_da)
    gamma2_1 = float(growth_rate(eq_data.replace(a=a1), diffmat, config))

    assert gamma2_1 < gamma2_0, (
        f"a descent step made gamma^2 worse: {gamma2_0:+.6e} -> {gamma2_1:+.6e}. "
        "Either the gradient sign is flipped or the step left the linear "
        "regime."
    )


def test_gradient_is_the_hellmann_feynman_contraction(eq_data, diffmat, config):
    """The gradient equals ``-v^T (dA/dq) v / v^T v`` with ``v`` held fixed.

    Computed here the long way -- freeze the eigenvector from one solve, then
    differentiate the Rayleigh quotient explicitly and negate (``gamma^2 =
    -lambda``) -- and compared against what ``growth_rate`` returns. They must
    be identical, not merely close: the custom VJP exists precisely to make the
    second expression compute the first, so any difference means gradient is
    leaking through the eigensolve or through the eigenvector-selection
    ``argmax``.
    """
    from agnimhd.assemble import matfree_operator

    _, v, _ = eigenpair(eq_data, diffmat, config)
    v = jax.lax.stop_gradient(v)

    def rayleigh(eq):
        op = matfree_operator(eq, diffmat, config)
        return jnp.real(jnp.vdot(v, op["Ax"](v)) / jnp.vdot(v, v))

    want = -float(jax.grad(rayleigh)(eq_data).a)
    got = float(
        jax.grad(growth_rate_of)({"a": eq_data.a}, a_map(eq_data), diffmat, config)["a"]
    )
    assert np.isclose(got, want, rtol=1e-10), (
        f"gradient is not the fixed-vector contraction: {got:+.6e} vs " f"{want:+.6e}"
    )


# ---------------------------------------------------------------------------
# Re-evaluation without an eigensolve, and warm starts
# ---------------------------------------------------------------------------


def test_v_fixed_skips_the_eigensolve_exactly(eq_data, diffmat, config):
    """``v_fixed`` gives the identical quotient (exactly, eagerly: same
    expression on the same vector) and the same Hellmann-Feynman gradient. A
    full-length vector is restricted, a wrong size raises, the vector may be
    traced under jit (traced evaluations differ at roundoff only)."""
    lam0, v, _ = eigenpair(eq_data, diffmat, config)
    assert float(growth_rate(eq_data, diffmat, config, v_fixed=v)) == float(lam0)
    full = jnp.zeros(3 * eq_data.n_nodes).at[keep_indices(*eq_data.resolution)]
    lam1 = growth_rate(eq_data, diffmat, config, v_fixed=full.set(v))
    assert float(lam1) == float(lam0)
    params, emap = {"a": eq_data.a}, a_map(eq_data)
    g0 = jax.grad(growth_rate_of)(params, emap, diffmat, config)["a"]
    lam2, g2 = growth_rate_and_grad(params, emap, diffmat, config, v_fixed=v)
    f = jax.jit(growth_rate, static_argnums=(2, 3))
    lam3 = f(eq_data, diffmat, config, SolverConfig(), v_fixed=v)
    for got, want in ((lam2, lam0), (g2["a"], g0), (lam3, lam0)):
        assert np.isclose(float(got), float(want), rtol=1e-10)
    with pytest.raises(ValueError, match="v_fixed"):
        growth_rate(eq_data, diffmat, config, v_fixed=v[:-1])


def test_v_guess_seeds_eigsh_and_cuts_the_lanczos_budget(
    eq_data, diffmat, config, eq_meta, monkeypatch
):
    """The warm start reaches ARPACK as ``v0`` (eagerly and under jit), and
    on the fixed-budget Lanczos a near-converged seed reaches the reference
    where the cold start does not. Measured at 20 matvecs, shift 1e-3, seed
    = eigenvector + 1e-3 noise: 7.6e-10 relative warm against 1.0e-3 cold
    (the stiff noise components must be damped first; 6 matvecs is too few
    for either). Budget, not wall time."""
    import scipy.sparse.linalg as ssl

    seen, real = [], ssl.eigsh
    monkeypatch.setattr(
        ssl, "eigsh", lambda A, **kw: seen.append(kw["v0"]) or real(A, **kw)
    )
    lam_ref, v, _ = eigenpair(eq_data, diffmat, config)
    v, ref = np.asarray(v), -float(eq_meta["dense_lambda3"])
    f = jax.jit(growth_rate, static_argnums=(2, 3))
    for lam in (
        growth_rate(eq_data, diffmat, config, v_guess=v),
        f(eq_data, diffmat, config, SolverConfig(), v_guess=jnp.asarray(v)),
    ):
        assert abs(float(lam) - ref) / abs(ref) < 2.8e-5
        np.testing.assert_allclose(seen[-1], v / np.linalg.norm(v), atol=1e-15)
    with pytest.raises(ValueError, match="v_guess"):
        growth_rate(eq_data, diffmat, config, v_guess=v[:-1])

    small = SolverConfig(eigensolver="dense", sigma=1e-3, num_matvecs=20)
    guess = v + 1e-3 * np.linalg.norm(v) * np.random.default_rng(0).standard_normal(
        v.size
    )
    lam_w, _, res_w = eigenpair(eq_data, diffmat, config, small, v_guess=guess)
    lam_c, _, res_c = eigenpair(eq_data, diffmat, config, small)
    rel = lambda lam: abs(float(lam) - float(lam_ref)) / abs(float(lam_ref))  # noqa
    assert rel(lam_w) < 2.8e-5 < rel(lam_c) and float(res_c) > 10 * float(res_w)


# ---------------------------------------------------------------------------
# Matrix-free Jacobi-Davidson
# ---------------------------------------------------------------------------
# JD needs its coarse level: the same equilibrium on fewer angular nodes, with
# the fine radial nodes and Fourier truncation, paired with the fine level by
# ``basis.coarse_level``. Every other angular node of the fixture is the
# equilibrium on the coarser uniform grid, exactly (``on_fewer_angles``).
# ``sigma = 1.3 gamma^2``, as in DESC's JD tests.


@pytest.mark.slow
@pytest.mark.parametrize(
    "family, mpol, theta_step",
    [(0, 5, 1), (1, 2, 2)],
    ids=["family_0", "complex_family_1"],
)
def test_jd_matches_the_dense_eigenpair(eq_data, config, family, mpol, theta_step):
    """JD with its coarse level (every other zeta node; for family 1 also every
    other theta node) gives eigsh's eigenvalue, an eigen-residual below
    ``jd_tol`` and, through its vector, eigsh's Hellmann-Feynman gradient.

    Family 0 at ``mpol = 5`` is the parity trap: under the stellarator
    reflection the softest fine mode is odd and the softest coarse mode even.
    Started from that coarse mode alone, JD returned the softest even mode
    (``gamma^2`` 2.274e-4 for 3.963e-4); the start is now the sum of the
    ``jd_keep`` softest coarse modes, which holds both parities."""
    basis = fixture_basis(eq_data.resolution, mpol=mpol, ntor=1)
    diffmat = basis.nodes_and_diffmat(eq_data.NFP, family)[1]
    eq_coarse = on_fewer_angles(eq_data, theta_step, zeta=slice(None, None, 2))
    coarse = basis.coarse_level(eq_coarse, family)
    dense = SolverConfig(sigma=1.3 * float(growth_rate(eq_data, diffmat, config)))
    gamma2, v, _ = eigenpair(eq_data, diffmat, config, dense)
    jd = dense.replace(eigensolver="jd", jd_tol=1e-4, jd_theta_tol=0.0, jd_outer=500)
    gamma2_jd, v_jd, resid = eigenpair(eq_data, diffmat, config, jd, coarse=coarse)
    assert float(gamma2_jd) == pytest.approx(float(gamma2), rel=1e-8)
    assert float(resid) <= jd.jd_tol, f"eigen-residual {float(resid):.3e}"

    fields = ("g_rr", "g_vv", "sqrtg", "iota", "p_r", "finite_n_instability_drive")
    params = {key: getattr(eq_data, key) for key in fields}
    grad = [
        jax.grad(growth_rate_of)(
            params, lambda p: eq_data.replace(**p), diffmat, config, v_fixed=vector
        )
        for vector in (v, v_jd)
    ]
    for key in fields:
        a, b = np.asarray(grad[0][key]), np.asarray(grad[1][key])
        assert np.linalg.norm(b - a) < 1e-3 * np.linalg.norm(a), key


def test_jd_refuses_a_missing_or_mismatched_coarse_level(eq_data, diffmat, config):
    """Without its coarse level JD stalls (200 outer iterations on a near-zero
    mode, measured), so it raises instead; a coarse level of a complex family
    for this real one is refused too."""
    jd = SolverConfig(eigensolver="jd", sigma=1e-3)
    with pytest.raises(ValueError, match="needs its coarse level"):
        growth_rate(eq_data, diffmat, config, jd)
    basis = fixture_basis(eq_data.resolution, mpol=4, ntor=1)
    eq_coarse = on_fewer_angles(eq_data, zeta=slice(None, None, 2))
    with pytest.raises(ValueError, match="coarse must be"):
        eigenpair(eq_data, diffmat, config, jd, coarse=basis.coarse_level(eq_coarse, 1))


@pytest.mark.slow
def test_jd_coarse_level_cuts_outer_iterations_and_jits(axisym_case):
    """Complex case (one zeta plane, axisym n=1, mpol 2), every other theta node
    as coarse level: measured 11 outer iterations from the coarse seed and
    deflation against 144 from a random start without, both to residual 1e-8.
    The coarse space can be built beforehand, ``coarse_space``, and passed as
    ``coarse=(v0, Z)``: a jitted ``growth_rate`` with it gives the same value."""
    from agnimhd.assemble import matfree_operator
    from agnimhd.objective import _ring_blocks, coarse_space
    from agnimhd.solvers import (
        factor_ring_blocks_traced,
        jacobi_davidson,
        make_block_precond,
    )

    eq, _, cfg = axisym_case
    basis = fixture_basis(eq.resolution, mpol=2)
    dm = basis.nodes_and_diffmat(eq.NFP)[1]
    coarse = basis.coarse_level(on_fewer_angles(eq, theta_step=2))
    _, ref = _dense_reference(eq, dm, cfg)
    sol = SolverConfig(eigensolver="jd", sigma=-1.3 * ref, jd_tol=1e-8, jd_theta_tol=0)
    op = matfree_operator(eq, dm, cfg)
    blocks, G = _ring_blocks(eq, dm, cfg, sol, op)
    M = make_block_precond(factor_ring_blocks_traced(blocks)[0], G, op["n_keep"])
    v0, Z = coarse_space(eq, dm, coarse, cfg, sol)
    rnd = np.random.default_rng(0).standard_normal(op["n_keep"]).astype(complex)
    kw = dict(sigma=sol.shift, tol=1e-8, theta_tol=0.0)
    th_c, _, info_c = jacobi_davidson(op["Ax"], M, v0, Z, **kw)
    th_r, _, info_r = jacobi_davidson(op["Ax"], M, jnp.asarray(rnd), **kw)
    for th, info in ((th_c, info_c), (th_r, info_r)):
        assert abs(float(th) - ref) / abs(ref) < 1e-7 and float(info["resid"]) < 1e-6
    assert int(info_c["iters"]) < int(info_r["iters"]), (info_c, info_r)
    lam = growth_rate(eq, dm, cfg, sol, coarse=coarse)
    f = jax.jit(growth_rate, static_argnums=(2, 3))
    assert abs(float(f(eq, dm, cfg, sol, coarse=(v0, Z))) - float(lam)) < 1e-8 * abs(
        ref
    )
    assert float(lam) == pytest.approx(-ref, rel=1e-7)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_public_names_are_importable_from_the_top_level():
    """A consumer should not have to know the module layout."""
    for name in (
        "EquilibriumData",
        "DiffMat",
        "AssemblyConfig",
        "SolverConfig",
        "growth_rate",
        "growth_rate_of",
        "growth_rate_and_grad",
        "eigenpair",
    ):
        assert hasattr(agnimhd, name), f"agnimhd.{name} is not exported"
        assert name in agnimhd.__all__


#: Run in a fresh interpreter: the session's sys.modules also holds whatever
#: other tests imported (test_adapters imports DESC on purpose).
_IMPORTS = """
import sys, agnimhd, agnimhd.cli, agnimhd.adapters
names = {m.split(".")[0] for m in sys.modules} - set(sys.stdlib_module_names)
print(",".join(sorted(n for n in names if not n.startswith("_"))))
"""


def _imported_by_agnimhd():
    """Top-level third-party modules loaded by importing the package."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", _IMPORTS], capture_output=True, text=True, check=True
    )
    return set(out.stdout.strip().split(","))


def test_desc_is_not_a_dependency():
    """Importing agnimhd (adapters included) must not import DESC."""
    assert "desc" not in _imported_by_agnimhd()


def test_optional_extras_are_not_imported_eagerly():
    """h5py and matplotlib are optional extras: only their functions import them."""
    assert not {"h5py", "matplotlib"} & _imported_by_agnimhd()


def test_dense_mg_path_with_a_stand_in_solve(
    eq_data, diffmat, config, eq_meta, monkeypatch
):
    """``dense_mg`` plumbing: row blocks, shift, identity padding, block inverse
    iteration with Rayleigh-Ritz. A Cholesky solve stands in for JAXMg (GPUs).
    Called directly and through ``eigenpair``, it returns ``gamma^2``, as does
    its ``log`` callback."""
    from agnimhd import multigpu

    def stand_in(M, B, mesh, tile):
        return jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(M), B)

    monkeypatch.setattr(multigpu, "solve_shifted", stand_in)
    gamma2_ref = -eq_meta["dense_lambda3"]
    sigma = 1.05 * gamma2_ref  # just above gamma^2: few iterations
    solver = SolverConfig(eigensolver="dense_mg", sigma=sigma, mg_tile=1000)
    logged = []
    v, gamma2 = multigpu.dense_mg(
        eq_data, diffmat, config, solver, log=lambda it, g2, res, v: logged.append(g2)
    )
    gamma2_ep, _, resid = eigenpair(eq_data, diffmat, config, solver, v_guess=v)
    for value in (gamma2, logged[-1], gamma2_ep):
        assert float(value) == pytest.approx(gamma2_ref, rel=2.8e-5)
    assert float(resid) < 1e-5
