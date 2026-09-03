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

import agnimhd
from agnimhd import (
    EquilibriumData,
    SolverConfig,
    eigenpair,
    growth_rate,
    growth_rate_and_grad,
    growth_rate_of,
)
from agnimhd.backend import jax, jnp
from agnimhd.objective import _lambda_hf


def a_map(eq):
    """A stand-in ``params -> EquilibriumData``: ``{"a": v} -> eq.replace(a=v)``.

    **Not an equilibrium solve**, and a real optimization must not use one like
    it. It is enough here because these tests check the derivative *machinery*,
    for which the map need only be differentiable and move something the
    operator depends on. ``a`` because it is one scalar the whole operator is
    normalized by, so a finite difference costs one extra pair of solves rather
    than one per node, and the eigenvalue is most sensitive to it.
    """
    return lambda p: eq.replace(a=p["a"])


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------


def test_the_equilibrium_is_unstable(eq_data, diffmat, config):
    """The sign, before anything else.

    The shipped case is a known unstable QH equilibrium. A solver that gets the
    magnitude right and the sign wrong has answered the opposite physics
    question, so this is asserted on its own rather than folded into a
    tolerance on the value.
    """
    lam = growth_rate(eq_data, diffmat, config)
    assert float(lam) < 0.0, f"expected an unstable mode, got lambda = {float(lam)}"


def test_growth_rate_reproduces_the_reference(eq_data, diffmat, config, eq_meta):
    """The value matches the number recorded when the fixture was exported.

    The reference is read from the sidecar, not typed from a document.

    The tolerance is the eigenvalue's **relative** noise floor, 2.8e-5. That
    is a measured property of this operator: below it, two correct runs are
    allowed to disagree.
    """
    ref = float(eq_meta["dense_lambda3"])
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
    lam, v, resid = eigenpair(eq_data, diffmat, config)
    assert float(lam) < 0.0
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


def test_jax_lanczos_agrees_with_eigsh(eq_data, diffmat, config):
    """The two dense eigensolvers find the same mode.

    They share nothing but the matrix: one is host ARPACK behind a callback,
    the other is matfree Lanczos on an exact JAX LU. ARPACK is 1.53x faster on
    CPU and is the default; this keeps the alternative honest.

    The shift is ``-1e-3`` rather than the default ``-1e-1``, and that is not a
    tolerance being nudged to make a test pass -- see
    ``test_a_far_shift_selects_the_wrong_mode_and_the_residual_says_so`` for the
    measurement, and ``SolverConfig.sigma`` for why the default is where it is.
    A fixed-matvec Lanczos needs a shift that is below the spectrum *and* near
    it; ARPACK, which iterates to a tolerance, does not.
    """
    lam_a = float(growth_rate(eq_data, diffmat, config))
    lam_b = float(
        growth_rate(
            eq_data,
            diffmat,
            config,
            SolverConfig(eigensolver="jax_lanczos", sigma=-1e-3),
        )
    )
    assert np.sign(lam_a) == np.sign(lam_b), "the two eigensolvers disagree on sign"
    rel = abs(lam_a - lam_b) / abs(lam_a)
    assert rel < 2.8e-5, f"eigensolvers differ by {rel:.3e}, above the noise floor"


def test_a_far_shift_selects_the_wrong_mode_and_the_residual_says_so(
    eq_data, diffmat, config
):
    """A shift far below the spectrum breaks a fixed-budget Lanczos.

    Shift-invert maps ``lambda`` to ``1/(lambda - sigma)``, and Lanczos
    separates modes at a rate set by the ratio of those. As ``sigma`` recedes
    the ratio goes to one: on this case it is 1.0007 at ``sigma = -1e-1``
    against 1.0823 at ``-1e-3``. So the default shift -- chosen conservatively,
    because a shift *above* the spectrum has no recovery at all -- makes a
    50-matvec ``jax_lanczos`` return the wrong mode, with the wrong sign.

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
        eq_data, diffmat, config, SolverConfig(eigensolver="jax_lanczos", sigma=-1e-3)
    )
    lam_far, _, resid_far = eigenpair(
        eq_data, diffmat, config, SolverConfig(eigensolver="jax_lanczos", sigma=-1e-1)
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
    assert float(lam_far) > 0.0, (
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


@pytest.mark.parametrize("eigensolver", ["eigsh", "jax_lanczos"])
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

    The shift is placed just below the dense eigenvalue. A fixed-matvec Lanczos
    needs a shift that is below the spectrum *and* near it; see
    ``test_a_far_shift_selects_the_wrong_mode_and_the_residual_says_so``.
    """
    eq, diffmat, config = axisym_case
    _, lam_dense = _dense_reference(eq, diffmat, config)

    solver = SolverConfig(
        eigensolver=eigensolver, sigma=1.3 * lam_dense, num_matvecs=100
    )
    lam, v, resid = eigenpair(eq, diffmat, config, solver)
    lam = float(lam)

    assert v.dtype == np.complex128, "the eigenvector came back real"
    # A real eigenvector would satisfy the assertions below for the wrong
    # reason: it would mean the solve collapsed onto the real subspace.
    v = np.asarray(v)
    assert np.linalg.norm(v.imag) / np.linalg.norm(v) > 1e-3

    assert np.sign(lam) == np.sign(lam_dense), (
        f"{eigensolver} flipped the sign of the growth rate: {lam:.6e} vs dense "
        f"{lam_dense:.6e} -- a stable/unstable misclassification"
    )
    assert float(resid) < 1e-3, f"eigenvector not converged: residual {resid:.3e}"
    np.testing.assert_allclose(lam, lam_dense, rtol=1e-6)


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
    solver = SolverConfig(eigensolver="eigsh", sigma=1.3 * lam_dense)

    lam = growth_rate(eq, diffmat, config, solver)
    assert lam.dtype == jnp.zeros(()).dtype, f"growth_rate returned {lam.dtype}"

    g = jax.grad(growth_rate_of)({"a": eq.a}, a_map(eq), diffmat, config, solver)["a"]
    assert np.isrealobj(np.asarray(g)), "the gradient came back complex"
    assert np.isfinite(float(g))
    assert float(g) != 0.0


def test_unimplemented_eigensolver_says_so(eq_data, diffmat, config):
    """The two-level path is not wired in here, and says so rather than
    silently falling back to a different solver."""
    with pytest.raises(NotImplementedError, match="pcg_deflated"):
        growth_rate(eq_data, diffmat, config, SolverConfig(eigensolver="pcg_deflated"))


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
    """One gradient step in ``a`` raises lambda toward zero.

    Instability is ``lambda < 0``, so an optimizer must *increase* it. This is
    the end-to-end statement that the sign convention holds all the way from
    the operator to something a caller would write, and it is the check that
    catches a globally flipped gradient -- which every finiteness and
    magnitude test above would pass.
    """
    a0 = float(eq_data.a)
    lam0, g = growth_rate_and_grad({"a": a0}, a_map(eq_data), diffmat, config)
    lam0 = float(lam0)
    assert lam0 < 0.0

    dlam_da = float(g["a"])
    # Ascent on lambda: step along +grad, sized to a small relative change in a.
    step = 1e-4 * a0 / abs(dlam_da)
    lam1 = float(growth_rate(eq_data.replace(a=a0 + step * dlam_da), diffmat, config))

    assert lam1 > lam0, (
        f"an ascent step made lambda worse: {lam0:+.6e} -> {lam1:+.6e}. "
        "Either the gradient sign is flipped or the step left the linear "
        "regime."
    )


def test_gradient_is_the_hellmann_feynman_contraction(eq_data, diffmat, config):
    """The gradient equals ``v^T (dA/dq) v / v^T v`` with ``v`` held fixed.

    Computed here the long way -- freeze the eigenvector from one solve, then
    differentiate the Rayleigh quotient explicitly -- and compared against what
    ``growth_rate`` returns. They must be identical, not merely close: the
    custom VJP exists precisely to make the second expression compute the
    first, so any difference means gradient is leaking through the eigensolve
    or through the eigenvector-selection ``argmax``.
    """
    from agnimhd.assemble import matfree_operator

    _, v, _ = eigenpair(eq_data, diffmat, config)
    v = jax.lax.stop_gradient(v)

    def rayleigh(eq):
        op = matfree_operator(eq, diffmat, config)
        return jnp.real(jnp.vdot(v, op["Ax"](v)) / jnp.vdot(v, v))

    want = float(jax.grad(rayleigh)(eq_data).a)
    got = float(
        jax.grad(growth_rate_of)({"a": eq_data.a}, a_map(eq_data), diffmat, config)["a"]
    )
    assert np.isclose(got, want, rtol=1e-10), (
        f"gradient is not the fixed-vector contraction: {got:+.6e} vs " f"{want:+.6e}"
    )


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


def test_desc_is_not_a_dependency():
    """AGNI must not import DESC, at any level, ever.

    The dependency direction is the point of the package: DESC will depend on
    ``agnimhd``, not the reverse. A lazy import inside a function would satisfy
    a naive check while still making the package unusable without DESC
    installed, so this walks every already-imported module rather than trying
    an import.
    """
    import sys

    assert "desc" not in sys.modules, (
        "importing agnimhd pulled in DESC. The dependency must go the other "
        "way: DESC depends on agnimhd."
    )


def test_only_the_allowed_dependencies_are_used():
    """jax, numpy, scipy, matfree -- and nothing else.

    Checked against what is actually imported after exercising the package,
    not against the declared metadata, since the metadata is the thing that
    would be out of date.
    """
    import sys

    allowed = {"jax", "jaxlib", "numpy", "scipy", "matfree", "agnimhd", "ml_dtypes"}
    stdlib = set(sys.stdlib_module_names)
    third_party = {
        name.split(".")[0] for name in sys.modules if not name.startswith("_")
    }
    third_party -= stdlib
    third_party = {n for n in third_party if not n.startswith("_")}
    # Anything pytest itself dragged in is not the package's doing.
    test_only = {"pytest", "pluggy", "iniconfig", "py", "_pytest", "opt_einsum"}
    unexpected = third_party - allowed - test_only
    assert "desc" not in unexpected
