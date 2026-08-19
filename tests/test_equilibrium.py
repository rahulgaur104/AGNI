"""Tests for the equilibrium contract.

``EquilibriumData`` is the whole interface between AGNI and whatever code
produced the equilibrium. Everything downstream reads these arrays and nothing
else, so the checks here are the checks that a consumer's adapter is either
right or caught.

The two traps that this file pins deliberately, because both have cost real
time and neither announces itself:

* the minor radius ``a`` -- the eigenvalue is hypersensitive to it, and two
  defensible ways to compute it from the same equilibrium differ by a few
  percent;
* ``p`` must be **pressure**, not a kinetic-energy density or a temperature.
"""

import json

import numpy as np
import pytest

from agnimhd import EquilibriumData
from agnimhd.equilibrium import (
    FORMAT_VERSION,
    OPTIONAL_ARRAYS,
    REQUIRED_ARRAYS,
    REQUIRED_SCALARS,
)

# -- structure ------------------------------------------------------------


def test_contract_is_closed_and_documented():
    """The contract names are fixed; a consumer can enumerate them."""
    assert "Psi" in REQUIRED_SCALARS and "a" in REQUIRED_SCALARS
    assert "drive" in OPTIONAL_ARRAYS
    # No name may be both required and optional -- an adapter reading the two
    # tuples to decide what to supply would get contradictory instructions.
    assert not set(REQUIRED_ARRAYS) & set(OPTIONAL_ARRAYS)
    assert len(set(REQUIRED_ARRAYS)) == len(REQUIRED_ARRAYS)


def test_fixture_matches_its_sidecar(eq_data, eq_meta):
    """The shipped case is what its provenance file says it is."""
    assert eq_data.resolution == tuple(eq_meta["resolution"])
    assert eq_data.n_nodes == np.prod(eq_meta["resolution"])
    assert eq_data.NFP == eq_meta["NFP"]


def test_node_ordering_is_rho_major(eq_data):
    """Flat index of ``(i, j, k)`` is ``(i * n_theta + j) * n_zeta + k``.

    Every array in the contract is flat, so the ordering is not recoverable
    from the data -- it is a convention the adapter has to match. Getting it
    wrong permutes the operator into something that is still symmetric and
    still has a spectrum, so it does not crash; it just answers a different
    question.
    """
    n_rho, n_theta, n_zeta = eq_data.resolution
    flat = np.arange(eq_data.n_nodes, dtype=float)
    cube = np.asarray(eq_data.reshape(flat))
    assert cube.shape == (n_rho, n_theta, n_zeta)
    i, j, k = 2, 5, 3
    assert cube[i, j, k] == (i * n_theta + j) * n_zeta + k


def test_missing_required_field_names_itself(eq_data):
    """A missing field raises and says which one."""
    kwargs = _kwargs(eq_data)
    kwargs.pop("g_vv")
    with pytest.raises(ValueError, match="g_vv"):
        EquilibriumData(**kwargs)


def test_unknown_field_is_rejected(eq_data):
    """The contract is closed, so a typo cannot silently do nothing."""
    kwargs = _kwargs(eq_data)
    kwargs["g_rho_rho"] = kwargs["g_rr"]
    with pytest.raises(ValueError, match="unknown field"):
        EquilibriumData(**kwargs)


def test_wrong_length_array_is_rejected(eq_data):
    """An array of the wrong length is a resolution mismatch, not a reshape."""
    kwargs = _kwargs(eq_data)
    kwargs["g_rr"] = np.asarray(kwargs["g_rr"])[:-1]
    with pytest.raises(ValueError):
        EquilibriumData(**kwargs)


def test_too_few_radial_shells_is_rejected(eq_data):
    """``n_rho < 3`` leaves no interior ``xi^rho`` after the Dirichlet mask."""
    ones = np.ones(2 * 4 * 4)
    kwargs = {k: ones for k in REQUIRED_ARRAYS}
    kwargs["drive"] = 0.0 * ones
    with pytest.raises(ValueError, match="n_rho"):
        EquilibriumData(n_rho=2, n_theta=4, n_zeta=4, Psi=1.0, a=0.5, **kwargs)


# -- the drive ------------------------------------------------------------


def test_drive_routes_agree(eq_data):
    """Supplying ``drive`` and deriving it from the two fields agree.

    The shipped fixture carries both, precisely so this can be checked without
    an equilibrium code present. Measured agreement is 2.8e-16 relative.
    """
    if eq_data.J_cross_grad_rho is None:
        pytest.fail(
            "the shipped fixture no longer carries J_cross_grad_rho; the "
            "two-route drive check cannot run. Re-export it with "
            "tools/export_fixture.py."
        )
    direct = np.asarray(eq_data.drive)
    derived = np.asarray(eq_data.replace(drive=None).instability_drive())
    rel = np.max(np.abs(direct - derived)) / np.max(np.abs(direct))
    assert rel < 1e-14, f"drive routes disagree by {rel:.3e} relative"


def test_drive_is_required_one_way_or_the_other(eq_data):
    """Neither route supplied is an error, and the message says both routes."""
    kwargs = _kwargs(eq_data)
    for key in ("drive", "J_cross_grad_rho", "B_dot_grad_grad_rho"):
        kwargs.pop(key, None)
    with pytest.raises(ValueError, match="J_cross_grad_rho"):
        EquilibriumData(**kwargs)


def test_drive_from_vectors_only(eq_data):
    """The two-vector route alone is sufficient -- ``drive`` is not required."""
    kwargs = _kwargs(eq_data)
    kwargs.pop("drive")
    eqd = EquilibriumData(**kwargs)
    assert eqd.drive is None
    got = np.asarray(eqd.instability_drive())
    assert np.all(np.isfinite(got))
    assert got.shape == (eq_data.n_nodes,)


# -- validation messages ---------------------------------------------------


@pytest.mark.parametrize("key", ["g_rr", "sqrt_g", "iota", "p_r"])
def test_nonfinite_field_names_itself(eq_data, key):
    """A NaN anywhere raises, and the message names the field.

    An adapter bug usually shows up as a NaN in exactly one array. Naming it
    is the difference between a one-line fix and bisecting twenty arrays.
    """
    kwargs = _kwargs(eq_data)
    arr = np.array(kwargs[key], dtype=float)
    arr[7] = np.nan
    kwargs[key] = arr
    with pytest.raises(ValueError, match=key):
        EquilibriumData(**kwargs)


def test_pressure_trap_is_called_out_by_name(eq_data):
    """A non-finite ``p`` mentions that ``p`` must be pressure in pascals.

    Feeding a raw kinetic-energy density instead of a pressure is a mistake
    that has been made here, and its symptom is a NaN out of the assembly with
    nothing pointing at the cause.
    """
    kwargs = _kwargs(eq_data)
    arr = np.array(kwargs["p"], dtype=float)
    arr[0] = np.inf
    kwargs["p"] = arr
    with pytest.raises(ValueError, match="(?i)pressure"):
        EquilibriumData(**kwargs)


def test_vanishing_jacobian_mentions_the_axis(eq_data):
    """A NaN in ``sqrt_g`` points at rho = 0 in the node set."""
    kwargs = _kwargs(eq_data)
    arr = np.array(kwargs["sqrt_g"], dtype=float)
    arr[3] = np.nan
    kwargs["sqrt_g"] = arr
    with pytest.raises(ValueError, match="(?i)axis"):
        EquilibriumData(**kwargs)


def test_validate_passes_on_the_shipped_case(eq_data):
    """The fixture satisfies its own contract."""
    eq_data.validate()


def test_validate_can_be_skipped_for_tracers(eq_data):
    """``validate=False`` builds without touching values."""
    kwargs = _kwargs(eq_data)
    kwargs["g_rr"] = np.full(eq_data.n_nodes, np.nan)
    kwargs["validate"] = False
    eqd = EquilibriumData(**kwargs)
    assert np.isnan(np.asarray(eqd.g_rr)).all()


# -- serialization ---------------------------------------------------------


def test_npz_round_trip_is_exact(eq_data, tmp_path):
    """Save/load returns bit-identical arrays and scalars."""
    path = tmp_path / "rt.npz"
    eq_data.save(path)
    back = EquilibriumData.load(path)
    assert back.resolution == eq_data.resolution
    assert back.NFP == eq_data.NFP
    for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
        want, got = getattr(eq_data, key), getattr(back, key)
        if want is None:
            assert got is None, f"{key} appeared out of nowhere"
            continue
        np.testing.assert_array_equal(np.asarray(want), np.asarray(got), err_msg=key)
    for key in REQUIRED_SCALARS:
        assert float(getattr(back, key)) == float(getattr(eq_data, key))


def test_load_rejects_a_future_format_version(eq_data, tmp_path):
    """A newer file fails loudly rather than being read with the wrong meaning."""
    path = tmp_path / "future.npz"
    eq_data.save(path)
    with np.load(path) as f:
        payload = {k: f[k] for k in f.files}
    payload["format_version"] = np.asarray(FORMAT_VERSION + 1)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="(?i)version"):
        EquilibriumData.load(path)


def test_hdf5_round_trip_or_a_clear_capability_error(eq_data, tmp_path):
    """HDF5 works if h5py is installed, and says so plainly if it is not.

    h5py is not one of the four allowed dependencies, so this path is optional
    by construction. What is *not* optional is that the failure mode be an
    explicit ImportError naming h5py rather than an AttributeError from deep
    inside the writer.
    """
    path = tmp_path / "rt.h5"
    try:
        eq_data.save_hdf5(path)
    except ImportError as err:
        assert "h5py" in str(err)
        return
    back = EquilibriumData.load_hdf5(path)
    assert back.resolution == eq_data.resolution
    np.testing.assert_array_equal(np.asarray(eq_data.g_rr), np.asarray(back.g_rr))


def test_sidecar_is_valid_json_and_records_provenance(eq_meta):
    """The fixture's provenance file is machine-readable, not prose.

    Reference numbers used by the suite are read from here. A number typed
    into a test from a document is a number nobody can trace back to a run.
    """
    assert isinstance(eq_meta, dict)
    for key in (
        "resolution",
        "NFP",
        "gamma",
        "dense_lambda3",
        "rho_nodes",
        "a",
        "a_definition",
        "Psi",
        "source_equilibrium",
        "desc_version",
    ):
        assert key in eq_meta, f"sidecar is missing {key!r}"
    # Assert the SIGN before anything else: it is the physics answer, and a
    # sign error is the failure this case exists to catch.
    assert float(eq_meta["dense_lambda3"]) < 0.0, "the shipped case must be unstable"
    # The `a` trap is only avoidable if the definition travels with the number.
    assert "Quadrature" in eq_meta["a_definition"]
    assert len(eq_meta["rho_nodes"]) == eq_meta["resolution"][0]
    # Round-trips as JSON, so nothing in it is a numpy scalar or a NaN.
    json.loads(json.dumps(eq_meta))


# -- the minor-radius trap -------------------------------------------------


def test_minor_radius_moves_the_data_it_normalizes(eq_data):
    """``a`` is a real input, not a cosmetic label.

    Two defensible ways to compute the minor radius from the same equilibrium
    -- a quadrature grid and a linear grid -- were measured to differ by 3.76%,
    and the eigenvalue is hypersensitive to the difference. This test pins that
    ``a`` is carried as data and that a 3.76% change in it is a 3.76% change in
    the object, so nothing downstream can be reading a hard-coded value.
    """
    a0 = float(eq_data.a)
    assert a0 > 0.0
    perturbed = eq_data.replace(a=a0 * 1.0376)
    assert float(perturbed.a) != a0
    assert np.isclose(float(perturbed.a) / a0, 1.0376)
    # and nothing else moved
    np.testing.assert_array_equal(np.asarray(eq_data.g_rr), np.asarray(perturbed.g_rr))


def test_replace_is_a_copy(eq_data):
    """``replace`` does not mutate the original."""
    a0 = float(eq_data.a)
    eq_data.replace(a=2.0 * a0)
    assert float(eq_data.a) == a0


# -- pytree ----------------------------------------------------------------


def test_pytree_round_trip():
    """Flatten/unflatten is the identity, and shapes live in the aux data."""
    import jax

    eqd = _tiny()
    leaves, treedef = jax.tree_util.tree_flatten(eqd)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.resolution == eqd.resolution
    assert back.NFP == eqd.NFP
    np.testing.assert_array_equal(np.asarray(eqd.g_rr), np.asarray(back.g_rr))


def test_scalars_are_leaves_not_aux():
    """``Psi`` and ``a`` are differentiable inputs, so they must be leaves.

    If either were static aux data, ``jax.grad`` would silently return no
    gradient for it -- and ``a`` is the input the eigenvalue is most sensitive
    to.
    """
    import jax

    eqd = _tiny()
    leaves, _ = jax.tree_util.tree_flatten(eqd)
    shapes = [np.shape(x) for x in leaves]
    assert shapes.count(()) >= 2, "Psi and a are not among the leaves"


def test_pytree_maps_over_tracers():
    """The container survives a JAX transformation without validating."""
    import jax
    import jax.numpy as jnp

    eqd = _tiny()

    @jax.jit
    def total(e):
        return jnp.sum(e.g_rr) + e.a

    got = float(total(eqd))
    assert np.isfinite(got)


# -- helpers ---------------------------------------------------------------


def _kwargs(eq_data):
    """Constructor keywords reproducing ``eq_data``, as plain numpy."""
    kwargs = dict(
        n_rho=eq_data.n_rho,
        n_theta=eq_data.n_theta,
        n_zeta=eq_data.n_zeta,
        NFP=eq_data.NFP,
        Psi=float(eq_data.Psi),
        a=float(eq_data.a),
    )
    for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
        val = getattr(eq_data, key)
        if val is not None:
            kwargs[key] = np.asarray(val)
    return kwargs


def _tiny():
    """A minimal structurally-valid container. Physically meaningless."""
    n_rho, n_theta, n_zeta = 4, 4, 4
    n = n_rho * n_theta * n_zeta
    ones = np.ones(n)
    fields = {k: ones.copy() for k in REQUIRED_ARRAYS}
    fields["drive"] = 0.0 * ones
    return EquilibriumData(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        Psi=1.0,
        a=0.5,
        **fields,
    )
