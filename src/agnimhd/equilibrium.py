"""The equilibrium interface: ``EquilibriumData``.

This is the **only** way the solver learns about an equilibrium. It holds arrays
and scalars -- nothing more. It is constructible from raw NumPy or JAX arrays
with no equilibrium code present, and serializable to a file.

``agnimhd`` ships **no adapters**. Converting a DESC ``Equilibrium``, a VMEC
``wout`` file, or a GVEC state into an ``EquilibriumData`` is the consumer's job
and lives in the consumer's repository. What this module owes them is a contract
precise enough to implement against without reading the solver source; see
``docs/adapters.md`` for the per-code checklist, and use ``agnimhd validate
<file>`` to check an adapter's output.

Coordinates
-----------
Everything is in **PEST straight-field-line coordinates** ``(rho, theta_PEST,
phi)``, abbreviated ``(r, v, p)`` in the field names.

- ``rho`` is the normalized radial coordinate, ``rho = sqrt(psi / psi_edge)``,
  running on ``(0, 1]``.
- ``theta_PEST`` is the straight-field-line poloidal angle on ``[0, 2*pi)``.
- ``phi`` is the geometric toroidal angle. For a field-period-symmetric device
  the nodes span one field period, ``[0, 2*pi/NFP)``.

Node ordering
-------------
**rho-major.** The flat index of node ``(i, j, k)`` -- radial ``i``, poloidal
``j``, toroidal ``k`` -- is::

    n = (i * n_theta + j) * n_zeta + k

equivalently ``numpy.reshape(arr, (n_rho, n_theta, n_zeta))`` recovers the
tensor-product structure. Every index map in the package assumes it. An adapter
that emits a different ordering will not error -- it will silently solve a
different problem -- so this is checked structurally where it can be
(see :meth:`EquilibriumData.validate`) and asserted in the docs where it cannot.

Units
-----
SI throughout, unnormalized. The solver applies its own normalization
internally, from ``a`` and ``B_N = |Psi| / (pi * a**2)``. Do not pre-normalize.

See Also
--------
docs/theory.md : the energy functional and its discretization.
docs/adapters.md : how to write an adapter for a new equilibrium code.
"""

import numpy as np

from .backend import errorif, jax, jnp

__all__ = ["EquilibriumData", "FORMAT_VERSION"]

#: Version of the on-disk format written by :meth:`EquilibriumData.save`.
#: Bumped whenever a field is added, removed, or reinterpreted. ``load``
#: refuses a file whose major version it does not know.
FORMAT_VERSION = 1

#: Node-resolved arrays every ``EquilibriumData`` must carry, each of shape
#: ``(n_nodes,)`` in the rho-major ordering documented above.
#:
#: The six covariant metric components are of the PEST coordinate basis:
#: ``g_ab = e_a . e_b`` with ``e_r = de/drho``, ``e_v = de/dtheta_PEST``,
#: ``e_p = de/dphi``.
REQUIRED_ARRAYS = (
    # covariant (lower) PEST metric, m^2 except g_rr which is dimensionless-ish
    "g_rr",
    "g_rv",
    "g_rp",
    "g_vv",
    "g_vp",
    "g_pp",
    # contravariant radial metric component, g^rr = grad(rho) . grad(rho), m^-2
    "g_sup_rr",
    # Jacobian sqrt(g) = e_r . (e_v x e_p) and its three partial derivatives,
    # all taken at fixed PEST coordinates. m^3.
    "sqrt_g",
    "sqrt_g_r",
    "sqrt_g_v",
    "sqrt_g_p",
    # current density: contravariant toroidal component J^zeta (A m^-3) and the
    # magnitude |J| (A m^-2)
    "J_sup_zeta",
    "abs_J",
    # profiles, evaluated on the same nodes (constant on each rho surface)
    "iota",
    "psi_r",
    "psi_rr",
    "p",
    "p_r",
)

#: Differentiable scalars. ``Psi`` is the parameter a consumer most often
#: optimizes against; ``a`` sets the entire normalization.
REQUIRED_SCALARS = ("Psi", "a")

#: Optional arrays. ``drive`` may be supplied directly, or derived by AGNI from
#: the two vector fields; see :meth:`EquilibriumData.instability_drive`.
OPTIONAL_ARRAYS = (
    "drive",
    "J_cross_grad_rho",
    "B_dot_grad_grad_rho",
)


def _as_1d(name, value, n_nodes):
    """Return ``value`` as a flat float array of length ``n_nodes``."""
    arr = jnp.asarray(value, dtype=jnp.float64).reshape(-1)
    errorif(
        arr.size != n_nodes,
        ValueError,
        f"{name} has {arr.size} entries but the grid has {n_nodes} nodes. "
        "Every field must be flattened to (n_rho * n_theta * n_zeta,) in "
        "rho-major order; see agnimhd.equilibrium's module docstring.",
    )
    return arr


@jax.tree_util.register_pytree_node_class
class EquilibriumData:
    """Equilibrium quantities on a PEST tensor-product grid.

    A JAX pytree: the node arrays and the two scalars are leaves, the
    resolution and ``NFP`` are static. That is what lets a consumer write

    .. code-block:: python

        jax.grad(lambda Psi: agnimhd.growth_rate(eq_data.replace(Psi=Psi), cfg))

    and have the derivative flow through the package boundary.

    Parameters
    ----------
    n_rho, n_theta, n_zeta : int
        Grid resolution. ``n_nodes = n_rho * n_theta * n_zeta``.
    Psi : float
        Total toroidal flux through the last closed flux surface, in webers.
        Signed: the sign convention follows the equilibrium code's own.
        Differentiable.
    a : float
        Minor radius in meters. **Not a free choice** -- see the warning below.
        Differentiable.
    NFP : int, optional
        Number of field periods. Used only to set the toroidal period of the
        node set, ``2*pi/NFP``. Default 1.
    g_rr, g_rv, g_rp, g_vv, g_vp, g_pp : ndarray, shape (n_nodes,)
        Covariant PEST metric components, ``g_ab = e_a . e_b``, in m^2.
    g_sup_rr : ndarray, shape (n_nodes,)
        ``grad(rho) . grad(rho)``, in m^-2.
    sqrt_g : ndarray, shape (n_nodes,)
        PEST Jacobian ``e_rho . (e_theta x e_phi)``, in m^3. Must be nonzero
        everywhere; the solver divides by it.
    sqrt_g_r, sqrt_g_v, sqrt_g_p : ndarray, shape (n_nodes,)
        Partial derivatives of ``sqrt_g`` with respect to ``rho``,
        ``theta_PEST`` and ``phi``, at fixed PEST coordinates. m^3.
    J_sup_zeta : ndarray, shape (n_nodes,)
        Contravariant toroidal current density ``J^zeta``, in A m^-3.
    abs_J : ndarray, shape (n_nodes,)
        Current density magnitude ``|J|``, in A m^-2.
    iota : ndarray, shape (n_nodes,)
        Rotational transform, dimensionless. Must be nonzero: the solver's
        variable change divides by it. A true mirror (``iota == 0``) is handled
        by a separate branch of the mass matrix and is detected automatically.
    psi_r, psi_rr : ndarray, shape (n_nodes,)
        First and second derivatives of the toroidal flux ``psi`` with respect
        to ``rho``, in Wb. ``psi_rr`` is part of the contract but is currently
        recomputed spectrally by the solver; supply it anyway so adapters do not
        have to change when that stops being true.
    p, p_r : ndarray, shape (n_nodes,)
        **Pressure** in pascals and its radial derivative in Pa per unit rho.
        See the second trap below.
    drive : ndarray, shape (n_nodes,), optional
        The instability drive, in T A m^-1. If omitted, supply
        ``J_cross_grad_rho`` and ``B_dot_grad_grad_rho`` instead and AGNI will
        form it; see :meth:`instability_drive`.
    J_cross_grad_rho : ndarray, shape (n_nodes, 3), optional
        ``J x grad(rho)`` in Cartesian components, A m^-2 (per unit rho).
    B_dot_grad_grad_rho : ndarray, shape (n_nodes, 3), optional
        ``(B . grad) grad(rho)`` in Cartesian components, T m^-2.
    validate : bool, optional
        Run :meth:`validate` in the constructor. Default True. Set False only
        when constructing from traced arrays inside a transformation, where the
        finiteness checks cannot be evaluated.

    Warnings
    --------
    **``a`` is not a free choice.** The eigenvalue is hypersensitive to the
    minor radius: it enters the normalization as ``B_N = |Psi| / (pi a^2)`` and
    the operator's terms carry ``a^2``, ``a^3`` and ``a^4``. Two different
    formulas in DESC -- the ``QuadratureGrid`` area integral and the
    ``LinearGrid`` boundary line integral -- give values of ``a`` that differ by
    **3.76%** on the shipped QH case, which moves lambda far more than any
    discretization error you will be chasing.

    Pin one definition. AGNI's is the **cross-section area integral**::

        A = (1 / n_zeta) * sum_zeta  integral_{S(zeta)} |e_rho x e_theta| dtheta drho
        a = sqrt(A / pi)

    i.e. the zeta-average of the enclosed constant-phi cross-sectional area,
    computed by direct area quadrature -- *not* by a boundary line integral, and
    *not* extrapolated from the outermost surface. ``a`` is an explicit field of
    this interface rather than something recomputed internally precisely so that
    an adapter must make this choice consciously.

    **Pressure, not kinetic energy.** ``p`` is the plasma pressure in pascals.
    Feeding a raw kinetic-energy density (e.g. ``(3/2) n T``, or a ``n*T`` in
    eV) into ``p`` produces ``NaN`` from the assembly rather than a wrong number,
    because the compressibility term takes ``sqrt`` of quantities built from it.
    Convert to pressure first. :meth:`validate` asserts finiteness and names this
    cause.

    Examples
    --------
    >>> import numpy as np, agnimhd
    >>> n_rho, n_theta, n_zeta = 4, 6, 5
    >>> n = n_rho * n_theta * n_zeta
    >>> ones = np.ones(n)
    >>> eqd = agnimhd.EquilibriumData(          # doctest: +SKIP
    ...     n_rho=n_rho, n_theta=n_theta, n_zeta=n_zeta, Psi=1.0, a=0.5,
    ...     g_rr=ones, g_rv=0*ones, ..., drive=0*ones,
    ... )
    """

    def __init__(
        self,
        *,
        n_rho,
        n_theta,
        n_zeta,
        Psi,
        a,
        NFP=1,
        validate=True,
        **fields,
    ):
        self.n_rho = int(n_rho)
        self.n_theta = int(n_theta)
        self.n_zeta = int(n_zeta)
        self.NFP = int(NFP)
        n_nodes = self.n_nodes

        missing = [k for k in REQUIRED_ARRAYS if k not in fields]
        errorif(
            bool(missing),
            ValueError,
            f"EquilibriumData is missing required field(s): {sorted(missing)}. "
            "See agnimhd.equilibrium.REQUIRED_ARRAYS and docs/adapters.md.",
        )
        unknown = set(fields) - set(REQUIRED_ARRAYS) - set(OPTIONAL_ARRAYS)
        errorif(
            bool(unknown),
            ValueError,
            f"EquilibriumData got unknown field(s): {sorted(unknown)}. "
            "The contract is closed: adding a field here has no effect on the "
            "solver, so a typo would silently do nothing.",
        )

        for key in REQUIRED_ARRAYS:
            setattr(self, key, _as_1d(key, fields[key], n_nodes))

        for key in ("drive",):
            val = fields.get(key, None)
            setattr(self, key, None if val is None else _as_1d(key, val, n_nodes))

        for key in ("J_cross_grad_rho", "B_dot_grad_grad_rho"):
            val = fields.get(key, None)
            if val is None:
                setattr(self, key, None)
                continue
            arr = jnp.asarray(val, dtype=jnp.float64).reshape(n_nodes, 3)
            setattr(self, key, arr)

        self.Psi = jnp.asarray(Psi, dtype=jnp.float64)
        self.a = jnp.asarray(a, dtype=jnp.float64)

        errorif(
            self.drive is None
            and (self.J_cross_grad_rho is None or self.B_dot_grad_grad_rho is None),
            ValueError,
            "EquilibriumData needs the instability drive. Supply `drive` "
            "directly, or supply both `J_cross_grad_rho` and "
            "`B_dot_grad_grad_rho` and AGNI will form it. See "
            "EquilibriumData.instability_drive for the definition and the "
            "s -> rho substitution it depends on.",
        )

        if validate:
            self.validate()

    # -- structure ---------------------------------------------------------

    @property
    def n_nodes(self):
        """int : Total number of grid nodes, ``n_rho * n_theta * n_zeta``."""
        return self.n_rho * self.n_theta * self.n_zeta

    @property
    def resolution(self):
        """tuple of int : ``(n_rho, n_theta, n_zeta)``."""
        return (self.n_rho, self.n_theta, self.n_zeta)

    def reshape(self, arr):
        """Reshape a flat node array to ``(n_rho, n_theta, n_zeta)``.

        Parameters
        ----------
        arr : ndarray, shape (n_nodes,)

        Returns
        -------
        ndarray, shape (n_rho, n_theta, n_zeta)
        """
        return jnp.asarray(arr).reshape(self.n_rho, self.n_theta, self.n_zeta)

    def replace(self, **changes):
        """Return a copy with some fields replaced.

        Anything not named keeps its current value. Validation is skipped, so
        this is safe to call on traced values inside ``jax.grad``.

        Parameters
        ----------
        **changes
            Any constructor keyword.

        Returns
        -------
        EquilibriumData
        """
        kwargs = dict(
            n_rho=self.n_rho,
            n_theta=self.n_theta,
            n_zeta=self.n_zeta,
            NFP=self.NFP,
            Psi=self.Psi,
            a=self.a,
            validate=False,
        )
        for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
            val = getattr(self, key)
            if val is not None:
                kwargs[key] = val
        kwargs.update(changes)
        return EquilibriumData(**kwargs)

    # -- pytree ------------------------------------------------------------

    def tree_flatten(self):
        """Flatten to (leaves, aux) for JAX. Arrays and scalars are leaves."""
        keys = tuple(
            k for k in REQUIRED_ARRAYS + OPTIONAL_ARRAYS if getattr(self, k) is not None
        )
        leaves = tuple(getattr(self, k) for k in keys) + (self.Psi, self.a)
        aux = (keys, self.n_rho, self.n_theta, self.n_zeta, self.NFP)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        """Rebuild from (aux, leaves). No validation: leaves may be tracers."""
        keys, n_rho, n_theta, n_zeta, NFP = aux
        obj = object.__new__(cls)
        obj.n_rho, obj.n_theta, obj.n_zeta, obj.NFP = n_rho, n_theta, n_zeta, NFP
        for key, leaf in zip(keys, leaves[: len(keys)]):
            setattr(obj, key, leaf)
        for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
            if key not in keys:
                setattr(obj, key, None)
        obj.Psi, obj.a = leaves[-2], leaves[-1]
        return obj

    # -- physics -----------------------------------------------------------

    def instability_drive(self):
        """Return the instability drive on the nodes, in T A m^-1.

        Uses the supplied ``drive`` if there is one, otherwise forms it from
        ``J_cross_grad_rho`` and ``B_dot_grad_grad_rho`` as

        .. math::

            \\mathrm{drive} = \\frac{2\\,
                (\\mathbf{J} \\times \\nabla\\rho) \\cdot
                (\\mathbf{B}\\cdot\\nabla)\\nabla\\rho}
                {(g^{\\rho\\rho})^2}

        Reference: Cooper, Correa-Restrepo et al., TERPSICHORE,
        doi:10.1007/978-1-4613-0659-7_8, Eq. (5) p. 162.

        Warnings
        --------
        **The paper writes this in terms of** ``s = rho**2``; **AGNI replaces
        every** ``s`` **with** ``rho``. That substitution is not a change of
        variable applied consistently -- it is a redefinition of the radial
        coordinate the whole functional is written in. Re-deriving the
        expression from the paper without making the same substitution produces
        a drive that is wrong by a rho-dependent factor of order two, which
        changes the eigenvalue's magnitude and can change its sign near
        marginality. If you are checking this term against the literature, check
        that first.

        Returns
        -------
        ndarray, shape (n_nodes,)
        """
        if self.drive is not None:
            return self.drive
        num = jnp.sum(self.J_cross_grad_rho * self.B_dot_grad_grad_rho, axis=-1)
        return 2.0 * num / self.g_sup_rr**2

    # -- validation --------------------------------------------------------

    def validate(self):
        """Check the data against the contract.

        Raises
        ------
        ValueError
            With a message naming the field and the likely cause.

        Notes
        -----
        Requires concrete (non-traced) arrays. Constructors called inside a JAX
        transformation should pass ``validate=False``.
        """
        errorif(
            min(self.n_rho, self.n_theta, self.n_zeta) < 1,
            ValueError,
            f"resolution must be positive, got {self.resolution}.",
        )
        errorif(
            self.n_rho < 3,
            ValueError,
            f"n_rho = {self.n_rho} is too small: the Dirichlet mask removes the "
            "innermost and outermost radial shells, leaving no interior "
            "xi^rho degrees of freedom. Use n_rho >= 3 to assemble at all, and "
            "see docs/resolution.md for the accuracy floor, which is much "
            "higher.",
        )

        for key in REQUIRED_ARRAYS:
            arr = np.asarray(getattr(self, key))
            bad = int(np.count_nonzero(~np.isfinite(arr)))
            if not bad:
                continue
            hint = ""
            if key in ("p", "p_r"):
                hint = (
                    " `p` must be PRESSURE in pascals. A raw kinetic-energy "
                    "density, or a temperature in eV, produces exactly this "
                    "failure -- convert to pressure first."
                )
            elif key == "sqrt_g":
                hint = (
                    " A vanishing Jacobian usually means the node set includes "
                    "the magnetic axis (rho = 0). AGNI's radial nodes must be "
                    "strictly inside (0, 1]."
                )
            raise ValueError(f"{key} has {bad} non-finite entries of {arr.size}.{hint}")

        sqrt_g = np.asarray(self.sqrt_g)
        errorif(
            bool(np.any(sqrt_g == 0.0)),
            ValueError,
            "sqrt_g vanishes at "
            f"{int(np.count_nonzero(sqrt_g == 0.0))} node(s). The solver "
            "divides by it; a node on the magnetic axis is the usual cause.",
        )
        errorif(
            bool(np.any(np.sign(sqrt_g) != np.sign(sqrt_g.flat[0]))),
            ValueError,
            "sqrt_g changes sign across the grid, which means the PEST basis "
            "flips handedness somewhere. Check the sign convention of "
            "theta_PEST and phi in the adapter.",
        )

        g_sup_rr = np.asarray(self.g_sup_rr)
        errorif(
            bool(np.any(g_sup_rr <= 0.0)),
            ValueError,
            "g_sup_rr must be strictly positive everywhere: it is "
            "|grad(rho)|^2. Non-positive entries mean the contravariant, not "
            "covariant, metric was supplied, or the sign was flipped.",
        )

        for key in ("g_rr", "g_vv", "g_pp"):
            arr = np.asarray(getattr(self, key))
            errorif(
                bool(np.any(arr <= 0.0)),
                ValueError,
                f"{key} is a diagonal covariant metric component and must be "
                "strictly positive. Non-positive entries usually mean the "
                "CONTRAVARIANT metric was supplied by mistake.",
            )

        iota = np.asarray(self.iota)
        errorif(
            bool(np.any(iota == 0.0)) and not bool(np.all(np.abs(iota) < 1e-12)),
            ValueError,
            "iota vanishes on some but not all nodes. AGNI's variable change "
            "divides by iota; a true mirror (iota identically zero) takes a "
            "separate branch, but a partial zero has no meaning here.",
        )

        drive = np.asarray(self.instability_drive())
        errorif(
            bool(np.any(~np.isfinite(drive))),
            ValueError,
            f"the instability drive has {int(np.count_nonzero(~np.isfinite(drive)))} "
            "non-finite entries. If it was derived from J_cross_grad_rho and "
            "B_dot_grad_grad_rho, check g_sup_rr for near-zero values, which "
            "the 1/(g^rr)^2 amplifies.",
        )

        for name, val in (("Psi", self.Psi), ("a", self.a)):
            v = float(np.asarray(val))
            errorif(
                not np.isfinite(v),
                ValueError,
                f"{name} must be finite, got {v}.",
            )
        errorif(
            float(np.asarray(self.a)) <= 0.0,
            ValueError,
            f"a must be positive, got {float(np.asarray(self.a))}.",
        )
        errorif(
            float(np.asarray(self.Psi)) == 0.0,
            ValueError,
            "Psi must be nonzero: the normalization B_N = |Psi|/(pi a^2) "
            "divides by it.",
        )
        return self

    # -- serialization -----------------------------------------------------

    def save(self, path):
        """Write to a versioned ``.npz`` file.

        The format is a NumPy zip archive containing one array per field, the
        two scalars as 0-d arrays, and the integers ``n_rho``, ``n_theta``,
        ``n_zeta``, ``NFP`` and ``format_version``. It is plain NumPy on
        purpose: HDF5 and NetCDF would each add a dependency, and the package's
        dependency set (jax, numpy, scipy, matfree) is a hard constraint.
        :func:`save_hdf5` is available when ``h5py`` is installed.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination. ``.npz`` is appended if absent.

        Returns
        -------
        str
            The path written.
        """
        path = str(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        out = {
            "format_version": np.asarray(FORMAT_VERSION),
            "n_rho": np.asarray(self.n_rho),
            "n_theta": np.asarray(self.n_theta),
            "n_zeta": np.asarray(self.n_zeta),
            "NFP": np.asarray(self.NFP),
            "Psi": np.asarray(self.Psi),
            "a": np.asarray(self.a),
        }
        for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
            val = getattr(self, key)
            if val is not None:
                out[key] = np.asarray(val)
        np.savez_compressed(path, **out)
        return path

    @classmethod
    def load(cls, path, validate=True):
        """Read a file written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
        validate : bool
            Run :meth:`validate` after loading.

        Returns
        -------
        EquilibriumData

        Raises
        ------
        ValueError
            If the file's ``format_version`` is newer than this package knows.
        """
        with np.load(str(path)) as f:
            ver = int(f["format_version"])
            errorif(
                ver > FORMAT_VERSION,
                ValueError,
                f"{path} declares format_version {ver}, but this agnimhd "
                f"understands at most {FORMAT_VERSION}. Upgrade agnimhd.",
            )
            fields = {
                k: f[k] for k in REQUIRED_ARRAYS + OPTIONAL_ARRAYS if k in f.files
            }
            return cls(
                n_rho=int(f["n_rho"]),
                n_theta=int(f["n_theta"]),
                n_zeta=int(f["n_zeta"]),
                NFP=int(f["NFP"]),
                Psi=float(f["Psi"]),
                a=float(f["a"]),
                validate=validate,
                **fields,
            )

    def save_hdf5(self, path):
        """Write to HDF5. Requires the optional ``h5py`` package.

        Same contents as :meth:`save`: one dataset per field, with the scalars
        and the format version as root attributes.

        Parameters
        ----------
        path : str or pathlib.Path

        Returns
        -------
        str

        Raises
        ------
        ImportError
            Naming ``h5py`` and this feature. HDF5 is genuinely optional -- the
            native format is ``.npz`` -- so this is an explicit capability
            check rather than a silent fallback.
        """
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "EquilibriumData.save_hdf5 needs the optional package 'h5py', "
                "which is not installed. `pip install h5py`, or use "
                "EquilibriumData.save, which writes the native .npz format and "
                "needs nothing beyond numpy."
            ) from exc
        path = str(path)
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["n_rho"] = self.n_rho
            f.attrs["n_theta"] = self.n_theta
            f.attrs["n_zeta"] = self.n_zeta
            f.attrs["NFP"] = self.NFP
            f.attrs["Psi"] = float(np.asarray(self.Psi))
            f.attrs["a"] = float(np.asarray(self.a))
            for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
                val = getattr(self, key)
                if val is not None:
                    f.create_dataset(key, data=np.asarray(val))
        return path

    @classmethod
    def load_hdf5(cls, path, validate=True):
        """Read an HDF5 file written by :meth:`save_hdf5`.

        Parameters
        ----------
        path : str or pathlib.Path
        validate : bool

        Returns
        -------
        EquilibriumData

        Raises
        ------
        ImportError
            Naming ``h5py``.
        """
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "EquilibriumData.load_hdf5 needs the optional package 'h5py', "
                "which is not installed. `pip install h5py`, or use "
                "EquilibriumData.load on a native .npz file."
            ) from exc
        with h5py.File(str(path), "r") as f:
            ver = int(f.attrs["format_version"])
            errorif(
                ver > FORMAT_VERSION,
                ValueError,
                f"{path} declares format_version {ver}, but this agnimhd "
                f"understands at most {FORMAT_VERSION}.",
            )
            fields = {k: f[k][()] for k in REQUIRED_ARRAYS + OPTIONAL_ARRAYS if k in f}
            return cls(
                n_rho=int(f.attrs["n_rho"]),
                n_theta=int(f.attrs["n_theta"]),
                n_zeta=int(f.attrs["n_zeta"]),
                NFP=int(f.attrs["NFP"]),
                Psi=float(f.attrs["Psi"]),
                a=float(f.attrs["a"]),
                validate=validate,
                **fields,
            )

    def __repr__(self):
        """Short summary."""
        return (
            f"EquilibriumData(n_rho={self.n_rho}, n_theta={self.n_theta}, "
            f"n_zeta={self.n_zeta}, NFP={self.NFP}, "
            f"Psi={float(np.asarray(self.Psi)):.6g}, "
            f"a={float(np.asarray(self.a)):.6g})"
        )
