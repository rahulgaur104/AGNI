"""Stellarator symmetry for the DESC adapters: evaluate half the nodes, mirror the rest.

A stellarator-symmetric equilibrium is invariant under the rotation by pi about the
R axis at phi = 0, ``S: (x, y, z) -> (x, -y, -z)``, which in DESC's coordinates is
``(rho, theta, zeta) -> (rho, -theta, -zeta)``. AGNI's angular nodes are uniform from
0, so the reflection permutes them: node ``(j, k)`` goes to ``(-j mod n_theta, -k mod
n_zeta)``; over one field period a mirrored zeta also includes a rotation by
``2 pi / NFP``, which leaves scalars and cylindrical vector components unchanged.

So every coordinate map and geometry evaluation needs only one node of each mirror
pair; the other follows with the quantity's parity. Under ``S``, with
``M = diag(1, -1, -1)`` its matrix in (R, phi, Z) components:

- ``e_rho -> M e_rho``; ``e_theta, e_phi -> -M e_theta, -M e_phi``, so a metric
  component ``g_ab`` is odd exactly when one index is rho;
- ``sqrt(g)`` and its rho derivative are even, its theta and phi derivatives odd;
- ``grad(rho) -> M grad(rho)``; ``B, J -> -M B, -M J`` (the field and the current
  reverse their sense under the reflection);
- ``R`` even, ``Z`` odd, ``omega = phi - zeta`` odd.

Everything here is a gather and a sign over static NumPy index arrays: jit- and
grad-safe.
"""

import numpy as np

from ..backend import jnp

#: Parity under the reflection: +1 / -1 for a scalar, a sign per (R, phi, Z)
#: component for a vector in DESC's ``rpz`` basis. ``phi`` is handled separately
#: (``mirror_phi``): it is odd only up to the zeta of its node.
PARITY = {
    # PEST metric, its inverse rho-rho component and the Jacobian
    "g_rr|PEST": 1,
    "g_rv|PEST": -1,
    "g_rp|PEST": -1,
    "g_vv|PEST": 1,
    "g_vp|PEST": 1,
    "g_pp|PEST": 1,
    "g^rr": 1,
    "sqrt(g)_PEST": 1,
    "(sqrt(g)_PEST_r)|PEST": 1,
    "(sqrt(g)_PEST_v)|PEST": -1,
    "(sqrt(g)_PEST_p)|PEST": -1,
    # current and profiles
    "J^zeta": 1,
    "|J|": 1,
    "iota": 1,
    "psi_r": 1,
    "psi_rr": 1,
    "p": 1,
    "p_r": 1,
    "ni": 1,
    # (-M J) x (M grad rho) = -M (J x grad rho); (B.grad) grad rho -> -M (...)
    "J x grad(rho)": (-1, 1, 1),
    "(B*grad) grad(rho)": (-1, 1, 1),
    # boundary geometry of the vacuum response
    "R": 1,
    "Z": -1,
    "e_theta_PEST": (-1, 1, 1),
    "e_phi|r,v": (-1, 1, 1),
    "e_theta_PEST x e_phi|r,v": (1, -1, -1),
    "|e_theta_PEST x e_phi|r,v|": 1,
}


def half_and_mirror(n_theta, n_zeta, n_rho=1, order="rtz"):
    """``(half, src, flip)``: the nodes to evaluate and how to fill in the rest.

    Parameters
    ----------
    n_theta, n_zeta, n_rho : int
    order : {"rtz", "biest"}
        Node order: AGNI's rho-major ``(i * n_theta + j) * n_zeta + k``, or BIEST's
        shell order ``k * n_theta + j`` (zeta outer, theta fastest; ``n_rho = 1``).

    Returns
    -------
    half : ndarray of int
        One node of every mirror pair (fixed points included), ascending.
    src : ndarray of int, shape (n_nodes,)
        For every node, its position in ``half``: itself, or its mirror.
    flip : ndarray of bool, shape (n_nodes,)
        True where the value comes from the mirror and the parity applies.
    """
    j = np.arange(n_theta)
    k = np.arange(n_zeta)
    jm, km = (-j) % n_theta, (-k) % n_zeta
    if order == "rtz":
        i = np.arange(n_rho)
        idx = ((i[:, None, None] * n_theta + j[None, :, None]) * n_zeta + k).ravel()
        mir = ((i[:, None, None] * n_theta + jm[None, :, None]) * n_zeta + km).ravel()
    elif order == "biest":
        idx = (k[:, None] * n_theta + j[None, :]).ravel()
        mir = (km[:, None] * n_theta + jm[None, :]).ravel()
    else:
        raise ValueError(f"order must be 'rtz' or 'biest', got {order!r}")
    keep = idx <= mir
    half = np.sort(idx[keep])
    pos = np.full(idx.size, -1)
    pos[half] = np.arange(half.size)
    flip = np.zeros(idx.size, dtype=bool)
    flip[idx] = ~keep
    rep = np.where(keep, idx, mir)  # the evaluated node of each pair
    src = np.empty(idx.size, dtype=int)
    src[idx] = pos[rep]
    return half, src, flip


def mirror_field(val_half, src, flip, sign):
    """The full array of a field evaluated on ``half``, with parity ``sign``."""
    v = jnp.asarray(val_half)[src]
    s = jnp.asarray(sign, dtype=v.dtype)
    f = jnp.asarray(flip).reshape((-1,) + (1,) * (v.ndim - 1))
    return jnp.where(f, s * v, v)


def mirror_phi(phi_half, zeta_half, zeta, src, flip):
    """``phi`` on every node: ``zeta - (phi_s - zeta_s)`` at a mirrored node.

    ``omega = phi - zeta`` is odd; ``phi`` itself is odd only up to the zeta of the
    target node (the field-period rotation, or the 2 pi of a full torus).
    """
    phi_s = jnp.asarray(phi_half)[src]
    omega = phi_s - jnp.asarray(zeta_half)[src]
    return jnp.where(jnp.asarray(flip), jnp.asarray(zeta) - omega, phi_s)


def mirror_rtz(rtz_half, src, flip, nodes):
    """DESC ``(rho, theta, zeta)`` of every node from those of ``half``.

    ``nodes`` are the full node set being mapped (PEST ``(rho, theta, zeta)``): rho
    and zeta are left alone by the PEST -> DESC map, so they come from there exactly;
    theta is ``-theta_s mod 2 pi`` at a mirrored node.
    """
    rtz_half = jnp.asarray(rtz_half)
    nodes = jnp.asarray(nodes)
    theta = rtz_half[src, 1]
    theta = jnp.where(jnp.asarray(flip), jnp.mod(-theta, 2 * jnp.pi), theta)
    return jnp.stack([nodes[:, 0], theta, nodes[:, 2]], axis=-1)
