#!/usr/bin/env python3
"""Radial bases, compared the way the paper compares them.

Reproduces the convergence study behind the choice of Legendre-Lobatto as
AGNI's default radial basis (paper Fig. 7): differentiate the analytic test
function

    f(rho, theta, zeta) = rho^4 exp(-20 (rho - 0.4)^2)
                          (sin 3theta + sin 4theta) cos 5zeta

radially with each basis's differentiation matrix, and measure the sup-norm
error against the exact derivative as the radial resolution grows.

This is a *numerics* demonstration, not a physics one -- no equilibrium is
involved. It is here because choosing a radial basis is the first decision a
new user makes, and the answer is measurable.

Run::

    python examples/basis_comparison.py
"""

import numpy as np

from agnimhd.backend import jax, jnp
from agnimhd.basis import (
    bspline_diffmat,
    finite_difference_diffmat,
    jacobi_diffmat,
    legendre_diffmat,
)
from agnimhd.quadrature import (
    automorphism_staircase1,
    bspline_nodes_weights,
    gauss_radau_jacobi,
    leggauss_lob,
)

# The mapping used in the paper's figure. x_0 is where nodes are concentrated;
# put it where the function -- or, in a real run, the eigenfunction -- is sharp.
AUTO = dict(eps=0.0, x_0=0.4, m_1=2.0, m_2=3.0)


def f(rho):
    """The radial factor of the test function, and its exact derivative."""
    g = rho**4 * np.exp(-20.0 * (rho - 0.4) ** 2)
    dg = (4.0 * rho**3 - 40.0 * rho**4 * (rho - 0.4)) * np.exp(-20.0 * (rho - 0.4) ** 2)
    return g, dg


def _mapped(nodes_fn, diffmat_fn, n):
    """Nodes and D for a basis on [-1, 1], pushed through the automorphism."""
    x = np.asarray(nodes_fn(n))
    rho = np.asarray(automorphism_staircase1(x, **AUTO))
    dfa = np.asarray(
        jax.vmap(lambda t: jax.grad(automorphism_staircase1, argnums=0)(t, **AUTO))(
            jnp.asarray(x)
        )
    )
    D = np.asarray(diffmat_fn(n)) / dfa[:, None]
    return rho, D


def error_for(name, n):
    """Sup-norm error of d/drho for one basis at radial resolution ``n``."""
    if name == "legendre-lobatto":
        rho, D = _mapped(
            lambda m: leggauss_lob(m)[0], lambda m: legendre_diffmat(m)[0], n
        )
    elif name == "radau-jacobi":
        rho, D = _mapped(
            lambda m: gauss_radau_jacobi(m)[0], lambda m: jacobi_diffmat(m)[0], n
        )
    elif name == "b-spline":
        rho, D = _mapped(
            lambda m: bspline_nodes_weights(m)[0], lambda m: bspline_diffmat(m)[0], n
        )
    elif name == "finite-difference":
        # Uniform nodes on [0, 1]; no mapping, since the 4th-order SBP closure
        # is built for equal spacing.
        rho = np.linspace(0.0, 1.0, n)
        D = np.asarray(finite_difference_diffmat(n, rho[1] - rho[0])[0])
    else:  # pragma: no cover
        raise ValueError(name)

    g, dg = f(rho)
    return float(np.max(np.abs(D @ g - dg)))


def main():
    """Print the convergence table."""
    bases = ["legendre-lobatto", "radau-jacobi", "b-spline", "finite-difference"]
    sizes = [16, 24, 32, 48, 64, 96]

    print(f"{'n_rho':>6}" + "".join(f"{b:>20}" for b in bases))
    for n in sizes:
        row = f"{n:>6}"
        for b in bases:
            try:
                row += f"{error_for(b, n):>20.3e}"
            except Exception as err:  # a basis may not support every n
                row += f"{type(err).__name__:>20}"
        print(row)

    print()
    print("Legendre-Lobatto with the radial mapping converges fastest, which is")
    print("why it is the default. The mapping matters as much as the basis: it")
    print("moves collocation points toward x_0, where the mode is sharp.")
    print("A coupled Zernike-Fourier radial-poloidal basis is also available")
    print("(agnimhd.basis.zernike_fourier_diffmat); it is non-separable, gives a")
    print("DENSE operator with no sparsity to exploit, and its Jacobi radial")
    print("recurrence -- not the uniform variant -- is the trusted one.")


if __name__ == "__main__":
    main()
