#!/usr/bin/env python3

from desc.equilibrium import Equilibrium
from desc.grid import Grid
from jax import grad, vmap

from create_cheb_diffmatrix import cheb_D1, cheb_D2, chebpts_lobatto
from create_fourier_diffmatrix import fourier_diffmat, fourier_pts

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


r_cheb = chebpts_lobatto(nx)
t_four = fourier_pts(ny)
z_four = fourier_pts(nz)

x, scale_x1, scale_x2 = map_domain(x_cheb, option=1)

rho, theta, zeta = jnp.meshgrid(r_cheb, y_four, z_four, indexing="ij")

Grid = Grid(jnp.array([rho.flatten(), theta.flatten(), zeta.flatten()]).T)

data_keys = [
    "g_rr",
    "g_rt",
    "g_rz",
    "g_tt",
    "g_tz",
    "g_zz",
    "g^zr",
    "g^tr",
    "g^rr",
    "j^theta",
    "j_zeta",
    "sqrt(g)",
    "sqrt(g)_r",
    "sqrt(g)_t",
    "sqrt(g)_z",
    "psi_r",
    "psi_rr",
    "iota",
    "iota_rr",
]

data = eq.compute(data_keys, grid=Grid)

# Next we calculate the specific equilibrium coefficients
