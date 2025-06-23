#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from create_fourier_diffmatrix import *
from create_chebyshev_diffmatrix import *

"""
Build the full operator matrix for the spectral problem 
using tensor products.
+-----------+-----------+----------+
|           |           |          |
| A_ρρ      | A_ρθ      | Aρζ      |
|           |           |          |
+-----------+-----------+----------+
|           |           |          |
| A_θρ      | A_θθ      | A_θζ     |
|           |           |          |
+-----------+-----------+----------+
|           |           |          |
| A_ζρ      | A_ζθ      | A_ζζ     |
|           |           |          |
+-----------+-----------+----------+
Parameters:
- D_rho: 1D differentiation matrix for rho
- D_theta: 1D differentiation matrix for theta
- D_zeta: 1D differentiation matrix for zeta
- E: Equilibrium quantities at collocation points
"""
def derivative_1d(f_vals0, D, coord="r")
    if coord = "r":
        f_flat = jnp.reshape(f_vals0, (nx, ny * nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nx, ny, nz))

    elif coord = "t":
        f_vals = jnp.transpose(f_vals0, (1, 0, 2))
        f_flat = jnp.reshape(f_vals, (ny, nx * nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (ny, nx, nz))
        df_grid = jnp.transpose(df_grid, (1, 0, 2))

    else coord = "z":
        f_vals = jnp.transpose(f_vals0, (2, 0, 1))
        f_flat = jnp.reshape(f_vals, (nz, nx * ny))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nz, nx, ny))
        df_grid = jnp.transpose(df_grid, (1, 2, 0))

    return df_grid

# Get dimensions from input matrices
N_rho = D_rho.shape[0]
N_theta = D_theta.shape[0]
N_zeta = D_zeta.shape[0]
N_total = N_rho * N_theta * N_zeta

rho_pts = chebpts_lobatto(N_rho)

# For finite-difference, we use an explicity form of the
# derivative matrix
if rho_basis == "FD": 
    D_rho0          = D_rho0
    D_rho_D_rho     = D2_rho0
else: 
    # Differentiation matrix size (N_rho x N_rho)
    D_rho0 = diffmat_lobatto(N_rho)
    D_rho_D_rho     = D_rho0 @ D_rho0


# Generate points on [0, 2π] 
theta_pts = fourier_pts(N_theta)
zeta_pts = fourier_pts(N_zeta)

# Get differentiation matrices
D_theta0 = fourier_diffmat(N_theta)

# Get differentiation matrices
D_zeta0 = fourier_diffmat(N_zeta)

I_rho0 = jnp.eye(N_rho)
I_theta0 = jnp.eye(N_theta)
I_zeta0 = jnp.eye(N_zeta)

D_rho = jnp.kron(I_zeta0, jnp.kron(I_theta0, D_rho0))
D_theta = jnp.kron(I_zeta0, jnp.kron(D_theta0, I_rho0))
D_zeta = jnp.kron(D_zeta0, jnp.kron(I_theta0, I_rho0))

I = jnp.kron(I_zeta0, jnp.kron(I_theta0, I_rho0))

D_theta_D_theta = D_theta0 @ D_theta0
D_zeta_D_zeta = D_zeta0 @ D_zeta0

D_rho_D_theta = D_rho0 @ D_theta0
D_rho_D_zeta  = D_rho0 @ D_zeta0
D_theta_D_zeta = D_theta0 @ D_zeta0

# Create the full matrix
A = jnp.zeros((3*n_total, 3*n_total))

# Define field component indices
rho_idx = slice(0, n_total)
theta_idx = slice(n_total, 2*n_total)
zeta_idx = slice(2*n_total, 3*n_total)

all_idx = slice(0, 3*n_total)

# Need to create a PyTree-compatible dictionary from which the equilibrium data can be accessed
# Importing equilibrium data
sqrt_g = data["sqrt(g)"][:, None]
g_rr = data["g_rr"][:, None]
g_tt = data["g_tt"][:, None]
g_zz = data["g_zz"][:, None]
g_rt = data["g_rt"][:, None]    
g_rz = data["g_rz"][:, None]
g_tz = data["g_tz"][:, None]

if metric_derivatives:
    g_rr_t = data["g_rr_t"][:, None]
    g_rr_z = data["g_rr_z"][:, None]
    
    g_tt_r = data["g_tt_r"][:, None]
    g_tt_z = data["g_tt_z"][:, None]
    
    g_zz_r = data["g_zz_r"][:, None]
    g_zz_t = data["g_zz_t"][:, None]
    
    g_rt_r = data["g_rt_r"][:, None]    
    g_rz_r = data["g_rz_r"][:, None]
    g_tz_r = data["g_tz_r"][:, None]
    
    g_rt_t = data["g_rt_t"][:, None]    
    g_rz_t = data["g_rz_t"][:, None]
    g_tz_t = data["g_tz_t"][:, None]
    
    g_rt_z = data["g_rt_z"][:, None]    
    g_rz_z = data["g_rz_z"][:, None]
    g_tz_z = data["g_tz_z"][:, None]
    
    sqrt_g_r = data["sqrt(g)_r"][:, None]
    sqrt_g_t = data["sqrt(g)_t"][:, None]
    sqrt_g_z = data["sqrt(g)_z"][:, None]
else:
    g_rr_t = derivative_1d(data["g_rr"], D_theta0, coord="t")[:, None]
    g_rr_z = derivative_1d(data["g_rr"], D_zeta0, coord="z")[:, None]

    g_tt_r = derivative_1d(data["g_tt"], D_rho0, coord="r")[:, None]
    g_tt_z = derivative_1d(data["g_tt"], D_zeta0, coord="z")[:, None]

    g_zz_r = derivative_1d(data["g_zz"], D_rho0, coord="r")[:, None]
    g_zz_t = derivative_1d(data["g_zz"], D_theta0, coord="t")[:, None]

    g_rt_r = derivative_1d(data["g_rt"], D_rho0, coord="r")[:, None]   
    g_rz_r = derivative_1d(data["g_rz"], D_rho0, coord="r")[:, None]
    g_tz_r = derivative_1d(data["g_tz"], D_rho0, coord="r")[:, None]

    g_rt_t = derivative_1d(data["g_rt"], D_theta0, coord="t")[:, None]   
    g_rz_t = derivative_1d(data["g_rz"], D_theta0, coord="t")[:, None]
    g_tz_t = derivative_1d(data["g_tz"], D_theta0, coord="t")[:, None]

    g_rt_z = derivative_1d(data["g_rt"], D_zeta0, coord="z")[:, None]   
    g_rz_z = derivative_1d(data["g_rz"], D_zeta0, coord="z")[:, None]
    g_tz_z = derivative_1d(data["g_tz"], D_zeta0, coord="z")[:, None]

    sqrt_g_r = derivative_1d(data["sqrt(g)"], D_rho0, coord="r")[:, None]
    sqrt_g_t = derivative_1d(data["sqrt(g)"], D_theta0, coord="t")[:, None]
    sqrt_g_z = derivative_1d(data["sqrt(g)"], D_zeta0, coord="z")[:, None]

j_sup_theta = data["j_sup_theta"][:, None]
j_sup_zeta = data["j_sup_zeta"][:, None]


#Q^ρ
Q_sup_rho = Q_sup_rho.at[rho_idx, rho_idx].set(1/sqrt_g * (1/iota * D_theta + D_zeta))

#Q^θ
Q_sup_theta = Q_sup_theta.at[rho_idx, rho_idx].set(1/sqrt_g * (-1/iota * D_rho + iota_r/iota**2))
Q_sup_theta = Q_sup_theta.at[rho_idx, theta_idx].set(1/sqrt_g * D_zeta)
Q_sup_theta = Q_sup_theta.at[rho_idx, zeta_idx].set(-1/sqrt_g * D_zeta)

#Q^ζ
Q_sup_zeta = Q_sup_zeta.at[rho_idx, rho_idx].set(-1/sqrt_g * D_rho)
Q_sup_zeta = Q_sup_zeta.at[rho_idx, theta_idx].set(-1/sqrt_g * D_theta)
Q_sup_zeta = Q_sup_zeta.at[rho_idx, zeta_idx].set(1/sqrt_g * D_theta)


# rho block 
partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, rho_idx].set(g_rt / sqrt_g * (D_rho_D_theta -  iota_r/iota**2 * D_theta + D_rho_D_zeta) - g_tt / sqrt_g * (D_rho_D_rho - iota_r/iota**2) - g_tz / sqrt_g  * D_rho_D_rho + (g_tr_r/sqrt_g - g_tr * sqrt_g_r/sqrt_g ** 2) *  (D_theta/iota + D_zeta) + (g_tt_r/sqrt_g - g_tt * sqrt_g_r/sqrt_g ** 2) *  (iota_r/iota**2 - D_rho/iota) - (g_tz_r/sqrt_g - g_tz * sqrt_g_r/sqrt_g ** 2) * D_rho)

# theta block
partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, theta_idx].set(1/sqrt_g * (D_rho_D_zeta - D_rho_D_theta + (g_tt_r/sqrt_g - g_tt * sqrt_g_r/sqrt_g**2) * D_zeta - (g_tz_r/sqrt_g - g_tz * sqrt_g_r/sqrt_g) * D_theta))

# zeta block
partual_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, theta_idx].set(-1/sqrt_g * (D_rho_D_zeta - D_rho_D_theta + (g_tt_r/sqrt_g - g_tt * sqrt_g_r/sqrt_g**2) * D_zeta - (g_tz_r/sqrt_g - g_tz * sqrt_g_r/sqrt_g) * D_theta))


# rho bloc
partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, rho_idx].set(g_rr / sqrt_g * (D_theta_D_theta -  0*iota_r/iota**2 * D_theta + D_theta_D_zeta) - g_rt / sqrt_g * (D_rho_D_theta - 0*iota_r/iota**2) - g_rz / sqrt_g  * D_rho_D_theta + (g_rr_t/sqrt_g - g_rr * sqrt_g_t/sqrt_g ** 2) *  (D_theta/iota + D_zeta) + (g_rt_t/sqrt_g - g_rt * sqrt_g_t/sqrt_g ** 2) *  (0*iota_r/iota**2 - D_rho/iota) - (g_rz_t/sqrt_g - g_rz * sqrt_g_t/sqrt_g ** 2) * D_rho)

# theta block
partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, theta_idx].set(1/sqrt_g * (D_theta_D_zeta - D_theta_D_theta + (g_rt_t/sqrt_g - g_rt * sqrt_g_t/sqrt_g**2) * D_zeta - (g_rz_t/sqrt_g - g_rz * sqrt_g_t/sqrt_g) * D_theta))

# zeta block
partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, zeta_idx].set(-1/sqrt_g * (D_theta_D_zeta - D_theta_D_theta + (g_rt_t/sqrt_g - g_rt * sqrt_g_t/sqrt_g**2) * D_zeta - (g_rz_t/sqrt_g - g_rz * sqrt_g_t/sqrt_g) * D_theta))


# rho block
partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, rho_idx].set(g_rt / sqrt_g * (D_theta_D_zeta -  0*iota_r/iota**2 * D_theta + D_zeta_D_zeta) - g_tt / sqrt_g * (D_rho_D_zeta - 0*iota_r/iota**2) - g_tz / sqrt_g  * D_rho_D_zeta + (g_rt_z/sqrt_g - g_rt * sqrt_g_z/sqrt_g ** 2) *  (D_theta/iota + D_zeta) + (g_tt_z/sqrt_g - g_tt * sqrt_g_z/sqrt_g ** 2) *  (0*iota_r/iota**2 - D_rho/iota) - (g_tz_z/sqrt_g - g_tz * sqrt_g_z/sqrt_g ** 2) * D_rho)

# theta block
partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, theta_idx].set(1/sqrt_g * (D_zeta_D_zeta - D_zeta_D_theta + (g_tt_z/sqrt_g - g_tt * sqrt_g_z/sqrt_g**2) * D_zeta - (g_tz_z/sqrt_g - g_tz * sqrt_g_z/sqrt_g) * D_theta))

# zeta block
partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, zeta_idx].set(-1/sqrt_g * (D_zeta_D_zeta - D_zeta_D_theta + (g_tt_z/sqrt_g - g_tt * sqrt_g_z/sqrt_g**2) * D_zeta - (g_tz_z/sqrt_g - g_tz * sqrt_g_z/sqrt_g) * D_theta))


# rhobloc
partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, rho_idx].set(g_rt / sqrt_g * (D_theta_D_zeta -  0*iota_r/iota**2 * D_theta + D_zeta_D_zeta) - g_tt / sqrt_g * (D_rho_D_zeta - 0*iota_r/iota**2) - g_tz / sqrt_g  * D_rho_D_zeta + (g_rt_z/sqrt_g - g_rt * sqrt_g_z/sqrt_g ** 2) *  (D_theta/iota + D_zeta) + (g_tt_z/sqrt_g - g_tt * sqrt_g_z/sqrt_g ** 2) *  (0*iota_r/iota**2 - D_rho/iota) - (g_tz_z/sqrt_g - g_tz * sqrt_g_z/sqrt_g ** 2) * D_rho)

# theta block
partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, theta_idx].set(1/sqrt_g * (D_theta_D_zeta - D_theta_D_theta + (g_tz_t/sqrt_g - g_tz * sqrt_g_t/sqrt_g**2) * D_zeta - (g_zz_t/sqrt_g - g_zz * sqrt_g_t/sqrt_g) * D_theta))

# zeta block
partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, zeta_idx].set(-1/sqrt_g * (D_theta_D_zeta - D_theta_D_theta + (g_tz_t/sqrt_g - g_tz * sqrt_g_t/sqrt_g**2) * D_zeta - (g_zz_t/sqrt_g - g_zz * sqrt_g_t/sqrt_g) * D_theta))


# rho block
partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, rho_idx].set(g_rr / sqrt_g * (D_theta_D_zeta -  0*iota_r/iota**2 * D_theta + D_zeta_D_zeta) - g_rt / sqrt_g * (D_rho_D_zeta - 0*iota_r/iota**2) - g_rz / sqrt_g  * D_rho_D_zeta + (g_rr_z/sqrt_g - g_rr * sqrt_g_z/sqrt_g ** 2) *  (D_theta/iota + D_zeta) + (g_rt_z/sqrt_g - g_rt * sqrt_g_z/sqrt_g ** 2) *  (0*iota_r/iota**2 - D_rho/iota) - (g_rz_z/sqrt_g - g_rz * sqrt_g_z/sqrt_g ** 2) * D_rho)

# theta block
partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, theta_idx].set(1/sqrt_g * (D_zeta_D_zeta - D_theta_D_zeta + (g_rt_z/sqrt_g - g_rt * sqrt_g_z/sqrt_g**2) * D_zeta - (g_rz_z/sqrt_g - g_rz * sqrt_g_z/sqrt_g) * D_theta))

# zeta block
partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, zeta_idx].set(-1/sqrt_g * (D_zeta_D_zeta - D_theta_D_zeta + (g_rt_z/sqrt_g - g_rt * sqrt_g_z/sqrt_g**2) * D_zeta - (g_rz_z/sqrt_g - g_rz * sqrt_g_z/sqrt_g) * D_theta))


# rho block
partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, rho_idx].set(g_rz / sqrt_g * (D_rho_D_zeta -  iota_r/iota**2 * D_theta + D_rho_D_zeta) - g_tz / sqrt_g * (D_rho_D_rho - iota_r/iota**2) - g_zz / sqrt_g  * D_rho_D_rho + (g_rz_r/sqrt_g - g_rz * sqrt_g_r/sqrt_g ** 2) *  (D_theta/iota + D_zeta) + (g_tz_r/sqrt_g - g_tz * sqrt_g_r/sqrt_g ** 2) *  (iota_r/iota**2 - D_rho/iota) - (g_zz_r/sqrt_g - g_zz * sqrt_g_r/sqrt_g ** 2) * D_rho)

# theta block
partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, theta_idx].set(1/sqrt_g * (D_rho_D_zeta - D_rho_D_theta + (g_tz_r/sqrt_g - g_tz * sqrt_g_r/sqrt_g**2) * D_zeta - (g_zz_r/sqrt_g - g_zz * sqrt_g_r/sqrt_g) * D_theta))

# zeta block
partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, zeta_idx].set(-1/sqrt_g * (D_rho_D_zeta - D_rho_D_theta + (g_tz_r/sqrt_g - g_tz * sqrt_g_r/sqrt_g**2) * D_zeta - (g_zz_r/sqrt_g - g_zz * sqrt_g_r/sqrt_g) * D_theta))


A = A.at[rho_idx, all_idx].set(chi_r/sqrt_g * (partial_zeta_Q_rho - partial_rho_Q_zeta) - psi_r/sqrt_g * (partial_rho_Q_theta - partial_theta_Q_rho) + sqrt_g * (j_sup_theta * Q_sup_zeta - j_sup_zeta * Q_sup_theta))
A = A.at[theta_idx, all_idx].set(-psi_r/sqrt_g * (partial_theta_Q_zeta - partial_zeta_Q_theta) + sqrt_g * j_sup_zeta * Q_sup_rho)
A = A.at[theta_idx, all_idx].set( chi_r/sqrt_g * (partial_theta_Q_zeta - partial_zeta_Q_theta) + sqrt_g * j_sup_theta * Q_sup_rho)

# pressure-driven instability term
A = A.at[rho_idx, rho_idx].set(((p_rr/psi_r - p_r*psi_rr/psi_r**2) * I_rho + p_r/psi_r * D_rho))
A = A.at[theta_idx, theta_idx].set(p_r/psi_r * D_theta)
A = A.at[theta_idx, theta_idx].set(p_r/psi_r * D_zeta)

v, w = jax.lax.linalg.eig(A)
