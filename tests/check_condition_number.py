#!/usr/bin/env python3
"""A set of functions that test fourier differentiation."""
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import grad, jit, vmap

# Import the necessary functions from your module
from create_fourier_diffmatrix import fourier_diffmat, fourier_pts

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

n = 40
# Get differentiation matrix
D1 = fourier_diffmat(n)

print(jnp.linalg.cond(D1))

