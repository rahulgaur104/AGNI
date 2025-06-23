#!/usr/bin/env python3


import jax
import jax.numpy as jnp
from jax.sharding import PositionalSharding, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

# 1. Setup devices
gpu = jax.devices('gpu')[0]  # Primary GPU
cpus = jax.devices('cpu')    # All CPUs
num_cpus = len(cpus)

# 2. Create shardings
cpu_sharding = PositionalSharding(cpus).reshape((num_cpus, 1))

# 3. Create matrices (D stays on GPU, Ix can be on CPU)
D = jax.random.normal(jax.random.PRNGKey(0), (5000, 5000), device=gpu)
Ix = jnp.eye(48)  # Will be automatically replicated

# 4. Hybrid Kronecker product function
def distributed_kronecker(D_gpu, I_cpu):
    # Transfer D to CPU in sharded fashion
    D_cpu_shards = jax.device_put(D_gpu, cpu_sharding)
    
    # Define computation for each CPU shard
    def kron_chunk(D_block, I):
        return jnp.kron(D_block, I)
    
    # Perform distributed computation
    return jax.pmap(kron_chunk)(D_cpu_shards, I_cpu)

# 5. Build the full matrix A
def build_A(n_total, data, D_gpu, I_cpu):
    # Initialize distributed A matrix (on CPUs)
    A = jnp.zeros((3*n_total, 3*n_total))
    A = jax.device_put(A, cpu_sharding)
    
    # Compute distributed Kronecker product
    kron_term = distributed_kronecker(D_gpu, I_cpu)
    
    # Define indices
    rho_idx = slice(0, n_total)
    
    # Distributed update function
    def update_fn(A_slice, kron_slice):
        coeff = data["g_rr"]/data["sqrt(g)"]
        update = coeff * (data["chi"]**2 * kron_slice + 1)
        return A_slice.at[rho_idx, rho_idx].add(update)
    
    # Apply update across all CPU shards
    A_updated = jax.pmap(update_fn)(A, kron_term)
    
    return A_updated

# Usage
n_total = 5000 * 48
data = {
    "g_rr": jnp.ones((n_total, n_total)),
    "sqrt(g)": 1.0,
    "chi": 1.0
}

A = build_A(n_total, data, D, Ix)
