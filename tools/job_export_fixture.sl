#!/bin/bash
#SBATCH -J agni_fixture
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --account=m4505
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH -o /pscratch/sd/r/rgaur/AGNI/tools/export_fixture_%j.log

# MILESTONE 0. Export the shipped equilibrium to the agnimhd EquilibriumData
# format, in an environment that still has DESC. After this runs the fixture is
# committed and every remaining milestone is done with DESC absent.
#
# Cost: one PEST->DESC coordinate map at tol=1e-12 over 2304 nodes, one dense
# assembly at 24x12x8 (n_keep = 6720, so the matrix is 6720^2 = 361 MB), and one
# shift-invert ARPACK eigsh. Minutes, not hours -- the wall is generous because
# map_coordinates at tol=1e-12 is the unpredictable part.
#
# The MEASURED dense eigenvalue is printed as `[export] MEASURED dense
# finite-n lambda3 = ...` and written into the .json sidecar. The value the DESC
# test suite records at this resolution is -1.337622e-04; the test that pins it
# lives in tests/test_eigenvalue.py and reads the sidecar, so it asserts against
# what was actually measured here rather than a number typed from a document.

set -euo pipefail

module load conda
conda activate desc-env2
unset LD_LIBRARY_PATH

export JAX_ENABLE_X64=1
export DESC_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export PYTHONDONTWRITEBYTECODE=1

REPO=/pscratch/sd/r/rgaur/AGNI
EQ=/pscratch/sd/r/rgaur/DESC2/DESC/tests/inputs/AGNI_QH_lowres.h5

cd "$REPO"
python -u tools/export_fixture.py \
    --eq "$EQ" \
    --res "${AGNI_RES:-24,12,8}" \
    --out "tests/data/qh_lowres_${AGNI_RES_TAG:-24x12x8}.npz"

echo "[job] done"
