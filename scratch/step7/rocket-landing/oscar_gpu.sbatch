#!/bin/bash
#SBATCH --job-name=se7-rocket-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W6, ONE GPU job (per-user cap gpu=2 shared with W1/W3/W4), in order:
#   1. M2 on the card: the nvdiff unit tests, then the byte-identity replay of nvdiff's existing users
#      (the six locomotion visual cells) under PYTHONPATH=$SCR/se7/baseline (76dc511) and
#      PYTHONPATH=$SCR/se7/rocket-landing (this branch), same job, same card, digests compared;
#   2. M3: PFRocketLanding's visual fidelity against PyFlyt+TinyRenderer (4 draws x 64 states, 480 px),
#      the frozen-corpus replay, the mandatory checks, and M4's throughput at batches 1/16/64/256.
# nvdiffrast compiles its CUDA plugin on first import: nvcc from cuda/12.9.0, cache under $SCR.
# PyFlyt's numba JIT (~57 s per process) is paid inside the reference bridge and is never inside a
# render timing.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export NUMBA_CACHE_DIR=$SCR/.cache/numba TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=4
module load cuda/12.9.0 2>/dev/null || true
PY=$SCR/venvs/se7-rocket/bin/python
WS=$SCR/se7/rocket-landing
BASE=$SCR/se7/baseline
OUT=$WS/examples/results/step7/rocket-landing
mkdir -p "$OUT/tier" "$OUT/nvdiff_identity" "$SCR/logs"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
which nvcc; nvcc --version | tail -1
set -x
cd "$WS"

# 1a. nvdiff import + plugin build, then the CPU-side unit tests on this node
PYTHONPATH=$WS $PY -c "import nvdiffrast.torch as dr, torch; dr.RasterizeCudaContext(device='cuda'); print('nvdiffrast ok', torch.cuda.get_device_name(0))" \
  2>&1 | tail -3 | tee "$OUT/nvdiff_identity/plugin_build.txt"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_nvdiff_triangles.py tests/unit/test_nvdiff_texgen.py tests/unit/test_rocket_landing.py \
  -q -p no:cacheprovider 2>&1 | tail -5 | tee "$OUT/tier/unit_tests_gpu_node.txt"

# 1b. byte identity of nvdiff's existing users, baseline then branch, then compare
PYTHONPATH=$BASE $PY "$WS/scratch/step7/rocket-landing/nvdiff_identity.py" --phase baseline --out "$OUT/nvdiff_identity" \
  2>&1 | tail -12
PYTHONPATH=$WS $PY "$WS/scratch/step7/rocket-landing/nvdiff_identity.py" --phase branch --out "$OUT/nvdiff_identity" \
  2>&1 | tail -12
PYTHONPATH=$WS $PY "$WS/scratch/step7/rocket-landing/nvdiff_identity.py" --phase compare --out "$OUT/nvdiff_identity" \
  2>&1 | tail -20
rm -f "$OUT"/nvdiff_identity/baseline_*.npz "$OUT"/nvdiff_identity/branch_*.npz     # digests + compare.json stay

# 1c. the locomotion visual cells' own declared-fidelity test on the branch (asserts their thresholds)
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py -q -p no:cacheprovider \
  -k "visual_tier and Mujoco" 2>&1 | tail -8 | tee "$OUT/nvdiff_identity/locomotion_visual_tier_tests.txt"

# 2. PFRocketLanding: corpus replay, mandatory checks, throughput, visual fidelity vs PyFlyt
PYTHONPATH=$WS $PY "$WS/scratch/step7/rocket-landing/measure_tier.py" --draws 4 --n 64 2>&1 | tail -40

# 3. the conformance suite restricted to this cell (Gate A on the card; reference needs PyFlyt)
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py -q -p no:cacheprovider \
  -k "PFRocketLanding" 2>&1 | tail -25 | tee "$OUT/tier/conformance_PFRocketLanding.txt"
echo "S7GPU rc=$?"
