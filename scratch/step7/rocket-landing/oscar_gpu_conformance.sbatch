#!/bin/bash
#SBATCH --job-name=se7-rocket-conf
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2109,gpu2104,gpu2264
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W6 -- the final-tip observation the reviewer asked for, and nothing else.
#
# M6 established the re-freeze on tree 726241e (job 6410105). The tip then gained the filled
# declaration, the flipped unit test and the RD-1 changes. So the one thing not yet OBSERVED at the
# tip is the assertion that the filled declaration actually holds on a card:
# test_visual_tier_matches_its_declared_fidelity failed in job 6410105 only because it was comparing
# against the UNMEASURED placeholder (7.709 <= None), and "it will pass once the numbers are in" is an
# inference. This job runs the GPU conformance subset for PFRocketLanding AND the six locomotion
# nvdiff cells at the final tip so that it is a measurement instead.
#
# Two failures are EXPECTED and already recorded, and neither is this workstream's to fix:
#   test_color_knobs_apply_the_requested_color[PFRocketLanding]  -- TinyRenderer's multiplicative
#     shading law puts the requested colour in no pixel of the OFFICIAL frame either; escalated to
#     Sami, tests/ frozen.
#   test_resolution_downsample_agrees_within_the_measured_bound[PFRocketLanding] -- DOWNSAMPLE_EXACT_FRAC
#     has no PFRocketLanding entry; measured 0.95614, a Step-8 freeze item, tests/ frozen.
# Anything else failing is a real result and must be read as one.
#
# EGL is required: the six cells' official reference is MuJoCo through EGL. gpu2109's mujoco.GLContext
# is broken and gpu2264's nvdiffrast CUDA context failed, so both are excluded with the four the
# coordinator excludes and gpu2104; a failed probe exits rather than reporting a subset as if it were
# the whole.
#
# SYNC RECIPE: git -C <worktree> rev-parse HEAD > <worktree>/.git_sha, then the rsync that excludes .git.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export NUMBA_CACHE_DIR=$SCR/.cache/numba TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=4
module load cuda/12.9.0 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
PY=$SCR/venvs/se7-rocket/bin/python
WS=$SCR/se7/rocket-landing
OUT=$WS/examples/results/step7/rocket-landing
mkdir -p "$OUT/tier" "$SCR/logs"
[ -f "$WS/.git_sha" ] || { echo "MISSING $WS/.git_sha (see SYNC RECIPE); refusing to measure a tree of unknown commit"; exit 3; }
NODE=$(hostname -s)
echo "NODE $NODE JOB $SLURM_JOB_ID GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) SHA $(cat "$WS/.git_sha") LOAD $(cat /proc/loadavg | cut -d' ' -f1-3)"

PYTHONPATH=$WS $PY -c "
import torch, nvdiffrast.torch as dr
torch.cuda.init(); torch.cuda.set_device(0)
torch.zeros(1, device='cuda:0')
dr.RasterizeCudaContext(device=torch.device('cuda:0'))
print('nvdiffrast ok', torch.cuda.get_device_name(0))
" 2>&1 | tail -20 | grep -q "nvdiffrast ok" || { echo "NVDIFFRAST CONTEXT FAILED on $NODE"; exit 2; }
$PY -c "import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); print('egl ok')" 2>&1 | tail -20 \
  | grep -q "egl ok" || { echo "EGL CONTEXT FAILED on $NODE: the six cells' official reference cannot be rendered here; RESUBMIT with --exclude=...,$NODE"; exit 4; }
echo "EGL_OK on $NODE"
set -x

PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py -q -rA -p no:cacheprovider \
  -k "MujocoHalfCheetah or MujocoHopper or MujocoWalker2d or MujocoAnt or MujocoHumanoid or MujocoSwimmer or PFRocketLanding" \
  > "$OUT/tier/conformance_final_tip.txt" 2>&1
set +x
echo "===== the declared-fidelity assertions, per cell ====="
grep -E "test_visual_tier_matches_its_declared_fidelity" "$OUT/tier/conformance_final_tip.txt" | head -20
echo "===== failures ====="
grep -E "^FAILED" "$OUT/tier/conformance_final_tip.txt" || echo "(none)"
echo "===== pytest line ====="
tail -3 "$OUT/tier/conformance_final_tip.txt"
echo "S7CONF done on $NODE at $(cat "$WS/.git_sha")"
