#!/bin/bash
#SBATCH --job-name=se6-raycast3d-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6C ticket 1 -- DMCBallInCupCatch against DMCFingerSpin, ON THE SAME CARD, IN ONE JOB.
#
# BallInCup is the one raycast3d GPU row below tier (GPU_PARITY / BELOW_G4 / BATCH_SCALES_FAIL)
# while Finger and Reacher are far above it. A sibling comparison across two jobs on two cards is
# not a comparison, so both run here, back to back, and the card is recorded.
#
# Also: the ticket-1 correctness diagnosis for the two OGBench cells that fail Gate A on
# check_style_is_state. It was measured on cuda in Step 4, so it is reproduced on cuda.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

for E in DMCBallInCupCatch DMCFingerSpin; do
  $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda \
      --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 --cprofile 256 \
      --fk-counterfactual \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_phases_gpu" || echo "PHASES $E rc=$?"
done

# Host-attribution pass: same wrappers, NO cuda sync inside them, so a phase that is large here and
# small in the synchronised pass is host work rather than device work.
for E in DMCBallInCupCatch DMCFingerSpin; do
  $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cuda --no-sync-phases \
      --batch 256 4096 --repeat 10 --warmup 3 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_phases_gpu_hostonly" || echo "HOSTONLY $E rc=$?"
done

# Deliverable 1 on the device the frozen digests were produced on. The CPU census cannot construct
# the four trackcom cells (their renderer is MJWarp and needs CUDA), which is itself the answer, but
# an observation beats an inference so it is repeated here where they DO construct.
$PY scratch/step6/raycast3d/backend_census.py --device cuda --batch 2 \
    --out "$OUT/backend_census_cuda.json" || echo "CENSUS rc=$?"

$PY scratch/step6/raycast3d/style_bisect.py --device cuda --batch 64 \
    --configs AntmazeTeleportExplore HumanoidmazeMediumNavigate AntmazeTeleportNavigate \
              HumanoidmazeLargeNavigate PointmazeLargeNavigate \
    --out "$OUT/style_is_state_bisect.json" || echo "BISECT rc=$?"

echo "PHASES_GPU done"
ls -la "$OUT"/profile_phases_gpu*/ 2>/dev/null | tail -30
