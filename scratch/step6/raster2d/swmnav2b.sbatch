#!/bin/bash
#SBATCH --job-name=se6-raster2d-swmnav2b
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# SWMNAV-2b verification and sweep, in the order a failure is cheapest to find. The SHA is stamped by
# the submitting worktree (scratch/step6/raster2d/sync.sh).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=raster2d; BEF=$SCR/se6/$WS-preB; AFT=$SCR/se6/$WS
RES=$AFT/examples/results/step6/$WS/swmnav2b
CFG=TwoRoom,TwoRoomTarget,Piecewise,PiecewiseZones
P5=$SCR/venvs/se-step5/bin/python; P6=$SCR/venvs/se-step6/bin/python
sha () { cat "$1/GIT_COMMIT" 2>/dev/null || echo unknown; }
mkdir -p "$RES/replay" "$RES/sweep"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "node=$(hostname) job=$SLURM_JOB_ID"
echo "before tree $BEF at $(sha "$BEF")"
echo "after  tree $AFT at $(sha "$AFT")"
set -x
cd $AFT
FAILED=0

for S in 0 1 2 3 4 5 6 7; do
  PYTHONPATH=$AFT $P5 scratch/step6/raster2d/navdot_fuzz.py --cases 200 --seed $S \
    || FAILED=$((FAILED+1))
done

PYTHONPATH=$AFT $P6 -m pytest tests/gpu/test_navdot_fused.py -q -p no:randomly 2>&1 | tail -6
PYTHONPATH=$AFT $P6 -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
    tests/gpu/test_fast_paths.py -k "TwoRoom or TwoRoomTarget or Piecewise or PiecewiseZones" \
    -q -p no:randomly 2>&1 | tail -6

PYTHONPATH=$AFT $P5 examples/fidelity_step4.py --configs "$CFG" --device cuda \
    --states-dir "$AFT/examples/results/step4/states" --out "$RES/replay/swmnav2b.json"

PYTHONPATH=$AFT $P5 scratch/step6/raster2d/swmnav_gpu_profile.py --device cuda \
    --batch 256 1024 4096 --out "$RES/gpu_profile_after.json"

ST=$RES/sweep/states; mkdir -p "$ST"
STABLE_ENVS_GIT_COMMIT=$(sha "$BEF") PYTHONPATH=$BEF $P5 $BEF/examples/bench_step3.py \
    --device cuda --threads 0 --envs "$CFG" --reps 15 --max-cell-s 60 --optional-batches \
    --states-out "$ST" --out "$RES/sweep/gpu_before.json" || echo "SWEEP_BEFORE rc=$?"
STABLE_ENVS_GIT_COMMIT=$(sha "$AFT") PYTHONPATH=$AFT $P5 $AFT/examples/bench_step3.py \
    --device cuda --threads 0 --envs "$CFG" --reps 15 --max-cell-s 60 --optional-batches \
    --states-in "$ST" --out "$RES/sweep/gpu_after.json" || echo "SWEEP_AFTER rc=$?"

echo "SWMNAV2B_DONE failed=$FAILED"
test "$FAILED" -eq 0
