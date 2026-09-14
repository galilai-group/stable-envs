#!/bin/bash
#SBATCH --job-name=se6-raster2d-swmnav-verify
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# SWMNAV-2 verification, in the order a failure is cheapest to find:
#   1. the arithmetic probe, corrected  -- is every expression bit-identical at all;
#   2. the differential fuzz            -- is the KERNEL bit-identical on awkward inputs;
#   3. the targeted GPU suites          -- on the registered envs, through render();
#   4. the frozen Step 4 replay         -- the four configs' digests on cuda;
#   5. the mandatory sweep, before and after, same card same job.
set -uo pipefail
# The SHA is stamped by the submitting worktree (see scratch/step6/raster2d/sync.sh), not read from
# whatever the tree last carried: gpu_after.json, gpu_profile_after.json and replay/swmnav2.json
# were measured before their commit existed and name its parent. swmnav2/PROVENANCE_NOTE.md holds
# the md5s that say what they actually measured.
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=raster2d; BEF=$SCR/se6/$WS-preB; AFT=$SCR/se6/$WS
RES=$AFT/examples/results/step6/$WS/swmnav2
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

PYTHONPATH=$AFT $P5 scratch/step6/raster2d/swmnav_exact_probe.py --n 4194304 \
    --out "$RES/exact_probe_fixed.json"

for S in 0 1 2 3; do
  PYTHONPATH=$AFT $P5 scratch/step6/raster2d/navdot_fuzz.py --cases 200 --seed $S
done

PYTHONPATH=$AFT $P6 -m pytest tests/gpu/test_navdot_fused.py -q -p no:randomly 2>&1 | tail -6
PYTHONPATH=$AFT $P6 -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
    tests/gpu/test_fast_paths.py -k "TwoRoom or TwoRoomTarget or Piecewise or PiecewiseZones" \
    -q -p no:randomly 2>&1 | tail -6

PYTHONPATH=$AFT $P5 examples/fidelity_step4.py --configs "$CFG" --device cuda \
    --states-dir "$AFT/examples/results/step4/states" --out "$RES/replay/swmnav2.json"

ST=$RES/sweep/states; mkdir -p "$ST"
STABLE_ENVS_GIT_COMMIT=$(sha "$BEF") PYTHONPATH=$BEF $P5 $BEF/examples/bench_step3.py \
    --device cuda --threads 0 --envs "$CFG" --reps 15 --max-cell-s 60 --optional-batches \
    --states-out "$ST" --out "$RES/sweep/gpu_before.json" || echo "SWEEP_BEFORE rc=$?"
STABLE_ENVS_GIT_COMMIT=$(sha "$AFT") PYTHONPATH=$AFT $P5 $AFT/examples/bench_step3.py \
    --device cuda --threads 0 --envs "$CFG" --reps 15 --max-cell-s 60 --optional-batches \
    --states-in "$ST" --out "$RES/sweep/gpu_after.json" || echo "SWEEP_AFTER rc=$?"

echo "SWMNAV_VERIFY_DONE"
