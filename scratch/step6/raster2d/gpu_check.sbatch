#!/bin/bash
#SBATCH --job-name=se6-raster2d-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6 raster2d: everything that needs a GPU, in ONE job on ONE card.
#
#   1. the fused kernel still agrees with the torch reference, and the torch reference still draws
#      the same bytes on cuda that it drew before (the differential fuzz, on the device);
#   2. the frozen Step 4 replay for the six configurations, on cuda, where the frozen digests were
#      taken -- digest_ours must come back identical;
#   3. the targeted GPU suite for these envs;
#   4. the GPU sweep, BEFORE and AFTER on the SAME card, so "no regression of the healthy GPU rows"
#      is a measurement rather than an argument. render_triton is untouched, so this is the check
#      that it is untouched.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export TRITON_CACHE_DIR=$SCR/.cache/triton WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=raster2d
BEF=$SCR/se6/$WS-preB
AFT=$SCR/se6/$WS
RES=$AFT/examples/results/step6/$WS
CFG=PushT,SimplePointMaze,TwoRoom,TwoRoomTarget,Piecewise,PiecewiseZones
P6=$SCR/venvs/se-step6/bin/python        # the only interpreter here with pytest
P5=$SCR/venvs/se-step5/bin/python        # the only one with stable_worldmodel + matplotlib
mkdir -p "$RES/replay" "$RES/sweep"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "node=$(hostname) job=$SLURM_JOB_ID"
set -x
cd "$AFT"

# 1. the torch reference on the device, old implementation vs new, byte for byte
PYTHONPATH=$AFT $P6 scratch/step6/raster2d/fuzz_render_torch.py --cases 400 --device cuda --size 24
PYTHONPATH=$AFT $P6 scratch/step6/raster2d/fuzz_render_torch.py --cases 250 --device cuda --seed 3 --size 17

# 2. frozen Step 4 replay, on cuda, into the Step 6 directory (never over examples/results/step4)
PYTHONPATH=$AFT $P5 examples/fidelity_step4.py --configs "$CFG" --device cuda \
    --states-dir "$AFT/examples/results/step4/states" \
    --out "$RES/replay/raster2d_swm.json"
PYTHONPATH=$AFT $SCR/venvs/se-step3/bin/python examples/fidelity_step4.py --configs PushT \
    --device cuda --states-dir "$AFT/examples/results/step4/states" \
    --out "$RES/replay/raster2d_pusht.json" || echo "PUSHT_REPLAY_STEP3VENV rc=$?"

# 3. the targeted GPU suite
PYTHONPATH=$AFT $P6 -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
    tests/gpu/test_fast_paths.py \
    -k "PushT or SimplePointMaze or TwoRoom or TwoRoomTarget or Piecewise or PiecewiseZones" \
    -q -p no:randomly 2>&1 | tail -40

# 4. the GPU sweep, before and after, same card
ST=$RES/sweep/states_gpu; mkdir -p "$ST"
PYTHONPATH=$BEF $P5 $BEF/examples/bench_step3.py --device cuda --threads 0 --envs "$CFG" \
    --reps 15 --max-cell-s 12 --optional-batches --states-out "$ST" \
    --out "$RES/sweep/gpu_before.json" || echo "GPU_SWEEP_BEFORE rc=$?"
PYTHONPATH=$AFT $P5 $AFT/examples/bench_step3.py --device cuda --threads 0 --envs "$CFG" \
    --reps 15 --max-cell-s 12 --optional-batches --states-in "$ST" \
    --out "$RES/sweep/gpu_after.json" || echo "GPU_SWEEP_AFTER rc=$?"

echo "GPU_CHECK_DONE"
