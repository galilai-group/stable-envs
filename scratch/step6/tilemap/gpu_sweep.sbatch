#!/bin/bash
#SBATCH --job-name=se6-tilemap-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# TILEMAP — the GPU claim, before and after, on the SAME CARD in ONE job.
#
# GPU was already healthy for this env (GPU_FAST, NO_REFERENCE, cores 35.5 on a 3090) so the job of
# this run is to prove it STAYED healthy, and in particular that the two frozen qualifiers still
# hold. That is the live risk in this patch and it is worth stating: the change removes ~200 of the
# 216 operator dispatches per render, which lifts B=1 far more than B=4096, and BATCH_SCALES is
# fps(4096) >= 20 * fps(1). A patch can therefore make every batch faster and still fail a bar.
#
# Everything here is measured on one 3090 in one allocation; a before on a 3090 and an after on an
# L40S would not be a before and an after. The card is recorded in every artifact.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export WARP_CACHE_PATH=$SCR/.cache/warp TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/tilemap
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/tilemap
mkdir -p "$OUT/profile_before" "$OUT/profile_after" "$OUT/replay" "$OUT/sweep" "$SCR/logs"
cd "$WS"
echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "######## 1. operator-level profile note at every mandatory batch, interleaved"
for LEG in before after; do
  [ "$LEG" = before ] && TREE=$BASE DIR=profile_before || TREE=$WS DIR=profile_after
  PYTHONPATH=$TREE $PY scratch/step6/profile_note.py --env CraftaxClassic --device cuda \
    --batch 1 16 64 256 1024 4096 --repeat 20 --warmup 5 --workstream step6/tilemap \
    --states-dir "$WS/examples/results/step4/states" --out "$OUT/$DIR" || echo "NOTE $LEG rc=$?"
done

echo "######## 2. phase attribution on the GPU, interleaved"
for LEG in before after; do
  [ "$LEG" = before ] && TREE=$BASE DIR=profile_before || TREE=$WS DIR=profile_after
  PYTHONPATH=$TREE $PY scratch/step6/tilemap/attribute.py --device cuda --threads 4 \
    --batch 1 64 256 4096 --repeat 20 --warmup 5 --label "gpu-$LEG" \
    --states "$WS/examples/results/step4/states/CraftaxClassic.npz" \
    --out "$OUT/$DIR" || echo "ATTR $LEG rc=$?"
done

echo "######## 3. the frozen GPU sweep at 1/16/64/256/1024/4096, both legs, same card"
for LEG in before after; do
  [ "$LEG" = before ] && TREE=$BASE || TREE=$WS
  PYTHONPATH=$TREE $PY "$TREE/examples/bench_step3.py" --device cuda \
    --envs CraftaxClassic --out "$OUT/sweep/gpu_${LEG}.json" || echo "BENCH $LEG rc=$?"
done
$PY scratch/step6/tilemap/tier.py --device cuda --env CraftaxClassic \
  --before "$OUT/sweep/gpu_before.json" --after "$OUT/sweep/gpu_after.json" \
  --out "$OUT/sweep/tier_gpu.json" || echo "TIER rc=$?"

echo "######## 4. frozen Step-4 replay on CUDA — the oracle's own device"
PYTHONPATH=$BASE $PY "$BASE/examples/fidelity_step4.py" --configs CraftaxClassic --device cuda \
  --out "$OUT/replay/cuda_before.json" || echo "REPLAY before rc=$?"
PYTHONPATH=$WS $PY "$WS/examples/fidelity_step4.py" --configs CraftaxClassic --device cuda \
  --out "$OUT/replay/cuda_after.json" || echo "REPLAY after rc=$?"
$PY scratch/step6/tilemap/replay_check.py --replay "$OUT/replay/cuda_after.json" \
  --out "$OUT/replay/cuda_vs_frozen.json"
echo "CUDA vs frozen Step-4 oracle rc=$?"
$PY scratch/step6/tilemap/replay_check.py --replay "$OUT/replay/cuda_before.json" \
  --out "$OUT/replay/cuda_before_vs_frozen.json"
echo "CUDA baseline vs frozen Step-4 oracle rc=$?  (a non-zero here indicts the NODE, not the patch)"

echo "######## 5. the targeted GPU tests, selected to this env"
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
  tests/gpu/test_fast_paths.py -q -k CraftaxClassic 2>&1 | tail -20

echo "TILEMAP_GPU_DONE"
