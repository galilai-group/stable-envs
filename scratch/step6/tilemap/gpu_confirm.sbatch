#!/bin/bash
#SBATCH --job-name=se6-tilemap-gpuconfirm
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The GPU sweep, the CUDA replay and the targeted GPU tests, re-run on the FINAL commit.
#
# The first GPU job (se6-tilemap-gpu, 6349394) finished in 58 s and three docstring-only commits
# landed in the same minute, so which text the after leg imported is not certain from the artifact.
# The pixels and the timings cannot depend on a docstring, but "cannot" is an argument and this is a
# measurement, so the certification numbers are re-taken on the commit that is actually offered.
# Same partition, same constraint, both legs in one allocation on one card.
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
cd "$WS"
echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
md5sum stable_envs/envs/craftax.py

for LEG in before after; do
  [ "$LEG" = before ] && TREE=$BASE || TREE=$WS
  PYTHONPATH=$TREE $PY "$TREE/examples/bench_step3.py" --device cuda \
    --envs CraftaxClassic --out "$OUT/sweep/gpu_${LEG}_final.json" || echo "BENCH $LEG rc=$?"
done
$PY scratch/step6/tilemap/tier.py --device cuda --env CraftaxClassic \
  --before "$OUT/sweep/gpu_before_final.json" --after "$OUT/sweep/gpu_after_final.json" \
  --out "$OUT/sweep/tier_gpu_final.json" | tail -5

PYTHONPATH=$WS $PY "$WS/examples/fidelity_step4.py" --configs CraftaxClassic --device cuda \
  --out "$OUT/replay/cuda_after_final.json"
$PY scratch/step6/tilemap/replay_check.py --replay "$OUT/replay/cuda_after_final.json" \
  --out "$OUT/replay/cuda_final_vs_frozen.json" | grep -E "verdict|status|digest"
echo "CUDA vs frozen rc=$?"

PYTHONPATH=$WS $PY -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
  tests/gpu/test_fast_paths.py -q -k CraftaxClassic 2>&1 | tail -6
echo "TILEMAP_GPU_CONFIRM_DONE"
