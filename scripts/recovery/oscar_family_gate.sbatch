#!/bin/bash
#SBATCH --job-name=se-family-gate
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/logs/family-%j.out
set -euo pipefail
base=/oscar/scratch/sboughan/se-semantic-recovery
stage=${1:?source stage required}
cohort=${2:?dmc or ogbench required}
case "$cohort" in
  dmc)
    py=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
    deps=/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages
    gate=scripts/recovery/dmc_first_family_gate.py
    ;;
  ogbench)
    py=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
    deps=/oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages
    gate=scripts/recovery/ogbench_maze_gate.py
    ;;
  *) exit 2 ;;
esac
deps="$deps:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$base/cache/torch" TRITON_CACHE_DIR="$base/cache/triton"
export XDG_CACHE_HOME="$base/cache" WARP_CACHE_PATH="$base/cache/warp"
nvidia-smi
export PYTHONPATH="$base/baseline:$base/swm-reference:$deps"
cd "$base/baseline"
"$py" "$base/family_renderer_boundary.py" --cohort "$cohort" --device cuda --output "$base/$stage-renderer-baseline-cuda.json"
export PYTHONPATH="$base/$stage:$base/swm-reference:$deps"
cd "$base/$stage"
"$py" "$gate" --device cuda --output "$base/$stage-gate-cuda.json"
"$py" "$base/family_renderer_boundary.py" --cohort "$cohort" --device cuda --output "$base/$stage-renderer-cuda.json"
"$py" - "$base" "$stage" <<'PY'
import json
import sys
from pathlib import Path
base, stage = Path(sys.argv[1]), sys.argv[2]
baseline = json.loads((base / f"{stage}-renderer-baseline-cuda.json").read_text())
candidate = json.loads((base / f"{stage}-renderer-cuda.json").read_text())
assert baseline == candidate, (baseline, candidate)
print("FAMILY_FROZEN_RENDERER_GATE_PASS", stage, len(baseline), "cells")
PY
if [[ "$cohort" == dmc ]]; then
  "$py" scripts/recovery/reacher_gate.py --device cuda --output "$base/$stage-reacher-regression-cuda.json"
fi
echo "FAMILY_GPU_GATE_PASS $stage"
