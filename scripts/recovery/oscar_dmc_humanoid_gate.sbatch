#!/bin/bash
#SBATCH --job-name=se-dmc-humanoid
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/logs/humanoid-%j.out
set -euo pipefail
base=/oscar/scratch/sboughan/se-semantic-recovery
stage=${1:?source stage required}
driver="$base/${2:?immutable renderer driver filename required}"
py=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
deps=/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$base/cache/torch" TRITON_CACHE_DIR="$base/cache/triton"
export XDG_CACHE_HOME="$base/cache" WARP_CACHE_PATH="$base/cache/warp"
nvidia-smi
sha256sum "$base/$stage/scripts/recovery/dmc_humanoid_gate.py" "$driver" "$0"

export PYTHONPATH="$base/baseline:$base/swm-reference:$deps"
cd "$base/baseline"
"$py" "$driver" --cohort dmc_humanoid --device cuda --historical-baseline --output "$base/$stage-renderer-baseline-cuda.json"

export PYTHONPATH="$base/$stage:$base/swm-reference:$deps"
cd "$base/$stage"
"$py" scripts/recovery/dmc_humanoid_gate.py --device cuda --output "$base/$stage-gate-cuda.json"
"$py" "$driver" --cohort dmc_humanoid --device cuda --output "$base/$stage-renderer-cuda.json"
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
"$py" scripts/recovery/reacher_gate.py --device cuda --output "$base/$stage-reacher-regression-cuda.json"
"$py" scripts/recovery/dmc_first_family_gate.py --device cuda --output "$base/$stage-first-family-regression-cuda.json"
"$py" scripts/recovery/dmc_continuation_gate.py --device cuda --output "$base/$stage-continuation-regression-cuda.json"
"$py" scripts/recovery/dmc_aligned_gate.py --device cuda --output "$base/$stage-aligned-regression-cuda.json"
echo "FAMILY_GPU_GATE_PASS $stage"
