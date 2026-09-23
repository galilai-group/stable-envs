#!/bin/bash
#SBATCH --job-name=se-fetch-representative
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|a5000|a5500
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/logs/fetch-%j.out
set -euo pipefail
base=/oscar/scratch/sboughan/se-semantic-recovery
stage=${1:?immutable stage required}
py=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$base/cache" WARP_CACHE_PATH="$base/cache/warp"
export TORCH_EXTENSIONS_DIR="$base/cache/torch" TRITON_CACHE_DIR="$base/cache/triton"
export MPLBACKEND=Agg MPLCONFIGDIR="$base/cache/matplotlib"
nvidia-smi
driver="$base/$stage/scripts/recovery/fetch_renderer_gate.py"
export PYTHONPATH="$base/baseline"
cd "$base/baseline"
"$py" "$driver" --baseline --output "$base/$stage-renderer-baseline.json"
export PYTHONPATH="$base/$stage"
cd "$base/$stage"
"$py" "$driver" --fixture "$base/$stage/docs/research/semantic_recovery_evidence/fetch_public/canonical_state.json" --output "$base/$stage-renderer-candidate.json" --gate-output "$base/$stage-renderer-gate.json"
"$py" - "$base" "$stage" <<'PY'
import json
import sys
from pathlib import Path
base,stage=Path(sys.argv[1]),sys.argv[2]
a=json.loads((base/f'{stage}-renderer-baseline.json').read_text())
b=json.loads((base/f'{stage}-renderer-candidate.json').read_text())
assert a==b,(a,b)
print('FETCH_FROZEN_RENDERER_GATE_PASS',stage,flush=True)
PY
echo "FETCH_GPU_GATE_PASS $stage"
