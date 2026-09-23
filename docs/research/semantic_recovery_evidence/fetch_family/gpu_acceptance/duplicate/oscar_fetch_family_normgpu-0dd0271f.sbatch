#!/bin/bash
#SBATCH --job-name=se-fetch-family
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|a5000|a5500
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/logs/fetch-family-%j.out
set -euo pipefail
base=/oscar/scratch/sboughan/se-semantic-recovery
stage=${1:?immutable stage required}
results="$base/fetch-normgpu-${SLURM_JOB_ID}"
mkdir -p "$results"
py=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$base/cache" WARP_CACHE_PATH="$base/cache/warp"
export TORCH_EXTENSIONS_DIR="$base/cache/torch" TRITON_CACHE_DIR="$base/cache/triton"
export MPLBACKEND=Agg MPLCONFIGDIR="$base/cache/matplotlib"
nvidia-smi
driver="$base/$stage/scripts/recovery/fetch_renderer_gate.py"
for task in PickAndPlace Reach Push Slide; do
 export PYTHONPATH="$base/baseline"
 cd "$base/baseline"
 "$py" "$driver" --task "$task" --baseline --output "$results/$stage-$task-baseline.json"
 export PYTHONPATH="$base/$stage"
 cd "$base/$stage"
 "$py" "$driver" --task "$task" --fixture "$base/$stage/docs/research/semantic_recovery_evidence/fetch_family/canonical-$task.json" --output "$results/$stage-$task-candidate.json" --gate-output "$results/$stage-$task-gate.json"
 "$py" - "$results" "$stage" "$task" <<'PYTHON'
import json
import sys
from pathlib import Path
base, stage, task = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
a = json.loads((base / f'{stage}-{task}-baseline.json').read_text())
b = json.loads((base / f'{stage}-{task}-candidate.json').read_text())
assert a == b, (a, b)
print('FETCH_FROZEN_RENDERER_GATE_PASS', task, stage, flush=True)
PYTHON
done
echo "FETCH_FAMILY_GPU_GATE_PASS $stage"
