#!/bin/bash
#SBATCH --job-name=rocket-current-gate
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/rocket-current-gate-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/rocket-current-gate-v1-20261003
RUN="$BASE/results/$SLURM_ARRAY_TASK_ID"
mkdir -p "$RUN/cache"
cd "$BASE/code"
export PYTHONPATH="$PWD:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export XDG_CACHE_HOME="$RUN/cache" TORCH_EXTENSIONS_DIR="$RUN/cache/torch" CUDA_CACHE_PATH="$RUN/cache/cuda"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys,importlib.metadata as md
root=pathlib.Path(sys.argv[1])
for rel,digest in json.loads((root/'candidate_manifest.json').read_text())['files'].items():
    assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
assert md.version('PyFlyt')=='0.29.0'
assert md.version('pybullet')=='3.2.7'
print('Frozen candidate and official Rocket engine verified',flush=True)
VERIFY
nvidia-smi > "$RUN/nvidia-smi.txt"
lscpu > "$RUN/lscpu.txt"
MODES=(--render-cpu --gpu)
"$PY" scripts/research_rocket_production/public_gate.py "${MODES[$SLURM_ARRAY_TASK_ID]}" --output "$RUN/public" > "$RUN/public.log" 2>&1
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    "$PY" scripts/research_rocket_production/persistent_variation_gate.py --output "$RUN/variation" > "$RUN/variation.log" 2>&1
    "$PY" scripts/research_rocket_production/branch_source_gate.py --output "$RUN/suffix" > "$RUN/suffix.log" 2>&1
    "$PY" -m pytest tests/unit/test_rocket_replay_semantics.py -q --junitxml="$RUN/unit.xml" > "$RUN/unit.log" 2>&1
fi
