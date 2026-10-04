#!/bin/bash
#SBATCH --job-name=rocket-final-candidate
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|a5000|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclude=gpu3001
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/rocket-production-coverage-v2-20261004/online-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/rocket-production-coverage-v2-20261004
ORIGINAL=/oscar/scratch/sboughan/se-semantic-recovery/rocket-current-gate-v1-20261003
RUN="$BASE/results/$SLURM_JOB_ID"
mkdir -p "$RUN/cache"
cp -a "$ORIGINAL/code" "$RUN/code"
cd "$RUN/code"
export PYTHONPATH="$PWD:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export XDG_CACHE_HOME="$RUN/cache" TORCH_EXTENSIONS_DIR="/oscar/scratch/sboughan/se-semantic-recovery/rocket-lowres-v2-20261004/results/6995506/cache/torch" CUDA_CACHE_PATH="$RUN/cache/cuda"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
"$PY" - "$ORIGINAL" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for rel,digest in json.loads((root/'candidate_manifest.json').read_text())['files'].items():
    assert hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()==digest,rel
VERIFY
tar -xzf "$BASE/patch.tar.gz"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for rel,digest in json.loads((root/'patch_manifest.json').read_text())['files'].items():
    assert hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()==digest,rel
VERIFY
lscpu > "$RUN/lscpu.txt"
nvidia-smi > "$RUN/nvidia-smi.txt"
(
 "$PY" -m pytest -q tests/unit/test_rocket_cuda_coverage.py > "$RUN/cuda-unit.log" 2>&1
 "$PY" scripts/research_rocket_production/public_gate.py --gpu --output "$RUN/public-cuda" > "$RUN/public-cuda.log" 2>&1
 "$PY" scripts/research_rocket_production/online_source_gate.py --gpu --output "$RUN/online-cuda" > "$RUN/online-cuda.log" 2>&1
 echo GPU_LANE_PASS > "$RUN/gpu-terminal.txt"
) &
GPU_PID=$!
(
 "$PY" -m pytest -q tests/unit/test_rocket_replay_semantics.py tests/unit/test_rocket_landing.py tests/unit/test_nvdiff_triangles.py tests/unit/test_nvdiff_texgen.py > "$RUN/cpu-unit.log" 2>&1
 "$PY" scripts/research_rocket_production/public_gate.py --render-cpu --output "$RUN/public-cpu" > "$RUN/public-cpu.log" 2>&1
 "$PY" scripts/research_rocket_production/online_source_gate.py --output "$RUN/online-cpu" > "$RUN/online-cpu.log" 2>&1
 "$PY" scripts/research_rocket_production/branch_source_gate.py --output "$RUN/branch-source" > "$RUN/branch-source.log" 2>&1
 "$PY" scripts/research_rocket_production/persistent_variation_gate.py --output "$RUN/persistent-variation" > "$RUN/persistent-variation.log" 2>&1
 echo CPU_LANE_PASS > "$RUN/cpu-terminal.txt"
) &
CPU_PID=$!
set +e
wait "$GPU_PID"
GPU_STATUS=$?
wait "$CPU_PID"
CPU_STATUS=$?
set -e
printf 'GPU_EXIT=%s\nCPU_EXIT=%s\n' "$GPU_STATUS" "$CPU_STATUS" > "$RUN/terminal.txt"
if [ "$GPU_STATUS" -ne 0 ] || [ "$CPU_STATUS" -ne 0 ]; then exit 1; fi
echo ROCKET_CANDIDATE_GATES_PASS
