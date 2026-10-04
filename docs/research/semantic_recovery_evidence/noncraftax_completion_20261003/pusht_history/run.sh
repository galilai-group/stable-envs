#!/bin/bash
#SBATCH --job-name=pusht-history-api
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/pusht-history-api-v1-20261003/unit-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/pusht-history-api-v1-20261003
RUN="$BASE/results/$SLURM_JOB_ID"
mkdir -p "$RUN/cache" "$RUN/deps" "$RUN/code"
tar -xzf "$BASE/pusht-history-api-20261003.tar.gz" -C "$RUN/code"
cd "$RUN/code"
export PYTHONPATH="$PWD:$RUN/deps:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export XDG_CACHE_HOME="$RUN/cache" PIP_CACHE_DIR="$RUN/cache/pip" TORCH_EXTENSIONS_DIR="$RUN/cache/torch" CUDA_CACHE_PATH="$RUN/cache/cuda"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
"$PY" -m pip install --target "$RUN/deps" --no-deps --only-binary=:all: pymunk==7.2.0 > "$RUN/install.log" 2>&1
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys,torch,importlib.metadata as m
root=pathlib.Path(sys.argv[1])
for rel,digest in json.loads((root/'candidate_manifest.json').read_text())['files'].items():
    assert hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()==digest,rel
assert torch.cuda.is_available()
assert m.version('pymunk')=='7.2.0'
print({k:m.version(k) for k in ['pymunk','numpy','torch','tensordict','pygame']},flush=True)
VERIFY
nvidia-smi > "$RUN/nvidia-smi.txt"
"$PY" -m pytest tests/unit/test_pusht_history.py tests/unit/test_step_capture.py tests/unit/test_step_masking.py -q --junitxml="$RUN/unit.xml" > "$RUN/unit.log" 2>&1
