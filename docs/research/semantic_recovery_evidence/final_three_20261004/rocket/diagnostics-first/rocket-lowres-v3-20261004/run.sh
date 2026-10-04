#!/bin/bash
#SBATCH --job-name=rocket-lowres-probe
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|a5000|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/rocket-lowres-v3-20261004/online-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/rocket-lowres-v3-20261004
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
"$PY" scripts/research_rocket_production/lowres_render_probe.py --output "$RUN/probe" > "$RUN/probe.log" 2>&1
