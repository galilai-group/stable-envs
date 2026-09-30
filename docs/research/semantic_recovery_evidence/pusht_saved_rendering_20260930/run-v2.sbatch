#!/bin/bash
#SBATCH --job-name=se-pusht-saved
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/pusht-saved-rendering-v2-20260930/job-%A_%a.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/pusht-saved-rendering-v2-20260930
run=$root/run-$SLURM_ARRAY_TASK_ID
mkdir "$run"
python3 - "$root" <<'PY'
import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]); oracle=Path('/oscar/scratch/sboughan/se-semantic-recovery/render-rerun-df80c9c5-20260928T021357Z/run-6750648_2/oracle/stable_worldmodel')
for p,h in json.loads((root/'oracle-sha256.json').read_text()).items(): assert hashlib.sha256((oracle/p).read_bytes()).hexdigest()==h,p
for p,h in json.loads((root/'production-sha256.json').read_text()).items(): assert hashlib.sha256((root/'source'/p).read_bytes()).hexdigest()==h,p
PY
oracle=/oscar/scratch/sboughan/se-semantic-recovery/render-rerun-df80c9c5-20260928T021357Z/run-6750648_2/oracle
py=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
export PYTHONPATH="$root/source:/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586/measure-6739950/pusht-overlay:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$run/torch-cache" TRITON_CACHE_DIR="$run/triton-cache" XDG_CACHE_HOME="$run/cache"
registrations=(swm/PushT-v1 swm/PushT-Discrete-v1)
registration=${registrations[$SLURM_ARRAY_TASK_ID]}
cd "$root/source"
lscpu > "$run/lscpu.txt"
nvidia-smi -q > "$run/nvidia-smi.txt"
"$py" -m scripts.benchmarks.renderer_table.pusht_saved capture "$registration" "$run/corpus" --oracle "$oracle"
"$py" -m scripts.benchmarks.renderer_table.pusht_saved benchmark "$registration" "$run/cpu" --oracle "$oracle" --corpus "$run/corpus" --device cpu --batches 64 128 256
"$py" -m scripts.benchmarks.renderer_table.pusht_saved benchmark "$registration" "$run/cuda" --oracle "$oracle" --corpus "$run/corpus" --device cuda --batches 64 128 256 512 1024 2048
