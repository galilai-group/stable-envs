#!/bin/bash
#SBATCH --job-name=classic-acrobot-repeat
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-acrobot-repeat-v1-20261003/gate-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-acrobot-repeat-v1-20261003
FROZEN=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
cd "$FROZEN/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_VIDEODRIVER=dummy
export XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$FROZEN" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'source_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
VERIFY
nvidia-smi
lscpu
"$PY" scripts/benchmarks/online_stepping/classic_snapshot.py --domain acrobot --output "$BASE/acrobot.json"
"$PY" scripts/benchmarks/online_stepping/summarize_classic_snapshot.py "$BASE/acrobot.json" "$BASE/acrobot.csv" --source "$PWD"
