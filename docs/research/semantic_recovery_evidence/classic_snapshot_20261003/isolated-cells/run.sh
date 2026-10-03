#!/bin/bash
#SBATCH --job-name=classic-isolated-cells
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-6%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-isolated-cells-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-isolated-cells-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD" SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'source_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen source verified',len(manifest['files']),flush=True)
VERIFY
nvidia-smi
lscpu
MODES=(baseline candidate)
if (( SLURM_ARRAY_TASK_ID % 2 )); then MODES=(candidate baseline); fi
for MODE in "${MODES[@]}"; do
  "$PY" scripts/benchmarks/online_stepping/classic_snapshot_isolated_cells.py --cell "$SLURM_ARRAY_TASK_ID" --mode "$MODE" --output "$BASE/results/$SLURM_ARRAY_TASK_ID/$MODE.json"
done
"$PY" scripts/benchmarks/online_stepping/summarize_classic_snapshot_isolated_cells.py "$BASE/results/$SLURM_ARRAY_TASK_ID" "$BASE/results/$SLURM_ARRAY_TASK_ID.csv" --cell "$SLURM_ARRAY_TASK_ID" --source "$PWD"
