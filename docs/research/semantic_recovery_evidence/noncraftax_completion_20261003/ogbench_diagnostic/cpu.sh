#!/bin/bash
#SBATCH --job-name=ogb-source-cpu-errors
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=1-2%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-errors-v2-20261003/cpu-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-errors-v2-20261003
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cpu-cache/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
for name, prefix in [('candidate_manifest.json','code'),('source_manifest.json','source')]:
    for rel,digest in json.loads((root/name).read_text())['files'].items():
        assert hashlib.sha256((root/prefix/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen candidate/source verified',flush=True)
VERIFY
FAMILIES=(Maze Cube Scene)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
"$PY" "$BASE/cpu_diagnostic.py" "$BASE" "$FAMILY" "$BASE/cpu-results/$SLURM_ARRAY_JOB_ID/$FAMILY"
