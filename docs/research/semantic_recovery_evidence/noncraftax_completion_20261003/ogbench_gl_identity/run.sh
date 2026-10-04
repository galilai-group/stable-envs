#!/bin/bash
#SBATCH --job-name=ogb-source-isolation
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=1-2%1
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-identity-v3-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-identity-v3-20261003
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
for name, prefix in [('candidate_manifest.json','code'),('source_manifest.json','source')]:
    manifest=json.loads((root/name).read_text())
    for rel,digest in manifest['files'].items():
        assert hashlib.sha256((root/prefix/rel).read_bytes()).hexdigest()==digest,rel
    print(name, len(manifest['files']), 'verified', flush=True)
VERIFY
nvidia-smi
lscpu
FAMILIES=(Maze Cube Scene)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --benchmark ogbench_source_frame_gate --domain "$FAMILY" --source-root "$BASE/source" --source-manifest "$BASE/source_manifest.json" --output "$BASE/results/$FAMILY"
