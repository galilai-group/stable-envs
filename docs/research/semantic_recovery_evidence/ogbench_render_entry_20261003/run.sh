#!/bin/bash
#SBATCH --job-name=ogb-render-entry
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-render-entry-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-render-entry-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'source_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen production source verified',len(manifest['files']),flush=True)
VERIFY
FAMILIES=(Cube Scene)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
nvidia-smi
lscpu
"$PY" scripts/benchmarks/online_stepping/ogbench_render_entry_profile.py --family "$FAMILY" --output "$BASE/results/$FAMILY.json"
