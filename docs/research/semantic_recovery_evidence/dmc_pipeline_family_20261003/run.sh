#!/bin/bash
#SBATCH --job-name=dmc-pipeline-family
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-10%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-family-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-family-v1-20261003
EXTRAS=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-smoke-v3-20261003
cd "$BASE/code"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
export PYTHONPATH="$EXTRAS/extras:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$EXTRAS" <<'VERIFY'
import hashlib,importlib.metadata as md,json,sys,pathlib
root=pathlib.Path(sys.argv[1]);profile=json.loads((root/'dependency-install.json').read_text())
for name,record in profile.items():
 d=md.distribution(name)
 assert d.version==record['version']
 for rel,digest in record['files'].items():assert hashlib.sha256(d.locate_file(rel).read_bytes()).hexdigest()==digest,(name,rel)
print('Pinned renderer extras verified without modifying the accepted environment',flush=True)
VERIFY
DOMAINS=(acrobot pendulum cartpole reacher cheetah hopper walker ball_in_cup finger manipulator humanoid)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
nvidia-smi
lscpu
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --domain "$DOMAIN" --benchmark dmc_pipeline --batches 64 256 1024 --output "$BASE/results/$DOMAIN"
"$PY" scripts/benchmarks/online_stepping/summarize_dmc_pipeline.py "$BASE/results/$DOMAIN" "$BASE/tables/$DOMAIN" --source "$PWD"
