#!/bin/bash
# Re-measure the ss=2 rows of one decode tag at the env's OWN supersample.
#   sbatch --job-name=ssfix_dmc --export=ALL,PY=...,TAG=dmc[,EXTRA="..."] scratch/audit/ss_rerun.sbatch
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/audit/%x_%j.log
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export ${EXTRA:-DUMMY_UNUSED=1}
PY=${PY:-/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python}
echo "node=$(hostname) py=$PY tag=$TAG"

ENVS=$("$PY" scratch/audit/ss_remeasure.py list --tag "$TAG" | tail -1 | tr ' ' ',')
echo "affected: $ENVS"
[ -z "$ENVS" ] && { echo "nothing to do"; exit 0; }

"$PY" -u scratch/audit/ss_remeasure.py strip --tag "$TAG"
"$PY" -u examples/bench_vs_decode.py --tag "$TAG" --envs "$ENVS" --ss native \
      --out examples/results/decode --scratch /tmp/se_decode_$$
rc=$?
"$PY" -u scratch/audit/ss_remeasure.py restore --tag "$TAG"
echo "ss_rerun tag=$TAG rc=$rc node=$(hostname)"
