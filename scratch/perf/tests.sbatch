#!/bin/bash
# The GPU conformance suite restricted to the MJWarp families, in the interpreter that has their
# reference packages. usage: sbatch --job-name=<tag> tests.sbatch <venv> -k "<expr>"
#SBATCH --job-name=se_perf_tests
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/%x_%j.log
set -uo pipefail
VENV=$1; shift
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export WARP_CACHE_PATH=/mnt/vast/home/randall.balestriero/.cache/warp
PY=/mnt/vast/home/randall.balestriero/venvs/$VENV/bin/python
echo "node=$(hostname) venv=$VENV args=$*"
"$PY" -u -m pytest tests/gpu -v -rs -p no:randomly "$@"
