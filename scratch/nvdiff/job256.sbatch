#!/bin/bash
#SBATCH --job-name=nvd256
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/%x_%j.log

# One PROCESS per scene: nvdiffrast answers "too big" with an illegal memory access that kills the
# interpreter rather than raising, so a shared process loses every row after the first that does not fit.
R=/mnt/vast/home/randall.balestriero/code/stable-envs
V=/mnt/vast/home/randall.balestriero/venvs/senvdiff/bin/python
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONPATH=$R
cd $R
echo "node=$(hostname)"
for S in "$@"; do
  echo "===== $S"
  $V scratch/nvdiff/bench256.py --only "$S" --res 256 2>&1 | tail -30
done
