#!/bin/bash
#SBATCH --job-name=se_store
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/store_%x_%j.log

# usage: sbatch --job-name=st_<tag> storage.sbatch <venv> [extra args]
VENV=$1; shift
R=/mnt/vast/home/randall.balestriero/code/stable-envs
P=/mnt/vast/home/randall.balestriero/venvs/$VENV/bin/python
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=$R
LOCAL=/tmp/se_store_$SLURM_JOB_ID
SHARED=/mnt/vast/home/randall.balestriero/tmp_storage_bench/$SLURM_JOB_ID
mkdir -p $LOCAL $SHARED

cd $R/examples
$P bench_storage.py --local "$LOCAL" --shared "$SHARED" \
  --out $R/examples/results/storage "$@"
rm -rf $LOCAL $SHARED
