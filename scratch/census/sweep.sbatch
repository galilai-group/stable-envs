#!/bin/bash
#SBATCH --job-name=se_decode
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# gpu-068 earns its place here: it wedged one chunk for 5h41m with a zero-byte log, and on another it
# tripped Meta-World's static-plate guard — two MuJoCo renders of the SAME static scene disagreeing by
# 255/255. That guard's own message says to suspect the node first, and it was right.
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/sweep_%x_%j.log

# usage: sbatch --job-name=se_decode_<venv> sweep.sbatch <venv> [extra args]
VENV=$1; shift
R=/mnt/vast/home/randall.balestriero/code/stable-envs
P=/mnt/vast/home/randall.balestriero/venvs/$VENV/bin/python
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=$R
export SCRATCH=/tmp/se_decode_$SLURM_JOB_ID
mkdir -p $SCRATCH

cd $R/examples
$P bench_vs_decode.py --tag "$VENV" --scratch "$SCRATCH" \
  --out $R/examples/results/decode "$@"
rm -rf $SCRATCH
