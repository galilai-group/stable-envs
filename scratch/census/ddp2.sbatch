#!/bin/bash
#SBATCH --job-name=se_ddp2
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/ddp2_%j.log
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
echo "### one-factor diagnostic"
$PY -u scratch/census/diag_ddp.py 2>&1
echo "### full test output"
$PY -m pytest tests/gpu/test_distributed.py -q -rs --tb=line 2>&1 | tail -25
