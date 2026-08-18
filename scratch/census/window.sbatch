#!/bin/bash
#SBATCH --job-name=se_window
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/window_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
echo "### DM Control venv (16 cells)"
/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python -m pytest tests/gpu/test_mover_window.py \
    -q --no-header --tb=short 2>&1 | tail -18
echo "### suite venv (MuJoCo locomotion)"
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/gpu/test_mover_window.py \
    -q --no-header --tb=short 2>&1 | tail -12
