#!/bin/bash
#SBATCH --job-name=se_gfx
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/gfx_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
echo "### suite interpreter (no Box2D) — must SKIP, not fail"
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest \
   tests/gpu/test_gfxdraw_fused.py -q --no-header --tb=short 2>&1 | grep -vE "^\s*$" | tail -18
echo "### se-envs (has Box2D) — must still RUN and pass"
/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python -m pytest \
   tests/gpu/test_gfxdraw_fused.py -q --no-header --tb=short 2>&1 | grep -vE "^\s*$" | tail -18
