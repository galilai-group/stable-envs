#!/bin/bash
#SBATCH --job-name=se3d_p3
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/probe3_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
echo "############ shader shapes in Triton"
$DMC -u scratch/perf3d/shade_exact.py 2>&1; echo "shade_exact exit=$?"
echo
echo "############ kernel vs interpreter, corrected"
$DMC -u scratch/perf3d/residual.py --batch 256 --size 256 --top 10 2>&1
