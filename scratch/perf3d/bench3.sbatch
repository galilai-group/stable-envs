#!/bin/bash
#SBATCH --job-name=se3d_b3
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/bench3_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python

echo "############ windows + candidate sets vs the unbounded passes"
$DMC -u -m pytest tests/gpu/test_mover_window.py -x -q 2>&1 | tail -6
echo
echo "############ fused vs torch"
$DMC -u -m pytest tests/gpu/test_fused_cast.py -x -q 2>&1 | tail -6
echo
echo "############ dmc venv: 256px b1024"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_dmc_b3.json
echo
echo "############ where the remainder is"
$DMC -u scratch/perf3d/chunkprof.py --batch 256 --size 256
