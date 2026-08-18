#!/bin/bash
#SBATCH --job-name=se3d_bench
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/bench_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
HB=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python

echo "############ fused vs torch, whole renders (dmc venv)"
$DMC -u -m pytest tests/gpu/test_fused_cast.py -x -q 2>&1 | tail -25

echo
echo "############ windowed vs full frame (dmc venv)"
$DMC -u -m pytest tests/gpu/test_mover_window.py -x -q 2>&1 | tail -15

echo
echo "############ dmc venv: 256px b1024"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_dmc_after.json

echo
echo "############ dmc venv: per-pass at 256px"
$DMC -u scratch/perf3d/passes256.py --size 256 --batch 32

echo
echo "############ se-hb: fused vs torch"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_fused_cast.py -x -q 2>&1 | tail -20

echo
echo "############ se-hb: ogbench 256px b1024"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs PointmazeMediumNavigate,AntmazeMediumNavigate,AntmazeLargeNavigate,AntmazeGiantNavigate,HumanoidmazeMediumNavigate,HumanoidmazeGiantNavigate \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_hb_after.json
