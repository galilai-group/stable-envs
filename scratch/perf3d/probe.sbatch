#!/bin/bash
#SBATCH --job-name=se3d_probe
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/probe_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
HB=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python

echo "############ triton bit-exactness"
$DMC -u scratch/perf3d/triton_exact.py

echo
echo "############ dmc venv: split at 256px b1024"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_dmc.json

echo
echo "############ dmc venv: per-pass at 256px"
$DMC -u scratch/perf3d/passes256.py --size 256 --batch 32

echo
echo "############ se-hb: ogbench + maze2d + swm_nav split at 256px b1024"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs PointmazeMediumNavigate,AntmazeMediumNavigate,AntmazeLargeNavigate,HumanoidmazeMediumNavigate,Maze2dUmaze,Maze2dLarge \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_hb.json
echo "### registry sample"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -c "import stable_envs as se; r=sorted(se.REGISTRY); print(len(r)); print([n for n in r if n.startswith(('Maze2d','Swm','SWM','Nav','Point','Ant','Human'))])"
