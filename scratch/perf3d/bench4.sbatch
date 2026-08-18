#!/bin/bash
#SBATCH --job-name=se3d_b4
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/bench4_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
HB=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python
ENVS=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python

echo "############ dmc venv: parity of the two paths"
$DMC -u -m pytest tests/gpu/test_mover_window.py tests/gpu/test_fused_cast.py -q 2>&1 | tail -5
echo
echo "############ dmc venv: 256px b1024"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs DMCCartpoleBalance,DMCCartpoleSwingup,DMCPendulumSwingup,DMCAcrobotSwingup,DMCPointMassEasy,DMCReacherEasy,DMCReacherHard,DMCFingerSpin,DMCFingerTurnEasy,DMCFingerTurnHard,DMCBallInCupCatch,MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_dmc_b4.json
echo
echo "############ dmc venv: what is left"
$DMC -u scratch/perf3d/chunkprof.py --batch 256 --size 256 --rows 12
echo
echo "############ se-hb: parity"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_fused_cast.py tests/gpu/test_mover_window.py -q 2>&1 | tail -5
echo
echo "############ se-hb: ogbench 256px b1024"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs PointmazeMediumNavigate,PointmazeGiantNavigate,AntmazeMediumNavigate,AntmazeLargeStitch,AntmazeGiantNavigate,HumanoidmazeMediumNavigate,HumanoidmazeGiantNavigate \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_hb_b4.json
echo
echo "############ se-envs: maze2d 256px b1024 + parity"
$ENVS -u -m pytest tests/gpu/test_fused_cast.py -q 2>&1 | tail -5
$ENVS -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs PointMazeUMaze,PointMazeOpen,PointMazeMedium,PointMazeLarge \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_maze2d_b4.json
