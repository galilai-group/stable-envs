#!/bin/bash
#SBATCH --job-name=se3d_b5
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/bench5_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
HB=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python
ENVS=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python

echo "############ PARITY AGAINST THE OFFICIAL RENDERER (dmc + gym mujoco)"
$DMC -u -m pytest tests/gpu/test_conformance.py -q \
  -k "parity or batch_invariance or determinism or backend_agreement" 2>&1 | tail -12

echo
echo "############ chunk independence + batch invariance"
$DMC -u -m pytest tests/gpu/test_reproducibility.py -q \
  -k "batch_size or chunk_independent or out_buffer or batch_order" 2>&1 | tail -8

echo
echo "############ our own two paths"
$DMC -u -m pytest tests/gpu/test_mover_window.py tests/gpu/test_fused_cast.py -q 2>&1 | tail -5

echo
echo "############ dmc venv: 256px b1024, RAY_BUDGET 1<<25"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs DMCCartpoleBalance,DMCCartpoleSwingup,DMCPendulumSwingup,DMCAcrobotSwingup,DMCPointMassEasy,DMCReacherEasy,DMCReacherHard,DMCFingerSpin,DMCFingerTurnEasy,DMCFingerTurnHard,DMCBallInCupCatch,MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_dmc_b5.json

echo
echo "############ peak memory at b1024"
$DMC -u -c "
import numpy as np, torch, stable_envs as se
for n in ('DMCFingerTurnEasy','MujocoPusher'):
    env = se.make(n, batch=1024, size=256, device='cuda')
    s = env.sample_states(1024, rng=np.random.default_rng(0))
    env.render(s); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    env.render(s); torch.cuda.synchronize()
    print(f'{n:24s} peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    del env; torch.cuda.empty_cache()
"

echo
echo "############ se-hb: ogbench parity vs official + speed"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_conformance.py -q \
  -k "(parity or batch_invariance) and (maze or Maze)" 2>&1 | tail -8
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs PointmazeMediumNavigate,AntmazeMediumNavigate,AntmazeGiantNavigate,HumanoidmazeMediumNavigate,HumanoidmazeGiantNavigate \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_hb_b5.json

echo
echo "############ se-envs: maze2d parity + speed"
$ENVS -u -m pytest tests/gpu/test_conformance.py -q -k "(parity or batch_invariance) and PointMaze" 2>&1 | tail -8
$ENVS -u scratch/perf3d/split256.py --batch 1024 --size 256 \
     --envs PointMazeUMaze,PointMazeMedium,PointMazeLarge \
     --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_maze2d_b5.json
