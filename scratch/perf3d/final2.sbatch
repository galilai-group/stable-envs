#!/bin/bash
#SBATCH --job-name=se3d_fin2
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/final2_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
HB=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python
ENVS=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python
OUT=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf

echo "############ DM CONTROL + GYM MUJOCO (parity tier), 256px b1024"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
  --envs DMCCartpoleBalance,DMCCartpoleSwingup,DMCPendulumSwingup,DMCAcrobotSwingup,DMCPointMassEasy,DMCReacherEasy,DMCReacherHard,DMCFingerSpin,DMCFingerTurnEasy,DMCFingerTurnHard,DMCBallInCupCatch,MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum \
  --json $OUT/final_dmc.json

echo
echo "############ OGBENCH (all 25), 256px b1024"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u scratch/perf3d/split256.py --batch 1024 --size 256 \
  --envs PointmazeMediumNavigate,PointmazeMediumStitch,PointmazeLargeNavigate,PointmazeLargeStitch,PointmazeGiantNavigate,PointmazeGiantStitch,PointmazeTeleportNavigate,PointmazeTeleportStitch,AntmazeMediumNavigate,AntmazeMediumStitch,AntmazeMediumExplore,AntmazeLargeNavigate,AntmazeLargeStitch,AntmazeLargeExplore,AntmazeGiantNavigate,AntmazeGiantStitch,AntmazeTeleportNavigate,AntmazeTeleportStitch,AntmazeTeleportExplore,HumanoidmazeMediumNavigate,HumanoidmazeMediumStitch,HumanoidmazeLargeNavigate,HumanoidmazeLargeStitch,HumanoidmazeGiantNavigate,HumanoidmazeGiantStitch \
  --json $OUT/final_ogbench.json

echo
echo "############ D4RL MAZE2D, 256px b1024"
$ENVS -u scratch/perf3d/split256.py --batch 1024 --size 256 \
  --envs PointMazeUMaze,PointMazeOpen,PointMazeMedium,PointMazeLarge --json $OUT/final_maze2d.json

echo
echo "############ FULL PARITY GATE"
$DMC -u -m pytest tests/gpu/test_mover_window.py tests/gpu/test_fused_cast.py -q 2>&1 | tail -4
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf \
  -k "(parity or batch_invariance or determinism or backend_agreement) and (DMC or Mujoco)" 2>&1 \
  | grep -E "^(FAILED|[0-9]+ (passed|failed))" | head -20
$DMC -u -m pytest tests/gpu/test_reproducibility.py -q -rf \
  -k "(batch_size or chunk_independent or out_buffer or batch_order) and (DMC or Mujoco)" 2>&1 \
  | grep -E "^(FAILED|[0-9]+ (passed|failed))" | head -20
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_fused_cast.py tests/gpu/test_mover_window.py -q 2>&1 | tail -3
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py -q -rf \
  -k "(parity or batch_size or chunk_independent) and (maze or Maze)" 2>&1 \
  | grep -E "^(FAILED|[0-9]+ (passed|failed))" | head -20
$ENVS -u -m pytest tests/gpu/test_fused_cast.py -q 2>&1 | tail -3
