#!/bin/bash
#SBATCH --job-name=se3d_b7
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/b7_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
echo "############ bounded vs unbounded, fused vs torch"
$DMC -u -m pytest tests/gpu/test_mover_window.py tests/gpu/test_fused_cast.py -q -rf 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "############ official-renderer parity"
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf -k "(parity or batch_invariance or determinism) and (DMC or Mujoco)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "############ 256px b1024"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
  --envs DMCCartpoleBalance,DMCCartpoleSwingup,DMCPendulumSwingup,DMCAcrobotSwingup,DMCPointMassEasy,DMCReacherEasy,DMCReacherHard,DMCFingerSpin,DMCFingerTurnEasy,DMCFingerTurnHard,DMCBallInCupCatch,MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum \
  --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_b7.json
echo "############ per-pass"
$DMC -u scratch/perf3d/passes256.py --size 256 --batch 64 --envs DMCReacherEasy,DMCFingerTurnEasy,DMCCartpoleBalance
echo "############ kernel vs interpreter"
$DMC -u scratch/perf3d/residual.py --batch 256 --size 256 --top 6
