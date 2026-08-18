#!/bin/bash
#SBATCH --job-name=se3d_port
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/port_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
echo "### bounded-vs-unbounded + fused-vs-torch"
$DMC -u -m pytest tests/gpu/test_mover_window.py tests/gpu/test_fused_cast.py -q -rf 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### official-renderer parity + determinism"
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf -k "(parity or batch_invariance or determinism or backend_agreement) and (DMC or Mujoco)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### batch/chunk invariance"
$DMC -u -m pytest tests/gpu/test_reproducibility.py -q -rf -k "(batch_size or chunk_independent or out_buffer or batch_order) and (DMC or Mujoco)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### throughput 256px b1024"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
  --envs MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum,DMCCartpoleBalance,DMCReacherEasy \
  --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/split_port.json
echo "### ceiling decomposition after the port"
$DMC -u scratch/perf3d/ceiling.py --batch 1024 --size 256 \
  --envs MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum \
  --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/ceiling_port.json
