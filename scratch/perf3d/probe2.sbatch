#!/bin/bash
#SBATCH --job-name=se3d_p2
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/probe2_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python

echo "############ can the SHADER be reproduced in Triton bit for bit"
$DMC -u scratch/perf3d/shade_exact.py

echo
echo "############ kernel time vs interpreter time (the residual, measured)"
$DMC -u scratch/perf3d/residual.py --batch 256 --size 256

echo
echo "############ MujocoPusher seeded parity, at the declared floor"
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf -k "parity and (Pusher or Reacher)" 2>&1 \
  | grep -E "^(FAILED|[0-9]+ (passed|failed))|interior_exact_frac|relaxed"
echo "--- rate over many draws ---"
$DMC -u scratch/audit/pusher_parity_rate.py --envs MujocoPusher --draws 30 2>&1 | tail -40
