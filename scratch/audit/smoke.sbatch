#!/bin/bash
#SBATCH --job-name=se-smoke
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/audit/%x_%j.log
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
"$PY" -m pytest tests/unit tests/test_safety_net.py -q --no-header
echo "unit rc=$?"
"$PY" -m pytest tests/gpu/test_conformance.py -v -rs --no-header \
    -k "parity_floors or required_env_modules or (render_parity and (TwoRoom or Piecewise or MujocoPusher or MujocoReacher or PushT))"
echo "parity rc=$? node=$(hostname)"
