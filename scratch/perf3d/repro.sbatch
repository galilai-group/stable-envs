#!/bin/bash
#SBATCH --job-name=se3d_rep
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/repro_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
echo "###### reproducibility, only the families this session touched"
$DMC -u -m pytest tests/gpu/test_reproducibility.py -q -rf \
  -k "(batch_size or chunk_independent or out_buffer or batch_order) and (DMC or Mujoco)" 2>&1 \
  | grep -E "^(FAILED|ERROR|[0-9]+ (passed|failed))|assert|Error" | head -40
echo
echo "###### conformance parity, only those families"
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf \
  -k "(parity or batch_invariance or determinism or backend_agreement) and (DMC or Mujoco)" 2>&1 \
  | grep -E "^(FAILED|ERROR|[0-9]+ (passed|failed))|assert" | head -40
