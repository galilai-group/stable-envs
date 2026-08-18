#!/bin/bash
#SBATCH --job-name=se3d_gate
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/gate_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
HB=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python
ENVS=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python
echo "### fused-vs-torch + window-vs-unbounded (dmc)"
$DMC -u -m pytest tests/gpu/test_mover_window.py tests/gpu/test_fused_cast.py -q -rf 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### official-renderer parity, DMC + gym MuJoCo"
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf -k "(parity or batch_invariance or determinism or backend_agreement) and (DMC or Mujoco)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### batch/chunk invariance, DMC + gym MuJoCo"
$DMC -u -m pytest tests/gpu/test_reproducibility.py -q -rf -k "(batch_size or chunk_independent or out_buffer or batch_order) and (DMC or Mujoco)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### ogbench"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_fused_cast.py -q -rf 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py -q -rf -k "(parity or batch_size or chunk_independent) and (maze or Maze)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### maze2d"
$ENVS -u -m pytest tests/gpu/test_fused_cast.py -q -rf 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
$ENVS -u -m pytest tests/gpu/test_conformance.py -q -rf -k "parity and PointMaze" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
