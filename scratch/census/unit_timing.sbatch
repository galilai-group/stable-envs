#!/bin/bash
#SBATCH --job-name=se-unit
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_unit_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
PY=/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python
echo "### node=$(hostname)"

echo "-- cost of the import the unit suite pays --"
"$PY" -c "
import time; t=time.perf_counter(); import torch; t1=time.perf_counter()
import stable_envs as se; t2=time.perf_counter()
print(f'  import torch        {t1-t:6.2f}s')
print(f'  import stable_envs  {t2-t1:6.2f}s   (registry {len(se.REGISTRY)})')" 2>&1 | grep -v WARNING

echo; echo "-- the new collision guard, alone --"
"$PY" -u -m pytest tests/certification/test_declarations.py -q -p no:randomly -k "shadow" -rA --no-header 2>&1 | tail -8

echo; echo "-- whole unit suite, with the 15 slowest --"
"$PY" -u -m pytest tests/unit -q -p no:randomly --durations=15 --no-header 2>&1 | tail -25
echo "### unit done"
