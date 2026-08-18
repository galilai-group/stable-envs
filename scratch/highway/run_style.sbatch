#!/bin/bash
#SBATCH --job-name=se-hwstyle
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/se_logs/hwstyle_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
P=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python
L=/mnt/vast/home/randall.balestriero/se_logs

echo "===== Roundabout on the FUSED backend (604 primitives) ====="
$P scratch/highway/probe_roundabout_abort.py triton 2>&1 | grep -E "^backend|^OK|Error"

echo
echo "===== parity, 64 states, fused backend ====="
for e in Highway Merge Roundabout Intersection; do
  echo "--- $e"
  $P scratch/highway/calib.py $e 64 2>&1 | grep -vE "Deprecation|logger" \
    | grep -E "primitives|state_dim|reference |exact_frac|mean_abs|interior_exact|wrong "
done

echo
echo "===== requested tests ====="
$P -u -m pytest tests/ -q -rs -k "Highway or Merge or Roundabout or Intersection or PushT" \
  > $L/pytest_style_$SLURM_JOB_ID.txt 2>&1
echo "exit $?"
grep -E "passed|failed|FAILED|ERROR" $L/pytest_style_$SLURM_JOB_ID.txt | grep -v "Exception ignored" | tail -20

echo
echo "===== full suite ====="
$P -u -m pytest tests/ -q -rs > $L/pytest_fullstyle_$SLURM_JOB_ID.txt 2>&1
echo "exit $?"
grep -E "passed|failed|FAILED|ERROR" $L/pytest_fullstyle_$SLURM_JOB_ID.txt | grep -v "Exception ignored" | tail -20

echo
echo "===== throughput ====="
$P scratch/highway/bench.py Highway,Merge,Roundabout,Intersection 4096,16384 2>&1 | grep -vE "Deprecation|logger"
echo DONE
