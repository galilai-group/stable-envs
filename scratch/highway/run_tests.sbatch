#!/bin/bash
#SBATCH --job-name=se-tests
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/se_logs/tests_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
P=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python
L=/mnt/vast/home/randall.balestriero/se_logs

echo "===== is the Roundabout abort state-dependent? ====="
for m in torch triton zeros; do
  if out=$($P scratch/highway/probe_roundabout_abort.py $m 2>&1); then
    echo "[$m] $(echo "$out" | grep -E '^backend|^OK' | tr '\n' ' ')"
  else
    rc=$?
    echo "[$m] ABORTED exit $rc :: $(echo "$out" | grep -E '^backend' | head -1)"
  fi
done

echo
echo "===== registry ====="
$P -c "import stable_envs as se; print(sorted(se.REGISTRY)); print('import errors', se.envs.import_errors())"

echo
echo "===== highway-family subset ====="
$P -u -m pytest tests/ -q -rs --tb=short -k "Highway or Roundabout or Merge or Intersection or Racetrack or PushT" \
  > $L/pytest_subset_$SLURM_JOB_ID.txt 2>&1
echo "exit $?"
grep -E "^[0-9]+ passed|passed|failed|error" $L/pytest_subset_$SLURM_JOB_ID.txt | tail -5

echo
echo "===== full suite ====="
$P -u -m pytest tests/ -q -rs --tb=short > $L/pytest_full_$SLURM_JOB_ID.txt 2>&1
echo "exit $?"
grep -E "passed|failed|error|FAILED|ERROR" $L/pytest_full_$SLURM_JOB_ID.txt | grep -v "Exception ignored" | tail -20
echo TESTS_DONE
