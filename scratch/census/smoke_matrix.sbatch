#!/bin/bash
# Validate the matrix plumbing before spending 7 full runs on it: does the SE_REQUIRE_ENVS guard
# resolve, does every interpreter collect, and does the rewritten rendernet fixture behave.
#SBATCH --job-name=se-mx-smoke
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_mx_smoke_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1

echo "### node=$(hostname) gpus=$(nvidia-smi -L | wc -l)"

for V in dmc se-envs se-hb se-jumanji se-oc se-pgx se-tiles; do
  PY=/mnt/vast/home/randall.balestriero/venvs/$V/bin/python
  echo; echo "================ $V ================"
  [ -x "$PY" ] || { echo "MISSING INTERPRETER"; continue; }

  export SE_REQUIRE_ENVS=$("$PY" - "$V" <<'EOF'
import json, sys
a = json.load(open("scratch/census/assignment.json"))
print(",".join(a.get(sys.argv[1], [])))
EOF
)
  echo "owns $(tr ',' '\n' <<<"$SE_REQUIRE_ENVS" | grep -c .) envs"
  echo "-- registry / cover guard --"
  "$PY" -u -m pytest tests/gpu/test_conformance.py -q -rs -p no:randomly \
      -k "registry_is_not_empty or required_envs_are_all_reachable" 2>&1 | tail -20
  echo "-- collection totals --"
  "$PY" -u -m pytest tests/ -q --collect-only -p no:randomly 2>&1 | tail -4
  echo "-- rendernet fixture (the rewritten one) --"
  "$PY" -u -m pytest tests/gpu/test_rendernet.py -q -rs -p no:randomly 2>&1 | tail -12
  echo "-- distributed (2 GPUs are allocated, so these must NOT skip) --"
  "$PY" -u -m pytest tests/gpu/test_distributed.py -q -rs -p no:randomly 2>&1 | tail -12
done
echo; echo "### smoke done rc=$?"
