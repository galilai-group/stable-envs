#!/usr/bin/env bash
# Representative local CPU measurements for Workstream N. These are performance samples, not
# certification numbers. Override PYTHON when the repository's development interpreter differs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/Users/samib/se6-venv/bin/python}"

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$ROOT"

"$PYTHON" -m benchmarks.render \
  --envs PushT,CartPole,DMCReacherHard,DMCWalkerWalk \
  --device cpu \
  --batches 1,16,64 \
  --warmup 2 \
  --repeats 7 \
  --threads 1 \
  --output examples/results/step9/benchmarks/render_cpu_local.json \
  --pretty

"$PYTHON" -m benchmarks.pipeline \
  --envs PushT,DMCReacherHard \
  --device cpu \
  --batch 64 \
  --warmup 2 \
  --repeats 7 \
  --threads 1 \
  --output examples/results/step9/benchmarks/pipeline_cpu_local.json \
  --pretty
