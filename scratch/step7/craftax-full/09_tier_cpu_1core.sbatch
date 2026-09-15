#!/bin/bash
#SBATCH --job-name=se7-craftax-full-tier-cpu-1core
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The CPU tier and the official reference under the ONE-CORE condition S_cpu is defined at, enforced
# by the cgroup rather than asserted.
#
# Job 6397916 pinned torch to one thread and left --cpus-per-task=4, which is the one-core condition
# for OUR renderer and not for the reference: torch.set_num_threads(1) says nothing about jax, whose
# CPU backend has its own pool, so the official renderer could use four cores while ours used one.
# That understates S_cpu rather than flattering it, but it is not the condition the ladder is defined
# under either way. ``taskset -c 0`` does not fix it -- it fails inside a SLURM cpuset -- so the
# allocation itself is one core and /proc/self/status is logged to show the cgroup enforced it.
#
# The paired baseline legs are unchanged in purpose: CraftaxFull is a new cell with no counterpart on
# $SCR/se7/baseline, so they qualify the NODE against a known answer rather than forming a ratio.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
R=examples/results/step7/craftax-full/sweep
P=scratch/step7/craftax-full
H=$(hostname)
mkdir -p $R
echo "== host $H arch $(uname -m) =="
echo "-- the one-core condition, as the cgroup sees it --"
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}  nproc=$(nproc)  node cores=$(getconf _NPROCESSORS_ONLN)"
cat /proc/loadavg
"$VENV/bin/python" -c "
import os, jax, torch
print('jax', jax.__version__, 'default_backend', jax.default_backend(), 'devices', jax.devices())
print('sched_getaffinity', sorted(os.sched_getaffinity(0)), 'torch threads', torch.get_num_threads())
"
md5sum stable_envs/envs/craftax_full.py

echo "== baseline leg A (CraftaxClassic on \$SCR/se7/baseline) =="; cat /proc/loadavg
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic --out $R/bench_step3_cpu_baseA_$H.json

echo "== branch leg (CraftaxFull) =="; cat /proc/loadavg
"$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 --envs CraftaxFull \
  --out $R/bench_step3_cpu_branch_$H.json

echo "== baseline leg B =="; cat /proc/loadavg
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic --out $R/bench_step3_cpu_baseB_$H.json

echo "== bands + both official denominators, same core, same job =="; cat /proc/loadavg
"$VENV/bin/python" $P/bench.py --device cpu --tag "$H" --official --seconds 2.0

echo "== paired verdict =="; cat /proc/loadavg
"$VENV/bin/python" $P/paired_verdict.py --host "$H"
