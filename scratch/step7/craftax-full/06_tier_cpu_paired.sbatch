#!/bin/bash
#SBATCH --job-name=se7-craftax-full-tier-cpu-paired
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The CPU tier by the PAIRED-MEASUREMENT rule, because --exclusive cannot start.
#
# Job 6397698 asked for a whole node on batch/normal and SLURM never scheduled it (Start Unknown for
# over half an hour) -- the normal QOS's cpu=64 cap blocks exclusive allocations here, exactly as it
# did for W1's 6395167 and in Step 6. The fallback is the rule Step 6 and W1 used instead: qualify
# the node inside the same job, and judge the measurement rather than trusting the absolute number.
#
# The difference from W1's job: CraftaxFull is a NEW cell and does not exist on the baseline tree, so
# there is no before to form a ratio with. The baseline legs measure CraftaxClassic on
# $SCR/se7/baseline -- a pristine 76dc511, this workstream's own parent, which Step 6 certified at
# 115,166 f/s on this cluster -- purely to say whether the node was in a state where an absolute
# number means anything. NOT $SCR/se6/baseline, which is the PRE-Step-6 tree.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-full/scratch/step7/craftax-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
R=examples/results/step7/craftax-full/sweep
P=scratch/step7/craftax-full
H=$(hostname)
mkdir -p $R
echo "== host $H arch $(uname -m) =="; cat /proc/loadavg; nproc
md5sum stable_envs/envs/craftax_full.py stable_envs/envs/_craftax_full_tex.py stable_envs/render/prng.py

echo "== baseline leg A (CraftaxClassic on \$SCR/se7/baseline) =="; cat /proc/loadavg
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic --out $R/bench_step3_cpu_baseA_$H.json

echo "== branch leg (CraftaxFull on this branch) =="; cat /proc/loadavg
"$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 --envs CraftaxFull \
  --out $R/bench_step3_cpu_branch_$H.json

echo "== baseline leg B =="; cat /proc/loadavg
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic --out $R/bench_step3_cpu_baseB_$H.json

echo "== per-band diagnostics =="; cat /proc/loadavg
"$VENV/bin/python" $P/bench.py --device cpu --tag "$H" --official --seconds 2.0

echo "== paired verdict =="; cat /proc/loadavg
"$VENV/bin/python" $P/paired_verdict.py --host "$H"
