#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-tier-cpu-paired
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The CPU tier by the PAIRED-MEASUREMENT rule, because --exclusive cannot start.
#
# Job 6395167 asked for a whole node and SLURM scheduled it for 05:00 -- the normal QOS's cpu=64 cap
# blocks exclusive allocations here, exactly as it did in Step 6. The fallback is the rule Step 6
# used instead: measure the Step-6 tree and this branch in the SAME job on the SAME node, and judge
# the pair rather than the absolute number.
#
# The baseline is $SCR/se7/baseline -- a pristine 76dc511, this workstream's OWN parent. It is NOT
# $SCR/se6/baseline, which AGENT_COMMON provides for Step 6 and which is the PRE-Step-6 tree: no
# _Composite, the per-frame blend renderer Step 6 replaced. Job 6395321 used it by mistake and
# measured the "baseline" at 1,371 f/s -- about a hundredth of what Step 6 certified -- which read
# as node contention and was not: node1936 was at 0.29 load per core. The verdict script's baseline
# floor caught it because a leg that re-measures a known tree has a known answer.
#
# The baseline leg runs TWICE, on either side of the branch leg. Its two numbers are the noise floor
# of the node: if they differ by more than 1.25x the node moved under the measurement and the whole
# job is quarantined rather than reported. That is the check the first attempt (6394647, node2410)
# had no way to make, which is why it reported a daylight gather at 6,219 f/s -- twenty times slower
# than a laptop -- without flagging anything.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
R=examples/results/step7/craftax-classic-full/sweep
H=$(hostname)
echo "== host $H =="; cat /proc/loadavg; nproc
md5sum stable_envs/envs/craftax.py stable_envs/render/prng.py

echo "== baseline leg A =="; cat /proc/loadavg
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic --out $R/bench_step3_cpu_baseA_$H.json

echo "== branch leg =="; cat /proc/loadavg
"$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 --envs CraftaxClassic \
  --out $R/bench_step3_cpu_branch_$H.json

echo "== baseline leg B =="; cat /proc/loadavg
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic --out $R/bench_step3_cpu_baseB_$H.json

echo "== per-band diagnostics =="; cat /proc/loadavg
"$VENV/bin/python" scratch/step7/craftax-classic-full/bench_bands.py --device cpu --threads 1 \
  --tag "$H"

echo "== paired verdict =="; cat /proc/loadavg
"$VENV/bin/python" scratch/step7/craftax-classic-full/paired_verdict.py --host "$H"
