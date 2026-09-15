#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-tier-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The ONE certification tier, from Step 3's own harness and Step 3's own state draw.
#
# --exclusive because the first attempt (job 6394647, node2410) measured the DAYLIGHT path -- an
# unchanged uint8 gather -- at 6,219 f/s where a laptop measures 126,845 and Step 6 certified
# 115,166-154,086 on this cluster. That is contention, not this renderer, and a tier taken on a
# contended node is not a tier. Load average is printed before and after so the next reader can
# tell the difference without re-deriving it.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENV=$SCR/venvs/se7-craftax
echo "== host $(hostname) =="; cat /proc/loadavg; nproc
md5sum stable_envs/envs/craftax.py stable_envs/render/prng.py
# (1) the headline: Step 3's harness, which draws probe.sample_states(64) -- the registered default's
#     own distribution -- and labels the tier with Step 3's own ladder
"$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 --envs CraftaxClassic \
  --out examples/results/step7/craftax-classic-full/sweep/bench_step3_cpu_$(hostname).json \
  || echo "bench_step3 leg failed"
# (2) the diagnostics: the same states per lighting band
"$VENV/bin/python" scratch/step7/craftax-classic-full/bench_bands.py --device cpu --threads 1 \
  --tag "$(hostname)"
# (3) the Step-6 tree on the same node in the same job, so the daylight comparison is a comparison
PYTHONPATH=$SCR/se6/baseline "$VENV/bin/python" examples/bench_step3.py --device cpu --threads 1 \
  --envs CraftaxClassic \
  --out examples/results/step7/craftax-classic-full/sweep/bench_step3_cpu_baseline_$(hostname).json \
  || echo "baseline leg unavailable"
cat /proc/loadavg
