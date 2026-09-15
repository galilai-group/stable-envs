#!/bin/bash
#SBATCH --job-name=se7-craftax-classic-full-tier-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# The GPU tier from Step 3's own harness, so the harness's tier LABEL exists for both devices.
#
# Same paired discipline as the CPU job: the Step-6-closed tree either side of the branch, in one
# allocation on one card, so the baseline re-measures a known answer (Step 6 certified it at
# 1,206,822 f/s at B=4096 on a 3090, and 1,229,787 on a second card). A 3090 is requested by
# constraint for the same reason -- a before on one card and an after on another is not a pair.
#
# One GPU job per agent; never cancel a job this workstream did not submit, and never a
# simclr_cifar10 job.
set -euo pipefail
source /oscar/scratch/sboughan/se7/craftax-classic-full/scratch/step7/craftax-classic-full/env_header.sh
VENV=$SCR/venvs/se7-craftax
R=examples/results/step7/craftax-classic-full/sweep
H=$(hostname)
echo "== host $H =="; cat /proc/loadavg; nproc
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
md5sum stable_envs/envs/craftax.py stable_envs/render/prng.py

echo "== baseline leg A =="
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cuda \
  --envs CraftaxClassic --out $R/bench_step3_cuda_baseA_$H.json

echo "== branch leg =="
"$VENV/bin/python" examples/bench_step3.py --device cuda --envs CraftaxClassic \
  --out $R/bench_step3_cuda_branch_$H.json

echo "== baseline leg B =="
PYTHONPATH=$SCR/se7/baseline "$VENV/bin/python" examples/bench_step3.py --device cuda \
  --envs CraftaxClassic --out $R/bench_step3_cuda_baseB_$H.json

echo "== per-band diagnostics =="
"$VENV/bin/python" scratch/step7/craftax-classic-full/bench_bands.py --device cuda --tag "$H"

echo "== paired verdict =="
"$VENV/bin/python" scratch/step7/craftax-classic-full/paired_verdict.py --host "$H" --device cuda
cat /proc/loadavg
