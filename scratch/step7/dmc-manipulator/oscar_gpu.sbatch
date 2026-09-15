#!/bin/bash
#SBATCH --job-name=se7-dmc-manipulator-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The CUDA arm: the new cell on a 3090, with the BEFORE/AFTER of the 15 pre-existing DM Control cells
# measured in the SAME JOB on the SAME CARD -- a before on a 3090 and an after on an L40S is not a
# before/after (AGENT_COMMON addendum). $SCR/se7/baseline must be a pristine copy of 76dc511.
#
# ONE GPU job per agent; norm-gpu allows ~2 concurrent across ALL agents. Never cancel a job you did
# not submit, and never a simclr_cifar10 job.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=dmc-manipulator
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/se7/$WS/examples/results/step7/$WS
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# BEFORE: the 15 pre-existing cells on the baseline tree, same card, same job.
# The PROBE is taken from this branch by ABSOLUTE path -- $SCR/se7/baseline is a pristine 76dc511
# and does not contain scratch/step7/, so the relative path this script first used resolved to
# nothing and the BEFORE arm would have been skipped by the `|| true` with SLURM still COMPLETED.
# PYTHONPATH still points at the baseline, which is the thing being measured.
cd "$SCR/se7/baseline"
PYTHONPATH=$SCR/se7/baseline $PY "$SCR/se7/$WS/scratch/step7/$WS/probe_identity.py" \
    --out "$OUT/gpu/identity_base_cuda.json" --device cuda

# AFTER: the same 15 plus the new cell, on this branch
cd "$SCR/se7/$WS"
PYTHONPATH=$SCR/se7/$WS $PY scratch/step7/$WS/probe_identity.py \
    --out "$OUT/gpu/identity_head_cuda.json" --device cuda

# the new cell's GPU conformance + parity, and its throughput sweep
PYTHONPATH=$SCR/se7/$WS $PY -m pytest tests/gpu/test_conformance.py \
    -k DMCManipulatorBringBall -q 2>&1 | tail -40
PYTHONPATH=$SCR/se7/$WS $PY scratch/step6/profile_note.py --env DMCManipulatorBringBall \
    --device cuda --batch 1 16 64 256 1024 4096 --repeat 10 --warmup 3 \
    --workstream "step7/$WS" --out "$OUT/profile"
set +x
ls -la "$OUT/gpu" "$OUT/profile" || true
