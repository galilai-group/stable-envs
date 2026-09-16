#!/bin/bash
#SBATCH --job-name=se8-variations-fetch-gpu
# 3090-gcondo is the preferred queue, but the condo GRES budget is held by the simclr/lejepa
# campaign: the first submission (6451876) pended on QOSGrpGRES with no start estimate and was
# cancelled and re-submitted to --partition=gpu --qos=norm-gpu, which is the documented fallback.
# Swap the two SBATCH lines below to run it either way; the card constraint and the node
# exclusions are the same in both.
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
# gpu2108/gpu2113: the official OGBench renderer is nondeterministic at 1/255 on these two.
# gpu2110/gpu2116: Step-7 exclude list. Carried forward unchanged.
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
#
# Step-8 Workstream B evidence: the twelve Fetch VariationMaps, in ONE allocation on ONE card.
#
# Fetch renders only through mujoco_warp on cuda, so every claim that has a PIXEL in it has to run
# here. Three stages, each guarded so none can abort another:
#
#   1. gpu_evidence.py in se-s3b-fetch311 -- the interpreter the frozen Fetch corpora were measured
#      in (mujoco/mujoco_warp 3.11, gymnasium_robotics 1.4.2). Registered default byte-identical
#      through the mapping, unmapped leaves moving nothing, mapped leaves moving only their own
#      slot, and swm's light.intensity at the model's own 0.8 reproducing the default frame.
#   2. the same script against $SCR/se8/baseline on the SAME card in the SAME job, so
#      "no pixel moved" is a measurement rather than an argument from the diff. This workstream adds
#      a new module and one registration line and touches no renderer, so the expected answer is
#      0/255 -- which is exactly why it is cheap to prove instead of asserting.
#   3. tests/unit/test_variation_maps_fetch.py in se6-verify, which is the only interpreter here
#      with pytest. It has stable_worldmodel and no gymnasium_robotics, so tier 1 (the table) runs
#      and tiers 2/3 skip; tier 2 is covered locally against the live wrapper and tier 3 is what
#      stage 1 measures directly.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export WARP_CACHE_PATH=$SCR/.cache/warp TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1

TREE=${TREE:-$SCR/se8/variations-fetch}
BASE=${BASE:-$SCR/se8/baseline}
FETCHPY=$SCR/venvs/se-s3b-fetch311/bin/python
VERIFYPY=$SCR/venvs/se6-verify/bin/python
export STABLE_ENVS_GIT_COMMIT=$(cat "$TREE/.git_sha" 2>/dev/null || echo unknown)

echo "=============================================================================="
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) tree=$TREE sha=$STABLE_ENVS_GIT_COMMIT"
echo "date=$(date -u +%FT%TZ) partition=${SLURM_JOB_PARTITION:-?} load=$(cut -d' ' -f1-3 /proc/loadavg)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no gpu"
echo "=============================================================================="

cd "$TREE" || exit 1
mkdir -p examples/results/step8/variations-fetch

echo "########## STAGE 1 :: mapping evidence, se-s3b-fetch311 :: $(date -u +%FT%TZ)"
PYTHONPATH=$TREE $FETCHPY scratch/step8/variations-fetch/gpu_evidence.py
echo "STAGE1 rc=$?"

echo "########## STAGE 2 :: default frames vs the Step-8 baseline tree :: $(date -u +%FT%TZ)"
PYTHONPATH=$TREE BASE_TREE=$BASE $FETCHPY scratch/step8/variations-fetch/pixel_parity.py
echo "STAGE2 rc=$?"

echo "########## STAGE 3 :: unit tests, se6-verify :: $(date -u +%FT%TZ)"
PYTHONPATH=$TREE $VERIFYPY -m pytest tests/unit/test_variation_maps_fetch.py -q --no-header \
  -p no:cacheprovider 2>&1 | grep -E "^(tests/\S+ (PASSED|FAILED|SKIPPED)|[0-9]+ (passed|failed).*|FAILED |ERROR )" \
  | tee examples/results/step8/variations-fetch/unit_tests.txt
echo "STAGE3 rc=$?"

echo "########## DONE :: $(date -u +%FT%TZ)"
ls -la examples/results/step8/variations-fetch/
