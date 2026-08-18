#!/bin/bash
# stable-envs UNIT subset: no GPU work, no reference env, no checkpoint. Seconds, not minutes.
#   sbatch scripts/run_unit_tests.sbatch                    # tests/unit + the safety net
#   sbatch scripts/run_unit_tests.sbatch -k style           # extra args go straight to pytest
#
# Everything here runs on CPU and is the loop to use while editing a contract rather than a kernel:
# the style batching rule, the dataset indexing, the policy/rollout API, the fidelity and decoder
# declarations, the corrector's refusals, the benchmark scripts' guards, and the repo's own citations
# and docs nav. It still asks the scheduler for a node -- nothing in this project runs on the login
# node, and `import torch` plus the env auto-discovery is not free.
#
# It does NOT replace scripts/run_gpu_tests.sbatch. Nothing here renders a frame, so a green run says
# the contracts hold and says nothing about the pixels.
#SBATCH --job-name=se-unit
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# EXCLUDED NODES, and they must be the same list in all three launchers here. Every one of these has
# been measured returning DIFFERENT PIXELS for the same scene from its own offscreen EGL rasterizer,
# before any stable-envs code is involved -- 004, 058 and 068 are the ones caught red-handed, and
# gpu-058 is written up twice (docs/research/metaworld.md and docs/research/ogbench.md both
# record it); the rest were condemned by the scheduler during a run.
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_unit_%j.out
set -uo pipefail
# Repo root is hardcoded: $BASH_SOURCE is the staged copy under SLURM, not this file.
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python

RUN=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_unit_${SLURM_JOB_ID:-manual}
mkdir -p "$RUN"

"$PY" -u -m stable_envs.testing.supervisor \
    --rundir "$RUN" --json "$RUN/result.json" \
    -- tests/unit tests/test_safety_net.py -v -rs --durations=15 "$@"
rc=$?
echo "se-unit job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc artifacts=$RUN"
exit $rc
