#!/bin/bash
# The conformance suite for the DM Control family, in the interpreter that can import dm_control.
#
# WHY THIS EXISTS. The primary suite interpreter (miniconda3/envs/test) carries mujoco 3.9.0, and NO
# dm_control release works with it -- the module raises at import with "'MjModel' object has no
# attribute 'flex_bandwidth'". stable_envs reports that as an import error and carries on, so the
# 16 DM Control envs are simply absent from the registry there and every test parameterised over them
# collects ZERO cases. The suite was green and the family had no reference-based parity coverage at
# all. That is not an EGL problem and no amount of context accounting would have found it.
#
# MEASURED, venvs/dmc (mujoco 3.11.0 + dm_control 1.0.44), gb-west-a1-gpu-074: 208 passed, 48 skipped,
# 2.4 min, 16 EGL contexts created and 0 released. All 11 parity-tier domains pass
# test_render_parity_with_official_renderer; the 5 trackcom domains are VISUAL tier and pass their own
# declared fidelity. There is no EGL abort in this interpreter.
#
# SE_REQUIRE_ENV_MODULES turns "dm_control did not import" from a printed note into a failure, so a
# run launched to cover this family cannot come back green having covered nothing.
#SBATCH --job-name=se-gpu-dmc
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# EXCLUDED NODES, and they must be the same list in all three launchers here. Every one of these has
# been measured returning DIFFERENT PIXELS for the same scene from its own offscreen EGL rasterizer,
# before any stable-envs code is involved -- 004, 058 and 068 are the ones caught red-handed, and
# gpu-058 was caught twice, in the MetaWorld and OGBench
# investigations; the rest were condemned by the scheduler during a run. A parity number measured on any of them is meaningless, and the failure reads as a
# state-sufficiency violation in the renderer, which is where an afternoon goes. The node gate in
# tests/_runner/node_health.py catches a node that goes bad DURING a run and exits 78; this list
# is the ones already known. tests/unit/test_repo_contracts.py asserts the three lists agree.
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_gpu_dmc_%j.out
set -uo pipefail
# Repo root is hardcoded: $BASH_SOURCE is the staged copy under SLURM, not this file.
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export SE_REQUIRE_ENV_MODULES=dmcontrol
PY=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python

RUN=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_gpu_dmc_${SLURM_JOB_ID:-manual}
mkdir -p "$RUN"

"$PY" -u -m tests._runner.supervisor \
    --rundir "$RUN" --json "$RUN/result.json" \
    -- tests/gpu -v -rs --durations=25 \
    -k "DMC or required_env_modules or registry_is_not_empty or parity_floors" "$@"
rc=$?
echo "se-gpu-dmc job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc artifacts=$RUN"
exit $rc
