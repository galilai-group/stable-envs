#!/bin/bash
# stable-envs FULL conformance suite. Requires a CUDA device by design.
#   sbatch scripts/run_gpu_tests.sbatch                 # everything
#   sbatch scripts/run_gpu_tests.sbatch -k PushT        # extra args go straight to pytest
#
# Run through stable_envs.testing.supervisor rather than through pytest directly. The suite has aborted
# itself four times in a week -- SIGABRT/SIGSEGV out of MuJoCo, EGL or the CUDA driver -- and a bare
# pytest reports nothing at all when that happens: the test that died is not named, every test after it
# is never run, and the whole artifact of the run is a shell exit code. The supervisor turns that into
# one CRASHED test and finishes the run. It costs one extra process and nothing else when nothing
# crashes. See that module's docstring for why --forked cannot be used here.
#
# MEASURED, H200 (gb-west-a1-gpu-021): 2580 passed, 648 skipped, 0 failed, 1 segment, 45.5 min, 3228
# collected over a registry of 64, 133 EGL contexts created and 0 released. The wall time tracks the
# registry and the suite size, so it is quoted with both. scripts/run_smoke_tests.sbatch is the
# pre-commit subset (3.4 min, or 0.9 min with --fast) and scripts/run_unit_tests.sbatch is the
# CPU-only contract subset (2.9 min).
#SBATCH --job-name=se-gpu
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
# TWO GPUs, and the second one is not spare capacity. tests/gpu/test_distributed.py asserts that the
# union of what two ranks render is BIT-IDENTICAL to one process, and that the same states give the
# same pixels on cuda:0 and cuda:1 -- the two properties a device-blind cache breaks. Under
# `--gres=gpu:1` all six of those tests skipped, and had skipped every run this suite has ever made:
# they are green on 2 GPUs, so nothing was broken, but nothing was checked either.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
# EXCLUDED NODES, and they must be the same list in all three launchers here. Every one of these has
# been measured returning DIFFERENT PIXELS for the same scene from its own offscreen EGL rasterizer,
# before any stable-envs code is involved -- 004, 058 and 068 are the ones caught red-handed, and
# gpu-058 is written up twice (docs/research/metaworld.md and docs/research/ogbench.md both
# record it); the rest were condemned by the scheduler during a run. A parity number measured on any of them is meaningless, and the failure reads as a
# state-sufficiency violation in the renderer, which is where an afternoon goes. The node gate in
# stable_envs/testing/node_health.py catches a node that goes bad DURING a run and exits 78; this list
# is the ones already known. tests/unit/test_repo_contracts.py asserts the three lists agree.
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_gpu_%j.out
set -uo pipefail
# Repo root is hardcoded: $BASH_SOURCE is the staged copy under SLURM, not this file.
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export STABLE_ENVS_HUMANOIDBENCH=1        # the family does not register by default; see pyproject
# THE interpreter, holding 191 of 196 envs. It used to be miniconda3/envs/test, whose registry held
# 64: every test parameterised over the other 132 skipped with an individually honest reason, and the
# run was green having never exercised two thirds of the library. Built by
# Built by the documented extras recipe; the four --no-deps/--override packages it needs
# are documented on the `all` extra in pyproject.toml.
PY=/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python

# What this run is REQUIRED to reach. Without it a venv that lost a dependency skips its whole
# assignment and reports success, which is the failure this gate exists to make impossible.
# stable_envs/testing/gate.py subtracts the one declared exclusion (Overcooked, which cannot share an
# interpreter -- jaxmarl needs scipy<=1.12 against jax's >=1.14) and nothing else.
export SE_REQUIRE_ENVS=$("$PY" -m stable_envs.testing.gate)
echo "se-gpu gate covers $(tr ',' '\n' <<<"$SE_REQUIRE_ENVS" | grep -c .) envs"

# Artifacts live on shared storage, not in the node's /tmp: a crashed or preempted job still leaves the
# event log and the aggregated json behind to be read afterwards.
RUN=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_gpu_${SLURM_JOB_ID:-manual}
mkdir -p "$RUN"

"$PY" -u -m stable_envs.testing.supervisor \
    --rundir "$RUN" --json "$RUN/result.json" \
    -- tests/ -v -rs --durations=25 "$@"
rc=$?

# One line, greppable, at the very end of the log. Exit 78 means the NODE was unfit, not the code --
# rerun elsewhere and do not go looking for a renderer bug.
echo "se-gpu job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc artifacts=$RUN"
exit $rc
