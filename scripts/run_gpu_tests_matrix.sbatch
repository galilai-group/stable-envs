#!/bin/bash
# stable-envs FULL conformance suite, in ONE interpreter of the set cover.
#
#   sbatch --job-name=se-mx-dmc scripts/run_gpu_tests_matrix.sbatch dmc
#   scripts/submit_test_matrix.sh                      # all of them
#
# WHY THIS EXISTS. scripts/run_gpu_tests.sbatch runs in a single interpreter
# (miniconda3/envs/test), whose registry holds 64 of the library's 196 envs. Every test
# parameterised over the other 132 SKIPPED, and the run was green: 2580 passed, 648 skipped. The
# skips were honest one-by-one -- "reference env not installed", "not registered in this
# interpreter" -- and collectively they meant two thirds of the product had never been through the
# release gate. No single interpreter can fix that: robocasa pins mujoco==3.3.1, mujoco_warp needs
# >=3.11, and dm_control needs a build in between, so the deps are mutually exclusive BY CONSTRUCTION
# and the suite has to be a fan-out or it is a subset.
#
# The cover is scratch/census/assignment.json -- the same one the throughput census uses, so the
# envs that were MEASURED and the envs that are TESTED are the same set by construction rather than
# by coincidence. 7 interpreters, 196 envs, no overlap.
#
# SE_REQUIRE_ENVS is what stops this degenerating back into the thing it replaces: it names the envs
# this interpreter was launched to cover, and the run FAILS if any of them is not reachable here.
# Without it a venv that lost a dependency would skip its whole assignment and report green, which is
# the exact failure this file exists to end.
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
# Same list as the other three launchers; tests/unit/test_repo_contracts.py asserts they agree.
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_mx_%x_%j.out
set -uo pipefail
VENV=$1; shift

cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export STABLE_ENVS_HUMANOIDBENCH=1
PY=/mnt/vast/home/randall.balestriero/venvs/$VENV/bin/python
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 2; }

# The envs this interpreter OWNS in the cover. Read here rather than hardcoded so the cover has one
# definition; a venv absent from the assignment is a launch error, not an empty run.
export SE_REQUIRE_ENVS=$("$PY" - "$VENV" <<'EOF'
import json, sys
a = json.load(open("scratch/census/assignment.json"))
if sys.argv[1] not in a:
    sys.exit(f"{sys.argv[1]} is not in the cover: {sorted(a)}")
print(",".join(a[sys.argv[1]]))
EOF
) || { echo "could not resolve the assignment for $VENV"; exit 2; }
echo "se-matrix venv=$VENV owns $(tr ',' '\n' <<<"$SE_REQUIRE_ENVS" | wc -l) envs"

RUN=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_mx_${VENV}_${SLURM_JOB_ID:-manual}
mkdir -p "$RUN"

"$PY" -u -m stable_envs.testing.supervisor \
    --rundir "$RUN" --json "$RUN/result.json" \
    -- tests/ -v -rs --durations=25 "$@"
rc=$?

echo "se-matrix venv=$VENV job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc artifacts=$RUN"
exit $rc
