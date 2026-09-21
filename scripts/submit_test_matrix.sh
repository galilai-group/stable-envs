#!/bin/bash
# Fan the FULL conformance suite out over the interpreters that cover the registry.
#
#   scripts/submit_test_matrix.sh                  # all 7, the whole 196-env library
#   scripts/submit_test_matrix.sh dmc se-pgx       # just these
#   scripts/submit_test_matrix.sh -- -x -k Antmaze # extra args go straight to pytest
#
# One interpreter cannot hold the library: robocasa pins mujoco==3.3.1, mujoco_warp needs >=3.11 and
# dm_control needs a build in between, so the dependency sets are mutually exclusive by construction.
# tests/fixtures/regression/census/assignment.json is the set cover -- 7 interpreters, 196 envs, no overlap -- and it is
# the SAME file the throughput census fans out over, so the envs that are measured and the envs that
# are tested are one set rather than two that drift.
#
# Each job runs on 2 GPUs so the distributed tests execute rather than skip, and exports
# SE_REQUIRE_ENVS for its own slice so an interpreter that lost a dependency FAILS instead of
# skipping its whole assignment and reporting green.
set -uo pipefail
cd "$(dirname "$0")/.." || exit

PYTEST_ARGS=()
VENVS=()
for a in "$@"; do
    if [ "$a" = "--" ]; then shift; PYTEST_ARGS=("$@"); break; fi
    VENVS+=("$a"); shift
done
if [ ${#VENVS[@]} -eq 0 ]; then
    mapfile -t VENVS < <(python3 -c "import json;print('\n'.join(sorted(json.load(open('tests/fixtures/regression/census/assignment.json')))))")
fi

for v in "${VENVS[@]}"; do
    echo -n "$v: "
    sbatch --job-name="se-mx-$v" scripts/run_gpu_tests_matrix.sbatch "$v" "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
done
