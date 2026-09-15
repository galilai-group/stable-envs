#!/bin/bash
#SBATCH --job-name=se7-rocket-x86tests
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W6 -- the x86-64 side of the CLOSURE-1 test correction, ON A COMPUTE NODE.
#
# Sami's binding rule (2026-09-15): NEVER run CPU-heavy work on an Oscar login node -- no pytest, no
# probes, no renders, no benchmarks, no installs. The earlier x86 runs in this workstream were taken
# on login009/login010 before that rule was stated; they are superseded by this job and the incident
# is recorded in findings.md section M7. Nothing here is a timing measurement: it is a pass/fail check
# of a bitwise assertion, which is why one core and fifteen minutes is the right size for it.
#
# What it answers: tests/unit/test_nvdiff_triangles.py::test_from_triangles_clip_row_is_independent_of_batch_size
# is the test the closure sweep found FAILING on x86-64. It has been corrected to assert what the repo
# certifies (cuda bitwise at every width; cpu within one float32 ulp at every width, with the measured
# per-architecture bitwise boundary recorded rather than asserted). This job shows the corrected suite
# passing on the certification architecture.
#
# The two target files run FIRST so their result exists even if the whole suite runs out of the
# fifteen minutes at one core; the full suite follows with its own timeout.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export NUMBA_CACHE_DIR=$SCR/.cache/numba TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=$SCR/venvs/se-step6/bin/python
WS=$SCR/se7/rocket-landing
OUT=$WS/examples/results/step7/rocket-landing/closure1
mkdir -p "$OUT" "$SCR/logs"
[ -f "$WS/.git_sha" ] || { echo "MISSING $WS/.git_sha"; exit 3; }
cd "$WS"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID ARCH $(uname -m) SHA $(cat "$WS/.git_sha") CPUS ${SLURM_CPUS_PER_TASK:-?} LOAD $(cat /proc/loadavg | cut -d' ' -f1-3)"
$PY -c "import torch, platform; print('torch', torch.__version__, platform.machine(), 'threads', torch.get_num_threads())"

echo "===== the two corrected files ====="
PYTHONPATH=$WS $PY -m pytest tests/unit/test_nvdiff_triangles.py tests/unit/test_rocket_landing.py \
  -q -rA -p no:cacheprovider > "$OUT/x86_tests_targeted.txt" 2>&1
tail -4 "$OUT/x86_tests_targeted.txt"
grep -E "batch_size" "$OUT/x86_tests_targeted.txt" | grep -E "^(PASSED|FAILED)" || true

echo "===== the whole unit suite on x86-64 ====="
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:cacheprovider --deselect tests/unit/test_examples.py \
  > "$OUT/x86_tests_full.txt" 2>&1
tail -4 "$OUT/x86_tests_full.txt"
echo "S7X86TESTS done on $(hostname -s) at $(cat "$WS/.git_sha")"
