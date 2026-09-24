#!/bin/bash
#SBATCH --job-name=se-craftax
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090|l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
bundle=${1:?absolute immutable bundle directory required}
cd "$bundle"
sha256sum -c SHA256SUMS
run="$bundle/run-${SLURM_JOB_ID:?}"
mkdir "$run"
mkdir "$run/source" "$run/oracle" "$run/evidence"
tar -xzf source.tar.gz -C "$run/source"
tar -xzf oracle.tar.gz -C "$run/oracle"
cp bundle.json SHA256SUMS "$run/evidence/"
nvidia-smi > "$run/evidence/nvidia-smi.txt"
export UV_CACHE_DIR=/oscar/scratch/sboughan/.cache/uv-craftax
export XDG_CACHE_HOME="$run/cache" MPLCONFIGDIR="$run/cache/matplotlib"
export JAX_PLATFORMS=cuda CRAFTAX_JAX_BACKEND=gpu CRAFTAX_DEVICE=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=false
export OMP_NUM_THREADS=8 MPLBACKEND=Agg
export PYTHONPATH="$run/source" SWM_ROOT="$run/oracle"
uv=/oscar/home/sboughan/.local/bin/uv
"$uv" venv --python /oscar/scratch/sboughan/venvs/se7-ogb/bin/python "$run/venv"
py="$run/venv/bin/python"
"$uv" pip sync --python "$py" --require-hashes --torch-backend cu126 \
  "$run/source/scripts/recovery/profiles/craftax-gpu.txt" > "$run/evidence/install.log" 2>&1
"$uv" pip freeze --python "$py" > "$run/evidence/freeze.txt"
cd "$run/source"
failures=0
for name in CraftaxClassicPixels CraftaxClassicSymbolic CraftaxPixels CraftaxSymbolic; do
  selector="$name"
  count=28
  if [[ "$name" == CraftaxClassicSymbolic ]]; then count=17; fi
  if [[ "$name" == CraftaxSymbolic ]]; then
    selector='CraftaxSymbolic or full_symbolic_preserves'
    count=18
  fi
  if "$py" -m pytest -xq tests/unit/test_craftax_semantics.py tests/unit/test_craftax_rendering.py \
       -k "$selector" --basetemp="$run/evidence/$name-pytest-tmp" \
       --junitxml="$run/evidence/$name.xml" > "$run/evidence/$name-tests.log" 2>&1 &&
     "$py" - "$run/evidence/$name.xml" "$count" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).findall('.//testcase')
assert len(cases) == int(sys.argv[2]), (len(cases), sys.argv[2])
assert not any(c.find(tag) is not None for c in cases for tag in ('failure', 'error', 'skipped'))
PY
  then
    if "$py" scripts/recovery/craftax_gate.py --name "$name" --device cuda --performance \
         --output "$run/evidence/$name" > "$run/evidence/$name-arrays.log" 2>&1; then
      echo "CRAFTAX_REGISTRATION_ACCEPTANCE_PASS $name $SLURM_JOB_ID"
    else failures=$((failures + 1)); fi
  else failures=$((failures + 1)); fi
done
cd "$run/evidence"
find . -type f ! -name GPU-SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > GPU-SHA256SUMS
cd "$run"
tar -czf "$bundle/craftax-evidence-$SLURM_JOB_ID.tar.gz" evidence
sha256sum "$bundle/craftax-evidence-$SLURM_JOB_ID.tar.gz"
if (( failures )); then
  echo "CRAFTAX_GPU_GATE_FAILED registrations=$failures"
  exit 1
fi
echo "CRAFTAX_FAMILY_GPU_GATE_PASS $SLURM_JOB_ID"
