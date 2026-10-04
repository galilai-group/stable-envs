#!/bin/bash
#SBATCH --job-name=rocket-unit-complete-bundle
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/rocket-unit-retry-v1-20261003/unit-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/rocket-unit-retry-v1-20261003
ORIGINAL=/oscar/scratch/sboughan/se-semantic-recovery/rocket-current-gate-v1-20261003
RUN="$BASE/results/$SLURM_JOB_ID"
mkdir -p "$RUN"
cd "$RUN"
tar -xzf "$ORIGINAL/bundle.tar.gz"
mkdir -p code/tests/fixtures/regression
cp "$BASE/support_matrix.json" code/tests/fixtures/regression/
cd code
export PYTHONPATH="$PWD:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se7-rocket/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python
"$PY" - "$ORIGINAL" "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
original,base=map(pathlib.Path,sys.argv[1:])
for rel,digest in json.loads((original/'candidate_manifest.json').read_text())['files'].items():
    assert hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()==digest,rel
for rel,digest in json.loads((base/'fixture-manifest.json').read_text()).items():
    assert hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()==digest,rel
print('Original candidate unchanged; missing test fixture supplied and verified',flush=True)
VERIFY
"$PY" -m pytest tests/unit/test_rocket_replay_semantics.py -q --junitxml="$RUN/unit.xml" > "$RUN/unit.log" 2>&1
