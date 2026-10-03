#!/bin/bash
#SBATCH --job-name=dmc-source-first-three
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-source-comparison-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-source-comparison-v1-20261003
EXTRAS=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-smoke-v3-20261003
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
cd "$BASE/code"
export PYTHONPATH="$EXTRAS/extras:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" "$EXTRAS" <<'VERIFY'
import hashlib, importlib.metadata as md, json, pathlib, sys
root, extras = map(pathlib.Path, sys.argv[1:])
candidate = json.loads((root / 'candidate_manifest.json').read_text())
for rel, digest in candidate['files'].items():
    assert hashlib.sha256((root / 'code' / rel).read_bytes()).hexdigest() == digest, rel
source = json.loads((root / 'source_manifest.json').read_text())
for rel, digest in source['files'].items():
    assert hashlib.sha256((root / 'source' / rel).read_bytes()).hexdigest() == digest, rel
profile = json.loads((extras / 'dependency-install.json').read_text())
for name, record in profile.items():
    distribution = md.distribution(name)
    assert distribution.version == record['version']
    for rel, digest in record['files'].items():
        assert hashlib.sha256(distribution.locate_file(rel).read_bytes()).hexdigest() == digest, (name, rel)
print('Frozen candidate/source and existing renderer extras verified', len(candidate['files']), len(source['files']), flush=True)
VERIFY
nvidia-smi
lscpu
DOMAINS=(acrobot pendulum cartpole)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
"$PY" scripts/benchmarks/online_stepping/dmc_source_oscar.py --domain "$DOMAIN" --source-root "$BASE/source" --source-manifest "$BASE/source_manifest.json" --batches 64 --order-offset "$SLURM_ARRAY_TASK_ID" --output "$BASE/results/$DOMAIN"
"$PY" scripts/benchmarks/online_stepping/summarize_dmc_source.py "$BASE/results/$DOMAIN" "$BASE/tables/$DOMAIN" --source "$PWD" --source-root "$BASE/source" --source-manifest "$BASE/source_manifest.json" --batches 64
