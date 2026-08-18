#!/bin/bash
#SBATCH --job-name=se-patch-all
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_patch_all_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
UV=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/uv
PY=/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python
SRC=/mnt/vast/home/randall.balestriero/src

echo "### node=$(hostname)"
echo "-- nvdiffrast, NOT editable this time --"
"$UV" pip install --python "$PY" --no-deps "$SRC/nvdiffrast" 2>&1 | tail -8
"$UV" pip freeze --python "$PY" 2>/dev/null | grep -i nvdiffrast || echo "STILL ABSENT"

echo; echo "-- pins must be unchanged --"
"$UV" pip freeze --python "$PY" 2>/dev/null | grep -iE "^(torch|dm-control|mujoco|gymnasium|numpy|jax|scipy)==" | sed 's/^/   /'

echo; echo "-- registry --"
PYTHONPATH=. "$PY" - <<'EOF' 2>&1 | grep -vE "^Gym has|^Please upgrade|^Users of|^See the migration|WARNING:jax"
import json, stable_envs as se
errs = se.envs.import_errors()
print("REGISTRY:", len(se.REGISTRY), "of 196")
print("modules that FAILED:", {k: str(v)[:90] for k, v in errs.items()})
cover = json.load(open("scratch/census/assignment.json"))
want = {e for es in cover.values() for e in es}
missing = sorted(want - set(se.REGISTRY))
print(f"unreachable: {len(missing)}")
for m in missing: print("  ", m)
EOF
echo "### patch done"
