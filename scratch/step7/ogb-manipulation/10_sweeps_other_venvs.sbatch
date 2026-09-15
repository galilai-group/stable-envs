#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-sweepvenvs
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Reviewer item 5: RoboCasa, MetaWorld and HumanoidBench register conditionally, call prepare_model
# (RoboCasa also texture_gaps), and appear in NEITHER sweep -- they do not build in se7-ogb. Run the
# same sweeps in the venvs where they DO build, and record which still do not, so the coverage claim
# is a denominator rather than an impression.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
OUT=$WS/examples/results/step7/ogb-manipulation/m2/venvs
mkdir -p "$OUT"
set -x
cd "$WS"
for V in se-step5 se-step6 se7-ogb; do
  P=$SCR/venvs/$V/bin/python
  [ -x "$P" ] || { echo "SKIP $V: no interpreter"; continue; }
  # EGL on a batch node is the wrong backend (it has no GPU); probe as the other jobs do.
  eval "$(PYTHONPATH=$WS:$SCR/se7/swm-src "$P" \
    scratch/step7/ogb-manipulation/gl_probe.py egl osmesa glfw 2>/dev/null | tail -1)" || true
  echo "=== $V  MUJOCO_GL=$MUJOCO_GL"
  mkdir -p "$OUT/$V"
  PYTHONPATH=$WS:$SCR/se7/swm-src "$P" scratch/step7/ogb-manipulation/sweep_spot_semantics.py \
    --out "$OUT/$V" --bake || echo "SWEEP $V rc=$?"
  PYTHONPATH=$WS:$SCR/se7/swm-src "$P" scratch/step7/ogb-manipulation/sweep_texture_gaps.py \
    --out "$OUT/$V" || echo "TEXSWEEP $V rc=$?"
  # Which of the three named families reached an MjModel here, and which did not?
  PYTHONPATH=$WS:$SCR/se7/swm-src "$P" - "$OUT/$V" <<'PY' || true
import json, sys, pathlib
import stable_envs as se
from stable_envs.envs import import_errors
want = ("RoboCasa", "MetaWorld", "Meta", "Humanoid", "Sawyer")
ids = sorted(se.REGISTRY)
hit = [n for n in ids if any(w.lower() in n.lower() for w in want)]
doc = {"n_registry_ids": len(ids), "named_family_ids": hit,
       "import_errors": import_errors()}
pathlib.Path(sys.argv[1], "families_present.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
print("registry ids:", len(ids), "| named-family ids:", len(hit), hit[:8])
print("import errors:", {k: v[:80] for k, v in import_errors().items()})
PY
done
ls -R "$OUT" | tail -30
echo OK
