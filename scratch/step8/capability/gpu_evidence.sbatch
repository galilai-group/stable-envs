#!/bin/bash
#SBATCH --job-name=se8-capability-plate-texels
#SBATCH --partition=3090-gcondo
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# The GPU evidence for step8/capability-plate-texels, in ONE job as the brief requires.
# NOT RUN in the authoring session: Oscar SSH returned "Permission denied (keyboard-interactive)"
# on 26 attempts over ~2 h and allocation 6443480 closed at ~22:45 EDT. No job id exists.
#
# Before submitting, rsync this branch to $SCR/se8/capability (no --delete):
#   rsync -az --exclude .git --exclude __pycache__ --exclude '*.pyc' \
#     /Users/samib/stable-envs-step8-capability/ \
#     sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se8/capability/
#
# The affected family is the EIGHT DMControlSuite cells that gained floor.color. One representative
# per domain is NOT sufficient here: PLATE_PATHS is every leaf and the plate is baked per
# (domain, task, camera, size, appearance), so these cells share the CODE and not the bake.
set -euo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
mkdir -p "$XDG_CACHE_HOME" "$SCR/logs"

WS=$SCR/se8/capability
BASE=$SCR/se8/baseline
OUT=$WS/examples/results/step8/capability/gpu
mkdir -p "$OUT"
# se-vis311 and se-step6 both carry the DM Control stack; the Step-7 DMC cells used them. Pick the
# one that has mujoco_warp at this node's driver, as the Step-7 drivers do.
PY=$SCR/venvs/se-step6/bin/python
$PY -c "import mujoco_warp" 2>/dev/null || PY=$SCR/venvs/se-vis311/bin/python

CELLS=DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCFingerTurnHard,DMCManipulatorBringBall,DMCPendulumSwingup,DMCReacherHard

nvidia-smi --query-gpu=name,driver_version --format=csv | tee "$OUT/gpu.txt"
{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "sha $(git -C "$WS" rev-parse HEAD 2>/dev/null || echo unknown)";
  $PY -c "import torch,mujoco;print('torch',torch.__version__,'mujoco',mujoco.__version__)"; } | tee "$OUT/provenance.txt"

# (a) default-frame digest identity, AFTER vs the pristine baseline tree, same job, same card.
for T in "$BASE:before" "$WS:after"; do
  P=${T%%:*}; N=${T##*:}
  ( cd "$P" && PYTHONPATH="$P" $PY "$WS/scratch/step8/capability/declaration_delta.py" \
      "$OUT/declaration_$N.json" ) || echo "declaration_$N FAILED"
done

# (b) bake-vs-recompile byte equality on the GPU device path, three probe colours.
for P in 0 1 2; do
  ( cd "$WS" && PYTHONPATH="$WS" $PY scratch/step8/capability/bake_vs_recompile.py \
      --mode recompile --probe $P --cells "$CELLS" --out "$OUT/rc$P.json" )
  ( cd "$WS" && PYTHONPATH="$WS" $PY scratch/step8/capability/bake_vs_recompile.py \
      --mode substitute --probe $P --cells "$CELLS" --out "$OUT/sb$P.json" )
  ( cd "$WS" && PYTHONPATH="$WS" $PY scratch/step8/capability/bake_vs_recompile.py \
      --compare "$OUT/rc$P.json" "$OUT/sb$P.json" --out "$OUT/bake_vs_recompile_gpu_probe$P.json" )
done

# (c) isolation, custom->default sequences and Gate A with the new style dims, through the existing
#     Workstream-C drivers (see $SCR/se8/gputests.sh and $SCR/se8/run_final4.sh for the C protocol).
( cd "$WS" && PYTHONPATH="$WS" $PY scratch/step8/variations-newcells/gpu_colors.py \
    --cells "$CELLS" --leaf floor.color --out "$OUT/isolation.json" ) || echo "isolation driver needs --cells wiring for DMC"

# (d) paired throughput at B=1 and B=4096 at the DEFAULT style, plus ONE high-cardinality floor
#     batch: the risk is plate-cache thrash (_SCENE_CACHE_MAX = 96, one bake per distinct QUANTISED
#     colour), not kernel cost.
( cd "$WS" && PYTHONPATH="$WS" $PY - <<'PY' | tee "$OUT/throughput.txt"
import json, os, time
import numpy as np, torch
from stable_envs.core.env import make
rows = []
for cell in os.environ.get("CELLS", "DMCCartpoleBalance").split(","):
    for B in (1, 4096):
        e = make(cell, batch=B, size=84, device="cuda")
        st = e.sample_states(B, np.random.default_rng(0))
        for label, var in (("default", None),
                           ("random_floor", {"floor.color": np.random.default_rng(1).random((2, 3))})):
            e.render(st, variation=var); torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                e.render(st, variation=var)
            torch.cuda.synchronize()
            rows.append({"cell": cell, "B": B, "style": label, "fps": 5 * B / (time.perf_counter() - t)})
            print(rows[-1], flush=True)
print(json.dumps(rows))
PY
) || echo "throughput FAILED"

echo "DONE $(date -Is)"
