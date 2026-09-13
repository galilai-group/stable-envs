#!/bin/bash
#SBATCH --job-name=se6-raycast3d-2cp
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2c' (b): _SCENE_CACHE evicts one entry instead of all of them.
#
# Three things, in one job on one card:
#   1. the cache-pressure scenario that made Step 4 observe 1 -- run against the pristine baseline
#      (wholesale clear) and against this branch (LRU). The baseline leg is the control: if it also
#      returns 0 the probe is not reproducing anything and the after-number means nothing.
#   2. the 11 OGBench configs replayed against their frozen digests, before and after, in pixels.
#   3. the host-determinism probe the conformance suite already owns, run on the same configs on
#      this node, because RAYCAST-2c' (a) is an argument about what that probe is FOR.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/2cp
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
rm -rf "$D"; mkdir -p "$D/before" "$D/after"

OGB="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate"
FAIL2="AntmazeTeleportExplore HumanoidmazeMediumNavigate"

echo "######## 1. the cache-pressure scenario: baseline (wholesale clear) vs branch (LRU)"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/cache_pressure_probe.py --device cuda --batch 64 \
    --configs $FAIL2 AntmazeTeleportNavigate HumanoidmazeLargeNavigate PointmazeLargeNavigate \
    --out "$OUT/cache_pressure_baseline.json" || echo "PRESSURE_BASE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/cache_pressure_probe.py --device cuda --batch 64 \
    --configs $FAIL2 AntmazeTeleportNavigate HumanoidmazeLargeNavigate PointmazeLargeNavigate \
    --out "$OUT/cache_pressure_branch.json" || echo "PRESSURE_BRANCH rc=$?"

echo "######## 2. the 11 OGBench configs, frozen digests and pixels, before vs after"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $OGB \
    --dump-dir "$D/before" --out "$OUT/replay/2cp_ogb_before_baseline.json" || echo "B rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cuda --configs $OGB \
    --dump-dir "$D/after" --out "$OUT/replay/2cp_ogb_after.json" || echo "A rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/before" "$D/after" \
    --out "$OUT/replay/2cp_ogb_pixel_diff.json" || echo "D rc=$?"

echo "######## 3. the suite's own host-determinism probe, on this node, n=4 and n=32"
HOSTPROBE_OUT="$OUT/host_rasterizer_determinism.json" PYTHONPATH=$WS $PY - <<'PROBE'
import json, os, socket
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import stable_envs as se
from stable_envs.testing import conformance as C

out = {"host": socket.gethostname(), "what": (
    "conformance.check_reference_renderer_is_stable is 'is THIS MACHINE's offscreen rasterizer "
    "deterministic at all?'. It renders the same states twice through ONE reference env. Two samples "
    "give it ~50% power against a per-bake coin flip, so it is repeated here 12 times per config and "
    "the number of distinct outcomes is reported beside it."), "configs": {}}
for n in ("AntmazeTeleportExplore", "HumanoidmazeMediumNavigate", "PointmazeTeleportNavigate",
          "AntmazeGiantNavigate", "PointmazeLargeNavigate"):
    cls = se.REGISTRY[n]
    rec = {"repeats": []}
    for _ in range(12):
        try:
            rec["repeats"].append(C.check_reference_renderer_is_stable(cls, 224)["max_abs_diff"])
        except Exception as exc:                                            # noqa: BLE001
            rec["error"] = repr(exc)[:200]; break
    if rec["repeats"]:
        rec["max"] = int(max(rec["repeats"]))
        rec["n_nonzero"] = int(sum(1 for v in rec["repeats"] if v))
        rec["fires_in"] = f"{rec['n_nonzero']}/{len(rec['repeats'])}"
    out["configs"][n] = rec
    print(f"HOSTPROBE {n:30s} {rec.get('fires_in')} max={rec.get('max')}", flush=True)
open(os.environ["HOSTPROBE_OUT"], "w").write(json.dumps(out, indent=2))
print("wrote", os.environ["HOSTPROBE_OUT"], flush=True)
PROBE

echo "2CP done"
