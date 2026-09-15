#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-combined
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Three things that all need a GPU, in one job because only one may be in flight.
#
# (1) Reviewer item 5, PROPERLY. The batch-node sweep (6398427) could not cover RoboCasa, MetaWorld
#     or HumanoidBench: all three refuse at IMPORT without CUDA, so a CPU node can never reach them
#     however many venvs it tries. On a GPU node they build, and the sweeps then say something about
#     the three families the reviewer named instead of recording them as import errors.
# (2) The Fetch leg of the post-review byte-identity replay, which 6398465 lost to a bug in this
#     script rather than to the correction: the Fetch config was not in SPOTCFG, so no state file was
#     written for it and both legs failed on FileNotFoundError.
# (3) Re-measure the OGBCube/OGBScene cells after the state layout changed (qvel dropped) and the
#     Scene gained its texture bake.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
WS=$SCR/se7/ogb-manipulation
BASE=$SCR/se7/baseline
VENV=$SCR/venvs/se7-ogb
FVENV=$SCR/venvs/se-s3b-fetch311
OUT=$WS/examples/results/step7/ogb-manipulation
export PYTHONPATH=$WS:$SCR/se7/swm-src
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true
eval "$(PYTHONPATH=$PYTHONPATH "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py --require egl 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: EGL unusable on $(hostname)"; exit 9; }

# ---- (1) the sweeps, on a node where the conditional families build -------------------------
mkdir -p "$OUT/m2/gpu"
for V in se7-ogb se-step5; do
  P=$SCR/venvs/$V/bin/python
  [ -x "$P" ] || continue
  echo "=== sweeps on $V (GPU node)"
  mkdir -p "$OUT/m2/gpu/$V"
  PYTHONPATH=$WS:$SCR/se7/swm-src "$P" scratch/step7/ogb-manipulation/sweep_spot_semantics.py \
    --out "$OUT/m2/gpu/$V" --bake || echo "SWEEP $V rc=$?"
  PYTHONPATH=$WS:$SCR/se7/swm-src "$P" scratch/step7/ogb-manipulation/sweep_texture_gaps.py \
    --out "$OUT/m2/gpu/$V" || echo "TEXSWEEP $V rc=$?"
  PYTHONPATH=$WS:$SCR/se7/swm-src "$P" - "$OUT/m2/gpu/$V" <<'PY' || true
import json, sys, pathlib
import stable_envs as se
from stable_envs.envs import import_errors
ids = sorted(se.REGISTRY)
want = ("robocasa", "metaworld", "sawyer", "humanoidbench", "h1")
hit = [n for n in ids if any(w in n.lower() for w in want)]
doc = {"n_registry_ids": len(ids), "named_family_ids": hit, "import_errors": import_errors()}
pathlib.Path(sys.argv[1], "families_present.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
print("GPU-node registry ids:", len(ids), "| robocasa/metaworld/humanoidbench ids:", len(hit), hit[:6])
print("still-unimportable:", sorted(import_errors()))
PY
done

# ---- (2) the Fetch leg the last job lost ------------------------------------------------------
FOUT=$OUT/m2/spot
mkdir -p "$FOUT"
for TREE in "$BASE" "$WS"; do
  TAG=$([ "$TREE" = "$WS" ] && echo after || echo before)
  PYTHONPATH=$TREE "$FVENV/bin/python" - "$TAG" "$FOUT" <<'PY'
import sys, numpy as np, stable_envs as se
tag, out = sys.argv[1], sys.argv[2]
for name in ("FetchReach", "FetchPickAndPlace"):
    try:
        st = np.load(f"/oscar/scratch/sboughan/se7/ogb-manipulation/examples/results/step4/states/{name}.npz")
        s = st[list(st.keys())[0]][:8]
        env = se.make(name, batch=len(s), device="cuda"); env.set_state(s)
        f = env.render(); f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        np.save(f"{out}/fetch_{name}_{tag}.npy", f)
        print(f"FETCH {tag} {name} {f.shape} sum={int(f.astype(np.int64).sum())}", flush=True)
    except Exception as e:
        print(f"FETCH {tag} {name} FAILED {type(e).__name__}: {e}"[:250], flush=True)
PY
done
"$VENV/bin/python" - "$FOUT" <<'PY'
import sys, json, pathlib, numpy as np
out = pathlib.Path(sys.argv[1]); rows = []
for name in ("FetchReach", "FetchPickAndPlace"):
    a, b = out / f"fetch_{name}_before.npy", out / f"fetch_{name}_after.npy"
    if a.exists() and b.exists():
        x, y = np.load(a), np.load(b)
        rows.append({"config": name, "identical": bool(np.array_equal(x, y)),
                     "max_abs_diff": int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()),
                     "shape": list(x.shape), "note": "Fetch has NO spot light, so the cutoff=180 half "
                     "of the rule cannot fire for it; this is the unaffected-family control."})
    else:
        rows.append({"config": name, "identical": None, "missing": True})
(out / "byte_identity_fetch_postreview.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print("BYTE-IDENTITY", r, flush=True)
PY
rm -f "$FOUT"/fetch_*.npy

# ---- (3) re-measure the cells -----------------------------------------------------------------
export SE7_OGB_UNMEASURED=1
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_m3_fidelity.py \
  --out "$OUT/oscar/m3fid" --cells OGBCube OGBScene --n 64 --draws 3 || echo "M3 rc=$?"
[ -s "$OUT/oscar/m3fid/m3_fidelity.json" ] || { echo "MISSING m3_fidelity.json"; exit 4; }
echo OK
