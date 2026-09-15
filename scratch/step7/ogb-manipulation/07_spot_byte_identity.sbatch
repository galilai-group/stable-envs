#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-spotbyte
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The spot correction's cluster obligation: a registered family with a VALID spot cutoff must be
# byte-identical before and after. The registry sweep already proves statically that no family's
# light_exponent changes; this proves it in pixels, on the same card, in one job.
#
# DMCReacherEasy, DMCCartpoleSwingup, DMCPointMassEasy and MujocoPusher are chosen because they
# CARRY a spot light (7 DM Control cells plus MujocoPusher do) -- a byte-identity test on a family
# with no spot light would pass vacuously. Fetch runs too, on its own 3.11 stack, because it is the
# family whose corrections this module exists to serve.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
WS=$SCR/se7/ogb-manipulation
BASE=$SCR/se7/baseline
VENV=$SCR/venvs/se7-ogb
FVENV=$SCR/venvs/se-s3b-fetch311
OUT=$WS/examples/results/step7/ogb-manipulation/m2/spot
mkdir -p "$OUT"
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true
[ -d "$BASE/stable_envs" ] || { echo "FATAL: no pristine baseline at $BASE"; exit 3; }

run_leg () {  # venv, tree, tag, configs...
  local V=$1 TREE=$2 TAG=$3; shift 3
  PYTHONPATH=$TREE "$V/bin/python" - "$TAG" "$OUT" "$@" <<'PY'
import sys, numpy as np, stable_envs as se
tag, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
for name in names:
    try:
        st = np.load(f"/oscar/scratch/sboughan/se7/ogb-manipulation/examples/results/step4/states/{name}.npz")
        s = st[list(st.keys())[0]][:8]
        env = se.make(name, batch=len(s), device="cuda")
        env.set_state(s)
        f = env.render()
        f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        np.save(f"{out}/{name}_{tag}.npy", f)
        print(f"LEG {tag} {name} {f.shape} sum={int(f.astype(np.int64).sum())}", flush=True)
    except Exception as e:
        print(f"LEG {tag} {name} FAILED {type(e).__name__}: {e}"[:300], flush=True)
PY
}
SPOTCFG="DMCReacherEasy DMCCartpoleSwingup DMCPointMassEasy MujocoPusher"
run_leg "$VENV"  "$BASE" before $SPOTCFG
run_leg "$VENV"  "$WS"   after  $SPOTCFG
run_leg "$FVENV" "$BASE" before FetchReach
run_leg "$FVENV" "$WS"   after  FetchReach

"$VENV/bin/python" - "$OUT" $SPOTCFG FetchReach <<'PY'
import sys, json, pathlib, numpy as np
out = pathlib.Path(sys.argv[1]); rows = []
for name in sys.argv[2:]:
    a, b = out / f"{name}_before.npy", out / f"{name}_after.npy"
    if a.exists() and b.exists():
        x, y = np.load(a), np.load(b)
        rows.append({"config": name, "identical": bool(np.array_equal(x, y)),
                     "max_abs_diff": int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()),
                     "shape": list(x.shape)})
    else:
        rows.append({"config": name, "identical": None, "missing": True})
(out / "byte_identity_spot.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print("BYTE-IDENTITY", r, flush=True)
bad = [r for r in rows if r.get("identical") is False]
print("VERDICT", "ALL IDENTICAL" if not bad else f"CHANGED: {[r['config'] for r in bad]}")
PY
rm -f "$OUT"/*.npy
[ -s "$OUT/byte_identity_spot.json" ] || { echo "MISSING byte_identity_spot.json"; exit 4; }
echo OK
