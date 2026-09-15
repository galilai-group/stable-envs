#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-spotgap
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Close the proof gap named in the last report. DMCPointMassEasy and MujocoPusher CARRY spot lights
# but have no frozen Step-4 state file, so job 6397885 could not replay them and the spot
# correction's byte-identity proof rested on two DM Control cells plus Fetch. Both envs can produce
# their own states, so the states come from `sample_states` with a fixed seed instead -- the same
# states on both legs, same card, same job.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
WS=$SCR/se7/ogb-manipulation
BASE=$SCR/se7/baseline
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m2/spot
mkdir -p "$OUT"
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true
[ -d "$BASE/stable_envs" ] || { echo "FATAL: no pristine baseline at $BASE"; exit 3; }

# The states are generated ONCE, from the pristine baseline, and reused on both legs: sampling them
# separately per leg would compare two different scenes and pass or fail for the wrong reason.
PYTHONPATH=$BASE "$VENV/bin/python" - "$OUT" <<'PY'
import sys, numpy as np, stable_envs as se
out = sys.argv[1]
for name in ("DMCPointMassEasy", "MujocoPusher"):
    try:
        env = se.make(name, batch=8, device="cpu")
        s = env.sample_states(8, np.random.default_rng(0))
        np.save(f"{out}/{name}_states.npy", np.asarray(s))
        print(f"STATES {name} {np.asarray(s).shape}", flush=True)
    except Exception as e:
        print(f"STATES {name} FAILED {type(e).__name__}: {e}"[:300], flush=True)
PY

for TREE in "$BASE" "$WS"; do
  TAG=$([ "$TREE" = "$WS" ] && echo after || echo before)
  PYTHONPATH=$TREE "$VENV/bin/python" - "$TAG" "$OUT" <<'PY'
import sys, os, numpy as np, stable_envs as se
tag, out = sys.argv[1], sys.argv[2]
for name in ("DMCPointMassEasy", "MujocoPusher"):
    f = f"{out}/{name}_states.npy"
    if not os.path.exists(f):
        print(f"LEG {tag} {name} SKIPPED (no states)", flush=True); continue
    try:
        s = np.load(f)
        env = se.make(name, batch=len(s), device="cuda")
        env.set_state(s)
        fr = env.render()
        fr = fr.cpu().numpy() if hasattr(fr, "cpu") else np.asarray(fr)
        np.save(f"{out}/{name}_{tag}.npy", fr)
        print(f"LEG {tag} {name} {fr.shape} sum={int(fr.astype(np.int64).sum())}", flush=True)
    except Exception as e:
        print(f"LEG {tag} {name} FAILED {type(e).__name__}: {e}"[:300], flush=True)
PY
done

"$VENV/bin/python" - "$OUT" <<'PY'
import sys, json, pathlib, numpy as np
out = pathlib.Path(sys.argv[1]); rows = []
for name in ("DMCPointMassEasy", "MujocoPusher"):
    a, b = out / f"{name}_before.npy", out / f"{name}_after.npy"
    if a.exists() and b.exists():
        x, y = np.load(a), np.load(b)
        rows.append({"config": name, "states": "sample_states(8, seed 0)",
                     "identical": bool(np.array_equal(x, y)),
                     "max_abs_diff": int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()),
                     "shape": list(x.shape)})
    else:
        rows.append({"config": name, "identical": None, "missing": True})
(out / "byte_identity_spot_sampled.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print("BYTE-IDENTITY", r, flush=True)
bad = [r for r in rows if r.get("identical") is False]
print("VERDICT", "ALL IDENTICAL" if not bad else f"CHANGED: {[r['config'] for r in bad]}")
PY
rm -f "$OUT"/*_before.npy "$OUT"/*_after.npy "$OUT"/*_states.npy
[ -s "$OUT/byte_identity_spot_sampled.json" ] || { echo "MISSING json"; exit 4; }
echo OK
