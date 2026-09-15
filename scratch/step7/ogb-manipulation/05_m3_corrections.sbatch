#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-m3corr
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# M3 design gate: does re-typing the out-of-range spot to POINT make MJWarp agree with MuJoCo?
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/oscar/m3
mkdir -p "$OUT"
export PYTHONPATH=$WS:$SCR/se7/swm-src
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true
eval "$(PYTHONPATH=$PYTHONPATH "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py --require egl 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: EGL unusable on $(hostname)"; exit 9; }
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_m3_corrections.py --out "$OUT" \
  --env swm/OGBCube-v0 --states 8 || echo "M3 rc=$?"
[ -s "$OUT/m3_corrections.json" ] || { echo "MISSING m3_corrections.json"; exit 4; }

# ---- M2's last obligation, folded in to stay within one GPU job -------------------------------
# Fetch refuses to construct on mujoco 3.12 (its Step-3B pin asserts 3.11), so the byte-identity
# replay for the family whose corrections mjwarp.py exists to serve has to run on the Fetch stack.
# se-s3b-fetch311 is a SHARED venv: it is only executed here, never written to.
FVENV=$SCR/venvs/se-s3b-fetch311
M2OUT=$WS/examples/results/step7/ogb-manipulation/m2/cuda
mkdir -p "$M2OUT"
for TREE in "$SCR/se7/baseline" "$WS"; do
  TAG=$([ "$TREE" = "$WS" ] && echo after || echo before)
  PYTHONPATH=$TREE "$FVENV/bin/python" - "$TAG" "$M2OUT" <<'PY'
import sys, numpy as np, stable_envs as se
tag, out = sys.argv[1], sys.argv[2]
for name in ("FetchReach", "FetchPickAndPlace"):
    try:
        st = np.load(f"/oscar/scratch/sboughan/se7/ogb-manipulation/examples/results/step4/states/{name}.npz")
        s = st[list(st.keys())[0]][:8]
        env = se.make(name, batch=len(s), device="cuda")
        env.set_state(s)
        f = env.render()
        f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        np.save(f"{out}/{name}_{tag}.npy", f)
        print(f"FETCH {tag} {name} {f.shape} sum={int(f.astype(np.int64).sum())}", flush=True)
    except Exception as e:
        print(f"FETCH {tag} {name} FAILED {type(e).__name__}: {e}"[:300], flush=True)
PY
done
"$FVENV/bin/python" - "$M2OUT" <<'PY'
import sys, json, pathlib, numpy as np
out = pathlib.Path(sys.argv[1]); rows = []
for name in ("FetchReach", "FetchPickAndPlace"):
    a, b = out / f"{name}_before.npy", out / f"{name}_after.npy"
    if a.exists() and b.exists():
        x, y = np.load(a), np.load(b)
        rows.append({"config": name, "identical": bool(np.array_equal(x, y)),
                     "max_abs_diff": int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()),
                     "shape": list(x.shape)})
    else:
        rows.append({"config": name, "identical": None, "missing": True})
(out / "byte_identity_fetch.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print("BYTE-IDENTITY", r, flush=True)
PY
echo OK
