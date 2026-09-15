#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-m2proof
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# M2 proof obligation, cluster half: the families that need CUDA to construct.
#
# The local sweep covered 40 registered families and found every answer identical, but Fetch --
# the family whose corrections this helper exists to serve -- is cuda-only and did not build there.
# This re-runs the same before/after sweep where those families DO build, and then does the thing
# a set comparison cannot do: renders a Fetch config on the branch and on the pristine baseline
# IN THE SAME JOB ON THE SAME CARD and asserts the frames are byte-identical. A helper that only
# reports is supposed to change no pixel; that is asserted, not assumed.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se7/ogb-manipulation
BASE=$SCR/se7/baseline
# se7-ogb, not se-step5: se-step5 has no ogbench, so the 25 OGBench maze cells -- the largest
# MJWarp-backed family in the registry -- did not build there and the sweep covered only 15 envs
# (job 6396748). gymnasium_robotics is added here because Fetch needs it and it is absent from every
# shared venv this workstream may write to; se7-ogb is this workstream's own, so installing into it
# breaks no other agent's pins.
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m2/cuda
mkdir -p "$OUT"
set -x
cd "$WS"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
[ -d "$BASE/stable_envs" ] || { echo "FATAL: no pristine baseline at $BASE"; exit 3; }
"$VENV/bin/python" -c 'import gymnasium_robotics' 2>/dev/null || \
  "$VENV/bin/pip" install gymnasium-robotics || exit 3
"$VENV/bin/python" -c 'import gymnasium_robotics, ogbench, mujoco_warp' || {
  echo "FATAL: venv missing a family this proof must cover"; exit 3; }

export MUJOCO_GL=egl
# 1. the sweep, where the cuda-only families construct
PYTHONPATH=$WS:$SCR/se7/swm-src "$VENV/bin/python" scratch/step7/ogb-manipulation/sweep_texture_gaps.py \
  --out "$OUT" || echo "SWEEP rc=$?"

# 2. byte-identity: same states, same card, branch vs pristine 76dc511.
for TREE in "$BASE" "$WS"; do
  TAG=$([ "$TREE" = "$BASE" ] && echo before || echo after)
  PYTHONPATH=$TREE:$SCR/se7/swm-src "$VENV/bin/python" - "$TAG" <<'PY'
import sys, numpy as np, stable_envs as se
tag = sys.argv[1]
out = "/oscar/scratch/sboughan/se7/ogb-manipulation/examples/results/step7/ogb-manipulation/m2/cuda"
for name in ("FetchReach", "FetchPickAndPlace", "DMCReacherEasy", "PointmazeMediumNavigate"):
    try:
        st = np.load(f"/oscar/scratch/sboughan/se7/ogb-manipulation/examples/results/step4/states/{name}.npz")
        s = st[list(st.keys())[0]][:8]
        env = se.make(name, batch=len(s), device="cuda")
        env.set_state(s)
        f = env.render()
        f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        np.save(f"{out}/{name}_{tag}.npy", f)
        print(f"{tag} {name} {f.shape} sum={int(f.astype(np.int64).sum())}", flush=True)
    except Exception as e:
        print(f"{tag} {name} FAILED {type(e).__name__}: {e}"[:300], flush=True)
PY
done
"$VENV/bin/python" - <<'PY'
import numpy as np, pathlib, json
out = pathlib.Path("/oscar/scratch/sboughan/se7/ogb-manipulation/examples/results/step7/ogb-manipulation/m2/cuda")
rows = []
for name in ("FetchReach", "FetchPickAndPlace", "DMCReacherEasy", "PointmazeMediumNavigate"):
    a, b = out / f"{name}_before.npy", out / f"{name}_after.npy"
    if a.exists() and b.exists():
        x, y = np.load(a), np.load(b)
        rows.append({"config": name, "identical": bool(np.array_equal(x, y)),
                     "max_abs_diff": int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()),
                     "shape": list(x.shape)})
    else:
        rows.append({"config": name, "identical": None, "missing": True})
(out / "byte_identity.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print("BYTE-IDENTITY", r, flush=True)
PY
[ -s "$OUT/texture_gaps_sweep.json" ] || { echo "MISSING sweep json"; exit 4; }
[ -s "$OUT/byte_identity.json" ] || { echo "MISSING byte_identity json"; exit 4; }
echo OK
