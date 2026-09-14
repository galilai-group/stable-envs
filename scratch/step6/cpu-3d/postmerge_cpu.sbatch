#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-postmerge
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Post-merge (RAYCAST-4/5): cpu3d is a new consumer of raycast3d.cast_nearest/any_blocking, which now
# take the per-kind bounded path. Identity is claimed; this verifies it: the cross-backend comparison
# against the same-tree cuda stack on the 64 frozen states (se-step6), and the 3.11 Gate-A arm (se-vis311).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/cpu-3d
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
OUT=$WS/examples/results/step6/cpu-3d/replay/postmerge; mkdir -p "$OUT"
STATES=$WS/examples/results/step4/states
CONFIGS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
NODE=$(hostname -s); cd "$WS"
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
PY=$SCR/venvs/se-step6/bin/python
$PY -c "from stable_envs.render import raycast3d as R; print('PRIMITIVE_BOUNDS', getattr(R, 'PRIMITIVE_BOUNDS', 'absent'))"
$PY - <<'PY'
import numpy as np, glob, os
src = glob.glob(os.environ["WS"] + "/examples/results/step6/cpu-3d/replay/gpu/frames_cuda_*.npz")[0]
tf = os.environ["SCR"] + "/se6/texture-fidelity/examples/results/step6/texture-fidelity/localize312/frames"
out = os.environ["OUT"] if "OUT" in os.environ else None
out = os.environ["WS"] + "/examples/results/step6/cpu-3d/replay/postmerge/oracle_frames"; os.makedirs(out, exist_ok=True)
z = np.load(src)
for c in z.files:
    t = np.load(f"{tf}/{c}.npz"); np.savez_compressed(f"{out}/{c}.npz", frozen_ours=z[c], frozen_official=t["frozen_official"], frozen_seg=t["frozen_seg"])
print("oracle staged from", src)
PY
set -x
$PY scratch/step6/cpu-3d/compare_frozen.py --frames-dir "$OUT/oracle_frames" --n 64 --out "$OUT/cross_backend_$NODE" --quant trunc 2>&1 | grep -v 'mean \\|diff' | tail -6
echo "CROSS rc=$?"; rm -rf "$OUT/oracle_frames"
D=$OUT/certify311_cpu; rm -rf "$D"; mkdir -p "$D/conformance"
$SCR/venvs/se-vis311/bin/python examples/fidelity_step4.py --shard dmc --configs "$CONFIGS" --device cpu --states-dir "$STATES" --out "$D/conformance/dmc.json" 2>&1 | tail -20
echo "FIDELITY 311 rc=$?"
$PY scratch/step4/certify.py --shards "$D" --out "$D" --doc "$D/report.md" 2>&1 | tail -4
echo "CERTIFY 311 rc=$?"
set +x; echo "POSTMERGE CPU DONE"
