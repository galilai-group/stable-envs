#!/bin/bash
#SBATCH --job-name=se6-cpu-3d-certify
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# CPU certification of the four trackcom cells' CPU backend, 1 torch thread, on a batch node:
#  (i)  Gate-A mandatory + visual checks via examples/fidelity_step4.py under the DECLARED reference
#       pair (se-vis311: mujoco 3.11.0 + dm_control 1.0.44) and, for the record, under 3.12 (se-step6,
#       where the certifier BLOCKS the reference-judged checks); scratch/step4/certify.py labels both.
#  (ii) cross-backend: CPU frames vs OUR MJWarp frames on the frozen states (same tree, gpu2106 stack).
#  (iii) the paired Step-3 sweep (official llvmpipe interleaved, decode bar, spread, 1024 cells).
# Provenance: the tree SHA reaches every artifact through STABLE_ENVS_GIT_COMMIT / $WS/GIT_COMMIT
# (rsync carries no .git; scratch/step6/cpu-3d/sync.sh writes the file). The GL vendor string of the
# reference renderer on this node is recorded beside the arms.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/cpu-3d
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
OUT=$WS/examples/results/step6/cpu-3d
STATES=$WS/examples/results/step4/states
CONFIGS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
NODE=$(hostname -s)
cd "$WS"
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
mkdir -p "$OUT/replay"
$SCR/venvs/se-step6/bin/python - "$OUT/replay/gl_renderer_${NODE}.json" <<'PY'
import json, os, socket, sys, mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 16, 16); r.update_scene(mujoco.MjData(m)); r.render()
from OpenGL import GL
doc = {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "mujoco_gl": os.environ.get("MUJOCO_GL"),
       "gl_renderer": GL.glGetString(GL.GL_RENDERER).decode(), "gl_vendor": GL.glGetString(GL.GL_VENDOR).decode(),
       "gl_version": GL.glGetString(GL.GL_VERSION).decode(), "commit": os.environ.get("STABLE_ENVS_GIT_COMMIT")}
json.dump(doc, open(sys.argv[1], "w"), indent=1); print(doc)
PY
set -x
for ARM in 311 312; do
  if [ "$ARM" = 311 ]; then PY=$SCR/venvs/se-vis311/bin/python; else PY=$SCR/venvs/se-step6/bin/python; fi
  D=$OUT/replay/certify${ARM}_cpu; rm -rf "$D"; mkdir -p "$D/conformance"
  $PY examples/fidelity_step4.py --shard dmc --configs "$CONFIGS" --device cpu --states-dir "$STATES" \
      --out "$D/conformance/dmc.json" 2>&1 | tail -25
  echo "FIDELITY $ARM rc=$?"
  $SCR/venvs/se-step6/bin/python scratch/step4/certify.py --shards "$D" --out "$D" --doc "$D/report.md" 2>&1 | tail -5
  echo "CERTIFY $ARM rc=$?"
done
PY=$SCR/venvs/se-step6/bin/python
$PY - <<'PY'
# the same-tree cuda stack is the oracle: expose it under the key compare_frozen.py reads
import numpy as np, glob, os
src = glob.glob(os.environ["WS"] + "/examples/results/step6/cpu-3d/replay/gpu/frames_cuda_*.npz")[0]
tf = os.environ["SCR"] + "/se6/texture-fidelity/examples/results/step6/texture-fidelity/localize312/frames"
out = os.environ["WS"] + "/examples/results/step6/cpu-3d/replay/oracle_frames"; os.makedirs(out, exist_ok=True)
z = np.load(src)
for c in z.files:
    seg = np.load(f"{tf}/{c}.npz")["frozen_seg"]
    np.savez_compressed(f"{out}/{c}.npz", frozen_ours=z[c], frozen_official=np.load(f"{tf}/{c}.npz")["frozen_official"], frozen_seg=seg)
print("oracle frames staged from", src)
PY
$PY scratch/step6/cpu-3d/compare_frozen.py --frames-dir "$OUT/replay/oracle_frames" --n 64 --out "$OUT/replay/cross_backend_$NODE" --sheet 2>&1 | grep -v 'mean \\|diff' | tail -6
echo "CROSS rc=$?"
rm -rf "$OUT/replay/oracle_frames"
mkdir -p "$OUT/sweep/after"
$PY scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --envs "$CONFIGS" \
    --out "$OUT/sweep/after/cpu_1thread_paired_after_${NODE}.json" 2>&1 | tail -12
echo "SWEEP rc=$?"
set +x; echo "CERTIFY CPU DONE"
