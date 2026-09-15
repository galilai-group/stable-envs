#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-gpu2
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2104,gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Steps 4-6 of gpu_declare.sbatch, resubmitted: on gpu2104 (job 6394721) steps 1-3 passed (cuda
# identity 21/21 + 4/4, twins, fast paths) and every call that opens the OFFICIAL renderer's EGL
# context under se-vis311 / se-step5 died with ``OpenGL.raw.EGL._errors.EGLError`` (a shared
# multi-GPU node; W3's job saw the same). Here: gpu2104 excluded, an EGL probe per venv that tries
# each EGL device index and exports the first that renders (``MUJOCO_EGL_DEVICE_ID``), and full
# tracebacks kept (no ``tail``).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se7/dmc-quadruped
PY5=$SCR/venvs/se-step5/bin/python
PY311=$SCR/venvs/se-vis311/bin/python
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/gpu
STATES7=$WS/examples/results/step7/dmc-quadruped/phase1/corpus/states
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
export PYTHONPATH=$WS
mkdir -p "$OUT"; cd "$WS"
NODE=$(hostname -s)
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT merged $(cat "$WS/MERGED_COMMIT" 2>/dev/null || echo ?) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/card2.txt"
{ echo "job2 $SLURM_JOB_ID node $NODE"; echo "GIT_COMMIT $STABLE_ENVS_GIT_COMMIT"; echo "MERGED_COMMIT $(cat "$WS/MERGED_COMMIT" 2>/dev/null)"; date -u; } >> "$OUT/provenance.txt"
# EGL probe: which device index renders, per venv
probe() {  # $1 = python
  "$1" - <<'PY'
import os, sys, traceback
import mujoco
xml = "<mujoco><worldbody><light pos='0 0 3'/><geom size='1'/></worldbody></mujoco>"
ok = None
for dev in [None, 0, 1, 2, 3]:
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    if dev is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
    try:
        import importlib
        for k in [k for k in sys.modules if k.startswith("mujoco.egl")]:
            del sys.modules[k]
        m = mujoco.MjModel.from_xml_string(xml)
        r = mujoco.Renderer(m, 16, 16); r.update_scene(mujoco.MjData(m)); r.render()
        print(f"EGL_OK device={dev}", flush=True); ok = dev; break
    except BaseException as e:
        print(f"EGL_FAIL device={dev}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        traceback.print_exc(limit=3)
print("EGL_RESULT", "none" if ok is None else ok)
PY
}
set -x
probe $PY311 2>&1 | tee "$OUT/egl_probe_311_${NODE}.txt"
probe $PY5   2>&1 | tee "$OUT/egl_probe_312_${NODE}.txt"
DEV=$(grep -h "^EGL_RESULT" "$OUT/egl_probe_311_${NODE}.txt" | awk '{print $2}')
if [ "$DEV" != "none" ] && [ "$DEV" != "None" ] && [ -n "$DEV" ]; then export MUJOCO_EGL_DEVICE_ID=$DEV; fi
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-unset}"
# 4. the declaration draws (3 x 64), both arms; the 3.11 arm is the one the entry is declared from
$PY311 scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 224 --out "$OUT/visual_calib_311" 2>&1 | tee "$OUT/visual_calib_311.log" | tail -12
$PY5   scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 224 --out "$OUT/visual_calib_312" 2>&1 | tee "$OUT/visual_calib_312.log" | tail -12
$PY311 scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --n 64 --draws 3 --size 84  --out "$OUT/visual_calib_311_84px" 2>&1 | tee "$OUT/visual_calib_311_84px.log" | tail -12
# 5. Gate A on cuda on the new corpus, both arms
$PY311 scratch/step7/dmc-quadruped/gateA_local.py --config DMCQuadrupedWalk --device cuda --states-dir "$STATES7" --out "$OUT/gateA_cuda_mj3110.json" 2>&1 | tee "$OUT/gateA_cuda_mj3110.log" | tail -26
$PY5   scratch/step7/dmc-quadruped/gateA_local.py --config DMCQuadrupedWalk --device cuda --states-dir "$STATES7" --out "$OUT/gateA_cuda_mj3120.json" 2>&1 | tee "$OUT/gateA_cuda_mj3120.log" | tail -26
# 6. the cuda stack on the corpus states, per surface vs official, both arms
$PY311 scratch/step6/texture-fidelity/localize.py --configs DMCQuadrupedWalk --states-dir "$STATES7" --out "$OUT/localize_311" 2>&1 | tee "$OUT/localize_311.log" | tail -6
$PY5   scratch/step6/texture-fidelity/localize.py --configs DMCQuadrupedWalk --states-dir "$STATES7" --out "$OUT/localize_312" 2>&1 | tee "$OUT/localize_312.log" | tail -6
set +x; echo "GPU DECLARE 2 DONE"
