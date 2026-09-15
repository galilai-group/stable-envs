#!/bin/bash
#SBATCH --job-name=se7-dmc-manipulator-official
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2104
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The OFFICIAL-RENDERER arm, so R and cores are computable and the GPU tier stops being
# GPU_UNMEASURED. Uses the repo's own economics harness (examples/bench_vs_decode.py), which measures
# all three columns in ONE process on ONE node: our GPU f/s, the env's OFFICIAL renderer f/s
# (--cpu-frames), and one-core JPEG-q90 decode f/s of this env's own frames (--jpeg-quality 90). That
# is the Step-6 protocol and the definition certification.md S6.5 uses:
#     R     = stable-envs GPU f/s / official renderer f/s
#     cores = stable-envs GPU f/s / one-core JPEG-q90 decode f/s
# Measuring them in separate jobs, or on separate nodes, is the different-machines violation.
#
# NODE EXCLUSIONS. gpu2108/2113/2110/2116 are the coordinator's standing list. gpu2104 is ADDED
# because W4's GPU job failed there inside the OFFICIAL renderer under se-vis311 and se-step5. Note
# for the record: MY job 6394689 ran the official renderer on gpu2104 successfully under se-step6
# (check_reference_renderer_is_stable reported max_abs_diff 0), so the evidence points at those two
# venvs' EGL stack rather than at the node -- the probe below is what will tell them apart, and it
# runs BEFORE any measurement so a failure cannot be mistaken for a number.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=dmc-manipulator
export PYTHONPATH=$SCR/se7/$WS
OUT=$SCR/se7/$WS/examples/results/step7/$WS
cd "$SCR/se7/$WS"

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ---- EGL DEVICE PROBE, before any measurement -------------------------------
# Creates a real offscreen context and renders one frame through mujoco.Renderer in EACH venv. A
# venv that cannot do this is reported HERE, by name, instead of producing a silent gap in a table.
probe () {
  local venv=$1
  "$SCR/venvs/$venv/bin/python" - <<'PY' 2>&1 | sed "s/^/[EGL $venv] /"
import os, sys, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
try:
    import mujoco
    from importlib.metadata import version as v
    try:
        dmv = v("dm_control")
    except Exception:
        dmv = "absent"
    print(f"mujoco {mujoco.__version__} dm_control {dmv}")
    n = mujoco.egl.EGL_ok if hasattr(mujoco, "egl") and hasattr(mujoco.egl, "EGL_ok") else "?"
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 1"/>'
        '<geom name="g" type="sphere" size=".1"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=64, width=64)   # creates the EGL context
    r.update_scene(d)
    img = r.render()
    print("OK  offscreen render", img.shape, "mean", float(img.mean()))
except Exception as e:
    print("FAIL", type(e).__name__, e)
    traceback.print_exc()
    sys.exit(3)
PY
  return ${PIPESTATUS[0]}
}
probe se-step6;   S6=$?
probe se-vis311;  V3=$?
probe se-step5;   S5=$?
echo "EGL probe rc: se-step6=$S6 se-vis311=$V3 se-step5=$S5"
if [ $S6 -ne 0 ]; then
  echo "ABORT: se-step6 cannot create an EGL context on $(hostname); this is the venv the measurement"
  echo "       uses, so every number below would be missing rather than wrong. Not measuring."
  exit $S6
fi

# ---- the measurement --------------------------------------------------------
set -x
mkdir -p "$OUT/economics"
$SCR/venvs/se-step6/bin/python examples/bench_vs_decode.py \
    --envs DMCManipulatorBringBall \
    --batches 1,16,64,256,1024,4096 --reps 10 --ss native \
    --jpeg-quality 90 --cpu-frames 32 \
    --tag se-step6-3090 --scratch "$SCR/.cache/se7bench" \
    --out "$OUT/economics"
rc=$?
set +x
echo "bench_vs_decode rc=$rc"
ls -la "$OUT/economics" || true
exit $rc
