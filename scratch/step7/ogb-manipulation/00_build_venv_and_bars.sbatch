#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-venv-bars
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W5 phase 0, cluster half 1: build the interpreter and take the CERTIFICATION bars.
#
# WHY A NEW VENV. `$SCR/venvs/se-step5` has mujoco_warp 3.12 + stable_worldmodel but NO ogbench;
# `$SCR/venvs/se-step6` has ogbench but neither mujoco_warp nor stable_worldmodel. The two
# environments this workstream measures need ALL THREE in one interpreter, and AGENT_COMMON.md
# forbids installing into either existing venv. `$SCR/venvs/se7-ogb` is that interpreter, pinned to
# the same mujoco 3.12.0 / warp 1.17.0 / torch 2.14.0+cu126 as se-step5 so a GPU number taken here
# is comparable with the Step-6 MJWarp rows.
#
# WHY THE BARS MUST BE RETAKEN HERE. The local M5 numbers in examples/results/step7/ogb-manipulation
# are a laptop's; Step 2 certifies `S_cpu` and `cores_cpu` against the official renderer and a JPEG
# decode measured ON THE SAME certification host at ONE thread. This job re-runs the identical
# probes under EGL on a `batch` node, which is also the first proof that these two envs render
# headless at all (Step 1 recorded only that they raise under GLFW on a headless node).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip UV_CACHE_DIR=$SCR/.cache/uv
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/oscar
mkdir -p "$OUT" "$SCR/logs"
set -x

# ---- 1. the interpreter -----------------------------------------------------
# A bare `python3` on a batch node is 3.9 and every `pip install -q` for a 3.12-only wheel
# fails SILENTLY into a venv that then imports nothing; job 6394700 burned itself proving it.
# Load the module explicitly, assert the version, and let pip print.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.12.8-zr3c 2>/dev/null || true
PY312=$(command -v python3.12 || command -v python3)
"$PY312" -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version' || {
  echo "FATAL: no python3.12 on this node"; exit 3; }

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import numpy, mujoco, mujoco_warp' 2>/dev/null; then
  rm -rf "$VENV"
  "$PY312" -m venv "$VENV" || { echo "FATAL: venv"; exit 3; }
  "$VENV/bin/pip" install --upgrade pip || exit 3
  # torch AND torchvision from the same cu126 index, in one resolve. Installing torchvision later
  # from PyPI picks the default-variant wheel against a cu126 torch and `import torchvision` then
  # dies with `operator torchvision::nms does not exist`, which is what killed job 6394784.
  "$VENV/bin/pip" install torch==2.14.0 torchvision==0.29.0 \
      --index-url https://download.pytorch.org/whl/cu126 || exit 3
  "$VENV/bin/pip" install "mujoco==3.12.0" "mujoco-warp==3.12.0" "warp-lang==1.17.0" \
      ogbench "dm_control==1.0.45" pytest scipy pillow || exit 3
  # stable_worldmodel: the registration under test. Its heavy extras are not needed to render.
  "$VENV/bin/pip" install loguru tabulate einops tqdm typer rich pyarrow \
      "lancedb>=0.30.0" "pylance>=4.0.0" opencv-python-headless pygame pymunk shapely || exit 3
  # stable_worldmodel comes from the SOURCE CHECKOUT, not PyPI. Job 6394991 proved they are
  # different code: PyPI 0.0.6's `data/dataset.py` imports h5py, the checkout's `data/__init__.py`
  # imports its own `formats.lance` and never touches h5py. The registration under test must be the
  # one the local census measured, so the checkout is rsynced to $SCR/se7/swm-src and put on
  # PYTHONPATH; h5py is installed anyway as cheap insurance against the other import path.
  "$VENV/bin/pip" install h5py || exit 3
fi
# Unconditional, because the venv may already exist from an earlier attempt that pip-installed the
# PyPI build: remove it so nothing can shadow the checkout, and add h5py for the other import path.
"$VENV/bin/pip" uninstall -y stable-worldmodel >/dev/null 2>&1 || true
"$VENV/bin/pip" install h5py || exit 3

SWM_SRC=$SCR/se7/swm-src
[ -d "$SWM_SRC/stable_worldmodel" ] || { echo "FATAL: no swm source at $SWM_SRC"; exit 3; }
PYTHONPATH=$WS:$SWM_SRC "$VENV/bin/python" \
  -c 'import numpy, mujoco, mujoco_warp, ogbench, torchvision, stable_worldmodel' || {
  echo "FATAL: venv incomplete"; exit 3; }

# ---- 1b. provenance -----------------------------------------------------------
# This block was accidentally deleted by an edit to section 1 and job 6395175 therefore shipped a
# STALE versions.txt from an earlier, failed job -- python 3.9, everything ABSENT -- next to numbers
# taken under python 3.12 with everything present. It is restored here and writes a job-tagged file
# so a stale one can never again sit beside a fresh result.
export SWM_SRC
PYTHONPATH=$WS:$SWM_SRC "$VENV/bin/python" - <<'PY' | tee "$OUT/provenance_${SLURM_JOB_ID}.txt"
import hashlib, importlib.metadata as md, os, pathlib, socket, sys
print("job", os.environ.get("SLURM_JOB_ID"), "host", socket.gethostname())
print("python", sys.version.split()[0])
for p in ("torch","torchvision","mujoco","mujoco-warp","warp-lang","ogbench","dm-control",
          "numpy","scipy","pillow","gymnasium","h5py"):
    try: print(p, md.version(p))
    except Exception: print(p, "ABSENT")
# stable_worldmodel comes from the rsynced SOURCE CHECKOUT, not PyPI (they are different code under
# the same version string). Print the digests so they can be compared with the local census's
# cube_env 143e5e81f424ff7a / scene_env d0eebdce0131f86e / envs__init__ 118a262677a647c6.
import stable_worldmodel as swm
root = pathlib.Path(swm.__file__).parent
print("stable_worldmodel", root)
for f in ("envs/ogbench/cube_env.py", "envs/ogbench/scene_env.py", "envs/__init__.py"):
    q = root / f
    print("sha256", f, hashlib.sha256(q.read_bytes()).hexdigest()[:16] if q.exists() else "MISSING")
PY

# Pick a GL backend that actually works on THIS node, and refuse to measure without one.
# A `batch` node has no GPU, so EGL's PLATFORM_DEVICE path cannot initialise -- job 6395139 proved
# that on node2326 -- and Mesa surfaceless EGL or osmesa (llvmpipe) is what renders there, exactly
# as the Step-6 CPU scripts probe for. A backend that creates a context but paints a constant frame
# is treated as a fault, not a pass.
eval "$(PYTHONPATH=$WS:$SWM_SRC "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py egl osmesa glfw 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: no GL backend on $(hostname)"; exit 9; }
echo "MUJOCO_GL=$MUJOCO_GL on $(hostname)" | tee "$OUT/gl_backend.txt"

# ---- 2. does this node render the two envs headless at all? -----------------
export PYTHONPATH=$WS:$SWM_SRC
cd "$WS"
"$VENV/bin/python" scratch/step7/ogb-manipulation/census_ogb.py --out "$OUT/census" \
  || echo "CENSUS rc=$?"

# ---- 3. the bars, one thread, on a certification node ------------------------
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_determinism_bars.py --out "$OUT/bars" \
  --reps 100 --blocks 7 || echo "BARS rc=$?"
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_purity_fps.py --out "$OUT/purity" \
  --reps 200 || echo "PURITY rc=$?"
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_plate_shadow.py --out "$OUT/plate" \
  || echo "PLATE rc=$?"
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_interior_alpha.py --out "$OUT/interior" \
  || echo "INTERIOR rc=$?"

# COMPLETED is not proof: name the files that must exist, and FAIL if they do not, so the
# dependent GPU job is not chained onto an empty result.
[ -s "$OUT/provenance_${SLURM_JOB_ID}.txt" ] || { echo "MISSING provenance"; exit 4; }
for f in "$OUT/census/census_OGBCube.json" "$OUT/census/census_OGBScene.json" \
         "$OUT/bars/bars_OGBCube.json" "$OUT/bars/bars_OGBScene.json"; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 4; }
done
ls -lR "$OUT" | tail -40
echo OK
