#!/bin/bash
# Offline and online speed of stable-envs against the source, every arm pinned to the same CPU cores.
# usage (from the repository root): sbatch -p qed -n 1 -c 8 --hint=nomultithread --mem=128G --gres=gpu:1 -t 48:00:00 benchmarks/paper/run.sh OUT FAMILY=ENV,ENV ...
#   SMOKE=1 uses small state, batch and episode counts. THROUGHPUT=1 runs only the throughput arms (see the THROUGHPUT block). CORES (default: the first NPIN, default 8, cores the job may use) is the
#   core set of every arm. STEP_LIMIT (default 6h) stops one arm that does not exit.
set -uo pipefail
OUT=$(realpath -m "$1"); shift
# Slurm runs a copy of this script, so the repository is the submit directory.
REPO=${SNV_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
[ -f "$REPO/pyproject.toml" ] || { echo "no repository at $REPO" >&2; exit 2; }
SWM=${SWM:-$HOME/lewm-hyd/swm}
CORES=${CORES:-$(python3 -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:${NPIN:-8}])))")}
NCORES=$(taskset -c "$CORES" nproc)
PIN="timeout -k 60 ${STEP_LIMIT:-6h} taskset -c $CORES"
UV=${UV:-$HOME/.local/bin/uv}
WORK=${TMPDIR:-/tmp}/snv-bench-$$
mkdir -p "$OUT"/{offline,online,logs} "$WORK"
trap 'rm -rf "$WORK"' EXIT
if [ "${SMOKE:-0}" = 1 ]; then
  N=256; BATCHES="1 64 256"; EPISODES=64; LENGTH=32; LANES=64; SLANES=8; SWEPISODES=64
else
  N=2048; BATCHES="1 256 1024 2048"; EPISODES=256; LENGTH=64; LANES=256; SLANES=8; SWEPISODES=64
fi
# Arms that draw with software OpenGL play the first SWEPISODES episodes only; steps/s does not depend on the count.
REPEATS=${REPEATS:-3}
# SRC_LANES envs per source worker (one avoids the multi-env corruption some sources show);
# ONLY_SOURCE=1 runs only the source arms (for example a second core budget); SKIP_OFFLINE=1 skips the offline arms.
SLANES=${SRC_LANES:-$SLANES}
ONLY_SOURCE=${ONLY_SOURCE:-0}
export PYTHONPATH=$REPO SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
# The uv cache stays in home: node-local disk holds one family venv at a time.
export UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv} XDG_CACHE_HOME=$WORK/cache
# One GPU only: --exclusive exposes all of them, and torch and JAX may number several differently.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(echo "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}" | cut -d, -f1)
export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES
egl() { env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$@"; }
sw() { env LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe PYOPENGL_PLATFORM=glx MUJOCO_GL=glfw xvfb-run -a "$@"; }

{
  echo "{\"commit\": \"$(git -C "$REPO" rev-parse HEAD)\", \"main\": \"$(git -C "$REPO" merge-base HEAD origin/main 2>/dev/null)\","
  echo " \"dirty\": $(git -C "$REPO" diff --quiet && echo false || echo true), \"swm\": \"$(git -C "$SWM" rev-parse HEAD)\","
  echo " \"host\": \"$(hostname)\", \"job\": \"${SLURM_JOB_ID:-}\", \"start\": \"$(date -Is)\", \"cores\": \"$CORES\", \"ncores\": $NCORES,"
  echo " \"cuda_visible_devices\": \"$CUDA_VISIBLE_DEVICES\", \"gpu\": \"$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name,driver_version --format=csv,noheader)\", \"smoke\": ${SMOKE:-0},"
  echo " \"src_lanes\": ${SRC_LANES:-null}, \"only_source\": ${ONLY_SOURCE:-0},"
  echo " \"node_jobs_at_start\": \"$(squeue -h -w "$(hostname -s)" -o '%i %u %b %C' 2>/dev/null | tr '\n' ';')\","
  echo " \"n_states\": $N, \"batches\": \"$BATCHES\", \"episodes\": $EPISODES, \"software_gl_episodes\": $SWEPISODES, \"length\": $LENGTH, \"lanes\": $LANES, \"repeats\": $REPEATS}"
} > "$OUT/run.json"
cat "$OUT/run.json"

venv() {  # venv FAMILY -> python path of a fresh family venv in $WORK
  local fam=$1 extras groups
  case $fam in
    classic) extras="classic"; groups="pin-classic";;
    navigation) extras="navigation navigation-gpu"; groups="pin-navigation";;
    pusht) extras="pusht"; groups="pin-pusht";;
    dmc) extras="dmcontrol dmcontrol-gpu"; groups="pin-dmc gpu-mjwarp";;
    ogbench) extras="ogbench ogbench-manipulation ogbench-manipulation-gpu"; groups="pin-ogbench gpu-mjwarp";;
    fetch) extras="robotics mjwarp"; groups="pin-fetch";;
    rocket) extras="rocket"; groups="pin-rocket";;
    craftax) extras="craftax craftax-gpu"; groups="pin-craftax";;
    *) echo "unknown family $fam" >&2; return 1;;
  esac
  local v=$WORK/venv-$fam
  ( cd "$REPO" && UV_PROJECT_ENVIRONMENT=$v $UV sync --frozen --python 3.12 --no-default-groups \
      $(for e in $extras; do echo --extra $e; done) $(for g in $groups torch-cuda test; do echo --group $g; done) ) \
      > "$OUT/logs/venv-$fam.log" 2>&1 || { echo "venv $fam failed" >&2; return 1; }
  # Every paper number uses torch 2.14.1+cu126; main locks 2.15.0 for CUDA 13.2. TORCH_VERSION overrides.
  local tv=${TORCH_VERSION:-2.14.1}
  local cu=(--extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match)
  $UV pip install --python $v/bin/python "${cu[@]}" --reinstall-package torch "torch==$tv+cu126" \
      >> "$OUT/logs/venv-$fam.log" 2>&1 || { echo "torch $fam failed" >&2; return 1; }
  $UV pip freeze --python $v/bin/python | grep "==" > $WORK/constraints-$fam.txt
  # Importing stable_worldmodel registers every swm env; its PushT module needs the pusht extra.
  # torchvision must be the build made for this torch (torch 2.X.Y pairs with torchvision 0.(X+15).Y).
  local tvv; tvv=$(echo "$tv" | awk -F. '{print "0." $2 + 15 "." $3}')
  $UV pip install --python $v/bin/python "${cu[@]}" -c $WORK/constraints-$fam.txt -e "$SWM" "pymunk==7.2.0" shapely opencv-python-headless "torchvision==$tvv+cu126" \
      >> "$OUT/logs/venv-$fam.log" 2>&1 || { echo "swm $fam failed" >&2; return 1; }
  if [ $fam = rocket ]; then
    # nvdiffrast is not package metadata; reuse the build made for this node's CUDA toolkit. SWM's Rocket needs cvxpy.
    local sp; sp=$($v/bin/python -c "import site; print(site.getsitepackages()[0])")
    cp -r ${NVDIFFRAST_SITE:-$HOME/lewm-hyd/lewm/venv/lib/python3.12/site-packages}/{nvdiffrast,nvdiffrast-0.4.0.dist-info,_nvdiffrast_c*.so} "$sp"/ 2>/dev/null
    $UV pip install --python $v/bin/python "${cu[@]}" -c $WORK/constraints-$fam.txt cvxpy >> "$OUT/logs/venv-$fam.log" 2>&1
    $v/bin/python -c "import nvdiffrast.torch" >> "$OUT/logs/venv-$fam.log" 2>&1 || { echo "nvdiffrast $fam failed" >&2; return 1; }
  fi
  $v/bin/python -c "import torch, stable_envs, stable_worldmodel" >> "$OUT/logs/venv-$fam.log" 2>&1 \
      || { echo "imports $fam failed" >&2; return 1; }
  $UV pip freeze --python $v/bin/python > "$OUT/logs/freeze-$fam.txt"
  echo $v/bin/python
}

for spec in "$@"; do
  fam=${spec%%=*}; envs=(${spec#*=}); envs=(${envs[@]//,/ })
  echo "=== family $fam: ${envs[*]} $(date -Is)"
  df -h "$WORK" | tail -1
  P=$(venv $fam) || { rm -rf $WORK/venv-$fam; continue; }
  $P -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
  if [ $fam = craftax ]; then
    # Craftax has no source renderer to restore States into; its comparison is the upstream jitted step.
    mkdir -p $OUT/craftax
    $PIN $P -m benchmarks.craftax_bench --envs "${envs[@]}" --device cuda --repeats $REPEATS --output $OUT/craftax/gpu.json
    JAX_PLATFORMS=cpu $PIN $P -m benchmarks.craftax_bench --envs "${envs[@]}" --device cpu --batches 64 256 \
      --repeats $REPEATS --calls 10 --output $OUT/craftax/cpu.json
    rm -rf $WORK/venv-$fam
    continue
  fi
  OFF="$PIN $P -m benchmarks.offline_sweep"; ON="$PIN $P -m benchmarks.online_ladder"
  S=$WORK/states-$fam; D=$WORK/episodes-$fam; mkdir -p $S $D
  [ "${SKIP_OFFLINE:-0}" = 1 ] || [ "${THROUGHPUT:-0}" = 1 ] || egl $OFF states --envs "${envs[@]}" --out $S --n $N --device cuda --lanes 1024
  egl $ON dataset --envs "${envs[@]}" --out $D --episodes $EPISODES --length $LENGTH
  if [ "${THROUGHPUT:-0}" = 1 ]; then
    # Online throughput with frames on one CPU budget or one GPU: the source as one env, a Gymnasium SyncVectorEnv
    # and an AsyncVectorEnv (CPU OpenGL), stable-envs on CPU and on CUDA, and decoding stored JPEGs of the same frames.
    for E in "${envs[@]}"; do
      echo "--- $E $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
      n=$OUT/throughput/$E; mkdir -p $n
      COMMON="--env $E --data $D --procs $NCORES --lanes $LANES --threads $NCORES --repeats $REPEATS"
      sw $ON run $COMMON --episodes $SWEPISODES --arms src-single-render src-sync-render src-async-render --output $n/source-cpu.json
      sw $ON run $COMMON --episodes $SWEPISODES --arms snv-render-cpu --output $n/snv-cpu.json
      egl $ON run $COMMON --arms snv-render-gpu --output $n/snv-gpu.json
      egl $ON run $COMMON --arms jpeg-decode-cpu jpeg-decode-gpu --output $n/jpeg.json
    done
    rm -rf $WORK/venv-$fam $S $D
    continue
  fi
  for E in "${envs[@]}"; do
    echo "--- $E $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
    o=$OUT/offline/$E; mkdir -p $o
    if [ "${SKIP_OFFLINE:-0}" = 1 ]; then :
    elif [ $ONLY_SOURCE != 1 ]; then
      egl $OFF ours --envs $E --states $S --device cuda --batches $BATCHES --repeats $REPEATS --output $o/snv-gpu.json
      sw $OFF ours --envs $E --states $S --device cpu --threads $NCORES --batches $BATCHES --repeats $REPEATS --output $o/snv-cpu.json
      egl $OFF mse --envs $E --states $S --device cuda --output $o/mse-gpu.json
      sw $OFF mse --envs $E --states $S --device cpu --output $o/mse-cpu.json
    fi
    [ "${SKIP_OFFLINE:-0}" = 1 ] || {
    case $E in OGBCube|OGBScene)
      egl $OFF source-upstream --envs $E --states $S --procs 1 --keep $o/upstream-gpu-frames.npy --repeats $REPEATS --output $o/src-upstream-gpu-1p.json
      egl $OFF source-upstream --envs $E --states $S --procs $NCORES --repeats $REPEATS --output $o/src-upstream-gpu-${NCORES}p.json
      sw $OFF source-upstream --envs $E --states $S --procs 1 --repeats $REPEATS --output $o/src-upstream-cpu-1p.json
      sw $OFF source-upstream --envs $E --states $S --procs $NCORES --repeats $REPEATS --output $o/src-upstream-cpu-${NCORES}p.json;;
    esac
    for p in 1 $NCORES; do
      egl $OFF source --envs $E --states $S --procs $p --repeats $REPEATS --output $o/src-gpu-${p}p.json
      sw $OFF source --envs $E --states $S --procs $p --repeats $REPEATS --output $o/src-cpu-${p}p.json
    done
    }
    n=$OUT/online/$E; mkdir -p $n
    COMMON="--env $E --data $D --procs $NCORES --source-lanes $SLANES --lanes $LANES --threads $NCORES --repeats $REPEATS"
    egl $ON run $COMMON --arms src-pipelined-render --src-label gpu --ours-device cuda --output $n/src-render-gpu.json
    sw $ON run $COMMON --episodes $SWEPISODES --arms src-pipelined-render --src-label cpu --ours-device cpu --output $n/src-render-cpu.json
    if [ "$E" = PyFlytRocketLanding ]; then
      SRC_PYBULLET_EGL=1 egl $ON run $COMMON --arms src-pipelined-render --src-label gpu-pybullet-egl --ours-device cuda --output $n/src-render-gpu-pybullet-egl.json
    fi
    if [ $ONLY_SOURCE = 1 ]; then
      sw $ON run $COMMON --arms src-pipelined-step --output $n/step.json
    else
      sw $ON run $COMMON --arms src-pipelined-step snv-step --output $n/step.json
      sw $ON run $COMMON --episodes $SWEPISODES --arms snv-render-cpu --output $n/snv-render-cpu.json
      egl $ON run $COMMON --arms snv-step-gpu snv-render-gpu --output $n/snv-gpu.json
    fi
  done
  rm -rf $WORK/venv-$fam $S $D
done
echo "{\"end\": \"$(date -Is)\"}" > "$OUT/end.json"
echo done
