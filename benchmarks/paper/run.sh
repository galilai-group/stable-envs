#!/bin/bash
# Offline and online speed of stable-envs against the source, every arm pinned to the same CPU cores.
# usage (from the repository root): sbatch -p qed -n 1 -c 8 --hint=nomultithread --mem=128G [--gres=gpu:1] -t 24:00:00 benchmarks/paper/run.sh OUT FAMILY=ENV,ENV ...
#   For each env, each device in DEVICES ("cpu cuda") and each arm in ARMS ("offline step frames"):
#     offline  stable-envs frames at each batch of the sweep from stored inputs: rows from the family sampler
#              (sample_states) where the class has one and draws rows on the device, else full States saved
#              along random-action rollouts; their frame MSE; and the source drawing the same inputs in 1 and
#              NCORES processes (with EGL on CUDA where reference_gpu_render is "mujoco-egl");
#     step     physics steps with L lanes for each L of the sweep, and the source stepping L envs on CPU;
#     frames   steps with frames, and the source drawing on CPU, plus on the GPU where the class has reference_gpu_render.
#   SWEEP_CPU (1:8:64:256:1024) and SWEEP_CUDA (1:256:512:1024:2048) are the batches and lane counts, colon-separated.
#   CPU_SOURCES=0 leaves out the source arms that run on CPU (for GPU jobs whose CPU source cells come from a CPU job).
#   An output file that exists is kept, so a rerun with the same OUT fills in only what is missing.
#   SMOKE=1 uses small counts. QUICK=1 is a go/no-go check: 1 repeat of a WINDOW (default 3 s) timed window and
#   lane counts 8:256:1024 unless SWEEP_* are set; paper numbers use the default (3 repeats of 10 s).
#   VENV_PYTHON=path reuses an existing family venv (no build, no delete); STATES_DIR=dir keeps the stored States and
#   episodes between runs; SRC_FROM=dir copies source cells (src-*.json) measured on the same machine into OUT first.
#   MODE=throughput runs the throughput arms, MODE=gifs one GIF per env.
#   CORES (default: the first NPIN, default 8, cores the job may use) is the core set of every arm. STEP_LIMIT
#   (default 6h) stops one arm that does not exit.
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
if [ "${QUICK:-0}" = 1 ]; then
  SWEEP_CPU=${SWEEP_CPU:-8:256:1024}; SWEEP_CUDA=${SWEEP_CUDA:-8:256:1024}; LENGTH=32; LANES=64; SWEPISODES=64
  REPEATS=${REPEATS:-1}; WINDOW=${WINDOW:-3}
elif [ "${SMOKE:-0}" = 1 ]; then
  SWEEP_CPU=${SWEEP_CPU:-1:8:64}; SWEEP_CUDA=${SWEEP_CUDA:-1:64:256}; LENGTH=32; LANES=64; SWEPISODES=64
else
  SWEEP_CPU=${SWEEP_CPU:-1:8:64:256:1024}; SWEEP_CUDA=${SWEEP_CUDA:-1:256:512:1024:2048}; LENGTH=64; LANES=256; SWEPISODES=64
fi
DEVICES=${DEVICES:-cpu cuda}; ARMS=${ARMS:-offline step frames}; REPEATS=${REPEATS:-3}; WINDOW=${WINDOW:-10}; CPU_SOURCES=${CPU_SOURCES:-1}
sweep() { [ $1 = cpu ] && echo $SWEEP_CPU | tr : " " || echo $SWEEP_CUDA | tr : " "; }
# The stored States hold the largest offline batch; the episode set holds one episode per lane of the largest count.
N=$(for d in $DEVICES; do sweep $d; done | tr " " "\n" | sort -n | tail -1)
has() { [[ " $ARMS " == *" $1 "* ]]; }
# Source workers can hold hundreds of envs whose reset takes seconds each (HumanoidMaze: 4 s).
export STABLE_ENVS_WORKER_TIMEOUT=${STABLE_ENVS_WORKER_TIMEOUT:-3600}
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
  echo " \"node_jobs_at_start\": \"$(squeue -h -w "$(hostname -s)" -o '%i %u %b %C' 2>/dev/null | tr '\n' ';')\","
  echo " \"cpu_sources\": $CPU_SOURCES, \"devices\": \"$DEVICES\", \"arms\": \"$ARMS\", \"sweep_cpu\": \"$SWEEP_CPU\", \"sweep_cuda\": \"$SWEEP_CUDA\","
  echo " \"n_states\": $N, \"length\": $LENGTH, \"repeats\": $REPEATS}"
} > "$OUT/run.json"
cat "$OUT/run.json"

venv() {  # venv FAMILY -> python path of a fresh family venv in $WORK
  local fam=$1 extras groups
  case $fam in
    classic) extras="classic"; groups="pin-classic";;
    navigation) extras="navigation navigation-gpu"; groups="pin-navigation";;
    pusht) extras="pusht pusht-gpu"; groups="pin-pusht";;
    dmc) extras="dmcontrol dmcontrol-gpu"; groups="pin-dmc gpu-mjwarp";;
    ogbench) extras="ogbench ogbench-manipulation ogbench-manipulation-gpu"; groups="pin-ogbench";;
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
    cp -r ${NVDIFFRAST_SITE:-$HOME/nvdiffrast-site}/{nvdiffrast,nvdiffrast-0.4.0.dist-info,_nvdiffrast_c*.so} "$sp"/ 2>/dev/null
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
  if [ -n "${VENV_PYTHON:-}" ]; then P=$VENV_PYTHON; else P=$(venv $fam) || { rm -rf $WORK/venv-$fam; continue; }; fi
  $P -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
  if [ $fam = craftax ]; then
    # Craftax has no source renderer to restore States into; its comparison is the upstream jitted step.
    mkdir -p $OUT/craftax
    $PIN $P -m benchmarks.craftax_bench --envs "${envs[@]}" --device cuda --repeats $REPEATS --output $OUT/craftax/gpu.json
    JAX_PLATFORMS=cpu $PIN $P -m benchmarks.craftax_bench --envs "${envs[@]}" --device cpu --batches 64 256 \
      --repeats $REPEATS --calls 10 --output $OUT/craftax/cpu.json
    rm -rf $WORK/venv-$fam
    continue
  fi
  if [ "${MODE:-sweep}" = gifs ]; then
    egl $PIN $P -m benchmarks.gifs --envs "${envs[@]}" --out $OUT/gifs --device cuda
    rm -rf $WORK/venv-$fam
    continue
  fi
  OFF="$PIN $P -m benchmarks.offline_sweep"; ON="$PIN $P -m benchmarks.online_ladder"
  S=${STATES_DIR:-$WORK}/states-$fam; D=${STATES_DIR:-$WORK}/episodes-$fam; mkdir -p $S $D
  if [ "${MODE:-sweep}" = throughput ]; then
    # Online throughput with frames on one CPU budget or one GPU: the source as one env, a Gymnasium SyncVectorEnv
    # and an AsyncVectorEnv (CPU OpenGL), and stable-envs on CPU and on CUDA.
    sw $ON dataset --envs "${envs[@]}" --out $D --episodes $LANES --length $LENGTH
    for E in "${envs[@]}"; do
      echo "--- $E $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
      n=$OUT/throughput/$E; mkdir -p $n
      COMMON="--env $E --data $D --procs $NCORES --lanes $LANES --threads $NCORES --repeats $REPEATS"
      sw $ON run $COMMON --episodes $SWEPISODES --arms src-single-render src-sync-render src-async-render --output $n/source-cpu.json
      sw $ON run $COMMON --episodes $SWEPISODES --arms snv-render-cpu --output $n/snv-cpu.json
      egl $ON run $COMMON --arms snv-render-gpu --output $n/snv-gpu.json
    done
    rm -rf $WORK/venv-$fam $S $D
    continue
  fi
  # peak_memory: the highest memory use of this job so far, from the job's cgroup (cgroup v2).
  peak_memory() {
    local d=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | tail -1)
    d=${d%/}
    while [[ $d == /sys/fs/cgroup/* ]] && [[ ${d##*/} != job_* ]]; do d=$(dirname "$d"); done
    [ -r "$d/memory.peak" ] && echo "peak_mem_GiB $(( $(cat "$d/memory.peak") >> 30 ))"
  }
  # gl DEVICE CMD...: run CMD under software OpenGL for CPU arms and under EGL for CUDA arms.
  gl() { local d=$1; shift; [ $d = cpu ] && sw "$@" || egl "$@"; }
  # wait_gpu: before a CUDA job, wait up to 10 minutes while an earlier arm's OpenGL contexts still fill the GPU;
  # a CUDA process cannot start while GPU memory is full.
  wait_gpu() {
    [[ $DEVICES == *cuda* ]] || return 0
    local i used
    for i in $(seq 60); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$used" -lt 4096 ] && return 0
      sleep 10
    done
    echo "gpu still holds ${used} MiB"
  }
  # cuda_lost: true when a new process can no longer initialise CUDA, as after a GPU driver fault.
  cuda_lost() { [[ $DEVICES == *cuda* ]] && [ "$(python3 -c 'import ctypes; print(ctypes.CDLL("libcuda.so.1").cuInit(0))' 2>&1)" != 0 ]; }
  # once FILE CMD...: run CMD unless FILE holds a result; stop the run if the GPU no longer initialises CUDA.
  once() {
    local f=$1; shift
    [ -s "$f" ] && { echo "have $f"; return 0; }
    wait_gpu; "$@"
    cuda_lost && { echo "CUDA LOST after $f"; exit 3; }
    return 0
  }
  has offline && sw $OFF states --envs "${envs[@]}" --out $S --n $N --devices $DEVICES
  if has step || has frames; then sw $ON dataset --envs "${envs[@]}" --out $D --episodes $N --length $LENGTH; fi
  for E in "${envs[@]}"; do
    echo "--- $E $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) $(peak_memory)"
    o=$OUT/offline/$E; n=$OUT/online/$E; mkdir -p $o $n
    if [ -n "${SRC_FROM:-}" ]; then
      for f in "$SRC_FROM"/offline/$E/src-*.json; do [ -s "$f" ] && cp -n "$f" $o/; done
      for f in "$SRC_FROM"/online/$E/src-*.json; do [ -s "$f" ] && cp -n "$f" $n/; done
    fi
    GPUSRC=$($P -c "import stable_envs as s; from benchmarks.compat import reference_gpu_render as r; print(r(s.info('$E')['class']) or '')")
    for dev in $DEVICES; do
      tag=$([ $dev = cpu ] && echo cpu || echo gpu)
      if has offline; then
        once $o/snv-$tag.json gl $dev $OFF ours --envs $E --states $S --device $dev --threads $NCORES --batches $(sweep $dev) --repeats $REPEATS --seconds $WINDOW --output $o/snv-$tag.json
        once $o/mse-$tag.json gl $dev $OFF mse --envs $E --states $S --device $dev --output $o/mse-$tag.json
        for p in 1 $NCORES; do
          [ $CPU_SOURCES = 1 ] && once $o/src-cpu-${p}p.json sw $OFF source --envs $E --states $S --procs $p --repeats $REPEATS --seconds $WINDOW --output $o/src-cpu-${p}p.json
          [ $dev = cuda ] && [ "$GPUSRC" = mujoco-egl ] && once $o/src-gpu-${p}p.json egl $OFF source --gpu --envs $E --states $S --procs $p --repeats $REPEATS --seconds $WINDOW --output $o/src-gpu-${p}p.json
        done
      fi
      # Online arms at every lane count L: ours with L lanes; the source with L envs over min(L, NCORES) workers.
      for L in $(sweep $dev); do
        p=$(( L < NCORES ? L : NCORES )); sl=$(( (L + p - 1) / p ))
        # OGBench maze sources take hours per whole episode at 1024 lanes and more; time their first 8 steps there,
        # and time ours on the same 8-step window as well, so both rates carry the same share of reset cost.
        slice=$([ $dev = cuda ] && [ $fam = ogbench ] && [ $L -ge 1024 ] && [[ $E == *Maze* ]] && echo "--source-steps 8")
        COMMON="--env $E --data $D --procs $p --source-lanes $sl --lanes $L --threads $NCORES --repeats $REPEATS --seconds $WINDOW --episodes $L"
        # Envs listed in SRC_CAP_FILE have a source that reaches its full rate by 8 lanes; their source cells stop at
        # SRC_CAP_L lanes, where the rate no longer grows, and ours keeps the whole sweep.
        srcok=1
        grep -qx "$E" "${SRC_CAP_FILE:-/workspace/src_cap_envs}" 2>/dev/null && [ $L -gt ${SRC_CAP_L:-256} ] && srcok=0
        if has step; then
          [ $CPU_SOURCES = 1 ] && [ $srcok = 1 ] && once $n/src-step-L$L.json sw $ON run $COMMON --arms src-pipelined-step --output $n/src-step-L$L.json
          arm=$([ $dev = cpu ] && echo snv-step || echo snv-step-gpu)
          once $n/snv-step-$tag-L$L.json gl $dev $ON run $COMMON --arms $arm --output $n/snv-step-$tag-L$L.json
          [ -n "$slice" ] && [ $srcok = 1 ] && once $n/snv-step-$tag-L$L-s8.json gl $dev $ON run $COMMON --arms $arm --chunks 1 8 $slice --output $n/snv-step-$tag-L$L-s8.json
        fi
        if has frames; then
          [ $CPU_SOURCES = 1 ] && once $n/src-render-cpu-L$L.json sw $ON run $COMMON --arms src-pipelined-render --src-label cpu --ours-device cpu --output $n/src-render-cpu-L$L.json
          [ $dev = cuda ] && [ -n "$GPUSRC" ] && once $n/src-render-gpu-L$L.json egl $ON run $COMMON --arms src-pipelined-render --src-label gpu --ours-device cuda $slice --output $n/src-render-gpu-L$L.json
          once $n/snv-render-$tag-L$L.json gl $dev $ON run $COMMON --arms snv-render-$tag --output $n/snv-render-$tag-L$L.json
          [ -n "$slice" ] && once $n/snv-render-$tag-L$L-s8.json gl $dev $ON run $COMMON --arms snv-render-$tag --chunks 1 8 $slice --output $n/snv-render-$tag-L$L-s8.json
        fi
      done
    done
  done
  echo "=== end of family $fam $(date -Is) $(peak_memory)"
  [ -z "${VENV_PYTHON:-}" ] && rm -rf $WORK/venv-$fam
  [ -z "${STATES_DIR:-}" ] && rm -rf $S $D
done
echo "{\"end\": \"$(date -Is)\"}" > "$OUT/end.json"
echo done
