#!/bin/bash
# Offline and online speed of stable-envs against the source, every arm pinned to the same CPU cores.
# usage (from the repository root): sbatch -p qed -n 1 --exclusive --gres=gpu:1 -t 48:00:00 benchmarks/paper/run.sh OUT FAMILY=ENV,ENV ...
#   SMOKE=1 uses small state, batch and episode counts. CORES (default 0-7) is the core set of every arm.
set -uo pipefail
OUT=$(realpath -m "$1"); shift
# Slurm runs a copy of this script, so the repository is the submit directory.
REPO=${SNV_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
[ -f "$REPO/pyproject.toml" ] || { echo "no repository at $REPO" >&2; exit 2; }
SWM=${SWM:-$HOME/lewm-hyd/swm}
CORES=${CORES:-0-7}
NCORES=$(taskset -c "$CORES" nproc)
PIN="taskset -c $CORES"
UV=${UV:-$HOME/.local/bin/uv}
WORK=${TMPDIR:-/tmp}/snv-bench-$$
mkdir -p "$OUT"/{offline,online,logs} "$WORK"
trap 'rm -rf "$WORK"' EXIT
if [ "${SMOKE:-0}" = 1 ]; then
  N=256; BATCHES="1 64 256"; EPISODES=64; LENGTH=32; LANES=64; SLANES=8
else
  N=10000; BATCHES="1 256 1024 4096"; EPISODES=1024; LENGTH=196; LANES=1024; SLANES=8
fi
REPEATS=${REPEATS:-10}
export PYTHONPATH=$REPO SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
# The uv cache stays in home: node-local disk holds one family venv at a time.
export UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv} XDG_CACHE_HOME=$WORK/cache
# One GPU only: --exclusive exposes all of them, and torch and JAX may number several differently.
export CUDA_VISIBLE_DEVICES=$(echo "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}" | cut -d, -f1) MUJOCO_EGL_DEVICE_ID=0
egl() { env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$@"; }
sw() { env LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe PYOPENGL_PLATFORM=glx MUJOCO_GL=glfw xvfb-run -a "$@"; }

{
  echo "{\"commit\": \"$(git -C "$REPO" rev-parse HEAD)\", \"main\": \"$(git -C "$REPO" merge-base HEAD origin/main 2>/dev/null)\","
  echo " \"dirty\": $(git -C "$REPO" diff --quiet && echo false || echo true), \"swm\": \"$(git -C "$SWM" rev-parse HEAD)\","
  echo " \"host\": \"$(hostname)\", \"job\": \"${SLURM_JOB_ID:-}\", \"start\": \"$(date -Is)\", \"cores\": \"$CORES\", \"ncores\": $NCORES,"
  echo " \"cuda_visible_devices\": \"$CUDA_VISIBLE_DEVICES\", \"gpu\": \"$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name,driver_version --format=csv,noheader)\", \"smoke\": ${SMOKE:-0},"
  echo " \"n_states\": $N, \"batches\": \"$BATCHES\", \"episodes\": $EPISODES, \"length\": $LENGTH, \"lanes\": $LANES, \"repeats\": $REPEATS}"
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
    fetch) extras="robotics"; groups="pin-fetch";;
    rocket) extras="rocket"; groups="pin-rocket";;
    *) echo "unknown family $fam" >&2; return 1;;
  esac
  local v=$WORK/venv-$fam
  ( cd "$REPO" && UV_PROJECT_ENVIRONMENT=$v $UV sync --frozen --python 3.12 --no-default-groups \
      $(for e in $extras; do echo --extra $e; done) $(for g in $groups torch-cuda test; do echo --group $g; done) ) \
      > "$OUT/logs/venv-$fam.log" 2>&1 || { echo "venv $fam failed" >&2; return 1; }
  local tv; tv=$($v/bin/python -c "import torch; print(torch.__version__.split('+')[0])")
  local cu=(--extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match)
  $UV pip install --python $v/bin/python "${cu[@]}" --reinstall-package torch "torch==$tv+cu126" \
      >> "$OUT/logs/venv-$fam.log" 2>&1 || { echo "torch $fam failed" >&2; return 1; }
  $UV pip freeze --python $v/bin/python | grep "==" > $WORK/constraints-$fam.txt
  # Importing stable_worldmodel registers every swm env, and its PushT module needs pymunk.
  $UV pip install --python $v/bin/python "${cu[@]}" -c $WORK/constraints-$fam.txt -e "$SWM" "pymunk==7.2.0" \
      >> "$OUT/logs/venv-$fam.log" 2>&1 || { echo "swm $fam failed" >&2; return 1; }
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
  OFF="$PIN $P -m benchmarks.offline_sweep"; ON="$PIN $P -m benchmarks.online_ladder"
  S=$WORK/states-$fam; D=$WORK/episodes-$fam; mkdir -p $S $D
  egl $OFF states --envs "${envs[@]}" --out $S --n $N --device cuda --lanes 1024
  egl $ON dataset --envs "${envs[@]}" --out $D --episodes $EPISODES --length $LENGTH
  for E in "${envs[@]}"; do
    echo "--- $E $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
    o=$OUT/offline/$E; mkdir -p $o
    egl $OFF ours --envs $E --states $S --device cuda --batches $BATCHES --repeats $REPEATS --output $o/snv-gpu.json
    sw $OFF ours --envs $E --states $S --device cpu --threads $NCORES --batches $BATCHES --repeats $REPEATS --output $o/snv-cpu.json
    egl $OFF mse --envs $E --states $S --device cuda --output $o/mse-gpu.json
    sw $OFF mse --envs $E --states $S --device cpu --output $o/mse-cpu.json
    for p in 1 $NCORES; do
      egl $OFF source --envs $E --states $S --procs $p --repeats $REPEATS --output $o/src-gpu-${p}p.json
      sw $OFF source --envs $E --states $S --procs $p --repeats $REPEATS --output $o/src-cpu-${p}p.json
    done
    n=$OUT/online/$E; mkdir -p $n
    COMMON="--env $E --data $D --procs $NCORES --source-lanes $SLANES --lanes $LANES --threads $NCORES --repeats $REPEATS"
    egl $ON run $COMMON --arms src-pipelined-render --src-label gpu --ours-device cuda --output $n/src-render-gpu.json
    sw $ON run $COMMON --arms src-pipelined-render --src-label cpu --ours-device cpu --output $n/src-render-cpu.json
    sw $ON run $COMMON --arms src-pipelined-step snv-step --output $n/step.json
    sw $ON run $COMMON --arms snv-render-cpu --output $n/snv-render-cpu.json
    egl $ON run $COMMON --arms snv-step-gpu snv-render-gpu --output $n/snv-gpu.json
  done
  rm -rf $WORK/venv-$fam $S $D
done
echo "{\"end\": \"$(date -Is)\"}" > "$OUT/end.json"
echo done
