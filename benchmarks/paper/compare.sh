#!/bin/bash
# stable-envs against the source libraries on this machine, in one command.
# usage: bash benchmarks/paper/compare.sh OUT [--gpu-only] [--sweep L:L...] [--smoke] [--ref REF] [--gpus N] [ENV ...]
#        bash benchmarks/paper/compare.sh --summary OUT
#   Default: every paper env (fleet/costs.json), CPU and GPU cells, the lane counts and batches of the paper.
#   --gpu-only     only the GPU cells (steps, frames+steps, offline), ours at the --sweep lane counts and batches
#   --sweep        ours' GPU lane counts and batches (default 1:8:64:256:1024:4096; with --gpu-only 1024:2048)
#   --smoke        short episodes and small lane counts, to check the machine
#   --ref REF      time the library at git REF (for example v0.1.0) with the harness of this checkout
#   --gpus N       use the first N GPUs (default: all); one env per GPU at a time, each on its own 8 or 16 cores
#   --summary OUT  print the headline of OUT again (also of a partial OUT) and run nothing
#   --dry-run      print the setup steps and every run.sh call of every env; install and run nothing
# It installs what it needs (apt packages, uv, stable-worldmodel, family venvs, nvdiffrast) under COMPARE_WORK
# (default ~/.cache/snv-compare), skips envs whose OUT/ENV/done.json exists, and ends with OUT/HEADLINE.md.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$(cd "$HERE/../.." && pwd)
SWM_SHA=21446f1ede6d5284e981bd7b47f432b994e6d812
MODE=full; SWEEP=; SMOKE=0; DRY=0; REF=; NGPU=; SUMMARY=0; ENVS=(); OUT=
while [ $# -gt 0 ]; do
  case $1 in
    --gpu-only) MODE=gpu;;
    --sweep) SWEEP=$2; shift;;
    --smoke) SMOKE=1;;
    --ref) REF=$2; shift;;
    --gpus) NGPU=$2; shift;;
    --summary) SUMMARY=1;;
    --dry-run) DRY=1;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    -*) echo "unknown option $1" >&2; exit 2;;
    *) if [ -z "$OUT" ]; then OUT=$1; else ENVS+=("$1"); fi;;
  esac
  shift
done
[ -n "$OUT" ] || { sed -n '2,16p' "$0"; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

summary() {
  local b; b=$(python3 -c "import json; print(json.load(open('$OUT/compare.json')).get('batches', ''))" 2>/dev/null)
  python3 "$HERE/headline.py" "$OUT" ${b:+--batches $b} --write
}
[ $SUMMARY = 1 ] && { summary; exit 0; }

if [ $MODE = gpu ]; then SWEEP_CUDA=${SWEEP:-1024:2048}; else SWEEP_CUDA=${SWEEP:-1:8:64:256:1024:4096}; fi
SWEEP_CPU=1:8:64:256:1024
[ $SMOKE = 1 ] && { SWEEP_CPU=1:8; [ -z "$SWEEP" ] && SWEEP_CUDA=8:64; }
W=${COMPARE_WORK:-$HOME/.cache/snv-compare}
mkdir -p "$W"/{tmp,venvs,uvcache,st}
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$OUT/compare.log"; }

env_list() {  # FAMILY ENV per line, the most expensive env first
  python3 - "$HERE/fleet/costs.json" "${ENVS[@]}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want = sys.argv[2:] or list(c["envs"])
for e in sorted(want, key=lambda e: -c["env_s"].get(e, c["family_s"].get(c["envs"].get(e, ""), 0))):
    if e not in c["envs"]:
        sys.exit(f"unknown env {e}")
    print(c["envs"][e], e)
PY
}

if [ "$DRY" = 1 ]; then
  echo "setup: apt-get xvfb xauth libgl1 libgl1-mesa-dri libegl1 libosmesa6 libglfw3 libglew2.2 libglu1-mesa git rsync python3-venv"
  echo "setup: uv 0.13.5 in $W/uvenv; stable-worldmodel $SWM_SHA in $W/swm; family venvs in $W/venvs (torch 2.14.1+cu126)"
  echo "setup: nvdiffrast 253ac4fc built with nvcc for the Rocket venv"
  [ -n "$REF" ] && echo "library: $REF = $(git -C "$SRC" rev-parse "$REF^{commit}" 2>/dev/null || echo 'NOT FOUND (git fetch --tags)'), harness of $(git -C "$SRC" rev-parse --short HEAD)"
  echo "GPUs: $(nvidia-smi -L 2>/dev/null | grep -c ^GPU) found; ${NGPU:-all} used, one env per GPU at a time"
  echo "mode $MODE, CPU lane counts $SWEEP_CPU, GPU lane counts and batches $SWEEP_CUDA, windows ${WINDOW:-3} s x ${REPEATS:-3}"
  env_list > "$W/tmp/dry-queue" || exit 2
  while read -r fam E; do
    [ -f "$OUT/$E/done.json" ] && { echo "$E: done, skipped"; continue; }
    DRY_RUN=1 W=$W MODE=$MODE SWEEP_CPU=$SWEEP_CPU SWEEP_CUDA=$SWEEP_CUDA CAPS=$HERE/fleet/src_cap_envs \
      bash "$HERE/job.sh" "$OUT/$E" "$fam" "$E"
  done < "$W/tmp/dry-queue"
  echo "end: $OUT/HEADLINE.md$([ $MODE = gpu ] && echo " for batches $SWEEP_CUDA")"
  exit 0
fi

# --- setup: system packages, uv, stable-worldmodel, the checkout under test
if command -v apt-get > /dev/null; then
  S=$([ "$(id -u)" = 0 ] || echo sudo)
  $S apt-get update -qq > /dev/null && DEBIAN_FRONTEND=noninteractive $S apt-get install -y -qq xvfb xauth libgl1 \
    libgl1-mesa-dri libegl1 libosmesa6 libglfw3 libglew2.2 libglu1-mesa git rsync python3-venv > /dev/null \
    || log "apt-get failed; continuing with the installed packages"
fi
[ -x "$W/uvenv/bin/uv" ] || { python3 -m venv "$W/uvenv" && "$W/uvenv/bin/pip" install -q "uv==0.13.5"; } || exit 1
if [ "$(git -C "$W/swm" rev-parse HEAD 2>/dev/null)" != $SWM_SHA ]; then
  rm -rf "$W/swm"; git init -q "$W/swm" && git -C "$W/swm" fetch -q --depth 1 https://github.com/galilai-group/stable-worldmodel $SWM_SHA \
    && git -C "$W/swm" checkout -q FETCH_HEAD || { log "stable-worldmodel fetch failed"; exit 1; }
fi
REPO=$SRC
if [ -n "$REF" ]; then
  sha=$(git -C "$SRC" rev-parse "$REF^{commit}") || exit 2
  REPO=$W/lib-${sha:0:8}
  [ -d "$REPO" ] || git -C "$SRC" worktree add -q --detach "$REPO" "$sha" || exit 1
  # The library at REF with the harness of this checkout.
  rm -rf "$REPO/benchmarks" && cp -r "$SRC/benchmarks" "$REPO/" && echo "$sha" > "$REPO/.bench_main_sha"
fi
export SNV_REPO=$REPO SWM=$W/swm UV=$W/uvenv/bin/uv VENVS=$W/venvs UV_CACHE_DIR=$W/uvcache UV_LINK_MODE=copy \
  UV_HTTP_TIMEOUT=600 TMPDIR=$W/tmp W NVDIFFRAST_SITE=${NVDIFFRAST_SITE:-} MODE SWEEP_CPU SWEEP_CUDA SMOKE \
  CAPS=$HERE/fleet/src_cap_envs WINDOW=${WINDOW:-3} REPEATS=${REPEATS:-3}
G=${NGPU:-$(nvidia-smi -L | grep -c ^GPU)}
python3 - "$OUT" "$REPO" "$SRC" "$MODE" "$SWEEP_CUDA" "$G" <<'EOF'
import json, os, subprocess, sys
out, repo, src, mode, sweep, g = sys.argv[1:]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
gpus = sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines()
m = {"commit": sh(f"git -C {repo} rev-parse HEAD"), "harness": sh(f"git -C {src} rev-parse HEAD"),
     "gpu": f"{len(gpus)}x {gpus[0]}" if gpus else "none", "gpus_used": int(g),
     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
     "cpu": sh("lscpu | sed -n 's/^Model name: *//p' | head -1"), "nproc": os.cpu_count(),
     "cpu_quota": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"), "mem_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30),
     "date": sh("date -Is")}
json.dump(m, open(f"{out}/machine.json", "w"), indent=1)
json.dump({"mode": mode, "sweep_cuda": sweep, "batches": sweep if mode == "gpu" else ""}, open(f"{out}/compare.json", "w"))
print(m)
EOF

# --- jobs: the most expensive env first; envs with done.json are skipped
env_list > "$OUT/.queue" || exit 2
[ -s "$OUT/.queue" ] || exit 2
log "$(wc -l < "$OUT/.queue" | tr -d ' ') envs, mode $MODE, GPU sweep $SWEEP_CUDA, $G GPUs, library $(git -C "$REPO" rev-parse --short HEAD)"

worker() {  # worker GPU: take envs from the queue until it is empty
  local gpu=$1 line fam E P rc
  while true; do
    line=$(flock "$OUT/.queue.lock" sh -c "head -1 '$OUT/.queue'; sed -i 1d '$OUT/.queue'")
    [ -z "$line" ] && return 0
    set -- $line; fam=$1; E=$2
    [ -f "$OUT/$E/done.json" ] && continue
    P=$(flock "$W/venvs/$fam.lock" bash "$HERE/fleet/venv.sh" $fam 2>> "$OUT/compare.log") || { log "venv $fam failed: $E"; continue; }
    log "GPU $gpu: $E"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu P=$P PYTHONPATH=$REPO bash "$REPO/benchmarks/paper/job.sh" "$OUT/$E" $fam $E )
    rc=$?
    log "GPU $gpu: $E rc=$rc"
    [ $rc = 3 ] && { log "GPU $gpu no longer initialises CUDA; its worker stops"; return 3; }
  done
}
for g in $(seq 0 $((G - 1))); do worker $g & sleep 5; done
wait
finished=(); for d in "$OUT"/*/; do [ -f "$d/done.json" ] && finished+=("$d"); done
python3 "$HERE/fleet/check_cells.py" "${finished[@]}" > "$OUT/problems.txt"
log "$(ls "$OUT"/*/done.json 2>/dev/null | wc -l | tr -d ' ') envs finished; cells not ok: $(grep -vc capped "$OUT/problems.txt")"
summary
