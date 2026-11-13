#!/bin/bash
# stable-envs v0.1.0 against the source libraries on this machine, in one command.
# usage: bash benchmarks/paper/compare.sh OUT [--missing-only] [options] [ENV ...]
#        bash benchmarks/paper/compare.sh --summary OUT
#   Default: the envs TwoRoom PushT DMCReacher OGBCube (or the ENV arguments); steps,
#   frames+steps and offline, on CPU and GPU, ours and the source, at lane counts and batches 1:16:256:1024 for both
#   devices, with no source caps. A cell that runs out of memory or time (STEP_LIMIT per cell process, default 20m) is
#   kept as a result with that status.
#   --missing-only  start from the rt4 L40S cells (rt4_l40s.tar.gz: steps and frames+steps at the lane counts rt4
#                   timed) and time only the other cells; the offline cells are all timed again on this machine
#   --summary       print the headline of OUT again (also of a partial OUT) and run nothing
#   --dry-run       print the setup steps and every run.sh call; install and run nothing
#   --gpu-only      only the GPU cells;  --sweep L:L  other lane counts and batches;  --smoke  a short check
#   --ref REF       the library to time (default v0.1.0; HEAD for this checkout), with the harness of this checkout
#   --gpus N        use the first N GPUs (default: all); one env per GPU at a time, each on its own 8 or 16 cores
# It installs what it needs (apt packages, uv, stable-worldmodel, family venvs, nvdiffrast) under COMPARE_WORK
# (default ~/.cache/snv-compare). It times every env's stable-envs cells first, then every env's source cells (an
# env whose OUT/ENV/done-ours.json or done.json exists is skipped in that phase), and ends with OUT/HEADLINE.md.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$(cd "$HERE/../.." && pwd)
SWM_SHA=21446f1ede6d5284e981bd7b47f432b994e6d812
MODE=full; SWEEP=; SMOKE=0; DRY=0; REF=v0.1.0; MISSING=0; NGPU=; SUMMARY=0; ENVS=(); OUT=
while [ $# -gt 0 ]; do
  case $1 in
    --gpu-only) MODE=gpu;;
    --missing-only) MISSING=1;;
    --sweep) SWEEP=$2; shift;;
    --smoke) SMOKE=1;;
    --ref) REF=$2; shift;;
    --gpus) NGPU=$2; shift;;
    --summary) SUMMARY=1;;
    --dry-run) DRY=1;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    -*) echo "unknown option $1" >&2; exit 2;;
    *) if [ -z "$OUT" ]; then OUT=$1; else ENVS+=("$1"); fi;;
  esac
  shift
done
[ -n "$OUT" ] || { sed -n '2,18p' "$0"; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

summary() {
  local b; b=$(python3 -c "import json; print(json.load(open('$OUT/compare.json')).get('sizes', ''))" 2>/dev/null)
  python3 "$HERE/headline.py" "$OUT" ${b:+--batches $b} --write
}
[ $SUMMARY = 1 ] && { summary; exit 0; }

SWEEP_CPU=${SWEEP:-1:16:256:1024}; SWEEP_CUDA=$SWEEP_CPU
[ $SMOKE = 1 ] && [ -z "$SWEEP" ] && { SWEEP_CPU=1:16; SWEEP_CUDA=1:16; }
[ ${#ENVS[@]} = 0 ] && ENVS=(TwoRoom PushT DMCReacher OGBCube)
# CPU frames at every lane count, with no cap.
export FRAMES_CPU=$SWEEP_CPU
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

# layout: GPUs, usable physical cores (affinity and CPU quota), concurrent jobs, and whether each job gets 16 cores.
layout() {
  python3 - "${NGPU:-}" <<'PY'
import os, subprocess, sys
try:
    g = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
except OSError:
    g = 0
g = int(sys.argv[1]) if sys.argv[1] else g
try:
    aff = sorted(os.sched_getaffinity(0))
    phys = len({open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip() for c in aff})
except (AttributeError, OSError):
    phys = os.cpu_count() or 0
try:
    q, p = open("/sys/fs/cgroup/cpu.max").read().split()
    quota = phys if q == "max" else int(int(q) / int(p))
except (OSError, ValueError):
    quota = phys
usable = min(phys, quota)
jobs = max(1, min(g, usable // 8)) if g else 0
par = int(usable >= 16 * jobs) if jobs else 0
print(g, usable, jobs, par)
PY
}
if [ "$DRY" = 1 ]; then
  echo "setup: apt-get xvfb xauth libgl1 libgl1-mesa-dri libegl1 libosmesa6 libglfw3 libglew2.2 libglu1-mesa git rsync python3-venv"
  echo "setup: uv 0.13.5 in $W/uvenv; stable-worldmodel $SWM_SHA in $W/swm; family venvs in $W/venvs (torch 2.14.1+cu126)"
  echo "setup: nvdiffrast 253ac4fc built with nvcc for the Rocket venv"
  [ -n "$REF" ] && echo "library: $REF = $(git -C "$SRC" rev-parse "$REF^{commit}" 2>/dev/null || echo 'NOT FOUND (git fetch --tags)'), harness of $(git -C "$SRC" rev-parse --short HEAD)"
  read -r g usable jobs par < <(layout)
  echo "layout: $g GPUs, $usable usable physical cores, $jobs jobs at a time (one per GPU), each on $([ "$par" = 1 ] && echo "16 cores (CPU and CUDA passes at the same time)" || echo "8 cores (passes one after the other)")"
  [ "$g" -gt "$jobs" ] && echo "layout: fewer jobs than GPUs: $usable physical cores hold $jobs sets of 8"
  echo "mode $MODE, CPU lane counts $SWEEP_CPU, GPU lane counts and batches $SWEEP_CUDA, windows ${WINDOW:-3} s x ${REPEATS:-3}"
  env_list > "$W/tmp/dry-queue" || exit 2
  for side in ours source; do
    echo "phase $side: every env's $side cells$([ $side = ours ] && echo ' (stable-envs and the frame MSE), on all GPUs' || echo ', after the ours phase; STEP_LIMIT per cell, else timeout')"
    while read -r fam E; do
      [ -f "$OUT/$E/done-$side.json" ] && { echo "  $E: done, skipped"; continue; }
      DRY_RUN=1 W=$W MODE=$MODE SIDE=$side SWEEP_CPU=$SWEEP_CPU SWEEP_CUDA=$SWEEP_CUDA CAPS=/dev/null \
        bash "$HERE/job.sh" "$OUT/$E" "$fam" "$E"
    done < "$W/tmp/dry-queue"
  done
  echo "cells to time ($([ "$MISSING" = 1 ] && echo "not in rt4_l40s.tar.gz, " )not in OUT), with an estimate:"
  dry_envs=(); while read -r _ e; do dry_envs+=("$e"); done < "$W/tmp/dry-queue"
  extra=(); [ "$MISSING" = 1 ] && extra=(--missing-only)
  python3 "$HERE/plan.py" --envs "${dry_envs[@]}" --sizes "$SWEEP_CUDA" --out "$OUT" --jobs "$jobs" ${extra[@]+"${extra[@]}"}
  echo "end: $OUT/HEADLINE.md"
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
  CAPS=/dev/null WINDOW=${WINDOW:-3} REPEATS=${REPEATS:-3} STEP_LIMIT=${STEP_LIMIT:-20m}
read -r G usable JOBS PAR < <(layout)
[ "$JOBS" -ge 1 ] || { log "no GPU found"; exit 2; }
[ "$PAR" = 1 ] || export PARALLEL_DEVICES=0
log "layout: $G GPUs, $usable usable physical cores, $JOBS jobs at a time, $([ "$PAR" = 1 ] && echo 16 || echo 8) cores per job"
[ "$G" -gt "$JOBS" ] && log "fewer jobs than GPUs: the CPU quota or the cores hold only $JOBS sets of 8 cores"
if [ "$MISSING" = 1 ]; then
  # The rt4 L40S cells; run.sh times only the cell files that are not there.
  tar xzf "$HERE/rt4_l40s.tar.gz" -C "$W/tmp" && env_list | while read -r _ e; do
    d=$W/tmp/rt4-l40s/$e
    [ -d "$d" ] && [ ! -f "$OUT/$e/hardware-rt4.json" ] || continue
    for p in "" pass-cpu/ pass-cuda/; do
      mkdir -p "$OUT/$e/${p}online/$e"
      for L in ${SWEEP_CPU//:/ }; do cp -n "$d/online/$e/"*-L"$L".json "$OUT/$e/${p}online/$e/" 2> /dev/null; done
    done
    cp "$d/hardware.json" "$OUT/$e/hardware-rt4.json"
  done
fi
python3 - "$OUT" "$REPO" "$SRC" "$MODE" "$SWEEP_CUDA" "$JOBS" <<'EOF'
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
json.dump({"mode": mode, "sizes": sweep}, open(f"{out}/compare.json", "w"))
print(m)
EOF

# --- jobs: every env's stable-envs cells first, on all GPUs; then every env's source cells. The most expensive env
# first in each phase; an env whose done-SIDE.json exists is skipped in that phase.
worker() {  # worker GPU SIDE: take envs from the queue until it is empty
  local gpu=$1 side=$2 line fam E P rc
  while true; do
    line=$(flock "$OUT/.queue.lock" sh -c "head -1 '$OUT/.queue'; sed -i 1d '$OUT/.queue'")
    [ -z "$line" ] && return 0
    set -- $line; fam=$1; E=$2
    [ -f "$OUT/$E/done-$side.json" ] && continue
    P=$(flock "$W/venvs/$fam.lock" bash "$HERE/fleet/venv.sh" "$fam" 2>> "$OUT/compare.log") || { log "venv $fam failed: $E"; continue; }
    log "GPU $gpu: $E ($side)"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu P=$P SIDE=$side PYTHONPATH=$REPO bash "$REPO/benchmarks/paper/job.sh" "$OUT/$E" "$fam" "$E" )
    rc=$?
    log "GPU $gpu: $E ($side) rc=$rc"
    [ $rc = 3 ] && { log "GPU $gpu no longer initialises CUDA; its worker stops"; return 3; }
  done
}
for side in ours source; do
  env_list > "$OUT/.queue" || exit 2
  log "$side cells: $(wc -l < "$OUT/.queue" | tr -d ' ') envs, mode $MODE, GPU sweep $SWEEP_CUDA, $JOBS jobs, library $(git -C "$REPO" rev-parse --short HEAD)"
  for g in $(seq 0 $((JOBS - 1))); do worker "$g" $side & sleep 5; done
  wait
done
finished=(); for d in "$OUT"/*/; do [ -f "$d/done.json" ] && finished+=("$d"); done
python3 "$HERE/fleet/check_cells.py" "${finished[@]}" > "$OUT/problems.txt"
log "$(ls "$OUT"/*/done.json 2>/dev/null | wc -l | tr -d ' ') envs finished; cells not ok: $(grep -c . "$OUT/problems.txt")"
summary
