#!/bin/bash
# One env against its source: several calls of run.sh into OUT, each with its own arms, lane counts and pinned cores.
# usage: bash benchmarks/paper/job.sh OUT FAMILY ENV      (from the root of the checkout under test)
# env: P (family venv python), W (work dir: swm, uv, tmp, cores.lock), MODE (full or gpu), SWEEP_CPU, SWEEP_CUDA,
#   CAPS (file of "ENV LANES": step sources of ENV stop at LANES), WINDOW, REPEATS, STEP_LIMIT, PARALLEL_DEVICES (1).
# MODE=full (default):
#   cpu-a   DEVICES=cpu ARMS="offline step" at SWEEP_CPU, with the CPU sources
#   cpu-f   DEVICES=cpu ARMS=frames up to the family's CPU frames limit (frames_cpu)
#   cuda-a  DEVICES=cuda ARMS="offline step" at SWEEP_CUDA, CPU_SOURCES=0
#   cuda-f  DEVICES=cuda ARMS=frames at 1:8:64:256, then at 1024 with the source GPU cell written as capped
#   The two passes run at the same time on two disjoint sets of 8 idle physical cores when 16 are free, else in turn.
# MODE=gpu: CUDA cells only, ours at SWEEP_CUDA; the sources at the lane counts where they reach their rate:
#   g-a     DEVICES=cuda ARMS="offline step" at SWEEP_CUDA, step sources up to the cap (default 1024)
#   g-s     ARMS=step at the cap, when SWEEP_CUDA does not hold it
#   g-f8    ARMS=frames at 8 lanes with the CPU and GPU sources; g-f64 at 64 lanes with the GPU source
#   g-f     ARMS=frames at SWEEP_CUDA, source GPU cells above 256 lanes written as capped
# done.json marks a finished job; cores.txt records the cores and how busy they were.
set -uo pipefail
OUT=$1; fam=$2; E=$3
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${MODE:-full}
SWEEP_CPU=${SWEEP_CPU:-1:8:64:256:1024}; SWEEP_CUDA=${SWEEP_CUDA:-1:8:64:256:1024:4096}
mkdir -p "$OUT" "$W/tmp"
LOG=$OUT/run.log
CAPL=$(awk -v e="$E" '$1 == e {print ($2 ? $2 : 256)}' "${CAPS:-/dev/null}" 2>/dev/null)
CAPF=$OUT/.cap; if [ -n "$CAPL" ]; then echo "$E" > $CAPF; else : > $CAPF; fi

frames_cpu() {
  case $E in OGBCube|OGBScene) echo 1:8; return;; esac
  case $fam in dmc|fetch|ogbench|rocket) echo 1:8:64;; *) echo 1:8:64:256;; esac
}

# reserve N: N idle physical cores, at most 10% busy, not held by another job on this host (W/cores.held).
# With TRIES=1 and PICK_STRICT=1 it fails instead of taking busy cores.
reserve() {
  local n=$1 i got busy
  for i in $(seq ${TRIES:-20}); do
    got=$(flock "$W/cores.lock" sh -c "PICK_EXCLUDE=\$(cut -d' ' -f2 '$W/cores.held' 2>/dev/null | tr '\n' ,) \
      PICK_MAX_BUSY=0.10 python3 '$HERE/pick_cores.py' $n 2> '$W/tmp/busy-$$' && true" ) && [ -n "$got" ] && break
    got=; sleep 30
  done
  [ -z "$got" ] && [ "${PICK_STRICT:-0}" = 1 ] && return 1
  [ -z "$got" ] && got=$(flock "$W/cores.lock" sh -c "PICK_EXCLUDE=\$(cut -d' ' -f2 '$W/cores.held' 2>/dev/null | tr '\n' ,) \
      python3 '$HERE/pick_cores.py' $n 2> '$W/tmp/busy-$$'")
  busy=$(sed -n 's/^busy //p' "$W/tmp/busy-$$")
  flock "$W/cores.lock" sh -c "echo '$$ $got' >> '$W/cores.held'"
  echo "$got $busy"
}
release() { flock "$W/cores.lock" sed -i "/^$$ /d" "$W/cores.held"; }
trap release EXIT

# call NAME DEV ARMS SWEEP CORES BUSY [VAR=VALUE ...]: one run.sh call into OUT/pass-DEV.
call() {
  local name=$1 dev=$2 arms=$3 sweep=$4 cores=$5 busy=$6 o=$OUT/pass-$2; shift 6
  echo "$name $(date -Is) cores $cores busy ${busy:-?} load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/cores.txt
  env "$@" CORES=$cores DEVICES=$dev ARMS="$arms" SWEEP_CPU=$sweep SWEEP_CUDA=$sweep VENV_PYTHON=$P \
    STATES_DIR=$W/st/$E-$dev WINDOW=${WINDOW:-3} REPEATS=${REPEATS:-3} STEP_LIMIT=${STEP_LIMIT:-40m} \
    SRC_CAP_FILE=$CAPF SRC_CAP_L=${CAPL:-256} bash "$HERE/run.sh" $o "$fam=$E" >> $LOG.$dev 2>&1
  local rc=$?
  cp $o/run.json $OUT/run-$name.json 2> /dev/null
  rm -f $o/end.json
  echo "call $name rc=$rc $(date -Is)" >> $LOG.$dev
  [ $rc = 3 ] && return 3
  return 0
}

gpusrc() {
  $P -c "import stable_envs as s; from tests.helpers.reference.sources import reference_gpu_render as r; print(r(s.info('$E')['class']) or '')" 2>/dev/null
}

# cap_gpu_frames LANES...: write the source GPU frames cells at these lane counts as capped (run.sh keeps a file it finds).
cap_gpu_frames() {
  local L f
  for L in "$@"; do
    f=$OUT/pass-cuda/online/$E/src-render-gpu-L$L.json
    [ -s $f ] && continue
    mkdir -p $(dirname $f)
    echo "{\"env\": \"$E\", \"arms\": {\"src-pipelined-render-gpu\": {\"status\": \"capped\", \"note\": \"source GPU frames stop at 256 lanes\"}}}" > $f
  done
}

cpu_pass() {
  call cpu-a cpu "offline step" $SWEEP_CPU "$1" "$2" CPU_SOURCES=1 || return 3
  call cpu-f cpu frames $(frames_cpu) "$1" "$2" CPU_SOURCES=1 || return 3
}

cuda_pass() {
  call cuda-a cuda "offline step" $SWEEP_CUDA "$1" "$2" CPU_SOURCES=0 || return 3
  call cuda-f cuda frames 1:8:64:256 "$1" "$2" CPU_SOURCES=0 || return 3
  [ -n "$(gpusrc)" ] && cap_gpu_frames 1024
  call cuda-f2 cuda frames 1024 "$1" "$2" CPU_SOURCES=0 || return 3
}

gpu_pass() {
  local c=$1 b=$2 sl=${CAPL:-1024} big
  [ $sl -gt 1024 ] && sl=1024
  # Step sources stop at sl lanes; ours keeps every lane count of the sweep.
  echo "$E" > $CAPF; CAPL=$sl
  call g-a cuda "offline step" $SWEEP_CUDA "$c" "$b" CPU_SOURCES=1 || return 3
  [[ ":$SWEEP_CUDA:" == *":$sl:"* ]] || call g-s cuda step $sl "$c" "$b" CPU_SOURCES=1 || return 3
  call g-f8 cuda frames 8 "$c" "$b" CPU_SOURCES=1 || return 3
  [ -n "$(gpusrc)" ] && { call g-f64 cuda frames 64 "$c" "$b" CPU_SOURCES=0 || return 3; }
  if [ -n "$(gpusrc)" ]; then
    big=$(echo $SWEEP_CUDA | tr : '\n' | awk '$1 > 256')
    [ -n "$big" ] && cap_gpu_frames $big
  fi
  call g-f cuda frames $SWEEP_CUDA "$c" "$b" CPU_SOURCES=0 || return 3
}

# hardware: the machine and software of this job; merged into run.json at the end.
hardware() {
  local g=${CUDA_VISIBLE_DEVICES:-0}
  $P - "$1" "$g" <<'PY'
import json, os, platform, subprocess, sys
import stable_envs, torch
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
cores, g = sys.argv[1], sys.argv[2]
smi = sh("nvidia-smi")
print(json.dumps({
    "gpu": sh(f"nvidia-smi -i {g} --query-gpu=name --format=csv,noheader"), "gpu_count": len(sh("nvidia-smi -L").splitlines()),
    "driver": sh(f"nvidia-smi -i {g} --query-gpu=driver_version --format=csv,noheader"),
    "cuda": smi.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in smi else None,
    "cpu": sh("lscpu | sed -n 's/^Model name: *//p' | head -1"), "cores": cores,
    "cpu_quota": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"), "nproc": os.cpu_count(),
    "ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30),
    "os": sh(". /etc/os-release && echo $PRETTY_NAME"), "python": platform.python_version(), "torch": torch.__version__,
    "stable_envs": getattr(stable_envs, "__version__", None), "commit": sh("git rev-parse HEAD"),
}))
PY
}

rc=0
if [ "$MODE" = gpu ]; then
  set -- $(reserve 8); gpu_pass "$1" "${2:-}" || rc=3
elif [ "${PARALLEL_DEVICES:-1}" = 1 ] && q=$(awk '{print ($1 == "max") ? 100000 : int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 100000) \
    && [ "${q:-100000}" -ge 16 ] && r=$(TRIES=1 PICK_STRICT=1 reserve 16); then
  set -- $r
  ( cpu_pass "$(echo $1 | cut -d, -f1-8)" "${2:-}" ) &
  ( cuda_pass "$(echo $1 | cut -d, -f9-16)" "${2:-}" ) || rc=3
  wait
  echo "parallel passes" >> $OUT/cores.txt
else
  set -- $(reserve 8)
  cpu_pass "$1" "${2:-}" && cuda_pass "$1" "${2:-}" || rc=3
fi
# One folder per env: the passes write disjoint files.
for d in $OUT/pass-*; do [ -d $d ] && cp -rn $d/offline $d/online $OUT/ 2> /dev/null; done
cp "$(ls $OUT/run-*.json 2>/dev/null | head -1)" $OUT/run.json 2> /dev/null
hardware "$(awk '$3 == "cores" {print $4}' $OUT/cores.txt | sort -u | tr '\n' ' ')" > $OUT/hardware.json 2>> $LOG
[ -s $OUT/run.json ] && python3 -c "
import json; r = json.load(open('$OUT/run.json')); r['hardware'] = json.load(open('$OUT/hardware.json'))
json.dump(r, open('$OUT/run.json', 'w'), indent=1)" 2>> $LOG
rm -rf "$W/st/$E-"*
[ $rc = 0 ] && { echo "{\"end\": \"$(date -Is)\"}" > $OUT/end.json; cp $OUT/end.json $OUT/done.json; }
exit $rc
