#!/bin/bash
# One env against its source: several calls of run.sh into OUT, each with its own arms, lane counts and pinned cores.
# usage: bash benchmarks/paper/job.sh OUT FAMILY ENV      (from the root of the checkout under test)
# env: P (family venv python), W (work dir: swm, uv, tmp, cores.lock), MODE (full or gpu), SWEEP_CPU, SWEEP_CUDA,
#   CAPS (file of "ENV LANES": step sources of ENV stop at LANES), WINDOW, REPEATS, STEP_LIMIT, PARALLEL_DEVICES (1).
# CPU pass (MODE=full only), with the CPU sources:
#   cpu-a   DEVICES=cpu ARMS="offline step" at SWEEP_CPU
#   cpu-f   DEVICES=cpu ARMS=frames up to the family's CPU frames limit (frames_cpu)
# CUDA pass, ours and the source at every lane count and batch of SWEEP_CUDA:
#   cuda-o  ARMS=offline; cuda-s ARMS=step; cuda-f ARMS=frames (the source's GPU renderer where it has one, else the
#           CPU source). Source cells that the CPU pass times are left to it; MODE=gpu times them here, and cuda-fc adds
#           the CPU frames source of an env with a GPU renderer, for the case that the GPU source fails its frame check.
# MODE=full runs the two passes at the same time on two disjoint sets of 8 idle physical cores when 16 are free, else
# one after the other. A cell that runs out of memory or time is kept as a result with that status (run.sh).
# SIDE=ours or SIDE=source (run.sh) times one side only; done-SIDE.json marks it finished, done.json the source side
# (compare.sh runs every env's ours side first). cores.txt records the cores and how busy they were. DRY_RUN=1 prints the
# calls only.
set -uo pipefail
OUT=$1; fam=$2; E=$3
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${MODE:-full}
SWEEP_CPU=${SWEEP_CPU:-1:8:64:256:1024}; SWEEP_CUDA=${SWEEP_CUDA:-1:8:64:256:1024:4096}
DRY=${DRY_RUN:-0}
LOG=$OUT/run.log
CAPL=$(awk -v e="$E" '$1 == e {print ($2 ? $2 : 256)}' "${CAPS:-/dev/null}" 2>/dev/null)
CAPF=$OUT/.cap
if [ "$DRY" != 1 ]; then
  mkdir -p "$OUT" "$W/tmp"
  if [ -n "$CAPL" ]; then echo "$E" > "$CAPF"; else : > "$CAPF"; fi
fi

frames_cpu() {
  case $E in OGBCube|OGBScene) echo 1:8; return;; esac
  case $fam in dmc|fetch|ogbench|rocket) echo 1:8:64;; *) echo 1:8:64:256;; esac
}

# reserve N: N idle physical cores, at most 10% busy, not held by another job on this host (W/cores.held).
# With TRIES=1 and PICK_STRICT=1 it fails instead of taking busy cores.
reserve() {
  local n=$1 got busy
  [ "$DRY" = 1 ] && { echo "<$n idle physical cores> -"; return 0; }
  for _ in $(seq "${TRIES:-20}"); do
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
release() { [ "$DRY" = 1 ] || flock "$W/cores.lock" sed -i "/^$$ /d" "$W/cores.held"; }
trap release EXIT

# call NAME DEV ARMS SWEEP CORES BUSY [VAR=VALUE ...]: one run.sh call into OUT/pass-DEV.
call() {
  local name=$1 dev=$2 arms=$3 sweep=$4 cores=$5 busy=$6 o=$OUT/pass-$2; shift 6
  if [ "$DRY" = 1 ]; then
    echo "  $E ${SIDE:-both} $name: run.sh DEVICES=$dev ARMS=\"$arms\" lanes $sweep $* step-source cap ${CAPL:-none} cores $cores"
    return 0
  fi
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
  [ "$DRY" = 1 ] && { echo "(if the source has a GPU renderer)"; return 0; }
  $P -c "import stable_envs as s; from tests.helpers.reference.sources import reference_gpu_render as r; print(r(s.info('$E')['class']) or '')" 2>/dev/null
}

# lanes SWEEP OP N: the lane counts of SWEEP with L OP N (le or gt), colon-separated.
lanes() { echo "$1" | tr : '\n' | awk -v n="$3" -v op="$2" '(op == "le" && $1 <= n) || (op == "gt" && $1 > n)' | paste -sd: -; }

cpu_pass() {
  call cpu-a cpu "offline step" "$SWEEP_CPU" "$1" "$2" CPU_SOURCES=1 || return 3
  call cpu-f cpu frames "$(frames_cpu)" "$1" "$2" CPU_SOURCES=1 || return 3
}

cuda_pass() {
  local c=$1 b=$2 cs g fmax small big
  cs=$([ "$MODE" = gpu ] && echo 1 || echo 0)
  call cuda-o cuda offline "$SWEEP_CUDA" "$c" "$b" CPU_SOURCES=$cs || return 3
  small=$(lanes "$SWEEP_CUDA" le 1024); big=$(lanes "$SWEEP_CUDA" gt 1024)
  [ -n "$small" ] && { call cuda-s cuda step "$small" "$c" "$b" CPU_SOURCES=$cs || return 3; }
  [ -n "$big" ] && { call cuda-s2 cuda step "$big" "$c" "$b" CPU_SOURCES=1 || return 3; }
  g=$(gpusrc)
  if [ -n "$g" ]; then
    call cuda-f cuda frames "$SWEEP_CUDA" "$c" "$b" CPU_SOURCES=0 || return 3
    [ "$MODE" = gpu ] && { call cuda-fc cuda frames "$(frames_cpu)" "$c" "$b" CPU_SOURCES=1 || return 3; }
  else
    fmax=$(frames_cpu | tr : '\n' | tail -1)
    small=$(lanes "$SWEEP_CUDA" le "$fmax"); big=$(lanes "$SWEEP_CUDA" gt "$fmax")
    [ -n "$small" ] && { call cuda-f cuda frames "$small" "$c" "$b" CPU_SOURCES=$cs || return 3; }
    [ -n "$big" ] && { call cuda-f2 cuda frames "$big" "$c" "$b" CPU_SOURCES=1 || return 3; }
  fi
  return 0
}

# hardware CORES: the machine and software of this job; merged into run.json at the end.
hardware() {
  $P - "$1" "${CUDA_VISIBLE_DEVICES:-0}" <<'PY'
import json, os, platform, subprocess, sys
import stable_envs, torch
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
cores, g = sys.argv[1], sys.argv[2]
smi = sh("nvidia-smi")
print(json.dumps({
    "gpu": sh(f"nvidia-smi -i {g} --query-gpu=name --format=csv,noheader"), "gpu_count": len(sh("nvidia-smi -L").splitlines()),
    "driver": sh(f"nvidia-smi -i {g} --query-gpu=driver_version --format=csv,noheader"),
    "cuda": smi.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in smi else None,
    "cpu": sh("lscpu | sed -n 's/^Model name: *//p' | head -1"), "cores": cores.strip(),
    "cpu_quota": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"), "nproc": os.cpu_count(),
    "ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30),
    "os": sh(". /etc/os-release && echo $PRETTY_NAME"), "python": platform.python_version(), "torch": torch.__version__,
    "stable_envs": getattr(stable_envs, "__version__", None), "commit": sh("git rev-parse HEAD"),
}))
PY
}

rc=0
if [ "$DRY" = 1 ]; then
  [ "$MODE" = gpu ] || { echo "  $E: CPU and CUDA passes at the same time on two sets of 8 idle cores when the quota allows"; cpu_pass "<8 cores>" -; }
  cuda_pass "<8 cores>" -
elif [ "$MODE" = gpu ]; then
  # shellcheck disable=SC2046
  set -- $(reserve 8); cuda_pass "$1" "${2:-}" || rc=3
elif [ "${PARALLEL_DEVICES:-1}" = 1 ] && q=$(awk '{print ($1 == "max") ? 100000 : int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 100000) \
    && [ "${q:-100000}" -ge 16 ] && r=$(TRIES=1 PICK_STRICT=1 reserve 16); then
  set -- $r
  ( cpu_pass "$(echo "$1" | cut -d, -f1-8)" "${2:-}" ) &
  ( cuda_pass "$(echo "$1" | cut -d, -f9-16)" "${2:-}" ) || rc=3
  wait
  echo "parallel passes" >> "$OUT/cores.txt"
else
  # shellcheck disable=SC2046
  set -- $(reserve 8)
  cpu_pass "$1" "${2:-}" && cuda_pass "$1" "${2:-}" || rc=3
fi
[ "$DRY" = 1 ] && exit 0
# One folder per env: the passes write disjoint files.
for d in $OUT/pass-*; do [ -d $d ] && cp -rn $d/offline $d/online $OUT/ 2> /dev/null; done
cp "$(ls $OUT/run-*.json 2>/dev/null | head -1)" $OUT/run.json 2> /dev/null
hardware "$(awk '$3 == "cores" {print $4}' $OUT/cores.txt | sort -u | tr '\n' ' ')" > $OUT/hardware.json 2>> $LOG
[ -s $OUT/run.json ] && python3 -c "
import json; r = json.load(open('$OUT/run.json')); r['hardware'] = json.load(open('$OUT/hardware.json'))
json.dump(r, open('$OUT/run.json', 'w'), indent=1)" 2>> $LOG
rm -rf "$W/st/$E-"*
if [ $rc = 0 ]; then
  echo "{\"end\": \"$(date -Is)\", \"side\": \"${SIDE:-both}\"}" > "$OUT/end.json"
  cp "$OUT/end.json" "$OUT/done-${SIDE:-both}.json"
  [ "${SIDE:-both}" = ours ] || cp "$OUT/end.json" "$OUT/done.json"
fi
exit $rc
