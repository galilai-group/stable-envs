#!/bin/bash
# Run the jobs that fleet.sh appends to /workspace/queue.txt ("TAG FAMILY=ENV"), one at a time, until /workspace/STOP
# exists. A job calls benchmarks/paper/run.sh (the paper harness, unchanged) several times into one folder
# /workspace/results/RUN/TAG, each call with its own cores, arms and lane counts:
#   cpu-a   DEVICES=cpu ARMS="offline step" SWEEP_CPU            with the CPU sources
#   cpu-f   DEVICES=cpu ARMS=frames, lane counts up to the family's CPU frames limit (frames_cpu)
#   cuda-a  DEVICES=cuda ARMS="offline step" SWEEP_CUDA, CPU_SOURCES=0 (the CPU source cells come from cpu-a)
#   cuda-f  DEVICES=cuda ARMS=frames at 1:8:64:256, CPU_SOURCES=0, then at 1024 where the source GPU cell is capped
# Every call pins all its arms to 8 idle physical cores (pick_cores.py); cores.txt records the cores and how busy they
# were before the call (the two passes share one 16-core pick when they run at the same time). done.json marks a finished job (run.sh writes end.json after every call).
# TAG ENV.cpu or ENV.cpuq: CPU calls only; ENV.cuda: CUDA calls only; ENV: both.
W=/workspace; source $W/env.sh
R=$W/results/$RUN
cd $W/se || exit 1
touch $W/queue.txt
SWEEP_CPU=${SWEEP_CPU:-1:8:64:256:1024}; SWEEP_CUDA=${SWEEP_CUDA:-1:8:64:256:1024:4096}
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 VENVS=$W/venvs SWM=$W/swm UV=$W/uvenv/bin/uv TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache NVDIFFRAST_SITE=$W/nvdiffrast-site SNV_REPO=$W/se

# frames_cpu FAMILY ENV: CPU frames lane counts (a CPU frame rate is flat from 8 lanes; larger cells take minutes).
frames_cpu() {
  case $2 in OGBCube|OGBScene) echo 1:8; return;; esac
  case $1 in dmc|fetch|ogbench|rocket) echo 1:8:64;; *) echo 1:8:64:256;; esac
}

# call NAME DEV ARMS SWEEP [VAR=VALUE ...]: one run.sh call into $OUT/pass-DEV, on $FIXED cores when set, else on idle
# cores picked now.
call() {
  local name=$1 dev=$2 arms=$3 sweep=$4 cores busy i o=$OUT/pass-$2; shift 4
  if [ -n "${FIXED:-}" ]; then
    cores=$FIXED; busy=$FIXED_BUSY
  else
    for i in 1 2 3 4 5 6 7 8 9 10; do
      cores=$(PICK_MAX_BUSY=0.10 python3 benchmarks/paper/pick_cores.py 8 2> $W/tmp/busy-$dev) && break
      sleep 30
    done
    [ -z "$cores" ] && cores=$(python3 benchmarks/paper/pick_cores.py 8 2> $W/tmp/busy-$dev)
    busy=$(sed -n 's/^busy //p' $W/tmp/busy-$dev)
  fi
  echo "$name $(date -Is) cores $cores busy ${busy:-?} load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/cores.txt
  env "$@" CORES=$cores DEVICES=$dev ARMS="$arms" SWEEP_CPU=$sweep SWEEP_CUDA=$sweep VENV_PYTHON=$P \
    STATES_DIR=$W/st/$TAG-$dev WINDOW=${WINDOW:-3} REPEATS=${REPEATS:-3} STEP_LIMIT=${STEP_LIMIT:-40m} \
    SRC_CAP_FILE=$CAPF SRC_CAP_L=${CAPL:-256} bash benchmarks/paper/run.sh $o "$fam=$E" >> $LOG.$dev 2>&1
  local rc=$?
  cp $o/run.json $OUT/run-$name.json 2> /dev/null
  rm -f $o/end.json
  echo "call $name rc=$rc $(date -Is)" >> $LOG.$dev
  [ $rc = 3 ] && return 3
  return 0
}

cpu_pass() {
  call cpu-a cpu "offline step" $SWEEP_CPU CPU_SOURCES=1 || return 3
  call cpu-f cpu frames $(frames_cpu $fam $E) CPU_SOURCES=1 || return 3
}

cuda_pass() {
  call cuda-a cuda "offline step" $SWEEP_CUDA CPU_SOURCES=0 || return 3
  call cuda-f cuda frames 1:8:64:256 CPU_SOURCES=0 || return 3
  # Source GPU frames stop at 256 lanes (with 128 EGL contexts per worker at 1024 lanes a source drew 45 frames/s
  # against 3,400 at 256): its 1024-lane cell is written as capped, and run.sh keeps a file it finds.
  local gpusrc f=$OUT/pass-cuda/online/$E/src-render-gpu-L1024.json
  gpusrc=$(PYTHONPATH=$W/se $P -c "import stable_envs as s; from tests.helpers.reference.sources import reference_gpu_render as r; print(r(s.info('$E')['class']) or '')" 2>/dev/null)
  if [ -n "$gpusrc" ] && [ ! -s $f ]; then
    mkdir -p $(dirname $f)
    echo "{\"env\": \"$E\", \"arms\": {\"src-pipelined-render-gpu\": {\"status\": \"capped\", \"note\": \"source GPU frames stop at 256 lanes\"}}}" > $f
  fi
  call cuda-f2 cuda frames 1024 CPU_SOURCES=0 || return 3
}

# Both passes at the same time on two disjoint sets of 8 idle physical cores when the CPU quota holds 16 cores and 16
# cores are at most 10% busy (RETIME.md: parallel against sequential, median ratio 1.001); else one after the other.
two_sets() {
  local q
  q=$(awk '{print ($1 == "max") ? 100000 : int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null)
  [ "${q:-100000}" -ge 16 ] || return 1
  PICK_MAX_BUSY=0.10 python3 benchmarks/paper/pick_cores.py 16 2> $W/tmp/busy-two
}

while true; do
  line=$(flock $W/queue.lock sh -c "head -1 $W/queue.txt; sed -i 1d $W/queue.txt")
  if [ -z "$line" ]; then
    echo idle > $W/state
    [ -f $W/STOP ] && break
    sleep 10; continue
  fi
  set -- $line; TAG=$1; spec=$2; fam=${spec%%=*}; E=${spec#*=}
  case $TAG in *.cpu|*.cpuq) DEVS=cpu;; *.cuda) DEVS=cuda;; *) DEVS="cpu cuda";; esac
  echo "running $TAG" > $W/state; echo "start $TAG $(date -Is)" >> $W/queue.log
  OUT=$R/$TAG; LOG=$R/run-$TAG.log; mkdir -p $OUT
  P=$(flock $W/venvs/$fam.lock bash benchmarks/paper/fleet/venv.sh $fam 2>> $LOG) || { P=; echo "venv $fam failed" >> $LOG; }
  # src_cap_envs: "ENV [LANES]"; the step source cells of ENV stop at LANES (run.sh SRC_CAP_FILE, SRC_CAP_L).
  CAPF=$W/tmp/cap-$E; CAPL=$(awk -v e="$E" '$1 == e {print ($2 ? $2 : 256)}' $W/src_cap_envs)
  if [ -n "$CAPL" ]; then echo "$E" > $CAPF; else : > $CAPF; fi
  rc=0; [ -z "$P" ] && rc=1
  if [ -n "$P" ]; then
    if [ "$DEVS" = "cpu cuda" ] && [ "${PARALLEL_DEVICES:-1}" = 1 ] && two=$(two_sets); then
      b=$(sed -n 's/^busy //p' $W/tmp/busy-two)
      ( FIXED=$(echo $two | cut -d, -f1-8) FIXED_BUSY=$b; cpu_pass ) &
      ( FIXED=$(echo $two | cut -d, -f9-16) FIXED_BUSY=$b; cuda_pass ) || rc=3
      wait
      echo "parallel passes" >> $OUT/cores.txt
    else
      for dev in $DEVS; do ${dev}_pass || { rc=3; break; }; done
    fi
    # One folder per job: the two passes write disjoint files (CPU and source-CPU cells; GPU and source-GPU cells).
    for d in $OUT/pass-*; do cp -rn $d/offline $d/online $OUT/ 2> /dev/null; done
    cp "$(ls $OUT/run-*.json | head -1)" $OUT/run.json 2> /dev/null
  fi
  rm -rf $W/st/$TAG-*
  [ $rc = 0 ] && { echo "{\"end\": \"$(date -Is)\"}" > $OUT/end.json; cp $OUT/end.json $OUT/done.json; }
  echo "done $TAG rc=$rc $(date -Is)" >> $W/queue.log
  # rc 1 (no venv): no done.json, so fleet.sh hands the job out again. rc 3: the GPU stopped initialising CUDA. Take no more jobs; fleet.sh hands this pod's jobs to the others.
  [ $rc = 3 ] && { echo broken > $W/state; echo "GPU lost, runner stops" >> $W/queue.log; exit 3; }
done
date -Is > $R/DONE
echo QUEUE_DONE >> $W/queue.log
