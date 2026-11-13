#!/bin/bash
# Re-time all paper envs for one stable-envs commit on a fleet of RunPod pods, from the Mac.
#
# usage: benchmarks/paper/fleet/fleet.sh run SHA PODS [RUN]       set up the pods, hand out the envs, pull the results
#        benchmarks/paper/fleet/fleet.sh {resume|pull|status} SHA PODS [RUN]   (resume: dispatch again, staged files kept)
#   SHA   stable-envs commit to time (in the object store of this checkout: git fetch origin first).
#   PODS  file with one pod per line: "NAME HOST PORT" (direct TCP SSH of a running pod). Pods that are added to the
#         file while the run goes on join it.
#   RUN   result label (default base-<sha8>); results land in $RESULTS/<NAME>/<RUN>/<ENV>/ (RESULTS defaults to
#         ~/stable-envs-paper-results/raw/runpod).
# Every env is one job: run.sh on both devices with the source on the same pod. Envs listed under "split" in
# costs.json run as two jobs, ENV.cpu and ENV.cuda, each on one device with the CPU sources of its own pod. The Mac keeps one job waiting on each
# pod and hands out the most expensive job first (seconds per env in costs.json), so the pods finish together. A job
# goes only to a pod with at least its min_mem_gib of RAM (costs.json; Rocket at 4096 envs needs about 188 GB). Jobs
# whose result folder already holds done.json are skipped, so a run that stops can be started again with the same RUN.
# ENVS (space-separated) limits the run to those envs. QUIET_CPU=1 makes one job ENV.cpuq per env: its CPU pass again
# (ours and the source), for a quieter pod; build_v2.py then uses those CPU cells. Knobs passed to every pod's run.sh: WINDOW_S (default 3),
# REPEATS (3), SMOKE, ARMS, SWEEP_CPU, SWEEP_CUDA, OFFLINE_BATCHES, FRAME_CHECK, SRC_MAX_L, FRAMES_CUDA_MAX, PARALLEL_DEVICES, EXTRAS_<fam>,
# GROUPS_<fam>, SRC_REUSE_TOL. SRC_REUSE_RUNS (space-separated earlier RUN labels): before a job goes to a pod, the
# source cells of the newest earlier result of its env on the same CPU and GPU model with the same source code
# (src_reuse.py) are uploaded; run.sh times the 8-lane and the largest source cells again and copies the others when
# they agree.
# This script never creates or deletes pods (RunPod MCP create-pod / delete-pod, or the console).
set -uo pipefail
CMD=$1; SHA_REF=$2; SHA=$(git rev-parse "$2^{commit}") || exit 2; PODS=$3; RUN=${4:-base-${SHA:0:8}}
HERE=$(cd "$(dirname "$0")" && pwd); HARNESS=$(cd "$HERE/../../.." && pwd)
RESULTS=${RESULTS:-$HOME/stable-envs-paper-results/raw/runpod}
STAGE=${STAGE:-${TMPDIR:-/tmp}/snv-fleet-$RUN}
# nvdiffrast 253ac4fc built for sm_89 (L40S) and sm_90 (H100) against torch 2.14.1+cu126.
NVD=${NVDIFFRAST_TGZ:-$HOME/.claude/jobs/bcdf52de/tmp/rp/nvdiffrast-site-sm89-90.tgz}
SSHO=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -o BatchMode=yes)
mkdir -p "$STAGE"
on() { local h=$1 p=$2; shift 2; ssh -n "${SSHO[@]}" -p "$p" root@"$h" "$@"; }
pods() { grep -v '^[[:space:]]*#' "$PODS" | grep -v '^[[:space:]]*$'; }
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$STAGE/fleet.log"; }

stage() {
  # The stable-envs tree at SHA, the harness files of this checkout's HEAD, and the job list.
  git -C "$HARNESS" bundle create -q "$STAGE/se.bundle" "$SHA_REF" || exit 1
  git -C "$HARNESS" archive --format=tar.gz -o "$STAGE/harness.tgz" HEAD benchmarks || exit 1
  git -C "$HARNESS" diff --quiet HEAD -- benchmarks || echo "warning: uncommitted harness changes are not shipped" >&2
  cp "$HERE/src_cap_envs" "$STAGE/" && cp "$NVD" "$STAGE/nvdiffrast-site.tgz" || exit 1
  : > "$STAGE/queue.txt"
  {
    echo "export SHA=$SHA HARNESS_SHA=$(git -C "$HARNESS" rev-parse HEAD) RUN=$RUN"
    # One process that hangs (an EGL source on an H100 hung for 2 h) is killed after STEP_LIMIT.
    echo "export WINDOW=${WINDOW:-3} REPEATS=${REPEATS:-3} STEP_LIMIT=${STEP_LIMIT:-40m}"
    [ -n "${SRC_REUSE_RUNS:-}" ] && echo "export SRC_REUSE=/workspace/srcreuse"
    for v in SWEEP_CPU SWEEP_CUDA FRAME_CHECK WARMUP TORCH_VERSION; do
      [ -n "${!v:-}" ] && echo "export $v=\"${!v}\""
    done
  } > "$STAGE/env.sh"
  python3 - "$HERE/costs.json" "${ENVS:-}" "${QUIET_CPU:-0}" > "$STAGE/jobs.txt" <<'EOF' || exit 1
import json, sys
c = json.load(open(sys.argv[1]))
quiet = sys.argv[3] == "1"
want = sys.argv[2].split()
envs = {e: f for e, f in c["envs"].items() if not want or e in want}
jobs = []
for e, f in envs.items():
    s, mem = c["env_s"].get(e, c["family_s"][f]), c.get("min_mem_gib", {}).get(e, 0)
    # A long env runs as two jobs, one per device, each with the CPU sources of its own pod.
    if quiet:
        jobs += [(s / 2, f, f"{e}.cpuq", mem)]
        continue
    jobs += [(s / 2, f, f"{e}.{d}", mem) for d in ("cpu", "cuda")] if e in c.get("split", []) else [(s, f, e, mem)]
for s, f, j, mem in sorted(jobs, key=lambda j: -j[0]):
    print(f, j, mem, int(f in c.get("gpu_heavy_families", [])))
EOF
  log "staged $(wc -l < "$STAGE/jobs.txt" | tr -d ' ') jobs for $SHA (harness $(git -C "$HARNESS" rev-parse --short HEAD))"
}

up_one() {  # up_one NAME HOST PORT: upload, set up and start the job runner of one pod
  local name=$1 h=$2 p=$3 i
  for i in 1 2 3 4 5 6; do on "$h" "$p" true 2> /dev/null && break; sleep 20; done
  { echo "export POD_NAME=$name"; cat "$STAGE/env.sh"; } > "$STAGE/env-$name.sh"
  scp "${SSHO[@]}" -q -P "$p" "$STAGE"/{se.bundle,harness.tgz,nvdiffrast-site.tgz,src_cap_envs,queue.txt} root@"$h":/workspace/ \
    && scp "${SSHO[@]}" -q -P "$p" "$STAGE/env-$name.sh" root@"$h":/workspace/env.sh || { log "$name: upload failed"; return 1; }
  on "$h" "$p" "rm -f /workspace/STOP /workspace/state; tar xzf /workspace/harness.tgz -C /tmp benchmarks/paper/fleet/pod_setup.sh &&
    nohup bash /tmp/benchmarks/paper/fleet/pod_setup.sh > /workspace/setup.out 2>&1 < /dev/null &" || { log "$name: start failed"; return 1; }
  log "$name: setup started"
}

send_reuse() {  # send_reuse NAME HOST PORT ENV: upload the earlier source cells of ENV that this pod may reuse
  local name=$1 h=$2 p=$3 e=$4 hw d dst=$STAGE/reuse/$1/$4
  [ -s "$STAGE/hw-$name" ] || on "$h" "$p" 'lscpu | sed -n "s/^Model name: *//p" | head -1; nvidia-smi --query-gpu=name --format=csv,noheader | head -1' > "$STAGE/hw-$name"
  rm -rf "$dst"; mkdir -p "$dst/online/$e"
  python3 "$HERE/src_reuse.py" "$RESULTS" "$SRC_REUSE_RUNS" "$e" "$(sed -n 1p "$STAGE/hw-$name")" "$(sed -n 2p "$STAGE/hw-$name")" \
      "$SHA" "$(git -C "$HARNESS" rev-parse HEAD)" > "$dst.list"
  [ -s "$dst.list" ] || { log "$name: no earlier source cells for $e"; return 0; }
  while read -r d; do cp "$d"/online/"$e"/src-*.json "$dst/online/$e/"; cp "$d/run.json" "$dst/"; done < "$dst.list"
  on "$h" "$p" "mkdir -p /workspace/srcreuse" && scp "${SSHO[@]}" -q -r -P "$p" "$dst" root@"$h":/workspace/srcreuse/ \
    && log "$name: source cells of $e from $(tr '\n' ' ' < "$dst.list" | sed "s|$RESULTS/||g")"
}

pull_one() {  # pull_one NAME HOST PORT
  local name=$1 h=$2 p=$3 dst=$RESULTS/$1/$RUN
  mkdir -p "$dst"
  rsync -a -e "ssh ${SSHO[*]} -p $p" root@"$h":/workspace/results/$RUN/ "$dst/" < /dev/null 2> /dev/null
  rsync -a -e "ssh ${SSHO[*]} -p $p" root@"$h":/workspace/{setup.log,setup.out,queue.log} "$dst/" < /dev/null 2> /dev/null
}

pull_all() {
  local name h p
  while read -r name h p; do pull_one "$name" "$h" "$p" & done < <(pods)
  wait
}

next_job() {  # next_job [MEM_GIB [GPU_CLASS]]: the first job not yet handed out, not done, and fitting the pod
  # GPU_CLASS 2: L40S or H100/H200 (GPU-heavy families first); 1: another GPU, which gets GPU-heavy families only when no
  # other job is left; 0: a GPU without nvdiffrast kernels (sm_80, A100), which never gets Rocket's CUDA job.
  local fam env need heavy mem=${1:-100000} cls=${2:-2} pass
  for pass in 1 2; do
    while read -r fam env need heavy; do
      grep -qx "$env" "$STAGE/handed.txt" 2> /dev/null && continue
      [ "${need:-0}" -gt "$mem" ] && continue
      [ $pass = 1 ] && [ "$cls" != 2 ] && [ "${heavy:-0}" = 1 ] && continue
      [ "$cls" = 0 ] && [[ $env == PyFlytRocketLanding* ]] && [[ $env != *.cpu ]] && continue
      ls "$RESULTS"/*/"$RUN/$env/done.json" > /dev/null 2>&1 && { echo "$env" >> "$STAGE/handed.txt"; continue; }
      echo "$fam $env"; return 0
    done < "$STAGE/jobs.txt"
  done
  return 1
}

left() {  # jobs not yet handed out
  local fam env need n=0
  while read -r fam env need heavy; do grep -qx "$env" "$STAGE/handed.txt" 2> /dev/null || n=$((n + 1)); done < "$STAGE/jobs.txt"
  echo $n
}

dispatch() {
  local t0 name h p state job last_pull running waiting idle mem gpucls
  t0=$(date +%s); last_pull=$t0; touch "$STAGE/handed.txt"
  while true; do
    running=0; waiting=0; idle=0; : > "$STAGE/active.new"
    while read -r name h p; do
      if [ ! -f "$STAGE/up-$name" ]; then
        touch "$STAGE/up-$name"; up_one "$name" "$h" "$p" < /dev/null &
        continue
      fi
      # The number of jobs waiting in the pod's queue, and its runner's state: setup, idle or running ENV.
      state=$(on "$h" "$p" 'echo $(cat /workspace/queue.txt 2>/dev/null | grep -c .) $(awk "{print (\$1 == \"max\") ? 100000 : int(\$1 / 2^30)}" /sys/fs/cgroup/memory.max 2>/dev/null || echo 100000) $(nvidia-smi --query-gpu=name --format=csv,noheader | awk "/L40S|H100|H200/{c=2} /A100/{c=0} END{print (c==\"\") ? 1 : c}") $(cat /workspace/state 2>/dev/null || echo setup) $(cut -d" " -f1 /workspace/queue.txt 2>/dev/null)' 2> /dev/null)
      set -- $state
      [ $# -lt 4 ] && { log "$name: no answer"; continue; }
      mem=$2; gpucls=$3; set -- "$1" "${@:4}"
      echo "${*:3}" | tr ' ' '\n' >> "$STAGE/active.new"
      [ "$2" = running ] && running=$((running + 1))
      [ "$2" = idle ] && [ "$1" = 0 ] && idle=$((idle + 1))
      [ "$1" -gt 0 ] && waiting=$((waiting + 1))
      # Near the end (fewer jobs left than pods) a job waits for an idle pod instead of queueing behind a long one. A job
      # handed out in this pass counts as waiting, so the run does not end before the pod starts it.
      if [ "$1" = 0 ] && [ "$2" != setup ] && { [ "$2" = idle ] || [ "$(left)" -gt "$(pods | wc -l)" ]; } \
          && job=$(next_job "$mem" "$gpucls"); then
        set -- $job
        [ -n "${SRC_REUSE_RUNS:-}" ] && send_reuse "$name" "$h" "$p" "${2%.*}"
        on "$h" "$p" "flock /workspace/queue.lock sh -c 'echo \"$2 $1=${2%.*}\" >> /workspace/queue.txt'" \
          && { echo "$2" >> "$STAGE/handed.txt"; echo "$2" >> "$STAGE/active.new"; echo "$2 $name" >> "$STAGE/assigned.txt"; log "$name <- $2"; waiting=$((waiting + 1)); }
      fi
    done < <(pods)
    if [ $(( $(date +%s) - last_pull )) -ge 300 ]; then pull_all; last_pull=$(date +%s); fi
    mv "$STAGE/active.new" "$STAGE/active.txt"
    if ! next_job > /dev/null && [ $idle -gt 0 ] && [ -s "$STAGE/handed.txt" ]; then
      # A handed-out job that no pod runs or queues and that has no done.json is lost (a pod set up again drops its
      # queue): hand it out again.
      pull_all; last_pull=$(date +%s)
      local lost=0 fam env need
      while read -r fam env need heavy; do
        ls "$RESULTS"/*/"$RUN/$env/done.json" > /dev/null 2>&1 && continue
        grep -qx "$env" "$STAGE/active.txt" && continue
        grep -vx "$env" "$STAGE/handed.txt" > "$STAGE/handed.tmp"; mv "$STAGE/handed.tmp" "$STAGE/handed.txt"
        lost=$((lost + 1))
      done < "$STAGE/jobs.txt"
      [ $lost -gt 0 ] && { log "handing out $lost unfinished jobs again"; continue; }
      if [ $running = 0 ] && [ $waiting = 0 ]; then
        log "all jobs done in $(( ($(date +%s) - t0) / 60 )) min"
        break
      fi
    fi
    sleep 30
  done
  while read -r name h p; do on "$h" "$p" 'touch /workspace/STOP'; done < <(pods)
  pull_all
  log "pulled into $RESULTS/<pod>/$RUN"
  # Every env must have a result, and every listed cell must be there and ok.
  local missing=0 fam env need heavy
  while read -r fam env need heavy; do
    ls "$RESULTS"/*/"$RUN/$env/done.json" > /dev/null 2>&1 || { log "MISSING $env"; missing=$((missing + 1)); }
  done < "$STAGE/jobs.txt"
  python3 "$HERE/check_cells.py" $(ls -d "$RESULTS"/*/"$RUN"/*/ | while read -r d; do [ -f $d/done.json ] && echo $d; done) \
    > "$STAGE/problems.txt"
  cp "$STAGE/problems.txt" "$RESULTS/problems-$RUN.txt"
  if [ $missing -gt 0 ] || [ -s "$STAGE/problems.txt" ]; then
    log "FAILED: $missing envs without a result, $(wc -l < "$STAGE/problems.txt" | tr -d ' ') problems in $RESULTS/problems-$RUN.txt"
    return 1
  fi
  log "all cells ok"
}

status() {
  local name h p
  while read -r name h p; do
    printf '%-16s %s\n' "$name" "$(on "$h" "$p" 'cat /workspace/state 2>/dev/null; grep -c . /workspace/queue.txt; tail -1 /workspace/queue.log 2>/dev/null' 2>&1 | tr '\n' ' ')"
  done < <(pods)
}

case $CMD in
  run) stage && dispatch;;
  resume) dispatch;;
  pull) pull_all;;
  status) status;;
  *) echo "unknown command $CMD" >&2; exit 2;;
esac
