#!/bin/bash
# On a fresh RunPod pod: install system packages, check out the stable-envs commit with the harness on top, build the
# family venvs of this pod's queue, then start the queue. Inputs in /workspace (uploaded by fleet.sh): se.bundle (git
# bundle of the commit under test), harness.tgz, nvdiffrast-site.tgz, src_cap_envs, queue.txt (may be empty), env.sh
# (SHA, HARNESS_SHA, POD_NAME, RUN, run.sh knobs).
set -uo pipefail
W=/workspace; cd $W || exit 1
source $W/env.sh
pkill -f pod_queue.sh
: > $W/queue.log
R=$W/results/$RUN
mkdir -p $W/tmp $W/uvcache $R $W/venvs
log() { echo "$(date -Is) $*" | tee -a $W/setup.log; }
log "setup start"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null && apt-get install -y -qq xvfb xauth libgl1 libgl1-mesa-dri libegl1 libosmesa6 libglfw3 \
  libglew2.2 git rsync > /dev/null || { log "apt failed"; exit 1; }
[ -x $W/uvenv/bin/uv ] || { python3 -m venv $W/uvenv && $W/uvenv/bin/pip install -q "uv==0.13.5"; } || { log "uv failed"; exit 1; }
[ -d $W/swm ] || { git clone -q https://github.com/galilai-group/stable-worldmodel $W/swm && git -C $W/swm checkout -q 21446f1; } \
  || { log "swm clone failed"; exit 1; }
rm -rf $W/nvdiffrast-site && tar xzf $W/nvdiffrast-site.tgz --no-same-owner -C $W
# The stable-envs commit under test as a git checkout (run.sh records its HEAD); the harness files replace benchmarks/.
rm -rf $W/se && git clone -q $W/se.bundle $W/se && git -C $W/se checkout -q "$SHA" || { log "checkout $SHA failed"; exit 1; }
[ "$(git -C $W/se rev-parse HEAD)" = "$SHA" ] || { log "HEAD is not $SHA"; exit 1; }
rm -rf $W/se/benchmarks && tar xzf $W/harness.tgz -C $W/se && echo "$SHA" > $W/se/.bench_main_sha
{
  echo "{\"pod\": \"$POD_NAME\", \"sha\": \"$SHA\", \"harness\": \"$HARNESS_SHA\", \"run\": \"$RUN\","
  echo " \"cpu_model\": \"$(lscpu | sed -n 's/^Model name: *//p' | head -1)\", \"cpu_quota\": \"$(cat /sys/fs/cgroup/cpu.max)\","
  echo " \"gpu\": \"$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)\","
  echo " \"host\": \"$(hostname)\", \"mem_gib\": $(free -g | awk '/Mem:/{print $2}'), \"setup_start\": \"$(date -Is)\"}"
} > $R/pod.json
cat $R/pod.json
# The family venvs of jobs already queued, in parallel, before any timing starts; later families build on demand.
specs=$(awk '{for (i = 2; i <= NF; i++) print $i}' $W/queue.txt | cut -d= -f1 | sort -u)
for f in $specs; do
  env SNV_REPO=$W/se SWM=$W/swm UV=$W/uvenv/bin/uv UV_CACHE_DIR=$W/uvcache UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 \
    NVDIFFRAST_SITE=$W/nvdiffrast-site VENVS=$W/venvs flock $W/venvs/$f.lock bash $W/se/benchmarks/paper/fleet/venv.sh $f &
done
wait
for f in $specs; do [ -s $W/venvs/venv-$f.ok ] || { log "venv $f failed"; tail -20 $W/venvs/venv-$f.log; }; done
log "setup done"
nohup bash $W/se/benchmarks/paper/fleet/pod_queue.sh > $W/queue.out 2>&1 &
log "queue started pid $!"
