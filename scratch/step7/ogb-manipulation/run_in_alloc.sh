#!/bin/bash
# Run one OGB GPU step inside the persistent Step-7 allocation instead of queueing an sbatch.
#
#   ./run_in_alloc.sh <step-tag> <script-to-run-on-the-node> <expected-md5-of-ogb_manip.py>
#
# The srun is issued FROM the login node and executes on gpu2101: nothing here computes locally. The
# md5 assert runs ON THE NODE, before the step, so the proof that the GPU imported the file at HEAD
# lands in the step's own log rather than in a claim made beside it.
set -uo pipefail
TAG=${1:?step tag}
BODY=${2:?script to run on the node}
WANT=${3:?expected md5 of stable_envs/envs/ogb_manip.py}
# WHICH ALLOCATION. Two are held for Step-7 closure and only one may be used at a time. Rather than
# hardcode one -- 6437813 expires first and a step started too close to its wall clock dies halfway --
# pick whichever is RUNNING with the MOST time left, and say so in the provenance line. Override with
# a 4th argument or ALLOC=<jobid>.
ALLOC_CANDIDATES=${ALLOC_CANDIDATES:-"6437813 6443480"}
pick_alloc() {
  local best="" best_s=-1
  for j in $ALLOC_CANDIDATES; do
    local row state left
    row=$(squeue -j "$j" -h -o '%T %L' 2>/dev/null) || continue
    state=$(echo "$row" | awk '{print $1}'); left=$(echo "$row" | awk '{print $2}')
    [ "$state" = "RUNNING" ] || continue
    # %L is [DD-]HH:MM:SS or MM:SS -- normalise to seconds
    local s
    s=$(echo "$left" | awk -F'[-:]' '{ if (NF==4) print (($1*24+$2)*60+$3)*60+$4;
                                       else if (NF==3) print (($1*60)+$2)*60+$3;
                                       else if (NF==2) print ($1*60)+$2; else print 0 }')
    if [ "${s:-0}" -gt "$best_s" ]; then best_s=$s; best=$j; fi
  done
  echo "$best $best_s"
}
ALLOC=${4:-${ALLOC:-}}
if [ -z "$ALLOC" ]; then
  read -r ALLOC ALLOC_LEFT <<<"$(pick_alloc)"
  [ -n "$ALLOC" ] || { echo "no allocation from '$ALLOC_CANDIDATES' is RUNNING; refusing"; exit 4; }
  echo "[run_in_alloc] chose allocation $ALLOC with ${ALLOC_LEFT}s left (candidates: $ALLOC_CANDIDATES)"
  # A step that cannot finish with margin must not start: the full gate is ~25 min, so refuse under 35.
  if [ "${ALLOC_LEFT:-0}" -lt 2100 ]; then
    echo "only ${ALLOC_LEFT}s left on $ALLOC -- under the 2100s margin a long step needs; refusing"
    exit 5
  fi
fi
SCR=/oscar/scratch/$USER
OUT=$SCR/se7/ogb-manipulation/examples/results/step7/ogb-manipulation/m6/$TAG
srun --jobid=$ALLOC --overlap -n 1 --cpus-per-task=6 bash -lc "
set -uo pipefail
source $SCR/se7/ogb-manipulation/scratch/step7/ogb-manipulation/env_header.sh
cd \$WS
mkdir -p '$OUT'
GOT=\$(md5sum stable_envs/envs/ogb_manip.py | cut -d' ' -f1)
echo \"[provenance] step=$TAG alloc_job=$ALLOC host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"
echo \"[provenance] ogb_manip.py md5 want=$WANT got=\$GOT\"
[ '$WANT' = \"\$GOT\" ] || { echo 'MD5 MISMATCH -- the node is not running HEAD; refusing'; exit 3; }
export SE7_STEP_TAG='$TAG' SE7_ALLOC_JOB='$ALLOC' SE7_OUT='$OUT'
bash '$BODY'
" 2>&1 | tee -a "$OUT/step.log"
rc=$?
echo "[run_in_alloc] step=$TAG rc=$rc out=$OUT"
exit $rc
