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
ALLOC=6437813
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
" 2>&1
rc=$?
echo "[run_in_alloc] step=$TAG rc=$rc out=$OUT"
exit $rc
