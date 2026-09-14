#!/bin/bash
#SBATCH --job-name=se6-verify-trackcom-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The four DM Control trackcom cells on cuda after the cpu-3d merge: does adding a CPU backend move
# the GPU picture? Four legs, one card, one allocation, so the only variable in each comparison is
# the one being asked about:
#
#   se-step5 x {baseline, integration}   the venv the coordinator named
#   se-step3 x {baseline, integration}   the venv the dmc shard was FROZEN in
#
# The second pair is what closed MJWARP-1: these four cells reproduce their frozen digest_ours AND
# digest_official byte for byte in se-step3 and did not in se-step5, so the venv was the variable,
# not MJWarp. Keeping both arms means a regression here could not be mistaken for that again.
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify5}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
TAG=${TAG:-_trackcom}
CFG=${CFG:-DMCCheetahRun,DMCHopperHop,DMCHumanoidStand,DMCWalkerWalk}
OUT=$OUTROOT/replay
D=$SCR/se6/_dumps/trackcom$TAG
mkdir -p "$OUT"; rm -rf "$D"
cd "$TREE" || exit 1
banner
echo "configs: $CFG"

for ARM in se-step5 se-step3; do
  PY=$SCR/venvs/$ARM/bin/python
  $PY -c "import mujoco,mujoco_warp;print('$ARM mujoco',mujoco.__version__,'mjwarp',mujoco_warp.__version__)"
  for LEG in baseline integration; do
    [ "$LEG" = baseline ] && T=$BASE || T=$TREE
    mkdir -p "$D/${ARM}_${LEG}"
    PYTHONPATH=$T $PY "$TREE/scratch/step6/raycast3d/replay_step4.py" --device cuda \
        --configs ${CFG//,/ } --dump-dir "$D/${ARM}_${LEG}" \
        --out "$OUT/trackcom_cuda_${ARM}_${LEG}${TAG}.json" 2>&1 | tail -8
    echo "ARM $ARM LEG $LEG rc=$?"
  done
  # the pixel diff, not just the hash, so a digest that moved arrives with a magnitude
  PYTHONPATH=$TREE $PY "$TREE/scratch/step6/raycast3d/replay_step4.py" \
      --compare-dumps "$D/${ARM}_baseline" "$D/${ARM}_integration" \
      --out "$OUT/trackcom_cuda_${ARM}_diff${TAG}.json" 2>&1 | tail -8
  echo "ARM $ARM DIFF rc=$?"
done

# The full check battery, in each shard's own frozen interpreter. The dmc shard rather than the four
# cells alone, because it costs nine minutes and it moves those four cells' Gate A onto the SAME SHA
# as everything else in the final table instead of leaving it to be inferred; raster and fetch for the
# same reason -- they were last measured at e0182f6/f7b07d7 and nothing between those and this SHA
# touches a gfxdraw, scene or mjwarp path, but "nothing touches it" is an argument and this is a
# report. ogb and swm stay where job 6358407 measured them: raycast3d.py and swm_nav.py are
# byte-identical across the two SHAs.
CDIR=$OUT/certify_trackcom${TAG}; mkdir -p "$CDIR/conformance"
for S in ${SHARDS:-"dmc raster fetch"}; do
  PY=$(venv_for "$S")/bin/python
  echo "######################## battery: shard $S via $PY"
  PYTHONPATH=$TREE $PY examples/fidelity_step4.py --shard "$S" --device cuda \
      --states-dir "$TREE/examples/results/step4/states" --out "$OUT/${S}${TAG}.json" 2>&1 | tail -25
  echo "BATTERY $S rc=$?"
  cp -f "$OUT/${S}${TAG}.json" "$CDIR/conformance/${S}.json" 2>/dev/null
done
PYTHONPATH=$TREE $SCR/venvs/se-step3/bin/python scratch/step4/certify.py --shards "$CDIR" \
    --out "$CDIR" --doc "$SCR/logs/trackcom_cuda_${SLURM_JOB_ID}.md" 2>&1 | tail -3
rm -rf "$D"
echo "TRACKCOM_GPU_DONE $(date -u +%FT%TZ)"
