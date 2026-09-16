#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-owner
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Two questions the last run raised and could not answer from its own output.
#
# (a) WHO owns the four Renderers that are still finalised? Every remaining close stack names the
#     garbage collector, not the creator, so creation stacks have to be recorded at __init__.
# (b) Did pinning the plate to the all-zero button bits MOVE THE PIXELS, or is the whole shift in
#     the numbers just the change of seeds? Scene went 5.21-5.38 (old tree, seeds 1000-1007) to
#     6.49-6.69 (new tree, seeds 0-7) and those two facts cannot be compared. Re-measuring the new
#     tree at seeds 1000-1007 isolates it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/m6
export PYTHONPATH=$WS:$SCR/se7/swm-src
mkdir -p "$OUT/owner" "$OUT/attrib"
cd "$WS"

WANT=ea8ba8ec961a429faa12a9f57c090e3e
GOT=$(md5sum stable_envs/envs/ogb_manip.py | cut -d' ' -f1)
echo "ogb_manip.py md5 want=$WANT got=$GOT"
[ "$WANT" = "$GOT" ] || { echo "MD5 MISMATCH -- the node is not running HEAD; refusing"; exit 3; }

set -x
nvidia-smi --query-gpu=name --format=csv || true

"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_renderer_owner.py \
  --out "$OUT/owner" --cells OGBCube OGBScene --n 8 > "$OUT/owner/owners.txt" 2>&1 || echo "OWNER rc=$?"
grep -cE "^\[BORN|^\[CLOSED|^\[FINALISED" "$OUT/owner/owners.txt" || true
sed -n '1,400p' "$OUT/owner/owners.txt"

"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_gate_fidelity.py \
  --out "$OUT/attrib" --cells OGBCube OGBScene --n 64 --draws 8 --seed-base 1000 \
  || echo "ATTRIB rc=$?"
cat "$OUT/attrib/paste_me.json" 2>/dev/null || true

cp "/oscar/scratch/$USER/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
gzip -9f "$OUT/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo OK
