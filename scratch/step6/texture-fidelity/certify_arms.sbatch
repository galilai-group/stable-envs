#!/bin/bash
#SBATCH --job-name=se6-texture-fidelity-certify
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# TEXTURE-2 item 3 + reviewer item 8, on ONE card:
#   certify311/conformance/dmc.json : Gate-A shard under the declared pair (se-vis311, mujoco 3.11.0)
#   certify312/conformance/dmc.json : the same under mujoco 3.12.0 (se-step5) -> BLOCKED, not FAIL
#   downsample_all/<venv>.json      : check_resolution_consistency on the bound's own protocol AND the
#                                     frozen corpus for every config with a DOWNSAMPLE_EXACT_FRAC bound,
#                                     measured in each venv that can build the env (reference-free check)
# scratch/step4/certify.py and the --merge of the table run on the Mac afterwards (no GPU needed).
# The commit SHA reaches the shard through STABLE_ENVS_GIT_COMMIT / $WS/GIT_COMMIT (rsync has no .git).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se6/texture-fidelity
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=$WS SDL_VIDEODRIVER=dummy
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
OUT=$WS/examples/results/step6/texture-fidelity
STATES=$WS/examples/results/step4/states
CONFIGS=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
set -x
echo "git commit: $STABLE_ENVS_GIT_COMMIT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd "$WS"
for ARM in 311 312; do
  if [ "$ARM" = 311 ]; then PY=$SCR/venvs/se-vis311/bin/python; else PY=$SCR/venvs/se-step5/bin/python; fi
  rm -rf "$OUT/certify$ARM"; mkdir -p "$OUT/certify$ARM/conformance"
  "$PY" examples/fidelity_step4.py --shard dmc --configs "$CONFIGS" --device cuda \
      --states-dir "$STATES" --out "$OUT/certify$ARM/conformance/dmc.json"
  echo "CERTIFY-SHARD $ARM rc=$?"
done
rm -rf "$OUT/downsample_all"; mkdir -p "$OUT/downsample_all"
for V in se-step5 se-step6 se-step4-swm; do
  "$SCR/venvs/$V/bin/python" scratch/step6/texture-fidelity/downsample_all.py --out "$OUT/downsample_all/$V.json"
  echo "DOWNSAMPLE-ALL $V rc=$?"
done
echo "CERTIFY DONE"
