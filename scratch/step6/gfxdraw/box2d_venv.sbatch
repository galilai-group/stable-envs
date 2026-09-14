#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-box2d
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The two other consumers of render/gfxdraw, on a host that can actually construct them.
#
# LunarLander and CarRacing need Box2D, which is in none of the Step 6 interpreters, so the
# consumer check has never run anywhere -- and a regression inside LunarLander.__init__ therefore
# sat behind a green "not registered" for a whole review cycle. This builds a venv that INHERITS
# se-step6's site-packages through a .pth (the pattern se6-raycast3d-maze2d already uses, so the
# torch/pygame/gymnasium pins are the certified ones and se-step6 is never written to) and adds
# swig + gymnasium[box2d] on top.
#
# Then it renders both envs from the BASELINE tree and from the branch and diffs the digests,
# which is the claim findings.md was making without evidence.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
V=$SCR/venvs/se6-gfxdraw-box2d
BASE=$SCR/venvs/se-step6
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/consumers
mkdir -p "$OUT"
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID"
set -x

if [ ! -x "$V/bin/pip" ]; then
  rm -rf "$V"
  # WITH pip: se-step6 was built by uv and has no pip of its own, so the first attempt could not
  # install anything. The new venv gets its own pip; se-step6 is still never written to.
  "$BASE/bin/python" -m venv "$V"
  # inherit the certified pins rather than resolving a second set (the se6-raycast3d-maze2d pattern)
  echo "$BASE/lib/python3.12/site-packages" > "$V/lib/python3.12/site-packages/_inherit_se_step6.pth"
  "$V/bin/pip" install --quiet --upgrade pip wheel setuptools || echo "pip bootstrap rc=$?"
  "$V/bin/pip" install --quiet swig || echo "swig rc=$?"
  PATH="$V/bin:$PATH" "$V/bin/pip" install --quiet --no-build-isolation box2d-py \
    || PATH="$V/bin:$PATH" "$V/bin/pip" install --quiet box2d-py \
    || echo "box2d-py rc=$?"
fi
"$V/bin/python" -c "import sys; print('site-packages order:', [p for p in sys.path if 'site-packages' in p])"
"$V/bin/python" -c "import Box2D, torch, pygame, gymnasium; print('Box2D', Box2D.__version__, 'torch', torch.__version__, 'pygame', pygame.version.ver)" \
  || { echo "BOX2D VENV UNUSABLE"; echo "BOX2D JOB DONE"; exit 0; }

for ARM in baseline gfxdraw; do
  cd "$SCR/se6/$ARM"
  PYTHONPATH=$SCR/se6/$ARM "$V/bin/python" \
      $SCR/se6/gfxdraw/scratch/step6/gfxdraw/consumer_digests.py \
      --envs LunarLander,CarRacing --device cpu --batch 6 \
      --snapshot-out "$OUT/consumers_${ARM}_$(hostname -s).json"
  echo "CONSUMERS $ARM rc=$?"
done
cd "$SCR/se6/gfxdraw"
PYTHONPATH=$SCR/se6/gfxdraw "$V/bin/python" scratch/step6/gfxdraw/consumer_digests.py \
    --envs LunarLander,CarRacing --device cpu --batch 6 \
    --snapshot-in "$OUT/consumers_baseline_$(hostname -s).json"
echo "BRANCH-VS-BASELINE rc=$?"

# the recorder write list, which is what LunarLander bakes its flag fringe from
for ARM in baseline gfxdraw; do
  cd "$SCR/se6/$ARM"
  PYTHONPATH=$SCR/se6/$ARM "$V/bin/python" -m pytest \
      $SCR/se6/gfxdraw/tests/unit/test_gfxdraw_blend_at_contract.py -q 2>&1 | tail -4
  echo "RECORDER CONTRACT $ARM rc=$?"
done
echo "BOX2D JOB DONE"
