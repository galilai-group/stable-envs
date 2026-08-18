#!/bin/bash
# Can ONE interpreter hold all 196 envs? Build the candidate and let it answer.
#
# The 7-way fan-out is justified by three claimed conflicts. Two of them dissolve on inspection:
#
#   * robocasa pins mujoco==3.3.1, BUT stable_envs/envs/robocasa.py never imports robocasa -- it
#     imports mujoco and reads a CACHED COMPILED KITCHEN. The pin binds the offline builder, not the
#     runtime.
#   * craftax / crafter / minihack embed their atlases in the module itself (zlib+base64), so their
#     upstream packages are build-time too.
#
# What is left is a set of DECLARED pins that production already violates successfully -- metaworld
# declares mujoco==3.3.0 while se-hb runs it on 3.11.0, procgen2 declares numpy<2 while se-hb runs
# 2.4.6, nle declares gymnasium==1.2.0 while se-hb runs 1.2.3 -- plus one real question: jaxmarl 0.1.0
# declares jax<=0.4.38, and the other jax families (craftax 1.6.1, jumanji 1.1.2, pgx 2.6.0) run on
# 0.10.2. Resolving that by downgrading jax is the wrong direction and would be checked by nothing;
# this script pins jax FORWARD at 0.10.2 and then tests whether Overcooked actually runs, which is the
# only way to find out.
#
# Writes a new venv and touches no existing one.
#SBATCH --job-name=se-build-all
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_build_all_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
UV=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/uv
VENV=/mnt/vast/home/randall.balestriero/venvs/se-all
export UV_CACHE_DIR=/mnt/vast/home/randall.balestriero/.cache/uv

echo "### node=$(hostname)"
rm -rf "$VENV"
"$UV" venv --python 3.11 "$VENV" || exit 1
PY=$VENV/bin/python
export VIRTUAL_ENV=$VENV

# The three declared-but-false pins, each already contradicted by a working interpreter, plus jax
# forced FORWARD so jaxmarl is tested against the jax the rest of the library uses.
cat > /tmp/over_all.txt <<'EOF'
mujoco==3.11.0
numpy==2.4.6
gymnasium==1.2.3
jax==0.10.2
jaxlib==0.10.2
EOF

cat > /tmp/req_all.txt <<'EOF'
mujoco==3.11.0
mujoco-warp==3.11.0
warp-lang==1.16.0
dm_control==1.0.44
gymnasium==1.2.3
gymnasium-robotics==1.4.2
metaworld==3.1.1
minigrid==3.1.0
procgen2==0.10.7
gym-pusht==0.1.6
Box2D==2.3.10
gym-sokoban==0.0.6
crafter==1.8.3
griddly==1.6.7
highway-env==1.12.1
ogbench==1.2.1
stable-worldmodel==0.1.1
jax==0.10.2
jaxlib==0.10.2
jumanji==1.1.2
pgx==2.6.0
resvg_py==0.3.4
craftax==1.6.1
minihack==1.0.2
nle==1.3.0
pygame
matplotlib
imageio
lance
pytest
EOF

echo; echo "######## 1. torch (cu130, the version every working interpreter runs) ########"
"$UV" pip install --python "$PY" torch==2.11.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -5

echo; echo "######## 2. the core union, with the three false pins overridden ########"
"$UV" pip install --python "$PY" --override /tmp/over_all.txt -r /tmp/req_all.txt 2>&1 | tail -30

echo; echo "######## 3. git-sourced: humanoid_bench, nvdiffrast — --no-deps ON PURPOSE ########"
# --no-deps, and EDITABLE for humanoid_bench, both for measured reasons.
#
# humanoid_bench declares gymnasium==0.29.1, mujoco==3.1.6 and an old torch. Installed normally it
# does not fail -- it SUCCEEDS and silently drags the environment backwards: the first run of this
# script came out with dm_control 1.0.20 (asked for 1.0.44) and torch 2.3.1 (asked for 2.11.0+cu130),
# which is what produced "'MjModel' object has no attribute 'flex_xvert0'" and read exactly like a
# real dm_control/mujoco incompatibility. It was not one; dmc and se-hb both run dm_control against
# mujoco 3.11.0 from plain PyPI wheels. Its declared pins are the same kind of stale-but-false pin as
# metaworld's and procgen2's, so its deps are supplied by the resolution above rather than by it.
#
# EDITABLE because a wheel build drops the humanoid_bench.dmc_deps subpackage -- the first run failed
# on exactly that -- and se-hb, which works, also has it editable.
# CLONED FIRST, because `uv pip install -e git+...` is rejected outright: "Editable must refer to a
# local directory, not a Git URL". Run 2 lost both packages to that one line -- 17 HumanoidBench and
# 6 Mujoco locomotion envs -- and the failure was two lines of stderr inside a `tail -6`.
SRC=/mnt/vast/home/randall.balestriero/src
mkdir -p $SRC
[ -d $SRC/humanoid-bench ] || git clone -q https://github.com/carlosferrazza/humanoid-bench.git $SRC/humanoid-bench
git -C $SRC/humanoid-bench checkout -q cb1189039151c8aadaaa987b442da54383c87fab
[ -d $SRC/nvdiffrast ] || git clone -q https://github.com/NVlabs/nvdiffrast.git $SRC/nvdiffrast
git -C $SRC/nvdiffrast checkout -q 253ac4fcea7de5f396371124af597e6cc957bfae

# humanoid_bench EDITABLE (a wheel build drops its humanoid_bench.dmc_deps subpackage), nvdiffrast
# NOT editable (it has no editable-installable layout; -e leaves it absent, which cost the 6 Mujoco
# locomotion envs in run 3). dmc, which works, also has it as a plain git install.
"$UV" pip install --python "$PY" --no-deps -e $SRC/humanoid-bench 2>&1 | tail -5
"$UV" pip install --python "$PY" --no-deps $SRC/nvdiffrast 2>&1 | tail -5
echo "-- both must now be present --"
"$UV" pip freeze --python "$PY" 2>/dev/null | grep -iE "humanoid|nvdiffrast" | sed 's/^/   /'

echo "-- the pins that must have SURVIVED step 3 --"
"$UV" pip freeze --python "$PY" 2>/dev/null | grep -iE "^(torch|dm-control|mujoco|gymnasium|numpy)==" | sed 's/^/   /'

echo; echo "######## 4. jaxmarl, WITHOUT letting it drag jax backwards ########"
# Run 1 established the blocker is NOT the jax pin but scipy: jaxmarl 0.1.0 requires scipy<=1.12 and
# jax 0.10.2 requires scipy>=1.14. Retried here with --no-deps to find out whether that pin is stale
# like the others, or load-bearing -- i.e. whether jaxmarl's Overcooked actually RUNS on modern jax
# once nothing is allowed to downgrade around it. That is the whole question for the last 5 envs.
"$UV" pip install --python "$PY" --no-deps jaxmarl==0.1.0 2>&1 | tail -8
# brax included: run 2 got past the scipy pin and then died on `No module named 'brax'`, so the
# question was still unanswered. Every one of these is --no-deps for the same reason as above -- they
# are here to satisfy jaxmarl's imports, not to be allowed a vote on jax, scipy or mujoco.
"$UV" pip install --python "$PY" --no-deps chex flax gymnax distrax brax orbax-checkpoint \
    optax etils ml_collections mujoco-mjx tensorflow-probability dm-env trimesh PyOpenGL 2>&1 | tail -4
echo "-- jax/scipy after jaxmarl (jax must still be 0.10.2) --"
"$UV" pip freeze --python "$PY" 2>/dev/null | grep -iE "^(jax|jaxlib|scipy)==" | sed 's/^/   /'

echo; echo "######## 5. pip check ########"
"$UV" pip check --python "$PY" 2>&1 | tail -25

echo; echo "######## 6. does Overcooked actually RUN on modern jax? ########"
"$PY" - <<'EOF' 2>&1 | tail -25
try:
    import jax, jax.numpy as jnp, numpy as np
    print("jax", jax.__version__, "numpy", np.__version__)
    from jaxmarl import make as jm_make
    env = jm_make("overcooked")
    k = jax.random.PRNGKey(0)
    obs, state = env.reset(k)
    print("OVERCOOKED RESET OK; agents:", list(obs)[:3])
    acts = {a: jnp.array(0) for a in env.agents}
    obs, state, r, d, i = env.step(k, state, acts)
    print("OVERCOOKED STEP OK; reward:", {a: float(v) for a, v in r.items()})
    print("VERDICT: jaxmarl 0.1.0 WORKS on jax", jax.__version__)
except Exception as e:
    import traceback; traceback.print_exc()
    print("VERDICT: jaxmarl FAILS forward on modern jax ->", type(e).__name__, e)
EOF

echo; echo "######## 7. which env modules import, and how big is the registry ########"
PYTHONPATH=. "$PY" - <<'EOF' 2>&1 | tail -45
import stable_envs as se
errs = se.envs.import_errors()
print("REGISTRY:", len(se.REGISTRY), "envs   (census union is 196)")
print("modules that FAILED to import:", len(errs))
for k, v in sorted(errs.items()):
    print(f"  {k}: {str(v)[:150]}")
import json
have = set(se.REGISTRY)
cover = json.load(open("scratch/census/assignment.json"))
want = {e for es in cover.values() for e in es}
missing = sorted(want - have)
print(f"\nassigned-but-unreachable: {len(missing)}")
for m in missing[:40]:
    print("  ", m)
EOF

echo "### build done"
