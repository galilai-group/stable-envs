#!/bin/bash
# The conformance suite for the Meta-World family, on the interpreter that can render it.
#
#   sbatch scratch/metaworld/run_tests.sbatch -k "MetaWorldReach or MetaWorldWindowOpen"
#   sbatch scratch/metaworld/run_tests.sbatch -k MetaWorld            # all 50
#
# NOT the shared conda env: that one carries mujoco 3.9 with no mujoco_warp, so this family raises
# ImportError there and vanishes from the registry (which is the designed behaviour -- see the module
# docstring). ~/venvs/se-fetch is the venv the numbers were measured in.
#
# `-k MetaWorld` matters for more than time. The env fixtures are module-scoped and every MuJoCo family
# holds EGL contexts that are never released; 50 of ours plus 50 reference envs is already ~100 live
# contexts, and adding the other 44 registered envs on top of that is how a suite starts failing with
# 69-90/255 differences that look like a renderer bug and are a driver limit.
#SBATCH --job-name=mw-test
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184,gb-west-a1-gpu-058
#SBATCH --output=/mnt/vast/home/randall.balestriero/se_logs/metaworld/pytest_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/venvs/se-fetch/bin/python

RUN=/mnt/vast/home/randall.balestriero/se_logs/metaworld/pytest_${SLURM_JOB_ID:-manual}
mkdir -p "$RUN"
"$PY" -u -m stable_envs.testing.supervisor --rundir "$RUN" --json "$RUN/result.json" \
    -- tests/ -v -rs --durations=20 "$@"
rc=$?
echo "mw-test job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc artifacts=$RUN"
exit $rc
