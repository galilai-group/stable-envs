#!/bin/bash
# stable-envs SMOKE subset, for use before a commit.
#   sbatch scripts/run_smoke_tests.sbatch            # one env per FAMILY          ~3.5 min
#   sbatch scripts/run_smoke_tests.sbatch --fast     # one env per RENDER BACKEND  ~1 min
#
# The full suite is the release gate and takes ~22 minutes, which is long enough that in practice it
# gets run after the fact rather than before the commit. These subsets run every test in the suite
# against a few environments -- the families are the code paths, the 25 OGBench cells are one renderer
# -- so a change that breaks the rasterizer, the tile atlas, the ray caster, the dataset path or the
# style plumbing fails here in minutes instead of in a report tomorrow.
#
# TIMINGS, MEASURED on an H200 (gb-west-a1-gpu-110), wall clock including collection and the node gate:
#   default  3.4 min, 279 passed / 34 skipped over the 9 registered families. NOT under two minutes,
#            and it cannot be with every family in: `test_per_env_variation_is_per_env` alone costs 25s
#            on SimplePointMaze, 23s on Acrobot, 19s on MiniHack and 13s on Highway, because it renders
#            once per declared variation leaf. The only ways under two minutes with all nine are to
#            weaken that check or to drop a family, and both cost more than the ninety seconds.
#   --fast   0.9 min, 133 passed / 10 skipped. Four envs, chosen by RENDER BACKEND rather than by
#            family, which is the partition that actually predicts what a code change can break:
#              PushT                    the analytic 2-D rasterizer (render/raster2d.py, Triton)
#              MiniGridEmpty            the tile-atlas path (render/tilemap.py)
#              PointmazeMediumNavigate  the 3-D ray caster and the MuJoCo plate bake (render/raycast3d)
#              CraftaxClassic           the native pixel gather
#            Every other family renders through one of those four. This is the loop to run while
#            editing a kernel; the default is the one to run before pushing.
#
# What NEITHER covers: the cross-process reproducibility check (it boots a second interpreter and
# rebuilds every selected env in it, which is most of its runtime and answers a release question, not a
# pre-commit one), the harness's own crash self-tests in tests/integration/test_safety_net.py (~80s of spawning
# interpreters, and they do not change when a renderer does), and the other 40 registered envs. Green
# here is not a release; scripts/run_gpu_tests.sbatch is.
#SBATCH --job-name=se-smoke
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# EXCLUDED NODES, and they must be the same list in all three launchers here. Every one of these has
# been measured returning DIFFERENT PIXELS for the same scene from its own offscreen EGL rasterizer,
# before any stable-envs code is involved -- 004, 058 and 068 are the ones caught red-handed, and
# gpu-058 was caught twice, in the MetaWorld and OGBench
# investigations; the rest were condemned by the scheduler during a run. A parity number measured on any of them is meaningless, and the failure reads as a
# state-sufficiency violation in the renderer, which is where an afternoon goes. The node gate in
# tests/_runner/node_health.py catches a node that goes bad DURING a run and exits 78; this list
# is the ones already known. tests/unit/test_repo_contracts.py asserts the three lists agree.
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_smoke_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python

if [ "${1:-}" = "--fast" ]; then
  shift
  export SE_SMOKE_ENVS=PushT,MiniGridEmpty,PointmazeMediumNavigate,CraftaxClassic
fi

RUN=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_smoke_${SLURM_JOB_ID:-manual}
mkdir -p "$RUN"

"$PY" -u -m tests._runner.supervisor \
    --rundir "$RUN" --json "$RUN/result.json" --smoke --stall-timeout 600 \
    -- tests/ -q -rs "$@"
rc=$?
echo "se-smoke job=${SLURM_JOB_ID:-manual} node=$(hostname) rc=$rc envs=${SE_SMOKE_ENVS:-per-family} artifacts=$RUN"
exit $rc
