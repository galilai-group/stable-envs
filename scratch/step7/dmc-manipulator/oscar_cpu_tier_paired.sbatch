#!/bin/bash
#SBATCH --job-name=se7-dmc-manipulator-cputier
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The CPU tier for DMCManipulatorBringBall, as ONE paired job: baseline, branch, baseline.
#
# The cell does not exist in the baseline tree -- it is what this branch adds -- so there is no
# before/after ratio for it and none is reported. The two baseline legs re-measure a FLOOR CELL whose
# Step-6 CPU throughput is on record (DMCReacherHard, 84 px, 1 thread, 111.0 f/s at B=16 on node1831
# in the shipped configuration), and their only job is to establish that this node was measuring a
# renderer at all. W1's three gates then judge the node: spread between the two baseline legs, the
# baseline floor against the known answer, and loadavg per NODE core. A number failing any of them is
# reported QUARANTINED rather than as a tier.
#
# The branch leg measures the cell at its REGISTERED DEFAULT (224 px) at one thread, and takes the
# official dm_control renderer and the one-core JPEG-q90 decode of its own frames IN THE SAME PROCESS
# ON THIS NODE, so S_cpu and cores_cpu are not a cross-machine ratio.
#
# $SCR/se7/baseline is the pristine 76dc511 archive. NOT $SCR/se6/baseline, which is the PRE-Step-6
# tree -- W1 lost a job to exactly that substitution and its gate 2 is what caught it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
# ONE thread, which is the budget the CPU tier is defined at.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WS=dmc-manipulator
PY=$SCR/venvs/se-vis311/bin/python
OUT=$SCR/se7/$WS/examples/results/step7/$WS/cputier
HOST=$(hostname -s)
FLOOR=DMCReacherHard
CELL=DMCManipulatorBringBall
mkdir -p "$OUT"

echo "node $(hostname)  cores $(nproc --all)  alloc ${SLURM_CPUS_PER_TASK}  loadavg $(cat /proc/loadavg)"
$PY -c "import mujoco; from importlib.metadata import version as v; \
        print('mujoco', mujoco.__version__, 'dm_control', v('dm_control')); \
        assert mujoco.__version__.startswith('3.11'), 'wrong pin'"
test -f "$SCR/se7/baseline/.GIT_COMMIT" && echo "baseline tree = $(cat $SCR/se7/baseline/.GIT_COMMIT)"

run () {  # leg, tree, env, size
  local leg=$1 tree=$2 env=$3 size=$4
  cd "$tree"
  PYTHONPATH=$tree $PY "$SCR/se7/$WS/scratch/step7/$WS/cpu_tier.py" \
      --env "$env" --size "$size" --leg "$leg" \
      --out "$OUT/cpu_tier_${leg}_${HOST}.json" \
      || echo "LEG $leg rc=$?"
  echo "--- loadavg after $leg: $(cat /proc/loadavg)"
}

set -x
run baseA  "$SCR/se7/baseline" "$FLOOR" 84
run branch "$SCR/se7/$WS"      "$CELL"  224
run baseB  "$SCR/se7/baseline" "$FLOOR" 84
set +x

cd "$SCR/se7/$WS"
$PY scratch/step7/$WS/paired_verdict_cpu.py --dir "$OUT" --host "$HOST" \
    --out "$OUT/paired_verdict_cpu_${HOST}.json"
ls -la "$OUT"
