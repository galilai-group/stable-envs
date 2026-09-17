#!/bin/bash
#SBATCH --job-name=se8-capability-plate-texels-3
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# T3 merge-gate item 4 for the seven parity DMC cells that are in the frozen Step-4 universe:
# Gate A on cuda with the new style_dim through examples/fidelity_step4.py, the driver that
# certified them in Steps 4-7 (gate_a_new_cells.py only knows the seven Step-7 cells).
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
WS=$SCR/se8/capability; OUT=$WS/examples/results/step8/capability/gpu; mkdir -p "$OUT"
PY=$SCR/venvs/se-step6/bin/python; $PY -c "import mujoco_warp" 2>/dev/null || PY=$SCR/venvs/se-vis311/bin/python
PARITY=DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCFingerTurnHard,DMCPendulumSwingup,DMCReacherHard
{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "sha $(cat $WS/.git_sha)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } | tee "$OUT/provenance3.txt"
( cd "$WS" && PYTHONPATH="$WS" $PY examples/fidelity_step4.py --configs "$PARITY" --device cuda --n-states 64 --pool 256 --out "$OUT/gate_a_cuda_dmc_parity.json" ) || echo "GATE_A_PARITY FAILED"
echo "ALLDONE3 $(date -Is)"
