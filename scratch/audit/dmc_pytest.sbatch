#!/bin/bash
# The conformance suite over the DM Control family, in the interpreter that can import dm_control.
#SBATCH --job-name=se-dmc-pytest
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/audit/%x_%j.log
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PY=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
RUN=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/audit/dmc_pytest_${SLURM_JOB_ID:-manual}
mkdir -p "$RUN"
"$PY" -u -m stable_envs.testing.supervisor --rundir "$RUN" --json "$RUN/result.json" \
    -- tests/gpu/test_conformance.py -v -rs -k DMC
echo "rc=$? node=$(hostname) artifacts=$RUN"
