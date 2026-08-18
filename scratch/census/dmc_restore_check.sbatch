#!/bin/bash
#SBATCH --job-name=se_dmcfix
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/dmcfix_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
P=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
echo "### import + registry census"
$P -c "
import stable_envs as se
dmc=[n for n in se.REGISTRY if n.startswith('DMC')]
print('registry:', len(se.REGISTRY), ' DMC cells:', len(dmc))
print('errors:', {k:v[:70] for k,v in se.envs.import_errors().items() if 'dmcontrol' in k})
import stable_envs.envs.dmcontrol as m
print('TrackCom present:', hasattr(m,'DMControlTrackCom'), '_tc_context:', hasattr(m,'_tc_context'), '_TCtx:', hasattr(m,'_TCtx'))
"
echo "### the 5 visual TrackCom cells actually render + declared fidelity"
$P -m pytest tests/gpu -q --no-header --tb=short \
   -k "DMC and (visual_tier or parity or batch_invariance or determinism)" 2>/dev/null | tail -6
