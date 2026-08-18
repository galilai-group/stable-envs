#!/bin/bash
#SBATCH --job-name=se3d_fin3
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/final3_%j.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
DMC=/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python
HB=/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python
ENVS=/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python
echo "### registry"
$DMC -u -c "import stable_envs as se; d=[n for n in sorted(se.REGISTRY) if n.startswith(('DMC','Mujoco'))]; print(len(d),'cells')"
echo "### VISUAL TIER: swimmer must meet its NEW declaration"
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf -k "visual_tier and DMC" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))|DMCSwimmer"
echo "### bounded-vs-unbounded + fused-vs-torch"
$DMC -u -m pytest tests/gpu/test_mover_window.py tests/gpu/test_fused_cast.py -q -rf 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### official-renderer parity + determinism + relaxed-floor explanations"
$DMC -u -m pytest tests/gpu/test_conformance.py -q -rf -k "(parity or batch_invariance or determinism or backend_agreement or relaxed) and (DMC or Mujoco or registry)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### batch/chunk invariance"
$DMC -u -m pytest tests/gpu/test_reproducibility.py -q -rf -k "(batch_size or chunk_independent or out_buffer or batch_order) and (DMC or Mujoco)" 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))"
echo "### throughput, all 15 parity cells + the 5 visual cells"
$DMC -u scratch/perf3d/split256.py --batch 1024 --size 256 \
  --envs MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum,DMCCartpoleBalance,DMCCartpoleSwingup,DMCPendulumSwingup,DMCAcrobotSwingup,DMCPointMassEasy,DMCReacherEasy,DMCReacherHard,DMCFingerSpin,DMCFingerTurnEasy,DMCFingerTurnHard,DMCBallInCupCatch,DMCSwimmerSwimmer6,DMCCheetahRun,DMCWalkerWalk,DMCHopperHop,DMCHumanoidStand \
  --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/final3.json
echo "### ceiling for the gym MuJoCo cells after the port"
$DMC -u scratch/perf3d/ceiling.py --batch 1024 --size 256 \
  --envs MujocoReacher,MujocoPusher,MujocoInvertedPendulum,MujocoInvertedDoublePendulum \
  --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/ceiling_port2.json
echo "### untouched families still green"
STABLE_ENVS_HUMANOIDBENCH=1 $HB -u -m pytest tests/gpu/test_fused_cast.py -q 2>&1 | tail -2
$ENVS -u -m pytest tests/gpu/test_fused_cast.py -q 2>&1 | tail -2
