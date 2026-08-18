#!/bin/bash
#SBATCH --job-name=se-gallery
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_gallery_%j.out
set -uo pipefail
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R; export PYTHONPATH=$R MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
echo "### node=$(hostname)"
# Media for the drop-in page. The nine that exist are all tile families, which under-sells the range:
# these five add the 2-D rasterizer, the 3-D ray caster, the MJWarp raytracer and the gfxdraw path, so
# a reader sees the claim hold across backends rather than across five MiniGrid variants.
/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python -u examples/gallery.py \
    --envs PushT,PointmazeMediumNavigate,CarRacing,MetaWorldAssembly,DMCWalkerWalk \
    --out docs/assets/media --video-frames 120 --target-px 288 2>&1 | tail -40
echo; echo "### produced ###"
for e in PushT PointmazeMediumNavigate CarRacing MetaWorldAssembly DMCWalkerWalk; do
  printf "%-26s " "$e"; ls docs/assets/media/$e 2>/dev/null | tr '\n' ' '; echo
done
du -sh docs/assets/media
echo "### done"
