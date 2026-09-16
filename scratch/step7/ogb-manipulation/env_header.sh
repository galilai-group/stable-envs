# The environment every OGB GPU step runs under. Sourced by both the sbatch scripts and
# `run_in_alloc.sh`, so a step run inside the persistent allocation is the same step that ran as a
# batch job -- one definition, not two that drift.
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WS=$SCR/se7/ogb-manipulation
export VENV=$SCR/venvs/se7-ogb
export PYTHONPATH=$WS:$SCR/se7/swm-src:$WS/scratch/step7/ogb-manipulation
