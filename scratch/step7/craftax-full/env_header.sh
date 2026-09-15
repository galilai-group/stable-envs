# Shared header for every se7-craftax-full job. HOME is at 100% of its quota on Oscar, so every
# cache that would otherwise land there is redirected to scratch BEFORE anything imports.
SCR=/oscar/scratch/sboughan
WS=$SCR/se7/craftax-full
export UV_CACHE_DIR=$SCR/cache/uv
export PIP_CACHE_DIR=$SCR/cache/pip
export XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch
export HF_HOME=$SCR/cache/hf
export WARP_CACHE_PATH=$SCR/cache/warp
export MPLCONFIGDIR=$SCR/cache/mpl
export JAX_PLATFORMS=cpu
mkdir -p $SCR/cache/{uv,pip,xdg,torch,hf,warp,mpl} $SCR/logs
cd $WS
export PYTHONPATH=$WS
