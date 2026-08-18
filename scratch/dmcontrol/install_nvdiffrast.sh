#!/bin/bash
# Add nvdiffrast to the DM Control venv, so the trackcom domains can be measured against BOTH
# full-scene backends in one interpreter. dm_control needs mujoco >= 3.10 (docs/research/dmcontrol.md
# S1) and `senvdiff` is pinned to mujoco 3.9.0, so the existing nvdiffrast venv cannot load a single
# DM Control domain -- the two studies cannot share an environment and the package has to be built
# into this one.
#
# Recipe verbatim from scratch/nvdiff/install_venv.sh, which was run from scratch and verified:
#   * CUDA_HOME with a real nvcc, matching torch's CUDA major version (13 here);
#   * CPATH at torch's OWN bundled CUDA headers, because nvdiffrast compiles against torch's
#     extension headers, which include <cusparse.h>, and the conda toolkit does not ship it.
# The only difference is the target venv: `dmc` was built WITHOUT --system-site-packages, so it
# carries its own mujoco 3.11.0 and nothing here can reach the shared conda env's mujoco 3.9.0.
set -euo pipefail

BASE=/mnt/vast/home/randall.balestriero/miniconda3/envs/test
VENV=/mnt/vast/home/randall.balestriero/venvs/dmc

SP=$("$VENV/bin/python" -c "import torch,os;print(os.path.dirname(torch.__file__))")
INC="$(dirname "$SP")/nvidia/cu13/include"
echo "torch      : $("$VENV/bin/python" -c 'import torch;print(torch.__version__)')"
echo "cuda hdrs  : $INC"
echo "nvcc       : $($BASE/bin/nvcc --version | tail -2 | head -1)"

CUDA_HOME="$BASE" CPATH="$INC" TORCH_CUDA_ARCH_LIST=9.0 \
  "$VENV/bin/pip" install --no-cache-dir --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"

"$VENV/bin/python" - <<'PY'
import torch, nvdiffrast, nvdiffrast.torch as dr
print("nvdiffrast", nvdiffrast.__version__)
ctx = dr.RasterizeCudaContext(device="cuda")
pos = torch.tensor([[[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]]],
                   device="cuda")
tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
rast, db = dr.rasterize(ctx, pos, tri, resolution=[64, 64], grad_db=True)
print("rast", tuple(rast.shape), "covered", int((rast[..., 3] > 0).sum()), "db", tuple(db.shape))
import dm_control, mujoco
from dm_control import suite
print("dm_control", dm_control.__version__, "mujoco", mujoco.__version__)
suite.load("cheetah", "run", task_kwargs={"random": 0})
print("dm_control still loads")
PY
echo "OK $VENV"
