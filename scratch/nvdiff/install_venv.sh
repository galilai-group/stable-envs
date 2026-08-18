#!/bin/bash
# Build the isolated venv this study runs in, from scratch, so the recipe is verified rather than
# remembered. Everything except nvdiffrast comes from the `test` conda env through
# --system-site-packages: that env already carries the exact reference stack the OGBench and
# locomotion parity numbers in this repo were taken under (mujoco 3.9.0, ogbench 1.2.1,
# gymnasium 1.2.3, torch 2.11.0+cu130), and a venv never writes into its base, so the shared env is
# untouched.
#
# The one thing that has to be built is nvdiffrast's CUDA extension. Two things it needs and does
# not find on its own:
#   * CUDA_HOME with a real nvcc. The `test` conda env has nvcc 13.0.88, matching torch's cu130.
#   * CPATH pointing at torch's own bundled CUDA headers. nvdiffrast compiles against torch's
#     extension headers, which include <cusparse.h>; the conda toolkit does not ship it, so the
#     build fails on a missing include that has nothing to do with nvdiffrast.
set -euo pipefail

BASE=/mnt/vast/home/randall.balestriero/miniconda3/envs/test
VENV=/mnt/vast/home/randall.balestriero/venvs/senvdiff

rm -rf "$VENV"
"$BASE/bin/python" -m venv --system-site-packages "$VENV"

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
PY
echo "OK $VENV"
