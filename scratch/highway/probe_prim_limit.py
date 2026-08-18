"""How many primitives can the fused kernel take before ptxas gives up?

``_scene_kernel`` walks the primitive list with ``tl.static_range``, so the list is unrolled at compile
time and the kernel grows linearly with it. PushT has 9 primitives; a highway scene has 130 and a
roundabout 604. This finds where the wall is, because "the env is too big" needs a number.

Run as a subprocess per N: the failure mode is an abort inside ptxas, which kills the interpreter.
"""
import sys
import time
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.render.scene import Scene2D
from stable_envs.render.raster2d import render_triton


def scene(B, N, dev="cuda"):
    g = torch.zeros(B, N, 8, device=dev)
    g[..., 0] = 10.0; g[..., 1] = 10.0; g[..., 2] = 30.0; g[..., 3] = 30.0
    g[..., 4] = 30.0; g[..., 5] = 10.0
    return Scene2D(kind=torch.full((B, N), 2, dtype=torch.int32, device=dev), geom=g,
                   fill=torch.full((B, N, 3), 200.0, device=dev),
                   line=torch.full((B, N, 3), 60.0, device=dev),
                   line_w=torch.zeros(B, N, device=dev),
                   background=torch.full((B, 3), 100.0, device=dev),
                   world=(0.0, 0.0, 600.0, 150.0))


if __name__ == "__main__":
    N = int(sys.argv[1])
    t0 = time.perf_counter()
    out = render_triton(scene(4, N), (150, 600))
    torch.cuda.synchronize()
    print(f"N={N:5d} OK  compile+run {time.perf_counter()-t0:6.2f}s  mean={out.float().mean():.1f}")
