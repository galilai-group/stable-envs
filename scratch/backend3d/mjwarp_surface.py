"""Reference integration surface: MuJoCo model + (B, nq) qpos -> (B, H, W, 3) uint8 on device.

This is the shape stable-envs needs from a 3-D backend, written against MJWarp so the
cost of each piece is measurable rather than assumed. It also answers three questions
the raw benchmark does not:

  * how much of the frame time is kinematics vs BVH refit vs the render kernel
  * what the BGRA -> RGB unpack costs (the packed buffer is `(a<<24)|(r<<16)|(g<<8)|b`,
    so a zero-copy byte view is BGRA, not RGB)
  * whether the whole thing survives CUDA graph capture as one replayable launch

`render(qpos)` is pure in the sense that matters here: the output depends only on the
state vector handed in, so nothing has to be stored on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw


class BatchRenderer:
    """Stateless-by-qpos batch renderer for one MuJoCo model."""

    def __init__(self, xml: str, batch: int, res: int = 224, cam: int = 0,
                 textures: bool = True, shadows: bool = False, skybox: bool = True):
        self.mjm = mujoco.MjModel.from_xml_path(xml)
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        self.batch, self.res = batch, res

        self.m = mjw.put_model(self.mjm)
        self.d = mjw.put_data(self.mjm, mjd, nworld=batch)
        self.rc = mjw.create_render_context(
            self.mjm, nworld=batch, cam_res=(res, res),
            render_rgb=True, render_depth=False,
            cam_active=[i == cam for i in range(self.mjm.ncam)],
            use_textures=textures, use_shadows=shadows, render_skybox=skybox,
        )
        # zero-copy uint8 view of the packed buffer: (B, H, W, 4) in BGRA order
        self.bgra = wp.to_torch(self.rc.rgb_data).view(torch.uint8).reshape(batch, res, res, 4)
        self.graph = None

    def _frame(self):
        mjw.kinematics(self.m, self.d)
        mjw.camlight(self.m, self.d)
        mjw.refit_bvh(self.m, self.d, self.rc)
        mjw.render(self.m, self.d, self.rc)

    def capture(self):
        self._frame()
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            self._frame()
        self.graph = cap.graph
        return self

    def render(self, qpos: torch.Tensor, rgb: bool = True) -> torch.Tensor:
        """qpos: (B, nq) float32 on device -> (B, H, W, 3) uint8 on device."""
        wp.to_torch(self.d.qpos).copy_(qpos)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()
        return self.bgra[..., [2, 1, 0]] if rgb else self.bgra


def timeit(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    wp.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    wp.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="antmaze_medium.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    xml = os.path.join(args.dir, args.scene)
    r = BatchRenderer(xml, args.batch, args.res)
    B = args.batch
    q = torch.as_tensor(np.tile(np.zeros(r.mjm.nq, np.float32), (B, 1)), device="cuda")
    q += torch.randn_like(q) * 0.1

    rows = []

    def add(label, sec, note=""):
        row = {"stage": label, "batch": B, "res": args.res, "scene": args.scene,
               "ms": 1e3 * sec, "fps": B / sec, "note": note}
        rows.append(row)
        print(json.dumps(row), flush=True)

    add("kinematics", timeit(lambda: mjw.kinematics(r.m, r.d)))
    add("camlight", timeit(lambda: mjw.camlight(r.m, r.d)))
    add("refit_bvh", timeit(lambda: mjw.refit_bvh(r.m, r.d, r.rc)))
    add("render_only", timeit(lambda: mjw.render(r.m, r.d, r.rc)))
    add("full_eager", timeit(lambda: r._frame()))
    r.capture()
    add("full_graph", timeit(lambda: wp.capture_launch(r.graph)))
    add("qpos->bgra_view", timeit(lambda: r.render(q, rgb=False)), "zero copy, BGRA")
    add("qpos->rgb_uint8", timeit(lambda: r.render(q, rgb=True)), "channel gather to RGB")

    out = r.render(q)
    print(json.dumps({"out_shape": list(out.shape), "out_dtype": str(out.dtype),
                      "device": str(out.device), "contiguous": out.is_contiguous()}))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
