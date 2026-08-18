"""Benchmark nvdiffrast as a MuJoCo batch rasterizer.

nvdiffrast rasterizes a triangle soup in clip space; it knows nothing about MuJoCo,
so this file supplies the missing half: tessellate every geom once (mjgeom.py), then
per frame transform the static soup by the per-world geom frames and a camera matrix,
rasterize, interpolate colour + normal, and shade.

The transform is the interesting part. `xmat[b, vgeom] @ vert` as a gather would
materialise a (B, V, 9) tensor, which is the dominant cost at B=1024. Instead the
gather is folded into a sparse matmul: S is (V, 4G) holding each vertex's homogeneous
local position in the 4 columns of its own geom, so `S @ MVP_reshaped` produces every
world's clip-space vertex in one cuSPARSE call with no intermediate.

What is timed is transform + rasterize + interpolate + shade -> uint8. Geom frames
(xpos/xmat) are taken as given on device: producing them is MJX/MJWarp's job, and
including MuJoCo's CPU kinematics here would measure the wrong thing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import numpy as np
import torch

import nvdiffrast.torch as dr
from mjgeom import camera_matrices, tessellate


def geom_frames(mjm, mjd, batch, seed=0):
    """A batch of (xpos, xmat) from real MuJoCo kinematics on a batch of qpos.

    One-time setup, deliberately outside the timed region.
    """
    rng = np.random.default_rng(seed)
    xpos = np.zeros((batch, mjm.ngeom, 3), dtype=np.float32)
    xmat = np.zeros((batch, mjm.ngeom, 3, 3), dtype=np.float32)
    q0 = np.asarray(mjd.qpos, dtype=np.float64).copy()
    for b in range(batch):
        q = q0.copy()
        q[:2] += rng.uniform(-4.0, 4.0, size=2)
        if mjm.nq > 7:
            q[7:] += rng.uniform(-0.3, 0.3, size=mjm.nq - 7)
        mjd.qpos[:] = q
        mujoco.mj_forward(mjm, mjd)
        xpos[b] = mjd.geom_xpos
        xmat[b] = np.asarray(mjd.geom_xmat).reshape(-1, 3, 3)
    return xpos, xmat


class NvdiffrastMJ:
    def __init__(self, mjm, mjd, res, cam, device="cuda"):
        self.res, self.dev = res, device
        soup = tessellate(mjm)
        self.V = len(soup["vert"])
        self.G = mjm.ngeom
        self.tri = torch.as_tensor(soup["face"], dtype=torch.int32, device=device)
        self.rgb = torch.as_tensor(soup["rgb"], device=device)
        vg = torch.as_tensor(soup["vgeom"], device=device)

        # S4: (V, 4G) sparse, row v = homogeneous local position in its geom's 4 columns
        v4 = torch.cat([torch.as_tensor(soup["vert"], device=device),
                        torch.ones(self.V, 1, device=device)], -1)
        rows = torch.arange(self.V, device=device).repeat_interleave(4)
        cols = (vg[:, None] * 4 + torch.arange(4, device=device)[None]).reshape(-1)
        self.S4 = torch.sparse_coo_tensor(torch.stack([rows, cols]), v4.reshape(-1),
                                          (self.V, 4 * self.G)).coalesce().to_sparse_csr()
        # S3: (V, 3G) same trick for normals (rotation only)
        n3 = torch.as_tensor(soup["norm"], device=device)
        rows3 = torch.arange(self.V, device=device).repeat_interleave(3)
        cols3 = (vg[:, None] * 3 + torch.arange(3, device=device)[None]).reshape(-1)
        self.S3 = torch.sparse_coo_tensor(torch.stack([rows3, cols3]), n3.reshape(-1),
                                          (self.V, 3 * self.G)).coalesce().to_sparse_csr()

        vp, campos = camera_matrices(mjm, mjd, cam, res)
        self.vp = torch.as_tensor(vp, device=device)
        self.campos = torch.as_tensor(campos, device=device)
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.bg = torch.tensor([0.1, 0.1, 0.2], device=device)

    def __call__(self, xpos, xmat):
        B, G = xpos.shape[0], self.G
        # per-geom model matrix, then MVP = VP @ M
        M = torch.zeros(B, G, 4, 4, device=self.dev)
        M[:, :, :3, :3] = xmat
        M[:, :, :3, 3] = xpos
        M[:, :, 3, 3] = 1.0
        mvp = self.vp @ M                                            # (B, G, 4, 4)

        rhs = mvp.permute(1, 3, 0, 2).reshape(4 * G, 4 * B)
        pos = torch.sparse.mm(self.S4, rhs).reshape(self.V, B, 4).permute(1, 0, 2).contiguous()

        rhs3 = xmat.permute(1, 3, 0, 2).reshape(3 * G, 3 * B)
        nrm = torch.sparse.mm(self.S3, rhs3).reshape(self.V, B, 3).permute(1, 0, 2).contiguous()

        rast, _ = dr.rasterize(self.ctx, pos, self.tri, resolution=[self.res, self.res], grad_db=False)
        col, _ = dr.interpolate(self.rgb[None].expand(B, -1, -1).contiguous(), rast, self.tri)
        nw, _ = dr.interpolate(nrm, rast, self.tri)

        nw = nw / nw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # headlight: light travels along the camera's view direction
        ldir = torch.nn.functional.normalize(self.campos.view(1, 1, 1, 3), dim=-1)
        diff = (nw * ldir).sum(-1, keepdim=True).abs().clamp(0.15, 1.0)
        shade = col * (0.35 + 0.65 * diff)
        hit = (rast[..., 3:4] > 0).float()
        img = shade * hit + self.bg.view(1, 1, 1, 3) * (1 - hit)
        # nvdiffrast returns row 0 at the bottom (OpenGL); MuJoCo returns it at the top
        return (img.clamp(0, 1) * 255).to(torch.uint8).flip(1)


def run(xml, batch, res, cam, iters=20, warmup=5, graph=False, label=""):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    r = NvdiffrastMJ(mjm, mjd, res, cam)
    xpos, xmat = geom_frames(mjm, mjd, batch)
    xpos = torch.as_tensor(xpos, device="cuda")
    xmat = torch.as_tensor(xmat, device="cuda")

    with torch.no_grad():
        for _ in range(warmup):
            out = r(xpos, xmat)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = r(xpos, xmat)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    fps = batch * iters / dt
    row = {
        "backend": "nvdiffrast",
        "scene": os.path.basename(xml),
        "label": label or "raster+shade",
        "batch": batch,
        "res": res,
        "nvert": r.V,
        "ntri": int(r.tri.shape[0]),
        "ms_per_batch": 1e3 * dt / iters,
        "fps": fps,
        "mpix_s": fps * res * res / 1e6,
        "out_shape": list(out.shape),
        "out_dtype": str(out.dtype),
    }
    print(json.dumps(row), flush=True)
    return row, r, (xpos, xmat)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="antmaze_medium.xml,cube_single.xml,humanoidmaze_medium.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batches", default="1,8,64,256,1024,4096")
    ap.add_argument("--out", default="")
    ap.add_argument("--images", default="")
    args = ap.parse_args()

    print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
    rows = []
    for scene in args.scenes.split(","):
        xml = os.path.join(args.dir, scene)
        if not os.path.exists(xml):
            continue
        for b in [int(x) for x in args.batches.split(",")]:
            try:
                row, r, _ = run(xml, b, args.res, 0)
                rows.append(row)
            except Exception as e:
                print(json.dumps({"backend": "nvdiffrast", "scene": scene, "batch": b,
                                  "error": f"{type(e).__name__}: {e}"}), flush=True)
    if args.images:
        import imageio.v2 as imageio

        os.makedirs(args.images, exist_ok=True)
        for scene in args.scenes.split(","):
            xml = os.path.join(args.dir, scene)
            if not os.path.exists(xml):
                continue
            mjm = mujoco.MjModel.from_xml_path(xml)
            mjd = mujoco.MjData(mjm)
            mujoco.mj_forward(mjm, mjd)
            r = NvdiffrastMJ(mjm, mjd, args.res, 0)
            xpos = torch.as_tensor(np.asarray(mjd.geom_xpos, np.float32)[None], device="cuda")
            xmat = torch.as_tensor(np.asarray(mjd.geom_xmat, np.float32).reshape(1, -1, 3, 3), device="cuda")
            with torch.no_grad():
                img = r(xpos, xmat)
            imageio.imwrite(os.path.join(args.images, scene.replace(".xml", "_nvdiffrast.png")),
                            img[0].cpu().numpy())
            print("wrote", scene)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
