"""Throughput of the proposed renderer: baked static plate + batched analytic agent, all on GPU.

The camera is fixed per env and the maze never moves, so floor + walls + skybox are one constant image
(and one constant depth buffer). Per frame the only work is intersecting the agent's primitives -- 1
sphere for pointmaze, 13 capsules+sphere for ant, 19 for humanoid -- and compositing.

Two variants, because the difference is the whole point: ``render`` casts every primitive against all
50176 rays, and ``render_tiled`` casts them only inside a TxT screen-space tile around the agent. The
agent covers 20-150 pixels of 50176, so the full-frame version spends ~99.9% of its work on rays that
cannot hit anything. Measured against MuJoCo's offscreen renderer on the same GPU.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast as RC  # noqa: E402
from general_check import make2, agent_geoms, cast_subset  # noqa: E402
from analytic_rgb import base_rgb  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF
AMB, DIFF, SPEC, SHINE = 0.09804, 1.3907, 0.26, 60.0


class GPURenderer:
    def __init__(self, loco, size, img=224, device="cuda"):
        import mujoco
        env, u = make2(loco, size, img)
        m, d = u.model, u.data
        m.vis.quality.offsamples = 4
        r = mujoco.Renderer(m, width=img, height=img)
        r.update_scene(d, camera=u.custom_camera)
        cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
        ag, _ = agent_geoms(m)
        static = [g for g in range(m.ngeom) if g not in ag]
        keep = m.geom_rgba[ag].copy()
        m.geom_rgba[ag, 3] = 0.0
        r2 = mujoco.Renderer(m, width=img, height=img)
        r2.update_scene(d, camera=u.custom_camera)
        plate = r2.render().copy()
        m.geom_rgba[ag] = keep
        _, bg_t, o, dr = cast_subset(m, d, cam, img, img, static)

        dev = torch.device(device)
        self.dev, self.img, self.env, self.u, self.m, self.d, self.ag = dev, img, env, u, m, d, ag
        self.o = torch.tensor(o.reshape(-1, 3), device=dev, dtype=torch.float32)
        self.dr = torch.tensor(dr.reshape(-1, 3), device=dev, dtype=torch.float32)
        self.plate = torch.tensor(plate, device=dev, dtype=torch.float32).reshape(-1, 3)
        self.bg_t = torch.tensor(np.where(np.isfinite(bg_t), bg_t, 1e30).reshape(-1),
                                 device=dev, dtype=torch.float32)
        self.types = [int(m.geom_type[g]) for g in ag]
        self.sizes = torch.tensor(np.stack([m.geom_size[g] for g in ag]), device=dev,
                                  dtype=torch.float32)
        self.cols = torch.tensor(np.stack([base_rgb(m, g) for g in ag]), device=dev,
                                 dtype=torch.float32)
        V = -self.dr / self.dr.norm(dim=-1, keepdim=True)
        L = torch.tensor([0.0, 0.0, 1.0], device=dev)
        hv = L + V
        self.hv = hv / hv.norm(dim=-1, keepdim=True)
        self.L = L
        self.renderer = r
        fwd = np.asarray(cam["forward"], np.float64)
        up = np.asarray(cam["up"], np.float64)
        right = np.cross(fwd, up)
        self.cam_pos = torch.tensor(cam["pos"], device=dev, dtype=torch.float32)
        self.cam_fwd = torch.tensor(fwd, device=dev, dtype=torch.float32)
        self.cam_up = torch.tensor(np.cross(right, fwd), device=dev, dtype=torch.float32)
        self.cam_right = torch.tensor(right, device=dev, dtype=torch.float32)
        self.tan_half = float(cam["frustum_top"] / cam["frustum_near"])

    def geom_poses(self, B):
        """Repeat the current MuJoCo agent poses B times (the FK output a real pipeline would supply)."""
        xp = torch.tensor(np.stack([self.d.geom_xpos[self.ag]] * B), device=self.dev,
                          dtype=torch.float32)
        xm = torch.tensor(np.stack([self.d.geom_xmat[self.ag].reshape(-1, 3, 3)] * B),
                          device=self.dev, dtype=torch.float32)
        return xp, xm

    def _shade(self, k, o, dr, hv, best_t, col, xpos, xmat):
        pos = xpos[:, k].unsqueeze(1)
        R = xmat[:, k]
        sz = self.sizes[k]
        oc = o - pos
        if self.types[k] == 2:                                     # sphere
            rad = sz[0]
            b = (oc * dr).sum(-1)
            cc = (oc * oc).sum(-1) - rad * rad
            disc = b * b - cc
            sq = torch.sqrt(torch.clamp(disc, min=0))
            t = -b - sq
            hit = (disc > 0) & (t > 0)
            nrm = (oc + t.unsqueeze(-1) * dr) / rad
        else:                                                      # capsule (cylindrical body)
            ol = torch.bmm(oc, R)
            dl = torch.bmm(dr, R)
            rad, hh = sz[0], sz[1]
            a = dl[..., 0] ** 2 + dl[..., 1] ** 2
            b = ol[..., 0] * dl[..., 0] + ol[..., 1] * dl[..., 1]
            cc = ol[..., 0] ** 2 + ol[..., 1] ** 2 - rad * rad
            disc = b * b - a * cc
            sq = torch.sqrt(torch.clamp(disc, min=0))
            t = (-b - sq) / torch.clamp(a, min=1e-9)
            zz = ol[..., 2] + t * dl[..., 2]
            hit = (disc > 0) & (t > 0) & (zz.abs() <= hh)
            nl = torch.stack([ol[..., 0] + t * dl[..., 0], ol[..., 1] + t * dl[..., 1],
                              torch.zeros_like(t)], -1) / rad
            nrm = torch.bmm(nl, R.transpose(1, 2))
        upd = hit & (t < best_t)
        ndl = torch.clamp((nrm * self.L).sum(-1), min=0)
        ndh = torch.clamp((nrm * hv).sum(-1), min=0)
        S = AMB + DIFF * ndl + SPEC * ndh ** SHINE
        cv = torch.clamp(self.cols[k].view(1, 1, 3) * S.unsqueeze(-1), 0, 1) * 255.0
        return torch.where(upd, t, best_t), torch.where(upd.unsqueeze(-1), cv, col)

    @torch.no_grad()
    def render(self, xpos, xmat):
        B, P, _ = xpos.shape
        N = self.o.shape[0]
        best_t = self.bg_t.unsqueeze(0).expand(B, N).clone()
        col = self.plate.unsqueeze(0).expand(B, N, 3).clone()
        o = self.o.unsqueeze(0).expand(B, N, 3)
        dr = self.dr.unsqueeze(0).expand(B, N, 3)
        hv = self.hv.unsqueeze(0).expand(B, N, 3)
        for k in range(P):
            best_t, col = self._shade(k, o, dr, hv, best_t, col, xpos, xmat)
        return col.round_().clamp_(0, 255).to(torch.uint8).view(B, self.img, self.img, 3)

    @torch.no_grad()
    def render_tiled(self, xpos, xmat, T=48):
        B, P, _ = xpos.shape
        H = W = self.img
        c = xpos.mean(1)
        rel = c - self.cam_pos
        z = (rel * self.cam_fwd).sum(-1)
        u = (rel * self.cam_right).sum(-1) / z / self.tan_half
        v = (rel * self.cam_up).sum(-1) / z / self.tan_half
        cx = ((u + 1) * 0.5 * W).long()
        cy = ((1 - v) * 0.5 * H).long()
        x0 = torch.clamp(cx - T // 2, 0, W - T)
        y0 = torch.clamp(cy - T // 2, 0, H - T)
        ar = torch.arange(T, device=self.dev)
        yy = (y0.view(B, 1, 1) + ar.view(1, T, 1)).expand(B, T, T)
        xx = (x0.view(B, 1, 1) + ar.view(1, 1, T)).expand(B, T, T)
        idx = (yy * W + xx).reshape(B, -1)
        o = self.o[idx]
        dr = self.dr[idx]
        hv = self.hv[idx]
        best_t = self.bg_t[idx].clone()
        col = self.plate[idx].clone()
        for k in range(P):
            best_t, col = self._shade(k, o, dr, hv, best_t, col, xpos, xmat)
        out = self.plate.unsqueeze(0).expand(B, H * W, 3).clone()
        out.scatter_(1, idx.unsqueeze(-1).expand(B, idx.shape[1], 3), col)
        return out.round_().clamp_(0, 255).to(torch.uint8).view(B, H, W, 3)


def timeit(fn, xp, xm, reps=10):
    for _ in range(3):
        fn(xp, xm)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn(xp, xm)
    torch.cuda.synchronize()
    return time.time() - t0


def bench(loco, size, img=224, batches=(64, 256, 1024)):
    g = GPURenderer(loco, size, img)
    out = dict(env=f"{loco}maze-{size}", img=img, n_prims=len(g.ag),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    # the tiled path must produce the SAME frame as the full path, or the speed is meaningless
    xp, xm = g.geom_poses(2)
    a = g.render(xp, xm)
    b = g.render_tiled(xp, xm)
    out["tiled_matches_full_frac"] = float((a == b).all(-1).float().mean())
    for B in batches:
        xp, xm = g.geom_poses(B)
        try:
            dt = timeit(g.render, xp, xm)
            out[f"full_fps_B{B}"] = round(B * 10 / dt, 1)
        except torch.cuda.OutOfMemoryError:
            out[f"full_fps_B{B}"] = "OOM"
            torch.cuda.empty_cache()
        try:
            dt = timeit(g.render_tiled, xp, xm)
            out[f"tiled_fps_B{B}"] = round(B * 10 / dt, 1)
        except torch.cuda.OutOfMemoryError:
            out[f"tiled_fps_B{B}"] = "OOM"
            torch.cuda.empty_cache()
    r = g.renderer
    for _ in range(10):
        r.update_scene(g.d, camera=g.u.custom_camera)
        r.render()
    t0 = time.time()
    n = 200
    for _ in range(n):
        r.update_scene(g.d, camera=g.u.custom_camera)
        r.render()
    out["mujoco_fps"] = round(n / (time.time() - t0), 1)
    best = max(v for k, v in out.items() if k.endswith(tuple(f"B{b}" for b in batches))
               and not isinstance(v, str))
    out["best_fps"] = best
    out["speedup_vs_mujoco"] = round(best / out["mujoco_fps"], 1)
    g.env.close()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = []
    for loco, size in [("point", "large"), ("point", "giant"), ("ant", "large"),
                       ("humanoid", "large")]:
        try:
            r = bench(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            with open(os.path.join(OUT, "bench_gpu.log"), "a") as f:
                f.write(f"FAIL {loco}-{size}\n{traceback.format_exc()}\n")
            continue
        res.append(r)
        print(json.dumps(r), flush=True)
        with open(os.path.join(OUT, "bench_gpu2.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
