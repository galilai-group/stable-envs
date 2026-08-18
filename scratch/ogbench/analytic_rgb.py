"""A complete analytic renderer for the OGBench maze scenes, diffed pixel-by-pixel against MuJoCo.

Everything below is read out of the compiled model (``probe_scene.py`` dumps), not guessed:

  camera   free camera, elevation -90 / azimuth 90 -> pos = (lookat_x, lookat_y, distance),
           forward (0,0,-1), up (0,1,0), perspective, fovy 45 deg. Validated to a 100% match on
           MuJoCo's own segmentation buffer in ``raycast.py``.
  lights   exactly two, both DIRECTIONAL along (0,0,-1): the model's 'global' light (ambient 0 after
           OGBench's pixels-mode XML surgery, diffuse 1, specular .1) and MuJoCo's headlight, which for
           a free camera points along the view axis (ambient .1, diffuse .4, specular .5).
  material every material in these scenes has specular .5 and shininess .5, so the specular exponent is
           128*.5 = 64 and the specular weight is (.5+.1)*.5 = .3.
  compose  fixed-function OpenGL adds the specular into the primary colour *before* the texture is
           modulated in, so the shading is a pure MULTIPLIER on the base colour, not an additive white
           highlight. That is what the data says: the measured ``obs/base`` ratio is identical across
           R, G and B on the floor to 4 decimals (shade2.py), which an additive white term cannot do.

  So:  rgb = round(255 * clip(base * (0.1 + 1.4*max(0,n.l) + 0.3*max(0,n.h)**64), 0, 1))

Two references are scored: MuJoCo with ``offsamples=0`` (no MSAA -- the bound a per-pixel rasterizer can
reach) and MuJoCo at its default ``offsamples=4``, which is what OGBench datasets actually contain.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_scene import make  # noqa: E402
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF

AMB, DIFF, SPEC, SHINE = 0.1, 1.4, 0.3, 64.0
LDIR = np.array([0.0, 0.0, 1.0])          # unit vector TOWARDS the light


def base_rgb(m, g):
    """Base colour MuJoCo modulates with the shading scalar."""
    mid = int(m.geom_matid[g])
    c = m.geom_rgba[g].astype(np.float64)[:3]
    if mid >= 0:
        if np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
            c = m.mat_rgba[mid].astype(np.float64)[:3]
        tid = m.mat_texid[mid]
        tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
        if tid >= 0:
            attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
            buf = getattr(m, attr)
            a0 = int(m.tex_adr[tid])
            h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
            reg = buf[a0:a0 + 3 * h * w].reshape(h, w, 3).astype(np.float64) / 255.0
            uniq = np.unique(reg.reshape(-1, 3), axis=0)
            c = c * (uniq[0] if len(uniq) == 1 else reg.reshape(-1, 3).mean(0))
    return c


def render_analytic(m, d, cam, H, W, ss=1):
    """(H,W,3) uint8. ``ss>1`` box-averages an ss-times supersampled render (an MSAA stand-in)."""
    R = H * ss
    gid, t, fwd, o, dr = RC.cast(m, d, cam, R, R)
    gid = np.where(t >= INF / 2, -1, gid)
    p = o + t[..., None] * dr
    n = normals(m, d, gid, p)
    V = -dr / np.linalg.norm(dr, axis=-1, keepdims=True)
    h = LDIR + V
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    ndl = np.clip(n @ LDIR, 0, None)
    ndh = np.clip((n * h).sum(-1), 0, None)
    S = AMB + DIFF * ndl + SPEC * ndh ** SHINE
    base = np.zeros((R, R, 3))
    for g in np.unique(gid):
        if g < 0:
            continue
        base[gid == g] = base_rgb(m, int(g))
    col = np.clip(base * S[..., None], 0, 1) * 255.0
    col[gid < 0] = 0.0                                   # skybox seen from directly above is black
    if ss > 1:
        col = col.reshape(H, ss, W, ss, 3).mean((1, 3))
    return np.clip(np.round(col), 0, 255).astype(np.uint8), gid, S


def stats(a, b):
    dd = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return dict(exact=float((dd == 0).mean()), le1=float((dd <= 1).mean()),
                le2=float((dd <= 2).mean()), le8=float((dd <= 8).mean()),
                mean=float(dd.mean()), max=int(dd.max()))


def run(loco, size, img=224, nstates=8, seed=0):
    import mujoco
    import imageio.v2 as imageio
    env, u = make(loco, size, img)
    m, d = u.model, u.data
    env.reset(options={"task_id": 1})
    for _ in range(20):
        env.step(u.action_space.sample())

    m.vis.quality.offsamples = 0
    r0 = mujoco.Renderer(m, width=img, height=img)
    m.vis.quality.offsamples = 4
    r4 = mujoco.Renderer(m, width=img, height=img)
    r0.update_scene(d, camera=u.custom_camera)
    cam = RC.avgcam(RC.glcam(r0.scene, 0), RC.glcam(r0.scene, 1))

    res = dict(env=f"{loco}maze-{size}", img=img, camera=cam, nstates=nstates)
    acc = {"vs_noaa_ss1": [], "vs_noaa_ss4": [], "vs_msaa_ss1": [], "vs_msaa_ss4": [],
           "interior_vs_noaa_ss1": []}
    for s in range(nstates):
        for _ in range(15):
            env.step(u.action_space.sample())
        r0.update_scene(d, camera=u.custom_camera)
        ref0 = r0.render().copy()
        r4.update_scene(d, camera=u.custom_camera)
        ref4 = r4.render().copy()
        mine1, gid, _S = render_analytic(m, d, cam, img, img, ss=1)
        mine4, _, _ = render_analytic(m, d, cam, img, img, ss=4)
        acc["vs_noaa_ss1"].append(stats(mine1, ref0))
        acc["vs_noaa_ss4"].append(stats(mine4, ref0))
        acc["vs_msaa_ss1"].append(stats(mine1, ref4))
        acc["vs_msaa_ss4"].append(stats(mine4, ref4))
        # interior only (pixels whose 4-neighbourhood shares one geom id)
        b = np.zeros(gid.shape, bool)
        b[1:, :] |= gid[1:, :] != gid[:-1, :]
        b[:-1, :] |= gid[1:, :] != gid[:-1, :]
        b[:, 1:] |= gid[:, 1:] != gid[:, :-1]
        b[:, :-1] |= gid[:, 1:] != gid[:, :-1]
        dd = np.abs(mine1.astype(np.int16) - ref0.astype(np.int16)).max(-1)
        acc["interior_vs_noaa_ss1"].append(dict(frac_boundary=float(b.mean()),
                                                exact=float((dd[~b] == 0).mean()),
                                                le1=float((dd[~b] <= 1).mean()),
                                                mean=float(dd[~b].mean()),
                                                max=int(dd[~b].max())))
        if s == 0:
            sheet = np.concatenate([ref4, mine4, np.clip(np.abs(mine4.astype(np.int16) -
                                                                ref4.astype(np.int16)) * 8, 0, 255)
                                    .astype(np.uint8)], 1)
            imageio.imwrite(os.path.join(OUT, f"analytic_{loco}_{size}.png"), sheet)
    for k, v in acc.items():
        res[k] = {kk: round(float(np.mean([x[kk] for x in v])), 6) for kk in v[0]}
    env.close()
    return res


def bench(loco, size, img=224, B=512, device="cuda"):
    """Throughput of the batched form: static background baked, agent primitives re-cast per frame."""
    import torch
    import mujoco
    env, u = make(loco, size, img)
    m, d = u.model, u.data
    env.reset(options={"task_id": 1})
    m.vis.quality.offsamples = 0
    r0 = mujoco.Renderer(m, width=img, height=img)
    r0.update_scene(d, camera=u.custom_camera)
    cam = RC.avgcam(RC.glcam(r0.scene, 0), RC.glcam(r0.scene, 1))

    names = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}
    agent = [g for g in range(m.ngeom)
             if not (names[g] or "").startswith(("block_", "teleport")) and names[g] != "floor"]

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    o, dr, fwd = RC.rays(cam, img, img)
    o = torch.tensor(o, device=dev, dtype=torch.float32).view(img, img, 3)
    dr = torch.tensor(dr, device=dev, dtype=torch.float32).view(img, img, 3)

    # static layer (floor + walls), baked once
    mine, gid, S = render_analytic(m, d, cam, img, img, ss=1)
    static_rgb = torch.tensor(mine, device=dev, dtype=torch.uint8)
    static_t = torch.tensor(np.where(np.isin(gid, agent), 1e30,
                                     RC.cast(m, d, cam, img, img)[1]), device=dev, dtype=torch.float32)

    # per-frame: intersect the agent primitives only
    prim = []
    for g in agent:
        prim.append((int(m.geom_type[g]), m.geom_size[g].astype(np.float32),
                     base_rgb(m, g).astype(np.float32)))
    P = len(prim)

    def frame_batch(xpos, xmat):
        """xpos (B,P,3), xmat (B,P,3,3) -> (B,H,W,3) uint8."""
        Bn = xpos.shape[0]
        best_t = static_t.expand(Bn, img, img).clone()
        col = static_rgb.expand(Bn, img, img, 3).clone().float()
        O = o.view(1, img, img, 3)
        D = dr.view(1, img, img, 3)
        for k, (gt, sz, c) in enumerate(prim):
            pos = xpos[:, k].view(Bn, 1, 1, 3)
            oc = O - pos
            if gt == 2:                                     # sphere
                r = float(sz[0])
                b = (oc * D).sum(-1)
                cc = (oc * oc).sum(-1) - r * r
                disc = b * b - cc
                sq = torch.sqrt(torch.clamp(disc, min=0))
                t = -b - sq
                hit = (disc > 0) & (t > 0)
                nrm = (O + t.unsqueeze(-1) * D - pos) / r
            else:                                           # capsule -> cylinder body only (bench)
                R = xmat[:, k].view(Bn, 1, 1, 3, 3)
                ol = torch.einsum("bhwij,bhwj->bhwi", R.expand(Bn, img, img, 3, 3), oc)
                dl = torch.einsum("bhwij,bhwj->bhwi", R.expand(Bn, img, img, 3, 3),
                                  D.expand(Bn, img, img, 3))
                r, hh = float(sz[0]), float(sz[1])
                a = dl[..., 0] ** 2 + dl[..., 1] ** 2
                b = ol[..., 0] * dl[..., 0] + ol[..., 1] * dl[..., 1]
                cc = ol[..., 0] ** 2 + ol[..., 1] ** 2 - r * r
                disc = b * b - a * cc
                sq = torch.sqrt(torch.clamp(disc, min=0))
                t = (-b - sq) / torch.clamp(a, min=1e-9)
                z = ol[..., 2] + t * dl[..., 2]
                hit = (disc > 0) & (t > 0) & (z.abs() <= hh)
                nl = torch.stack([ol[..., 0] + t * dl[..., 0], ol[..., 1] + t * dl[..., 1],
                                  torch.zeros_like(t)], -1) / r
                nrm = torch.einsum("bhwji,bhwj->bhwi", R.expand(Bn, img, img, 3, 3), nl)
            upd = hit & (t < best_t)
            V = -D / D.norm(dim=-1, keepdim=True)
            L = torch.tensor([0.0, 0.0, 1.0], device=dev)
            hv = L + V
            hv = hv / hv.norm(dim=-1, keepdim=True)
            ndl = torch.clamp((nrm * L).sum(-1), min=0)
            ndh = torch.clamp((nrm * hv).sum(-1), min=0)
            Sv = AMB + DIFF * ndl + SPEC * ndh ** SHINE
            cv = torch.tensor(c, device=dev).view(1, 1, 1, 3) * Sv.unsqueeze(-1)
            cv = torch.clamp(cv, 0, 1) * 255.0
            col = torch.where(upd.unsqueeze(-1), cv, col)
            best_t = torch.where(upd, t, best_t)
        return col.round().clamp(0, 255).to(torch.uint8)

    xpos = torch.tensor(np.stack([d.geom_xpos[agent]] * B), device=dev, dtype=torch.float32)
    xmat = torch.tensor(np.stack([d.geom_xmat[agent].reshape(-1, 3, 3)] * B),
                        device=dev, dtype=torch.float32)
    for _ in range(3):
        frame_batch(xpos, xmat)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    reps = 20
    for _ in range(reps):
        frame_batch(xpos, xmat)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    env.close()
    return dict(env=f"{loco}maze-{size}", batch=B, n_agent_prims=P, img=img,
                fps=float(B * reps / dt), ms_per_batch=float(1000 * dt / reps), device=str(dev))


def mujoco_bench(loco, size, img=224, n=200):
    import mujoco
    env, u = make(loco, size, img)
    m, d = u.model, u.data
    env.reset(options={"task_id": 1})
    r = mujoco.Renderer(m, width=img, height=img)
    for _ in range(10):
        r.update_scene(d, camera=u.custom_camera)
        r.render()
    t0 = time.time()
    for _ in range(n):
        r.update_scene(d, camera=u.custom_camera)
        r.render()
    dt = time.time() - t0
    env.close()
    return dict(env=f"{loco}maze-{size}", fps=float(n / dt), ms=float(1000 * dt / n))


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = [("point", "medium"), ("point", "large"), ("point", "giant"), ("point", "teleport"),
             ("ant", "large"), ("humanoid", "large")]
    allr = []
    for loco, size in cells:
        try:
            r = run(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            continue
        allr.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "camera"}, indent=1), flush=True)
    with open(os.path.join(OUT, "analytic_rgb.json"), "w") as f:
        json.dump(allr, f, indent=1)

    print("\n==== throughput ====", flush=True)
    bres = []
    for loco, size in [("point", "large"), ("ant", "large")]:
        try:
            b = bench(loco, size)
            mb = mujoco_bench(loco, size)
            b["mujoco_fps"] = mb["fps"]
            b["speedup"] = b["fps"] / mb["fps"]
            bres.append(b)
            print(json.dumps(b), flush=True)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
    with open(os.path.join(OUT, "bench.json"), "w") as f:
        json.dump(bres, f, indent=1)


if __name__ == "__main__":
    main()
