"""CPU feasibility, ONE bounded measurement (coordinator ruling 1): a throwaway torch CPU rasteriser of the
PFRocketLanding scene (~486 triangles) at the certified 480 px, one thread, reported as f/s against the
two frozen bars -- S_cpu >= 5 x the 34.25 f/s official (>= 171 f/s) and cores_cpu >= 1 (>= 1,191 f/s of
one-core JPEG decode). Whether a shared CPU triangle backend gets built is not decided here.

Design (a fair attempt, not a straw man): per frame, vertices are transformed in numpy, triangles are
rasterised in torch in two vectorised passes -- the two ground triangles over the full frame, every
other triangle over a padded bounding-box grid batched across triangles -- with a depth resolve by
``scatter_reduce(amin)``, perspective-correct barycentrics, TinyRenderer's shading law and nearest
texture lookup, truncation to uint8. It also scores itself against the reference corpus, which gives a
LOCAL preview of the visual-tier residual before the 3090 run (a different rasterizer again, so a
preview and not the number).

Usage: /Users/samib/se7-rocket-venv/bin/python scratch/step7/rocket-landing/cpu_raster_probe.py [--n 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from stable_envs.envs import rocket_landing as RL  # noqa: E402
from stable_envs.render import nvdiff as ND  # noqa: E402

REF = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "reference")
OUT = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "phase0", "cpu_raster_probe.json")
LAW = ND.TinyShading()
L = torch.as_tensor(LAW.unit_dir(), dtype=torch.float32)


class CpuRaster:
    def __init__(self, res: int = 480, small_pad: int = 96):
        self.res = res
        vert, norm, face, vg, uv = RL.build_scene_arrays()
        self.vert, self.norm, self.face, self.vg, self.uv = vert, norm, face, vg, uv
        self.fg = vg[face[:, 0]]
        self.tex = torch.as_tensor(RL.checker_texture().astype(np.float32) / 255.0)
        self.small_pad = small_pad

    def frame(self, state13: np.ndarray, style_env: np.ndarray) -> np.ndarray:
        R = self.res
        xp, xm = RL.geom_poses(state13[None])
        _, _, pos_prev, q_prev, _, _ = RL.lagged_pose(state13[None])
        eye, cmat = RL.camera_pose(pos_prev, q_prev)
        view, proj = ND.Camera(eye, cmat, RL.CAM_FOV, RL.CAM_NEAR, RL.CAM_FAR).matrices()
        # world-space vertices and normals
        wv = np.einsum("vij,vj->vi", xm[0][self.vg], self.vert) + xp[0][self.vg]
        wn = np.einsum("vij,vj->vi", xm[0][self.vg], self.norm)
        clip = (proj @ view[0] @ np.concatenate([wv, np.ones((len(wv), 1))], 1).T).T
        # NEAR-PLANE CLIPPING in clip space (z_c >= -w_c), Sutherland-Hodgman per triangle: a vertex behind
        # the camera projects to garbage and a triangle straddling the plane fills the frame. Attributes
        # are linear in clip space, so they interpolate with the same weights.
        attrs = np.concatenate([clip, wn, self.uv], 1)                          # (V, 4+3+2)
        fg_all = self.fg
        new_tris, new_fg = [], []
        for k, tri in enumerate(self.face):
            poly = [attrs[i] for i in tri]
            d = [pv[2] + pv[3] for pv in poly]
            if all(x >= 0 for x in d):
                new_tris.append(poly)
                new_fg.append(fg_all[k])
                continue
            if all(x < 0 for x in d):
                continue
            outp = []
            for j in range(3):
                a, b = poly[j], poly[(j + 1) % 3]
                da, db = d[j], d[(j + 1) % 3]
                if da >= 0:
                    outp.append(a)
                if (da >= 0) != (db >= 0):
                    t_ = da / (da - db)
                    outp.append(a + t_ * (b - a))
            for j in range(1, len(outp) - 1):
                new_tris.append([outp[0], outp[j], outp[j + 1]])
                new_fg.append(fg_all[k])
        A = np.asarray(new_tris, np.float64).reshape(-1, 9)                     # (3T, 9) corners
        fg = np.asarray(new_fg, np.int64)
        clip, wn, uv = A[:, :4], A[:, 4:7], A[:, 7:9]
        w = clip[:, 3]
        # TinyRenderer samples integer pixel corners bottom-up: shift by +0.5 px like the nvdiff path does
        sx = (clip[:, 0] / w + 1) * R / 2 + 0.5
        sy = (clip[:, 1] / w + 1) * R / 2 + 0.5
        sz = clip[:, 2] / w
        mat = RL.material_table(style_env[None])[0]                            # (13,3)
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32))               # noqa: E731
        SX, SY, SZ, IW = t(sx), t(sy), t(sz), t(1.0 / w)
        WN, UV = t(wn), t(uv)
        F = torch.arange(len(fg) * 3).view(-1, 3)
        self._fg_now = fg
        zbuf = torch.full((R * R,), float("inf"))
        best = torch.full((R * R,), -1, dtype=torch.long)                     # sample index of the min
        samples = []                                                          # (pixel, z, tri, b0, b1, b2)

        def rasterise(tris, pad):
            """Edge functions on a padded (T, pad, pad) grid anchored at each triangle's bbox corner."""
            i0, i1, i2 = F[tris, 0], F[tris, 1], F[tris, 2]
            x0, y0, x1, y1, x2, y2 = SX[i0], SY[i0], SX[i1], SY[i1], SX[i2], SY[i2]
            xmin = torch.clamp(torch.floor(torch.minimum(torch.minimum(x0, x1), x2)), 0, R - 1)
            ymin = torch.clamp(torch.floor(torch.minimum(torch.minimum(y0, y1), y2)), 0, R - 1)
            gx = xmin[:, None, None] + torch.arange(pad)[None, None, :] + 0.5
            gy = ymin[:, None, None] + torch.arange(pad)[None, :, None] + 0.5
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            e0 = (x1[:, None, None] - gx) * (y2[:, None, None] - gy) - (x2[:, None, None] - gx) * (y1[:, None, None] - gy)
            e1 = (x2[:, None, None] - gx) * (y0[:, None, None] - gy) - (x0[:, None, None] - gx) * (y2[:, None, None] - gy)
            e2 = area[:, None, None] - e0 - e1
            inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
            inside &= (gx < R) & (gy < R) & (area[:, None, None] != 0)
            b0, b1, b2 = e0 / area[:, None, None], e1 / area[:, None, None], e2 / area[:, None, None]
            z = b0 * SZ[i0][:, None, None] + b1 * SZ[i1][:, None, None] + b2 * SZ[i2][:, None, None]
            inside &= z <= 1.0                                                  # the FAR plane (255 m)
            idx = torch.nonzero(inside)
            tsel, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]
            px = (gy[tsel, ys, 0] - 0.5).long() * R + (gx[tsel, 0, xs] - 0.5).long()
            return (px, z[tsel, ys, xs], tris[tsel], b0[tsel, ys, xs], b1[tsel, ys, xs], b2[tsel, ys, xs])

        # chunk triangles by their on-screen extent (sorted), so the padded grid is the chunk's max extent
        i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
        xmin_ = torch.clamp(torch.minimum(torch.minimum(SX[i0], SX[i1]), SX[i2]), 0, R)
        xmax_ = torch.clamp(torch.maximum(torch.maximum(SX[i0], SX[i1]), SX[i2]), 0, R)
        ymin_ = torch.clamp(torch.minimum(torch.minimum(SY[i0], SY[i1]), SY[i2]), 0, R)
        ymax_ = torch.clamp(torch.maximum(torch.maximum(SY[i0], SY[i1]), SY[i2]), 0, R)
        ext = torch.maximum(xmax_ - xmin_, ymax_ - ymin_)
        order = torch.argsort(ext)
        visible = order[ext[order] > 0]
        pos_ = 0
        while pos_ < len(visible):
            pad = int(torch.ceil(ext[visible[pos_]]).item()) + 2
            # grow the chunk while the padded work stays bounded (~2.5 M samples per chunk)
            n_take = max(1, min(len(visible) - pos_, int(2_500_000 // (pad * pad))))
            sel = visible[pos_:pos_ + n_take]
            pad = int(torch.ceil(ext[sel].max()).item()) + 2
            samples.append(rasterise(sel, min(pad, R + 1)))
            pos_ += n_take
        px = torch.cat([s[0] for s in samples])
        z = torch.cat([s[1] for s in samples])
        tri = torch.cat([s[2] for s in samples])
        b = torch.stack([torch.cat([s[k] for s in samples]) for k in (3, 4, 5)], -1)
        zbuf.scatter_reduce_(0, px, z, reduce="amin")
        win = z == zbuf[px]
        px, tri, b, z = px[win], tri[win], b[win], z[win]
        best[px] = torch.arange(len(px))                                       # last writer wins ties
        keep = best[px] == torch.arange(len(px))
        px, tri, b = px[keep], tri[keep], b[keep]
        # perspective-correct attributes
        vi = F[tri]                                                            # (S,3)
        iw = IW[vi] * b                                                        # (S,3)
        bc = iw / iw.sum(-1, keepdim=True)
        n = (WN[vi] * bc[..., None]).sum(1)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        base = torch.as_tensor(mat[self._fg_now[tri.numpy()]], dtype=torch.float32)
        ndl = (n * L).sum(-1, keepdim=True)
        r = 2 * ndl * n - L
        col = base * (LAW.ambient + LAW.diffuse * ndl.clamp_min(0) + LAW.specular * r[:, 2:3].clamp_min(0) ** 2)
        g0 = torch.as_tensor(self._fg_now[tri.numpy()] == 0)
        if bool(g0.any()):
            uv = (UV[vi[g0]] * bc[g0][..., None]).sum(1)
            tu = (torch.floor(uv[:, 0] * 512).long() % 512)
            tv = (torch.floor(uv[:, 1] * 512).long() % 512)
            col[g0] = col[g0] * self.tex[tv, tu]
        img = torch.ones((R * R, 3))
        img[px] = col.clamp(0, 1)
        out = torch.floor(img * 255.0).clamp(0, 255).to(torch.uint8).view(R, R, 3)
        return out.flip(0).numpy()                                             # row 0 at the top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    S = np.load(os.path.join(REF, "states.npz"))
    F = np.load(os.path.join(REF, "frames.npz"))["frames"]
    st = S["obs30"][:, :13].astype(np.float64)
    sty = RL.corpus_style_to_env(S["style"])
    idx = np.linspace(0, len(st) - 1, a.n).astype(int)
    r = CpuRaster(480)
    r.frame(st[0], sty[0])                                                    # warm
    t0 = time.perf_counter()
    frames = [r.frame(st[i], sty[i]) for i in idx]
    dt = (time.perf_counter() - t0) / len(idx)
    fps = 1.0 / dt
    ours = np.stack(frames).astype(np.int16)
    ref = F[idx].astype(np.int16)
    d = np.abs(ours - ref)
    per = d.mean(axis=(1, 2, 3))
    dm = d.max(-1)
    hist = {"le2": float(((dm > 0) & (dm <= 2)).sum() / max(1, (dm > 0).sum())),
            "3-8": float(((dm > 2) & (dm <= 8)).sum() / max(1, (dm > 0).sum())),
            "gt8": float((dm > 8).sum() / max(1, (dm > 0).sum()))}
    within8 = (d.max(-1) <= 8).mean()
    exact = (d.max(-1) == 0).mean()
    from stable_envs.testing.conformance import ssim as _ssim
    try:
        ss = float(_ssim(ours.astype(np.uint8), ref.astype(np.uint8)))
    except Exception as e:                                                    # pragma: no cover
        ss = f"unavailable: {e}"
    out = {"resolution": 480, "threads": a.threads, "n_frames": int(len(idx)), "ms_per_frame": round(dt * 1e3, 2),
           "fps": round(fps, 2),
           "bars": {"official_fps": 34.25, "S_cpu_ge_5_needs_fps": 171.25, "one_core_jpeg_decode_fps": 1190.9,
                    "S_cpu": round(fps / 34.25, 3), "cores_cpu": round(fps / 1190.9, 4),
                    "meets_S_cpu_5": bool(fps >= 171.25), "meets_cores_1": bool(fps >= 1190.9)},
           "preview_vs_reference_corpus": {"mean_abs": round(float(d.mean()), 3), "within8_frac": round(float(within8), 4),
                                           "exact_frac": round(float(exact), 4), "ssim": ss,
                                           "worst_frame_mean_abs": round(float(per.max()), 3),
                                           "differing_px_by_magnitude": hist,
                                           "median_frame_mean_abs": round(float(np.median(per)), 3)},
           "note": "throwaway probe; the fidelity numbers are a LOCAL preview by a third rasterizer, not the tier's"}
    print(json.dumps(out, indent=1))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    from PIL import Image
    k = int(np.argmax(per))
    Image.fromarray(frames[k]).save(OUT.replace(".json", "_worst_ours.png"))
    Image.fromarray(F[idx[k]]).save(OUT.replace(".json", "_worst_ref.png"))
    Image.fromarray(frames[0]).save(OUT.replace(".json", "_first_ours.png"))


if __name__ == "__main__":
    main()
