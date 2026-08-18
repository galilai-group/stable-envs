"""Three-way diff: gymnasium vs the CPU reference vs the GPU port, on identical states.

The CPU reference already matches gymnasium exactly, so any GPU disagreement is a porting bug and
not a rule that was misread -- this narrows it to a primitive by drawing them one at a time.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym

import gpu_render as R
import ref_envs as E
import ref_gfx as G


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    sts = [(float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-np.pi, np.pi))) for _ in range(8)]
    gymf, reff = [], []
    for x, th in sts:
        env.state = np.array([x, 0.0, th, 0.0])
        gymf.append(env.render().copy())
        reff.append(E.render_cartpole(x, th, aa=True))
    gymf, reff = np.stack(gymf), np.stack(reff)
    gpuf = R.CartPoleGPU(device=dev, aa=True).render(np.asarray(sts)).cpu().numpy()
    print("gym vs ref :", int((np.abs(gymf.astype(int) - reff.astype(int)) > 0).sum()), "px")
    print("gym vs gpu :", int((np.abs(gymf.astype(int) - gpuf.astype(int)) > 0).sum()), "px")

    d = (np.abs(reff.astype(int) - gpuf.astype(int)).max(-1) > 0)
    print("ref vs gpu :", int(d.sum()), "px")
    for i in range(len(sts)):
        ys, xs = np.where(d[i])
        if not len(ys):
            continue
        print(f"\nframe {i} state={sts[i]}  {len(ys)} px")
        for j in range(min(10, len(ys))):
            y, x = int(ys[j]), int(xs[j])
            print(f"   flipped(y={y},x={x}) pre-flip y={399-y}   ref={reff[i,y,x].tolist()} "
                  f"gpu={gpuf[i,y,x].tolist()} gym={gymf[i,y,x].tolist()}")
        break

    # ---- primitive-by-primitive on ONE state -------------------------------------------------
    x, th = sts[0]
    print(f"\n{'='*70}\nper-primitive, state=({x:.4f},{th:.4f})\n{'='*70}")
    P = E.CARTPOLE
    cartx = x * P["scale"] + 300.0
    carty = 100.0
    cart = [(-25.0 + cartx, -15.0 + carty), (-25.0 + cartx, 15.0 + carty),
            (25.0 + cartx, 15.0 + carty), (25.0 + cartx, -15.0 + carty)]
    pole = []
    for c in [(-5.0, -5.0), (-5.0, 120.0), (5.0, 120.0), (5.0, -5.0)]:
        c = E._rot(c, -th)
        pole.append((c[0] + cartx, c[1] + carty + 7.5))

    steps = [("cart", cart, P["cart_c"], 52), ("pole", pole, P["pole_c"], 132)]
    cvr = np.full((400, 600, 3), 255, np.int64)
    for nm, verts, col, kmax in steps:
        # CPU
        vi = G.trunc_verts(verts)
        cpu_before = cvr.copy()
        G.draw_aapolygon(cvr, vi, col)
        aa_cpu = cvr.copy()
        cvr[G.filled_polygon_mask(vi, 400, 600)] = np.asarray(col, np.int64)
        # GPU
        cvg = R.Canvas(1, 400, 600, torch.device(dev))
        cvg.buf[0] = torch.as_tensor(cpu_before, dtype=torch.uint8, device=dev)
        vt = torch.as_tensor(np.asarray(verts), dtype=torch.float64, device=dev).view(1, -1, 2)
        vig = torch.trunc(vt).to(torch.int64)
        ct = torch.tensor(col, dtype=torch.int64, device=dev).view(1, 3)
        bidx = torch.arange(1, device=dev)
        for e in range(vig.shape[1]):
            R._aaline(cvg, vig[0:1, e, 0], vig[0:1, e, 1],
                      vig[0:1, (e + 1) % vig.shape[1], 0], vig[0:1, (e + 1) % vig.shape[1], 1],
                      ct, kmax, bidx)
        aa_gpu = cvg.buf[0].cpu().numpy().astype(np.int64)
        da = (np.abs(aa_cpu - aa_gpu).max(-1) > 0)
        mask_g = R.fill_polygon_mask(vig[..., 0], vig[..., 1], cvg.ys, cvg.xs, 600)[0].cpu().numpy()
        mask_c = G.filled_polygon_mask(vi, 400, 600)
        print(f"  {nm}: verts_int={vi.tolist()}")
        print(f"      aapolygon diff {int(da.sum())} px ; fill mask xor {int((mask_g ^ mask_c).sum())}")
        if da.sum():
            ys, xs = np.where(da)
            for j in range(min(8, len(ys))):
                y, xx = int(ys[j]), int(xs[j])
                print(f"        (y={y},x={xx}) cpu={aa_cpu[y,xx].tolist()} gpu={aa_gpu[y,xx].tolist()} "
                      f"before={cpu_before[y,xx].tolist()}")
    env.close()


if __name__ == "__main__":
    main()
