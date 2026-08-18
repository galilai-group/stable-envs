"""Probe 3: what is the parity CEILING, measured with the reference renderer on both sides?

Two terms decide whether any analytic renderer can be interior-exact against MiniWorld, and both
can be isolated without writing a renderer at all — by rendering the OFFICIAL scene twice with one
GL state changed. That makes the number a property of the reference rather than of my code.

  T1 TRILINEAR MIPMAPPING. Textures are uploaded with GL_LINEAR_MIPMAP_LINEAR and a driver-built
     mipmap chain (miniworld/opengl.py Texture.load). The per-pixel LOD is chosen from screen-space
     derivatives by a rule the GL spec leaves implementation-defined, and glGenerateMipmap's
     downsample filter is implementation-defined too. Re-render with GL_LINEAR (level 0, no
     mipmaps): every pixel where the two disagree is a pixel no analytic sampler can reproduce
     without reimplementing NVIDIA's chain.

  T2 PER-VERTEX (GOURAUD) LIGHTING. A whole wall is ONE quad (miniworld.Room._gen_static_data ->
     gen_seg_poly emits 4 vertices per wall segment, floor to ceiling), and GL_LIGHT0 is a
     POSITIONAL light at light_pos. Fixed-function GL evaluates the lighting equation at the 4
     corners and interpolates. A per-pixel evaluation of the same equation is a different function
     over the quad's interior. Computed numerically here on the real room geometry.

Also G4: throughput of the official renderer, and the render's share of one env step.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pyglet

pyglet.options["headless"] = True

import gymnasium as gym  # noqa: E402
import miniworld  # noqa: E402,F401
from pyglet.gl import (  # noqa: E402
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER,
    glBindTexture,
    glTexParameteri,
)

from miniworld.opengl import Texture  # noqa: E402

IDS = ["MiniWorld-OneRoom-v0", "MiniWorld-Hallway-v0", "MiniWorld-TMazeLeft-v0",
       "MiniWorld-FourRooms-v0", "MiniWorld-Sidewalk-v0", "MiniWorld-CollectHealth-v0"]


def interior_mask(frames):
    import scipy.ndimage as ndi
    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def set_min_filter(mode):
    """Re-set GL_TEXTURE_MIN_FILTER on every texture miniworld has cached."""
    n = 0
    for tex in Texture.tex_cache.values():
        glBindTexture(GL_TEXTURE_2D, tex.tex.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mode)
        n += 1
    glBindTexture(GL_TEXTURE_2D, 0)
    return n


def states_of(u, n, seed0=0):
    """n states from the env's own reset+random-action distribution."""
    out = []
    for i in range(n):
        u.reset(seed=seed0 + i)
        for _ in range(int(np.random.default_rng(seed0 + i).integers(0, 40))):
            u.step(int(np.random.default_rng(1000 + seed0 + i).integers(0, 3)))
        out.append((np.array(u.agent.pos), float(u.agent.dir),
                    [(np.array(e.pos), float(e.dir)) for e in u.entities if e is not u.agent]))
    return out


def restore(u, st):
    u.agent.pos, u.agent.dir = st[0].copy(), st[1]
    ents = [e for e in u.entities if e is not u.agent]
    for e, (p, d) in zip(ents, st[2]):
        e.pos, e.dir = p.copy(), d
    u._render_static()


def main():
    print("=" * 100)
    print("T1  TRILINEAR MIPMAP vs LEVEL-0 BILINEAR, official renderer on both sides")
    print("=" * 100)
    print(f"{'id':32s} {'all-exact':>10s} {'interior-exact':>15s} {'mean':>8s} {'max':>6s} "
          f"{'interior%':>10s}")
    for eid in IDS:
        env = gym.make(eid, render_mode="rgb_array")
        u = env.unwrapped
        sts = states_of(u, 8)
        A, B = [], []
        for st in sts:
            restore(u, st)
            set_min_filter(GL_LINEAR_MIPMAP_LINEAR)
            A.append(u.render())
            set_min_filter(GL_LINEAR)
            B.append(u.render())
        set_min_filter(GL_LINEAR_MIPMAP_LINEAR)
        A, B = np.stack(A), np.stack(B)
        d = np.abs(A.astype(np.int16) - B.astype(np.int16)).max(-1)
        m = interior_mask(A)
        di = d[:, :-1, :-1][m]
        print(f"{eid:32s} {(d == 0).mean():>10.4f} {(di == 0).mean():>15.4f} "
              f"{d.mean():>8.3f} {d.max():>6d} {100*m.mean():>9.2f}%")
        env.close()

    print()
    print("=" * 100)
    print("T2  PER-VERTEX (what GL does) vs PER-PIXEL (what an analytic shader does) DIFFUSE")
    print("=" * 100)
    print("fixed-function GL evaluates the lighting equation at each vertex and interpolates.")
    print("These are the levels/255 an analytic per-pixel shader would be off by, over each wall quad.")
    print(f"{'id':32s} {'quads':>6s} {'max|dP-dV|':>12s} {'levels/255':>12s} {'mean lv':>9s}")
    for eid in IDS:
        env = gym.make(eid, render_mode="rgb_array")
        u = env.unwrapped
        u.reset(seed=0)
        L = np.asarray(u.light_pos, np.float64)
        diffuse = float(np.asarray(u.light_color, np.float64)[0])
        worst, tot, nq = 0.0, [], 0
        for room in u.rooms:
            wv, wn = room.wall_verts, room.wall_norms
            for q in range(0, len(wv), 4):
                v = wv[q:q + 4]
                nrm = wn[q]
                if len(v) < 4:
                    continue
                nq += 1
                # GL's positional-light diffuse term at each vertex
                def dif(p):
                    ld = L - p
                    ld = ld / (np.linalg.norm(ld) + 1e-12)
                    return max(0.0, float(np.dot(nrm, ld))) * diffuse
                dv = np.array([dif(p) for p in v])
                # bilinear interpolation of the vertex values over the quad, vs the exact term
                gu, gw = np.meshgrid(np.linspace(0, 1, 17), np.linspace(0, 1, 17))
                P = (v[0][None, None] * ((1 - gu) * (1 - gw))[..., None]
                     + v[1][None, None] * ((1 - gu) * gw)[..., None]
                     + v[2][None, None] * (gu * gw)[..., None]
                     + v[3][None, None] * (gu * (1 - gw))[..., None])
                interp = (dv[0] * (1 - gu) * (1 - gw) + dv[1] * (1 - gu) * gw
                          + dv[2] * gu * gw + dv[3] * gu * (1 - gw))
                LD = L[None, None] - P
                LD = LD / (np.linalg.norm(LD, axis=-1, keepdims=True) + 1e-12)
                exact = np.clip((LD * nrm[None, None]).sum(-1), 0, None) * diffuse
                e = np.abs(exact - interp)
                worst = max(worst, float(e.max()))
                tot.append(float(e.mean()))
        print(f"{eid:32s} {nq:>6d} {worst:>12.4f} {255*worst:>12.1f} "
              f"{255*float(np.mean(tot)):>9.1f}")
        env.close()

    print()
    print("=" * 100)
    print("G4  official renderer throughput, and the render's share of one env step")
    print("=" * 100)
    print(f"{'id':32s} {'obs 80x60 f/s':>14s} {'vis 800x600 f/s':>16s} {'render % of step':>17s}")
    for eid in IDS:
        env = gym.make(eid, render_mode="rgb_array")
        u = env.unwrapped
        u.reset(seed=0)
        for _ in range(20):
            u.render_obs()
        t = time.perf_counter()
        for _ in range(300):
            u.render_obs()
        t_obs = (time.perf_counter() - t) / 300
        for _ in range(10):
            u.render()
        t = time.perf_counter()
        for _ in range(100):
            u.render()
        t_vis = (time.perf_counter() - t) / 100
        u.reset(seed=0)
        t = time.perf_counter()
        for _ in range(300):
            u.step(2)
        t_step = (time.perf_counter() - t) / 300
        print(f"{eid:32s} {1/t_obs:>14.1f} {1/t_vis:>16.1f} {100*t_obs/t_step:>16.1f}%")
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
