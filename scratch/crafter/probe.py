"""Crafter gate probe: G1 purity, G2 compression, G3 decoder stability, and an atlas parity check.

Run with the se-tiles venv (crafter installed there):
    /mnt/vast/home/randall.balestriero/venvs/se-tiles/bin/python scratch/crafter/probe.py
"""
from __future__ import annotations

import sys

import numpy as np

import crafter
from crafter import constants, engine, objects

MATS = [None] + list(constants.materials)                       # material id -> name
OBJTEX = [None, "player-left", "player-right", "player-up", "player-down", "player-sleep",
          "cow", "zombie", "skeleton", "arrow-left", "arrow-right", "arrow-up", "arrow-down",
          "plant", "plant-ripe", "fence"]
N_OBJ = len(OBJTEX)
N_MAT = len(MATS)
OUTSIDE = N_MAT * N_OBJ
N_CODE = OUTSIDE + 1
ITEMS = list(constants.items)
VIEW = (9, 9)
ITEM_ROWS = int(np.ceil(len(ITEMS) / VIEW[0]))
GW, GH = VIEW[0], VIEW[1] - ITEM_ROWS                            # 9 x 7 world tiles
OFF = (GW // 2, GH // 2)


def read_state(env):
    """(codes 9x7, inventory 16, daylight) straight off the live env."""
    w = env._world
    p = env._player
    cx, cy = int(p.pos[0]), int(p.pos[1])
    codes = np.full((GW, GH), OUTSIDE, np.int64)
    for x in range(GW):
        for y in range(GH):
            px, py = cx + x - OFF[0], cy + y - OFF[1]
            if not (0 <= px < w.area[0] and 0 <= py < w.area[1]):
                continue
            mat, obj = w[(px, py)]
            m = MATS.index(mat)
            o = 0 if obj is None else OBJTEX.index(obj.texture)
            codes[x, y] = m * N_OBJ + o
    inv = np.array([p.inventory[k] for k in ITEMS], np.int64)
    return codes, inv, float(w.daylight)


# ----------------------------------------------------------------- atlas, built with crafter's own code

def build_atlas(textures, unit):
    """(N_CODE, unit, unit, 3) uint8 world tiles, composited by crafter's own _draw/_draw_alpha."""
    u = np.array([unit, unit])
    out = np.zeros((N_CODE, unit, unit, 3), np.uint8)
    for m in range(N_MAT):
        for o in range(N_OBJ):
            c = np.zeros((unit, unit, 3), np.uint8) + 127
            engine._draw(c, np.array([0, 0]), textures.get(MATS[m], u))
            if o:
                engine._draw_alpha(c, np.array([0, 0]), textures.get(OBJTEX[o], u))
            out[m * N_OBJ + o] = c
    out[OUTSIDE] = 127
    return out


def build_item_atlas(textures, unit):
    """(16, 10, unit, unit, 3) uint8 inventory tiles, built with crafter's own ItemView."""
    view = engine.ItemView(textures, [1, 1])
    out = np.zeros((len(ITEMS), 10, unit, unit, 3), np.uint8)
    for i, name in enumerate(ITEMS):
        for a in range(10):
            out[i, a] = view({name: a}, np.array([unit, unit]))
    return out


# ----------------------------------------------------------------- our renderer, from the state alone

def our_render(codes, inv, daylight, atlas, iatlas, size=64):
    unit = size // VIEW[0]
    world = atlas[codes]                                        # (GW,GH,u,u,3)
    world = world.transpose(0, 2, 1, 3, 4).reshape(GW * unit, GH * unit, 3)
    items = iatlas[np.arange(len(ITEMS)), np.clip(inv, 0, 9)]    # (16,u,u,3)
    strip = np.zeros((VIEW[0] * unit, ITEM_ROWS * unit, 3), np.uint8)
    for i in range(len(ITEMS)):
        x, y = i % VIEW[0], i // VIEW[0]
        strip[x * unit:(x + 1) * unit, y * unit:(y + 1) * unit] = items[i]
    lit = light(world, daylight)
    if codes[OFF[0], OFF[1]] % N_OBJ == OBJTEX.index("player-sleep"):
        lit = sleep(lit)
    view = np.concatenate([lit, strip], 1)
    canvas = np.zeros((size, size, 3), np.uint8)
    b = (size - unit * VIEW[0]) // 2
    canvas[b:b + view.shape[0], b:b + view.shape[1]] = view
    return canvas.transpose(1, 0, 2)


def light(canvas, daylight, sat=0.4, tint=(0, 16, 64), amount=0.5):
    """crafter.engine.LocalView._light for daylight >= 0.5 (no noise), in explicit arithmetic."""
    c = canvas.astype(np.int64)
    L = (c[..., 0] * 19595 + c[..., 1] * 38470 + c[..., 2] * 7471 + 0x8000) >> 16
    deg = L[..., None].astype(np.float64)
    night = np.trunc(deg + sat * (c.astype(np.float64) - deg))              # PIL Image.blend
    night = (1 - amount) * night + amount * np.asarray(tint, np.float64)
    return daylight * canvas.astype(np.float64) + (1 - daylight) * night


def sleep(canvas, tint=(0, 0, 16), amount=0.5):
    """crafter.engine.LocalView._sleep: fully desaturate the (truncated) canvas, then tint."""
    c = np.trunc(canvas).astype(np.int64)
    L = (c[..., 0] * 19595 + c[..., 1] * 38470 + c[..., 2] * 7471 + 0x8000) >> 16
    deg = np.repeat(L[..., None], 3, -1).astype(np.float64)
    return (1 - amount) * deg + amount * np.asarray(tint, np.float64)


def main():
    env = crafter.Env(seed=0)
    unit = 64 // VIEW[0]
    atlas = build_atlas(env._textures, unit)
    iatlas = build_item_atlas(env._textures, unit)
    print(f"atlas {atlas.shape} {atlas.nbytes} B   item atlas {iatlas.shape} {iatlas.nbytes} B")

    # G3: the decoder is a constant of the family -> two envs, two resets, identical atlases
    env2 = crafter.Env(seed=7)
    env2.reset()
    a2 = build_atlas(env2._textures, unit)
    print("G3 atlas identical across two envs/resets:", np.array_equal(atlas, a2))

    rng = np.random.RandomState(0)
    env.reset()
    states, frames, daylights = [], [], []
    n_out, n_sleep = 0, 0
    # 600 uniform-random steps, then a scripted march into the north-west corner so the render
    # window actually leaves the world and the OUTSIDE code is exercised on real frames.
    plan = [rng.randint(0, 17) for _ in range(600)] + [1] * 45 + [3] * 45
    for a in plan:
        codes, inv, dl = read_state(env)
        f = env.render()
        if dl >= 0.5:
            states.append((codes, inv, dl))
            frames.append(f)
            n_out += int((codes == OUTSIDE).any())
            n_sleep += int(codes[OFF[0], OFF[1]] % N_OBJ == OBJTEX.index("player-sleep"))
        daylights.append(dl)
        obs, r, done, info = env.step(a)
        if done:
            env.reset()
    print(f"collected {len(states)} daylight>=0.5 frames of {len(plan)}; "
          f"daylight range {min(daylights):.4f}-{max(daylights):.4f}; "
          f"windows with OUTSIDE: {n_out}; sleeping: {n_sleep}")

    # parity of the reimplementation against the official renderer
    worst, exact, first_bad = 0, 0, None
    for i, ((codes, inv, dl), f) in enumerate(zip(states, frames)):
        g = our_render(codes, inv, dl, atlas, iatlas)
        d = np.abs(g.astype(np.int16) - f.astype(np.int16))
        worst = max(worst, int(d.max()))
        exact += int(d.max() == 0)
        if d.max() and first_bad is None:
            first_bad = (i, codes, inv, dl, d)
    print(f"G1/atlas parity: {exact}/{len(states)} frames bit-identical, max abs diff {worst}")
    if first_bad is not None:
        i, codes, inv, dl, d = first_bad
        ys, xs = np.nonzero(d.max(-1))
        print(f"  first mismatch at frame {i}, daylight {dl:.6f}, "
              f"rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()}, "
              f"tiles {sorted({(int(y)//unit, int(x)//unit) for y, x in zip(ys, xs)})}")
        print(f"  centre code {codes[OFF[0], OFF[1]]} "
              f"(mat {MATS[codes[OFF[0], OFF[1]]//N_OBJ]}, obj {OBJTEX[codes[OFF[0], OFF[1]]%N_OBJ]})")
        for ty, tx in sorted({(int(y) // unit, int(x) // unit) for y, x in zip(ys, xs)})[:6]:
            if ty < GH:
                c = int(codes[tx, ty])
                print(f"    world tile (row {ty}, col {tx}): code {c} "
                      f"mat {MATS[c//N_OBJ]} obj {OBJTEX[c%N_OBJ]}")
            else:
                print(f"    item tile (row {ty}, col {tx}): "
                      f"{ITEMS[(ty-GH)*VIEW[0]+tx]}={inv[(ty-GH)*VIEW[0]+tx]}")

    # split by daylight == 1 vs < 1 so a light-path error cannot hide in the aggregate
    for lo, hi in ((0.999999, 1.01), (0.5, 0.999999)):
        sel = [(s, f) for s, f in zip(states, frames) if lo <= s[2] < hi]
        if not sel:
            continue
        w = max(int(np.abs(our_render(*s, atlas, iatlas).astype(np.int16) - f.astype(np.int16)).max())
                for s, f in sel)
        e = sum(int(np.abs(our_render(*s, atlas, iatlas).astype(np.int16)
                           - f.astype(np.int16)).max() == 0) for s, f in sel)
        print(f"  daylight in [{lo},{hi}): {e}/{len(sel)} exact, max {w}")

    # G2
    codes, inv, dl = states[0]
    sd = codes.size + inv.size + 1
    print(f"G2: state_dim {sd}, float32 {4*sd} B vs frame {frames[0].nbytes} B "
          f"= {frames[0].nbytes/(4*sd):.1f}x; packed uint8 {sd} B = {frames[0].nbytes/sd:.1f}x")
    full = (env._world._mat_map.nbytes + env._world._obj_map.nbytes)
    print(f"    crafter's own world arrays alone: {full} B = {full/frames[0].nbytes:.2f}x the frame")

    # interior mask fraction (does crafter art leave any flat 3x3 neighbourhood?)
    import scipy.ndimage as ndi
    fr = np.stack(frames[:32]).astype(np.int16)
    flat = []
    for f in fr:
        m = (np.abs(np.diff(f, axis=0)).sum(-1)[:, :-1]
             + np.abs(np.diff(f, axis=1)).sum(-1)[:-1]) == 0
        flat.append(ndi.minimum_filter(m, size=3))
    print(f"interior mask fraction: {np.stack(flat).mean():.4f}")


if __name__ == "__main__":
    sys.exit(main())
