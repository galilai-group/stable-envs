"""Closed-form transcription of SDL_gfx ``_aalineColor(..., draw_endpoint=0)`` -- the only thing
``gfxdraw.aapolygon`` does -- plus the blend formula, both decided by pixel-diffing pygame.

The C routine is an incremental Wu line with a 32-bit fixed-point error accumulator, which looks
sequential and therefore GPU-hostile. It is not: with ``erradj = E << 16`` and ``erracc`` starting
at zero, the state after k steps is a closed form,

    E      = (minor << 16) // major
    wraps  = (k * E) >> 16          # how many times the minor axis has advanced
    weight = ((k * E) & 0xFFFF) >> 8

so every AA pixel of an edge can be emitted in parallel. Within one edge the major axis advances
every step, so no two steps touch the same pixel and a scatter is conflict-free.

The blend is fitted, not assumed: several plausible integer forms are scored against pygame.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
from pygame import gfxdraw

W = 96
SOLID = 256          # sentinel: pixelColor, not pixelColorWeight


def _surf(bg=(255, 255, 255)):
    s = pygame.Surface((W, W))
    s.fill(bg)
    return s


def _arr(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2)).astype(np.int64)


def aaline_ops(x1, y1, x2, y2):
    """(x, y, weight) triples in DRAW ORDER for _aalineColor(x1,y1,x2,y2,draw_endpoint=0).

    ``weight == SOLID`` marks a write that goes through ``pixelColor`` (exact colour); any other
    weight goes through ``pixelColorWeight``, whose alpha is ``(255*w) >> 8`` and therefore never
    reaches 255 -- a w=255 pixel is blended at alpha 254, not written solid.
    """
    xx0, yy0, xx1, yy1 = int(x1), int(y1), int(x2), int(y2)
    if yy0 > yy1:
        yy0, yy1 = yy1, yy0
        xx0, xx1 = xx1, xx0
    dx, dy = xx1 - xx0, yy1 - yy0
    if dx >= 0:
        xdir = 1
    else:
        xdir, dx = -1, -dx

    if dx == 0:                                        # vertical: solid vline, no AA
        if dy > 0:
            return [(int(x1), int(y1), SOLID)] + [(xx0, y, SOLID) for y in range(yy0, yy0 + dy + 1)]
        return [(int(x1), int(y1), SOLID)]
    if dy == 0:                                        # horizontal: solid hline, no AA
        if dx > 0:
            a, b = xx0, xx0 + xdir * dx
            lo, hi = min(a, b), max(a, b)
            return [(int(x1), int(y1), SOLID)] + [(x, yy0, SOLID) for x in range(lo, hi + 1)]
        return [(int(x1), int(y1), SOLID)]

    ops = [(int(x1), int(y1), SOLID)]                    # initial pixel, ORIGINAL endpoint, full colour
    if dy > dx:                                        # y-major
        E = (dx << 16) // dy
        for k in range(1, dy):
            acc = k * E
            w = acc >> 16
            wgt = (acc & 0xFFFF) >> 8
            ops.append((xx0 + xdir * w, yy0 + k, 255 - wgt))
            ops.append((xx0 + xdir * (w + 1), yy0 + k, wgt))
    else:                                              # x-major
        E = (dy << 16) // dx
        for k in range(1, dx):
            acc = k * E
            w = acc >> 16
            wgt = (acc & 0xFFFF) >> 8
            ops.append((xx0 + xdir * k, yy0 + w, 255 - wgt))
            ops.append((xx0 + xdir * k, yy0 + w + 1, wgt))
    return ops


CANDS = {
    "a=(255*w)>>8 ; out=(s*a+d*(255-a))//255": lambda s, d, w: (
        s * ((255 * w) >> 8) + d * (255 - ((255 * w) >> 8))) // 255,
    "a=(255*w)>>8 ; out=d+((s-d)*a)//255": lambda s, d, w: d + ((s - d) * ((255 * w) >> 8)) // 255,
    "a=w        ; out=(s*a+d*(255-a))//255": lambda s, d, w: (s * w + d * (255 - w)) // 255,
    "a=(255*w)>>8 ; out=d+((s-d)*a+128)//255": lambda s, d, w: d + ((s - d) * ((255 * w) >> 8) + 128) // 255,
    "a=(255*w)>>8 ; out=(s*a+d*(255-a)+255)>>8": lambda s, d, w: (
        (s * ((255 * w) >> 8) + d * (255 - ((255 * w) >> 8))) >> 8),
}


def main():
    pygame.init()
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------- fit the blend on single aalines
    print("=" * 74)
    print("blend formula fit: aapolygon-style aalines on a white and on a grey background")
    print("=" * 74)
    obs = []  # (src, dst, weight, out)
    for bg in ((255, 255, 255), (100, 160, 200), (0, 0, 0)):
        for _ in range(40):
            p = rng.integers(4, 92, 4)
            col = tuple(int(c) for c in rng.integers(0, 256, 3))
            s = _surf(bg)
            gfxdraw.aapolygon(s, [(int(p[0]), int(p[1])), (int(p[2]), int(p[3])),
                                  (int(p[0]), int(p[3]))], col)
            got = _arr(s)
            # replay the three edges in the same order aapolygonColor does
            canvas = np.tile(np.array(bg, np.int64), (W, W, 1))
            tri = [(p[0], p[1]), (p[2], p[3]), (p[0], p[3])]
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                for (x, y, wgt) in aaline_ops(a[0], a[1], b[0], b[1]):
                    if 0 <= x < W and 0 <= y < W:
                        obs.append((np.array(col, np.int64), canvas[y, x].copy(), wgt,
                                    got[y, x].copy(), (x, y)))
                        # provisional update with the leading candidate so later ops see it
                        aa = (255 * wgt) >> 8
                        canvas[y, x] = (np.array(col) * aa + canvas[y, x] * (255 - aa)) // 255 \
                            if wgt != 255 else np.array(col)
    src = np.stack([o[0] for o in obs])
    dst = np.stack([o[1] for o in obs])
    wg = np.array([o[2] for o in obs])[:, None]
    out = np.stack([o[3] for o in obs])
    print(f"  {len(obs)} sampled writes")
    for name, f in CANDS.items():
        pred = np.where(wg == 255, src, f(src, dst, wg))
        ok = (pred == out).all(-1)
        print(f"  {name:44s} exact {100*ok.mean():6.2f} %   max|err| {np.abs(pred-out).max()}")

    # ---------------------------------------------------------- full-primitive replay
    print("\n" + "=" * 74)
    print("full replay: aapolygon THEN filled_polygon, rotated quads, vs pygame")
    print("=" * 74)
    import probe_exact_rules as P

    def blend(sc, dc, w):
        if w >= 255:
            return sc.copy()
        a = (255 * w) >> 8
        return (sc * a + dc * (255 - a)) // 255

    nbad = 0
    tot_diff = 0
    tot_px = 0
    fringe_px = 0
    for _ in range(120):
        cx, cy = rng.uniform(25, 71, 2)
        th = rng.uniform(-np.pi, np.pi)
        hw, hh = rng.uniform(3, 25), rng.uniform(2, 12)
        loc = np.array([(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)])
        c, s_ = np.cos(th), np.sin(th)
        wv = loc @ np.array([[c, -s_], [s_, c]]).T + np.array([cx, cy])
        col = np.array([202, 152, 101], np.int64)
        s = _surf()
        gfxdraw.aapolygon(s, [tuple(map(float, p)) for p in wv], tuple(col))
        gfxdraw.filled_polygon(s, [tuple(map(float, p)) for p in wv], tuple(col))
        got = _arr(s)

        vi = np.trunc(wv).astype(np.int64)
        canvas = np.full((W, W, 3), 255, np.int64)
        for i in range(4):                                  # aapolygon edge order
            a, b = vi[i], vi[(i + 1) % 4]
            for (x, y, wgt) in aaline_ops(a[0], a[1], b[0], b[1]):
                if 0 <= x < W and 0 <= y < W:
                    canvas[y, x] = blend(col, canvas[y, x], wgt)
        fill = P.filled_polygon_ref(vi, W, W)
        canvas[fill] = col
        d = (canvas != got).any(-1)
        nbad += d.any()
        tot_diff += d.sum()
        tot_px += W * W
        fringe_px += ((got != 255).any(-1) & ~fill).sum()
    print(f"  quads with ANY mismatching pixel : {nbad}/120")
    print(f"  total mismatching pixels          : {tot_diff} / {tot_px} "
          f"({100*tot_diff/tot_px:.5f} %)")
    print(f"  AA fringe pixels (outside fill)   : {fringe_px}  "
          f"({fringe_px/120:.1f} per quad)")

    pygame.quit()


if __name__ == "__main__":
    main()
