"""Pin the last 30%: SDL_gfx blends on the MASKED channel word, so the rounding differs per channel.

``_putPixelAlpha`` does ``R = (dR + ((((color & Rmask) - dR) * alpha) >> 8)) & Rmask`` in *unsigned
32-bit* arithmetic on the masked value, not on the 0..255 channel. For the red mask (0x000000ff)
that is a plain ``>>8`` truncation; for green (0x0000ff00) and blue (0x00ff0000) the shift happens
below the channel's own bit position, so the truncation lands somewhere else -- and when the source
is darker than the destination the unsigned wrap turns the truncation into a ceiling. All of that is
testable, so it is tested rather than reasoned about.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
from pygame import gfxdraw

from probe_aaline import aaline_ops

W = 128
M32 = (1 << 32) - 1


def _surf(bg):
    s = pygame.Surface((W, W))
    s.fill(bg)
    return s


def _arr(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2)).astype(np.int64)


def c_blend(s, d, a, shift):
    """Exact unsigned-32 transcription of _putPixelAlpha for one masked channel."""
    S = np.left_shift(s, shift)
    D = np.left_shift(d, shift)
    mask = 0xFF << shift
    v = np.bitwise_and(S - D, M32)
    v = np.bitwise_and(v * a, M32)
    v = np.right_shift(v, 8)
    return np.right_shift(np.bitwise_and(D + v, mask), shift)


def main():
    pygame.init()
    probe = pygame.Surface((4, 4))
    print(f"  plain pygame.Surface masks = {probe.get_masks()}  bitsize {probe.get_bitsize()}")

    rng = np.random.default_rng(1)
    S, D, A, O = [], [], [], []
    bgs = [(255, 255, 255), (0, 0, 0), (100, 160, 200), (17, 240, 90), (128, 128, 128)]
    cols = [(0, 0, 0), (204, 77, 77), (202, 152, 101), (0, 204, 204), (255, 255, 255), (129, 132, 203)]
    for bg in bgs:
        for col in cols:
            for _ in range(30):
                v = rng.integers(8, 120, (3, 2))
                if len(set(map(tuple, v.tolist()))) < 3:
                    continue
                s = _surf(bg)
                gfxdraw.aapolygon(s, [tuple(int(q) for q in p) for p in v], col)
                got = _arr(s)
                cnt = np.zeros((W, W), np.int64)
                wmap = np.zeros((W, W), np.int64)
                for i in range(3):
                    for (x, y, w) in aaline_ops(v[i][0], v[i][1], v[(i + 1) % 3][0], v[(i + 1) % 3][1]):
                        if 0 <= x < W and 0 <= y < W:
                            cnt[y, x] += 1
                            wmap[y, x] = w
                ys, xs = np.where(cnt == 1)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    S.append(col)
                    D.append(bg)
                    A.append(wmap[y, x])
                    O.append(got[y, x])
    S = np.array(S, np.int64)
    D = np.array(D, np.int64)
    wg = np.array(A, np.int64)[:, None]
    O = np.array(O, np.int64)
    print(f"  {len(S)} clean single-write observations, "
          f"{len(np.unique(np.concatenate([S, D], 1), axis=0))} distinct (src,dst) pairs")

    a = (255 * wg) >> 8
    shifts = np.array([0, 8, 16])
    pred_c = np.stack([c_blend(S[:, k], D[:, k], a[:, 0], int(shifts[k])) for k in range(3)], 1)
    pred_c = np.where(wg == 255, S, pred_c)
    ok = (pred_c == O).all(-1)
    print(f"\n  SDL_gfx masked-word form (per-channel shift 0/8/16): exact {100*ok.mean():.3f} %  "
          f"max|err| {np.abs(pred_c-O).max()}")
    for k, nm in enumerate("RGB"):
        okk = pred_c[:, k] == O[:, k]
        print(f"     channel {nm}: {100*okk.mean():7.3f} %")

    pred_s = np.stack([c_blend(S[:, k], D[:, k], a[:, 0], 0) for k in range(3)], 1)
    pred_s = np.where(wg == 255, S, pred_s)
    print(f"  same form but shift=0 for every channel:              exact "
          f"{100*(pred_s == O).all(-1).mean():.3f} %")

    pred_f = np.where(wg == 255, S, (S * a + D * (255 - a)) // 255)
    print(f"  float-ish (s*a+d*(255-a))//255:                       exact "
          f"{100*(pred_f == O).all(-1).mean():.3f} %")

    bad = ~ok
    if bad.any():
        print(f"\n  residual mismatches: {bad.sum()}")
        idx = np.where(bad)[0][:8]
        for i in idx:
            print(f"    src={S[i].tolist()} dst={D[i].tolist()} w={int(wg[i,0])} "
                  f"a={int(a[i,0])} got={O[i].tolist()} pred={pred_c[i].tolist()}")
    pygame.quit()


if __name__ == "__main__":
    main()
