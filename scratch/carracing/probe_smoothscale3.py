"""smoothscale, second hypothesis: the taps are exact and only the FINAL divide truncates.

probe_smoothscale2 showed every disagreement was ours = ref - 1, and hand-checking two of them
(out[2] taps (46,43692)+(26,65534); out[4] taps (73,21848)+(97,65536)+(164,21842)) showed the
reference behaves as if the per-tap products are NOT truncated:

    A     = sum_i  v_i * w_i          (w_i are 16.16 weights summing to exactly ``space``)
    space = (0x10000 * n_src) // n_dst
    recip = (1 << 32) // space
    out   = (A * recip) >> 32          <- candidate (a)   /   A // space  <- candidate (b)

Both are tested here, per axis and end to end, against pygame.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()


def taps(n: int, m: int):
    """[(idx array, weight array)] * m, plus space/recip. Weights sum to ``space`` per output."""
    space = (0x10000 * n) // m
    recip = (1 << 32) // space
    rows, cur, counter = [], [], space
    for x in range(n):
        if counter > 0x10000:
            cur.append((x, 0x10000))
            counter -= 0x10000
        else:
            frac = 0x10000 - counter
            rows.append(cur + [(x, counter)])
            cur = [(x, frac)] if frac else []
            counter = space - frac
        if len(rows) == m:
            break
    assert len(rows) == m, f"{n}->{m}: {len(rows)} rows"
    for r in rows:
        assert sum(w for _, w in r) == space
    return rows, space, recip


def shrink(a: np.ndarray, m: int, axis: int, variant: str) -> np.ndarray:
    a = np.moveaxis(a, axis, 0).astype(np.int64)
    rows, space, recip = taps(a.shape[0], m)
    out = np.empty((m,) + a.shape[1:], np.int64)
    for j, r in enumerate(rows):
        A = np.zeros(a.shape[1:], np.int64)
        for i, w in r:
            A += a[i] * w
        out[j] = (A * recip) >> 32 if variant == "a" else A // space
    return np.moveaxis((out & 0xFF).astype(np.uint8), 0, axis)


def surf_from(a):
    h, w = a.shape[:2]
    s = pygame.Surface((w, h))
    pygame.surfarray.blit_array(s, np.transpose(a, (1, 0, 2)).copy())
    return s


def arr_from(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))


def main():
    rng = np.random.default_rng(11)
    print(f"pygame {pygame.version.ver}")
    worst = {"a": 0, "b": 0}
    for (W, H, w, h) in ((1000, 800, 96, 96), (1000, 800, 600, 400), (1000, 800, 400, 313),
                         (256, 128, 25, 37), (100, 100, 99, 51)):
        print(f"\n=== {W}x{H} -> {w}x{h} ===")
        for t, mk in enumerate((
                lambda: rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
                lambda: np.repeat(np.repeat(rng.integers(0, 256, (H // 8 + 1, W // 8 + 1, 3),
                                                         dtype=np.uint8), 8, 0), 8, 1)[:H, :W],
                lambda: np.full((H, W, 3), 102, np.uint8),
                lambda: (rng.integers(0, 2, (H, W, 3)) * 255).astype(np.uint8),
        )):
            a = np.ascontiguousarray(mk())
            ref = arr_from(pygame.transform.smoothscale(surf_from(a), (w, h)))
            refx = arr_from(pygame.transform.smoothscale(surf_from(a), (w, H)))
            refy = arr_from(pygame.transform.smoothscale(surf_from(a), (W, h)))
            for v in ("a", "b"):
                e2e = shrink(shrink(a, w, 1, v), h, 0, v)
                for nm, ours, r in (("end2end", e2e, ref), ("Xonly", shrink(a, w, 1, v), refx),
                                    ("Yonly", shrink(a, h, 0, v), refy)):
                    d = np.abs(ours.astype(np.int32) - r.astype(np.int32))
                    worst[v] = max(worst[v], int(d.max()))
                    if d.max():
                        print(f"  [{v}] t{t} {nm:8s} exact {(d == 0).mean()*100:7.3f}% "
                              f"max {d.max()} mean {d.mean():.5f}")
    print(f"\nworst deviation: variant a (A*recip>>32) = {worst['a']}, "
          f"variant b (A//space) = {worst['b']}")


if __name__ == "__main__":
    main()
