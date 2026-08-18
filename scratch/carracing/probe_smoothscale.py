"""Recover ``pygame.transform.smoothscale``'s downscale rule EXACTLY, by measurement.

Blocker 1 of docs/research/carracing.md says the 96x96 observation is a resample, not a render, so
matching it needs smoothscale reimplemented.  This asks what smoothscale actually computes.

Hypothesis under test (pygame's ``filter_shrink_X``/``filter_shrink_Y``, 16.16 fixed point):

    space = (0x10000 * n_src) // n_dst           # > 0x10000 for a shrink
    recip = (1 << 32) // space
    acc, counter = 0, space
    for each source sample v:
        if counter > 0x10000:  acc += v;  counter -= 0x10000
        else:
            frac = 0x10000 - counter
            emit( ((acc + ((v * counter) >> 16)) * recip) >> 16 )
            acc = (v * frac) >> 16
            counter = space - frac

Checked against pygame on random surfaces, per axis and end to end.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()


def surf_from(a: np.ndarray) -> pygame.Surface:
    """(H,W,3) uint8 -> a plain 32-bit Surface, the kind CarRacing draws into."""
    h, w = a.shape[:2]
    s = pygame.Surface((w, h))
    pygame.surfarray.blit_array(s, np.transpose(a, (1, 0, 2)).copy())
    return s


def arr_from(s: pygame.Surface) -> np.ndarray:
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))


def plan(n_src: int, n_dst: int):
    """(idx, weight) table for one shrink axis, plus ``recip``. weight 0x10000 == whole sample."""
    space = (0x10000 * n_src) // n_dst
    recip = (1 << 32) // space
    rows, acc_idx, acc_w = [], [], []
    counter = space
    for x in range(n_src):
        if counter > 0x10000:
            acc_idx.append(x)
            acc_w.append(0x10000)
            counter -= 0x10000
        else:
            frac = 0x10000 - counter
            rows.append((list(acc_idx) + [x], list(acc_w) + [counter]))
            acc_idx, acc_w = ([x], [frac]) if frac else ([], [])
            counter = space - frac
        if len(rows) == n_dst:
            break
    return rows, recip, space


def shrink_axis(a: np.ndarray, n_dst: int, axis: int) -> np.ndarray:
    """The hypothesis, in numpy. ``a`` is uint8 with the shrink axis at ``axis``."""
    a = np.moveaxis(a, axis, 0).astype(np.int64)
    rows, recip, _ = plan(a.shape[0], n_dst)
    out = np.empty((n_dst,) + a.shape[1:], np.int64)
    for j, (idx, w) in enumerate(rows):
        acc = np.zeros(a.shape[1:], np.int64)
        for i, wi in zip(idx, w):
            acc += a[i] if wi == 0x10000 else (a[i] * wi) >> 16
        out[j] = ((acc & 0xFFFF) * recip) >> 16
    return np.moveaxis((out & 0xFF).astype(np.uint8), 0, axis)


def smooth(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """X pass then Y pass, both through an 8-bit intermediate -- pygame's own order."""
    return shrink_axis(shrink_axis(a, w, 1), h, 0)


def report(name, ours, ref):
    d = np.abs(ours.astype(np.int32) - ref.astype(np.int32))
    print(f"  {name:28s} exact {(d == 0).mean()*100:7.3f}%  max {d.max():3d}  mean {d.mean():.5f}")
    return int(d.max())


def main():
    rng = np.random.default_rng(0)
    print(f"pygame {pygame.version.ver}")
    s = pygame.Surface((1000, 800))
    print(f"Surface((1000,800)): bitsize {s.get_bitsize()} bytesize {s.get_bytesize()} "
          f"masks {[hex(m) for m in s.get_masks()]}")

    worst = 0
    for (W, H, w, h) in ((1000, 800, 96, 96), (1000, 800, 600, 400)):
        print(f"\n=== {W}x{H} -> {w}x{h} ===")
        rows, recip, space = plan(W, w)
        print(f"  X: space {space} recip {recip}  taps/out {min(len(r[0]) for r in rows)}"
              f"..{max(len(r[0]) for r in rows)}  rows {len(rows)}")
        rows, recip, space = plan(H, h)
        print(f"  Y: space {space} recip {recip}  taps/out {min(len(r[0]) for r in rows)}"
              f"..{max(len(r[0]) for r in rows)}  rows {len(rows)}")

        for trial, mk in enumerate((
                lambda: rng.integers(0, 256, (H, W, 3), dtype=np.uint8),         # white noise
                lambda: np.repeat(np.repeat(rng.integers(0, 256, (H // 8 + 1, W // 8 + 1, 3),
                                                         dtype=np.uint8), 8, 0), 8, 1)[:H, :W],
                lambda: np.full((H, W, 3), 102, np.uint8),                       # CarRacing bg
                lambda: np.full((H, W, 3), 204, np.uint8),
        )):
            a = np.ascontiguousarray(mk())
            ref = arr_from(pygame.transform.smoothscale(surf_from(a), (w, h)))
            worst = max(worst, report(f"trial{trial} end-to-end", smooth(a, w, h), ref))

            # is it separable through an 8-bit intermediate?
            two = arr_from(pygame.transform.smoothscale(
                pygame.transform.smoothscale(surf_from(a), (w, H)), (w, h)))
            report(f"trial{trial} pygame X-then-Y", two, ref)

            # per-axis, against pygame doing only that axis
            refx = arr_from(pygame.transform.smoothscale(surf_from(a), (w, H)))
            worst = max(worst, report(f"trial{trial} X only", shrink_axis(a, w, 1), refx))
            refy = arr_from(pygame.transform.smoothscale(surf_from(a), (W, h)))
            worst = max(worst, report(f"trial{trial} Y only", shrink_axis(a, h, 0), refy))

    print(f"\nWORST end-to-end/per-axis deviation over all trials: {worst}")


if __name__ == "__main__":
    main()
