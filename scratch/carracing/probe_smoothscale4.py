"""smoothscale, third hypothesis: the fixed-point UNIT is 2^14, not 2^16 (the SIMD path).

Variant (a) of probe 3 (exact taps) and the 16.16 double-truncation of probe 2 both miss, in
opposite directions.  Hand-checking 1000->600 out[2] showed the reference lands on 33 where 16.16
gives 32 and the only difference is ``recip``: 2^32//((0x10000*1000)//600) = 39321 while
2^30//((0x4000*1000)//600) = 39322.  A 2^14 unit is what an ``_mm_mulhi_epu16`` SIMD path needs,
so this sweeps the unit.

    space = ((1 << bits) * n) // m
    recip = (1 << (16 + bits)) // space
    acc, counter = 0, space
    for v in src:
        if counter > (1 << bits):  acc += v; counter -= (1 << bits)
        else:
            frac = (1 << bits) - counter
            emit(((acc + ((v * counter) >> bits)) * recip) >> 16)
            acc = (v * frac) >> bits
            counter = space - frac
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()


def shrink(a: np.ndarray, m: int, axis: int, bits: int) -> np.ndarray:
    a = np.moveaxis(a, axis, 0).astype(np.int64)
    n = a.shape[0]
    U = 1 << bits
    space = (U * n) // m
    recip = (1 << (16 + bits)) // space
    out = np.zeros((m,) + a.shape[1:], np.int64)
    acc = np.zeros(a.shape[1:], np.int64)
    counter, j = space, 0
    for x in range(n):
        if counter > U:
            acc = acc + a[x]
            counter -= U
        else:
            frac = U - counter
            if j < m:
                out[j] = (((acc + ((a[x] * counter) >> bits)) & 0xFFFF) * recip) >> 16
            j += 1
            acc = (a[x] * frac) >> bits
            counter = space - frac
    assert j >= m, f"{n}->{m} bits {bits}: only {j} outputs"
    return np.moveaxis((out & 0xFF).astype(np.uint8), 0, axis)


def surf_from(a):
    h, w = a.shape[:2]
    s = pygame.Surface((w, h))
    pygame.surfarray.blit_array(s, np.transpose(a, (1, 0, 2)).copy())
    return s


def arr_from(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))


def main():
    rng = np.random.default_rng(3)
    print(f"pygame {pygame.version.ver}")
    BITS = (16, 15, 14, 13, 12, 8)
    tot = {b: [0, 0, 0] for b in BITS}          # exact, n, maxdev
    for (W, H, w, h) in ((1000, 800, 96, 96), (1000, 800, 600, 400), (1000, 800, 400, 313),
                         (256, 128, 25, 37), (100, 100, 99, 51), (777, 333, 111, 222)):
        for mk in (lambda: rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
                   lambda: np.repeat(np.repeat(rng.integers(0, 256, (H // 8 + 1, W // 8 + 1, 3),
                                                            dtype=np.uint8), 8, 0), 8, 1)[:H, :W],
                   lambda: np.full((H, W, 3), 102, np.uint8),
                   lambda: (rng.integers(0, 2, (H, W, 3)) * 255).astype(np.uint8)):
            a = np.ascontiguousarray(mk())
            for nm, tw, th in (("X", w, H), ("Y", W, h), ("XY", w, h)):
                if tw == W and th == H:
                    continue
                ref = arr_from(pygame.transform.smoothscale(surf_from(a), (tw, th)))
                for b in BITS:
                    o = a
                    if tw != W:
                        o = shrink(o, tw, 1, b)
                    if th != H:
                        o = shrink(o, th, 0, b)
                    d = np.abs(o.astype(np.int32) - ref.astype(np.int32))
                    tot[b][0] += int((d == 0).sum())
                    tot[b][1] += d.size
                    tot[b][2] = max(tot[b][2], int(d.max()))
    print("\n bits   exact%    maxdev")
    for b in BITS:
        e, n, mx = tot[b]
        print(f"  {b:2d}   {100*e/n:8.4f}%   {mx}")


if __name__ == "__main__":
    main()
