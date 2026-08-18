"""Nail the exact fixed-point details of ``smoothscale``'s 1-D shrink.

probe_smoothscale.py showed the STRUCTURE is right (X-then-Y through an 8-bit intermediate is
bit-exact against the single call) but the arithmetic is off by 1 on a few percent of samples, so
some rounding/tap detail is wrong.  This searches a small family of variants against pygame on
random 1-D data, and dumps the disagreeing taps for whichever survives longest.
"""
from __future__ import annotations

import itertools
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()

H = 8


def ref_shrink(row: np.ndarray, m: int) -> np.ndarray:
    """pygame's own answer for a 1-D shrink, taken from an (H, n) surface scaled to (m, H)."""
    n = row.shape[0]
    a = np.repeat(row.reshape(1, n, 1), H, 0).repeat(3, 2)
    s = pygame.Surface((n, H))
    pygame.surfarray.blit_array(s, np.transpose(a, (1, 0, 2)).copy())
    out = np.transpose(np.array(pygame.surfarray.pixels3d(
        pygame.transform.smoothscale(s, (m, H)))), (1, 0, 2))
    assert (out[:, :, 0] == out[0:1, :, 0]).all(), "rows disagree -- not a pure X shrink"
    return out[0, :, 0].astype(np.int64)


def run(row, m, part_round, final_round, cmp_ge, wrap16):
    n = row.shape[0]
    space = (0x10000 * n) // m
    recip = (1 << 32) // space
    half_p = 0x8000 if part_round else 0
    half_f = 0x8000 if final_round else 0
    out, taps = [], []
    acc, cur, counter = 0, [], space
    for x in range(n):
        v = int(row[x])
        hit = counter >= 0x10000 if cmp_ge else counter > 0x10000
        if hit:
            acc += v
            cur.append((x, 0x10000))
            counter -= 0x10000
        else:
            frac = 0x10000 - counter
            tot = acc + ((v * counter + half_p) >> 16)
            if wrap16:
                tot &= 0xFFFF
            out.append(((tot * recip + half_f) >> 16) & 0xFF)
            taps.append(cur + [(x, counter)])
            acc = (v * frac + half_p) >> 16
            cur = [(x, frac)] if frac else []
            counter = space - frac
        if len(out) == m:
            break
    return np.array(out, np.int64), taps


def main():
    rng = np.random.default_rng(7)
    cases = [(1000, 600), (1000, 96), (800, 96), (800, 400), (10, 3), (7, 3), (100, 37), (256, 25)]
    variants = list(itertools.product((0, 1), (0, 1), (0, 1), (0, 1)))
    print(f"pygame {pygame.version.ver}\n")
    score = {v: 0 for v in variants}
    total = 0
    for (n, m) in cases:
        rows = [rng.integers(0, 256, n) for _ in range(4)] + [np.full(n, 102), np.full(n, 255)]
        for row in rows:
            ref = ref_shrink(row.astype(np.uint8), m)
            total += ref.size
            for v in variants:
                got, _ = run(row, m, *v)
                k = min(got.size, ref.size)
                score[v] += int((got[:k] == ref[:k]).sum())
    print(f"{'part_rnd final_rnd cmp>= wrap16':34s}  exact")
    for v in sorted(variants, key=lambda k: -score[k]):
        print(f"  {str(v):32s} {100*score[v]/total:8.4f}%")

    best = max(variants, key=lambda k: score[k])
    print(f"\nbest {best}; dumping first disagreements at 1000->600")
    row = rng.integers(0, 256, 1000)
    ref = ref_shrink(row.astype(np.uint8), 600)
    got, taps = run(row, 600, *best)
    bad = np.nonzero(got != ref)[0]
    print(f"  {bad.size}/600 wrong")
    for i in bad[:8]:
        t = taps[i]
        print(f"  out[{i}] ours {got[i]:3d} ref {ref[i]:3d}   taps "
              f"{[(int(j), w, int(row[j])) for j, w in t]}")
    for i in range(6):
        t = taps[i]
        print(f"  ok? out[{i}] ours {got[i]:3d} ref {ref[i]:3d}  taps "
              f"{[(int(j), w) for j, w in t]}")


if __name__ == "__main__":
    main()
