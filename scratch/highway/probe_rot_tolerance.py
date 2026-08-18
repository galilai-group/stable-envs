"""How far from a multiple of 90 degrees does ``pygame.transform.rotate`` still take its fast path?

probe_pivot_near90.py showed that a rotation by -90.0000025 degrees -- what a float32 heading of
``pi/2`` becomes -- is BIT-IDENTICAL to a rotation by -90.0, while -90.001 is not. So the fast path has
a tolerance, and ``_pivot``'s exact-equality test for the axis-aligned branch is the wrong test: it
sends near-axis angles down the ``Wr``-parity branch, which is up to a pixel out.

Bisect it, at several sprite sizes and around all four quarter turns.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()


def sprite(S):
    s = pygame.Surface((S, S), pygame.SRCALPHA)
    pygame.draw.rect(s, (255, 0, 0), (1, S // 3, S - 3, max(1, S // 4)), 0)
    return s


def same(base, a, b):
    ra, rb = pygame.transform.rotate(base, a), pygame.transform.rotate(base, b)
    if ra.get_size() != rb.get_size():
        return False
    return np.array_equal(pygame.surfarray.pixels3d(ra), pygame.surfarray.pixels3d(rb))


for S in (11, 14, 21, 38, 49, 294, 490):
    base = sprite(S)
    out = []
    for anchor in (0.0, 90.0, -90.0, 180.0, 270.0):
        lo, hi = 0.0, 1.0
        if not same(base, anchor + 1e-9, anchor):
            out.append((anchor, 0.0))
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if same(base, anchor + mid, anchor):
                lo = mid
            else:
                hi = mid
        out.append((anchor, lo))
    print(f"S={S:4d}: " + "  ".join(f"{a:+7.1f}->{t:.3e}" for a, t in out))

# is it a tolerance on the ANGLE, or does the fast path key on the ROUNDED angle? Compare the
# threshold against some plausible constants.
print("\ncandidate constants: 1e-6 %.0e   FLT_EPSILON*90 %.3e   1e-5 %.0e"
      % (1e-6, np.finfo(np.float32).eps * 90, 1e-5))

# and confirm the whole point: what a float32 pi/2 and pi become in degrees
for v in (np.float32(np.pi / 2), np.float32(np.pi), np.float32(-np.pi / 2)):
    print(f"float32 {float(v)!r} -> rad2deg {np.rad2deg(float(v))!r} "
          f"(off by {abs(np.rad2deg(float(v))) % 90:.3e})")
