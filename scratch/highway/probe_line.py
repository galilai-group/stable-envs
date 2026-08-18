"""Which capsule reproduces a width-1 ``pygame.draw.line``?

Lane markings are 1 px lines at highway-env's default scaling, i.e. Bresenham, not a stroked segment.
A capsule of half-width 0.5 measures PERPENDICULAR distance, while Bresenham for an x-major line keeps
one pixel per column, i.e. it thresholds the VERTICAL distance at 0.5. The two agree only for axis
aligned lines, which is why highway-v0 came out exact and a diagonal marking would not. Setting the
capsule half-width to 0.5*|dx|/L converts the perpendicular threshold back into a vertical one; this
measures whether that is enough.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np
import pygame

pygame.display.init()
W = H = 120
BG, FG = (0, 0, 0), (255, 255, 255)
YY, XX = np.mgrid[0:H, 0:W]
PX, PY = XX + 0.5, YY + 0.5


def ref(a, b):
    s = pygame.Surface((W, H))
    s.fill(BG)
    pygame.draw.line(s, FG, a, b, 1)
    return np.moveaxis(pygame.surfarray.array3d(s), 0, 1)[..., 0] > 0


def capsule(a, b, rule, off=0.5, eps=0.0):
    ax, ay = a[0] + off, a[1] + off
    bx, by = b[0] + off, b[1] + off
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    if L2 == 0:
        L2 = 1e-20
    L = np.sqrt(L2)
    hw = {"perp": 0.5, "major": 0.5 * max(abs(ex), abs(ey)) / L}[rule]
    if eps:                       # break the tie at d == +0.5 the way an integer Bresenham does
        if abs(ex) >= abs(ey):
            ay += eps; by += eps
        else:
            ax += eps; bx += eps
    t = np.clip(((PX - ax) * ex + (PY - ay) * ey) / L2, 0, 1)
    dx, dy = PX - ax - t * ex, PY - ay - t * ey
    return dx * dx + dy * dy <= hw * hw


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    variants = {"perp": ("perp", 0.0), "major": ("major", 0.0),
                "major+eps": ("major", 1e-3), "major-eps": ("major", -1e-3),
                "major+.25": ("major", 0.25), "major-.25": ("major", -0.25)}
    tot = {k: [0, 0, 0] for k in variants}
    for _ in range(400):
        a = tuple(rng.integers(5, W - 5, 2))
        b = tuple(rng.integers(5, W - 5, 2))
        if a == b:
            continue
        R = ref(a, b)
        for k, (rule, eps) in variants.items():
            C = capsule(a, b, rule, eps=eps)
            tot[k][0] += int((C != R).sum())
            tot[k][1] += int(R.sum())
            tot[k][2] += int((C != R).sum() == 0)
    for rule, (bad, npix, perfect) in tot.items():
        print(f"  rule={rule:6s} wrong px {bad:7d} over {npix} lit px "
              f"({100*bad/npix:.2f}%), {perfect}/400 lines exact")

    print("\n  by slope (rule=major):")
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.1), (1.1, 5.0)):
        bad = npix = n = 0
        for _ in range(120):
            a = tuple(rng.integers(5, W - 5, 2))
            dx = rng.integers(-60, 60)
            dy = int(dx * rng.uniform(lo, hi)) if abs(dx) else 0
            b = (int(np.clip(a[0] + dx, 0, W - 1)), int(np.clip(a[1] + dy, 0, H - 1)))
            if a == b:
                continue
            R = ref(a, b)
            C = capsule(a, b, "major")
            bad += int((C != R).sum()); npix += int(R.sum()); n += 1
        print(f"    |slope| in [{lo},{hi}): {bad} wrong of {npix} lit ({100*bad/max(npix,1):.2f}%)")
