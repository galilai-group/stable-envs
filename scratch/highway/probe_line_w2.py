"""What does a width-2 ``pygame.draw.line`` actually paint?

intersection-v0 runs at scaling 5.5*1.3 = 7.15, so ``pix(STRIPE_WIDTH) = int(2.145) = 2`` and the lane
markings leave pygame's Bresenham path. This was the one uncalibrated thing left in the way of porting
that env, so it gets measured rather than assumed: several candidate capsules and quads are compared
against the reference on random segments, split by orientation because the axis-aligned case is the one
intersection-v0 mostly needs.
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


def ref(a, b, w):
    s = pygame.Surface((W, H))
    s.fill(BG)
    pygame.draw.line(s, FG, a, b, w)
    return np.moveaxis(pygame.surfarray.array3d(s), 0, 1)[..., 0] > 0


def cand(a, b, rule, off=0.5):
    ax, ay, bx, by = a[0] + off, a[1] + off, b[0] + off, b[1] + off
    ex, ey = bx - ax, by - ay
    L2 = max(ex * ex + ey * ey, 1e-20)
    L = np.sqrt(L2)
    t = np.clip(((PX - ax) * ex + (PY - ay) * ey) / L2, 0, 1)
    dx, dy = PX - ax - t * ex, PY - ay - t * ey
    d2 = dx * dx + dy * dy
    if rule == "cap1.0":
        return d2 <= 1.0
    if rule == "cap0.5":
        return d2 <= 0.25
    if rule == "cap_major":
        hw = 1.0 * max(abs(ex), abs(ey)) / L
        return d2 <= hw * hw
    if rule == "quad1.0":                       # infinite band, no round caps, clipped to the segment
        perp = np.abs((PX - ax) * (-ey / L) + (PY - ay) * (ex / L))
        s = ((PX - ax) * ex + (PY - ay) * ey) / L2
        return (perp <= 1.0) & (s >= 0) & (s <= 1)
    if rule == "cap0.75":
        return d2 <= 0.5625
    if rule.startswith("minor"):
        # Read off probe_line_w2b: a width-w line lights w consecutive pixels per major-axis column,
        # and for EVEN w the run is shifted one pixel toward +minor, because 2 pixels cannot straddle a
        # centre symmetrically. So: threshold the MINOR-axis distance at w/2 about a centre offset by
        # +0.5 when w is even. Independent of the drawing direction (measured both ways).
        w = float(rule.split(":")[1])
        shift = 0.5 if int(w) % 2 == 0 else 0.0
        if abs(ex) >= abs(ey):
            ay += shift; by += shift
        else:
            ax += shift; bx += shift
        t2 = np.clip(((PX - ax) * ex + (PY - ay) * ey) / L2, 0, 1)
        dx2, dy2 = PX - ax - t2 * ex, PY - ay - t2 * ey
        hw = (w / 2.0) * max(abs(ex), abs(ey)) / L
        return dx2 * dx2 + dy2 * dy2 <= hw * hw
    raise KeyError(rule)


RULES = ["cap0.5", "cap1.0", "cap_major", "quad1.0", "minor:1", "minor:2", "minor:3"]

if __name__ == "__main__":
    for w in (1, 2, 3):
        print(f"=== pygame.draw.line width={w} ===")
        for tag, gen in (
            ("axis-aligned", lambda r: ((10, 30), (100, 30)) if r.random() < 0.5 else ((40, 8), (40, 110))),
            ("random", None),
        ):
            rng = np.random.default_rng(0)
            tot = {k: [0, 0, 0] for k in RULES}
            n = 0
            for _ in range(300):
                if gen is not None:
                    a, b = gen(rng)
                    a = (a[0] + int(rng.integers(-4, 4)), a[1] + int(rng.integers(-4, 4)))
                    b = (b[0] + int(rng.integers(-4, 4)), b[1] + int(rng.integers(-4, 4)))
                else:
                    a = tuple(int(v) for v in rng.integers(6, W - 6, 2))
                    b = tuple(int(v) for v in rng.integers(6, W - 6, 2))
                if a == b:
                    continue
                R = ref(a, b, w)
                n += 1
                for k in RULES:
                    if k.startswith("minor") and int(k.split(":")[1]) != w:
                        continue
                    C = cand(a, b, k)
                    tot[k][0] += int((C != R).sum())
                    tot[k][1] += int(R.sum())
                    tot[k][2] += int((C != R).sum() == 0)
            print(f"  {tag} ({n} segments)")
            for k, (bad, lit, perfect) in tot.items():
                print(f"    {k:10s} wrong {bad:6d} / {lit:6d} lit ({100*bad/max(lit,1):6.2f}%), "
                      f"{perfect:3d}/{n} exact")
