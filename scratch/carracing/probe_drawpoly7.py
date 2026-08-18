"""Full table: exact crossings vs pygame's painted span, for one rotated rectangle."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np, pygame
pygame.init()
W, H = 160, 120


def ref(points):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), [(float(a), float(b)) for a, b in points])
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def table(px, py, tag):
    n = len(px)
    m = ref(list(zip(px, py)))
    miny, maxy = min(py), max(py)
    print(f"\n{tag}  px={px} py={py}")
    print("    y   exact_l  exact_r   ref_l ref_r   dl=ref_l-exact_l  dr=ref_r-exact_r")
    for y in range(max(miny, 0), min(maxy, H - 1) + 1):
        xs = []
        for i in range(n):
            ip = i - 1 if i else n - 1
            y1, y2 = py[ip], py[i]
            if y1 < y2:
                x1, x2 = px[ip], px[i]
            elif y1 > y2:
                y1, y2 = py[i], py[ip]
                x1, x2 = px[i], px[ip]
            else:
                continue
            if (y1 <= y < y2) or (y == maxy and y1 < y <= y2):
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        c = np.nonzero(m[y])[0]
        rl, rr = (int(c[0]), int(c[-1])) if c.size else (None, None)
        if len(xs) >= 2 and rl is not None:
            print(f"  {y:4d} {xs[0]:8.3f} {xs[-1]:8.3f}   {rl:5d} {rr:5d}"
                  f"   {rl - xs[0]:+7.3f} {rr - xs[-1]:+7.3f}")
        else:
            print(f"  {y:4d} xs={xs}  ref=({rl},{rr})")


table([74, 56, 95, 113], [102, 53, 39, 88], "E rotated rect")
