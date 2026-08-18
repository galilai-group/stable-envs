"""Does the pivot rule hold for an odd-sized sprite?

The rule was measured on the 38x38 vehicle sprite. Road objects use a sprite of ``pix(LENGTH)``, which
is 11x11 for the merge obstacle, and ``pos - S/2`` is then a half-integer that the blit truncates. If
the rule is parity-dependent it has to say so.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np
import pygame

pygame.display.init()
GREY, FILL, BLACK = (100, 100, 100), (200, 200, 0), (60, 60, 60)
WIN, POS = 60, (30, 30)
YY, XX = np.mgrid[0:WIN, 0:WIN]
PX, PY = XX + 0.5, YY + 0.5


def blit_origin(w, h, pos, angle):
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    br = [p.rotate(angle) for p in box]
    mn = (min(br, key=lambda p: p[0])[0], min(br, key=lambda p: p[1])[1])
    mx = (max(br, key=lambda p: p[0])[0], max(br, key=lambda p: p[1])[1])
    pv = pygame.math.Vector2(w / 2, -h / 2)
    pm = pv.rotate(angle) - pv
    return (pos[0] - w / 2 + mn[0] - pm[0], pos[1] - h / 2 - mx[1] + pm[1])


def run(S, rect):
    sp = pygame.Surface((S, S), pygame.SRCALPHA)
    pygame.draw.rect(sp, FILL, rect, 0)
    pygame.draw.rect(sp, BLACK, rect, 1)
    inset = (rect[0] + 1, rect[1] + 1, rect[2] - 2, rect[3] - 2)
    rects = [(rect, BLACK), (inset, FILL)]
    print(f"--- S={S} rect={rect} (S odd: {S % 2 == 1}) ---")
    grid = np.arange(-1.5, 0.51, 0.25)
    seen = {}
    for heading in np.linspace(-np.pi, np.pi, 181):
        angle = np.rad2deg(-heading)
        rot = pygame.transform.rotate(sp, angle)
        surf = pygame.Surface((WIN, WIN))
        surf.fill(GREY)
        surf.blit(rot, blit_origin(S, S, POS, angle))
        ref = np.moveaxis(pygame.surfarray.array3d(surf), 0, 1)
        hits = []
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])
        for dx in grid:
            for dy in grid:
                img = np.zeros((WIN, WIN, 3), np.uint8)
                img[:] = GREY
                for r, col in rects:
                    rc = np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]) - S / 2.0
                    w = np.array([POS[0] + dx, POS[1] + dy]) + R @ rc
                    d = np.stack([PX - w[0], PY - w[1]], -1) @ R
                    img[(np.abs(d[..., 0]) <= r[2] / 2.0) & (np.abs(d[..., 1]) <= r[3] / 2.0)] = col
                if np.abs(img.astype(np.int16) - ref.astype(np.int16)).max() == 0:
                    hits.append((round(float(dx), 2), round(float(dy), 2)))
        wr = rot.get_size()[0]
        seen.setdefault(wr, []).append(tuple(hits))
    for wr in sorted(seen):
        uniq = sorted(set(seen[wr]))
        print(f"  Wr={wr:3d} (Wr odd {wr%2==1}, dWr={wr-S:3d}) n={len(seen[wr]):3d} distinct hit-sets: {uniq[:4]}")


if __name__ == "__main__":
    run(11, (0, 0, 11, 11))
    run(14, (0, 0, 14, 6))
    run(21, (2, 5, 17, 9))
