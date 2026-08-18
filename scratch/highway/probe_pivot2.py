"""Exact pivot offset by direct search, over a dense angle sweep, plus the closed form it obeys.

The offset is searched on a small window so the sweep is cheap; only offsets that reproduce 100% of the
window's pixels are accepted, so this measures the truth rather than a best fit.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np
import pygame

pygame.display.init()
GREY = (100, 100, 100)
COL, LIGHT, BLACK = (100, 200, 255), (147, 255, 255), (60, 60, 60)
WIN = 80
POS = (40, 40)


def sprite_rects(scaling, LEN=5.0, WID=2.0, TIRE=1.0):
    pix = lambda v: int(v * scaling)
    S = pix(LEN + 2 * TIRE)
    body = (pix(TIRE), pix((LEN + 2 * TIRE) / 2 - WID / 2), pix(LEN), pix(WID))
    hl = (pix(TIRE + LEN - 0.72), pix((LEN + 2 * TIRE) / 2 - 1.4 * WID / 3), pix(0.72), pix(0.6))
    hr = (pix(TIRE + LEN - 0.72), pix((LEN + 2 * TIRE) / 2 + 0.6 * WID / 5), pix(0.72), pix(0.6))
    return S, body, hl, hr


def build_sprite(S, body, hl, hr):
    s = pygame.Surface((S, S), pygame.SRCALPHA)
    pygame.draw.rect(s, COL, body, 0)
    pygame.draw.rect(s, LIGHT, hl, 0)
    pygame.draw.rect(s, LIGHT, hr, 0)
    pygame.draw.rect(s, BLACK, body, 1)
    return s


def blit_rotate_origin(w, h, pos, angle):
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    br = [p.rotate(angle) for p in box]
    min_box = (min(br, key=lambda p: p[0])[0], min(br, key=lambda p: p[1])[1])
    max_box = (max(br, key=lambda p: p[0])[0], max(br, key=lambda p: p[1])[1])
    pv = pygame.math.Vector2(w / 2, -h / 2)
    pm = pv.rotate(angle) - pv
    return (pos[0] - w / 2 + min_box[0] - pm[0], pos[1] - h / 2 - max_box[1] + pm[1])


YY, XX = np.mgrid[0:WIN, 0:WIN]
PX, PY = XX + 0.5, YY + 0.5


def _clip(r, box):
    x0 = max(r[0], box[0]); y0 = max(r[1], box[1])
    x1 = min(r[0] + r[2], box[0] + box[2]); y1 = min(r[1] + r[3], box[1] + box[3])
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1 - x0, y1 - y0)


def analytic(S, rects, heading, dx, dy):
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    img = np.zeros((WIN, WIN, 3), np.uint8)
    img[:] = GREY
    cen = np.array([S / 2.0, S / 2.0])
    for r, col in rects:
        if r is None:
            continue
        rc = np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]) - cen
        w = np.array([POS[0] + dx, POS[1] + dy]) + R @ rc
        d = np.stack([PX - w[0], PY - w[1]], -1) @ R
        img[(np.abs(d[..., 0]) <= r[2] / 2.0) & (np.abs(d[..., 1]) <= r[3] / 2.0)] = col
    return img


def run(scaling, n_ang=97):
    S, body, hl, hr = sprite_rects(scaling)
    sp = build_sprite(S, body, hl, hr)
    inset = (body[0] + 1, body[1] + 1, body[2] - 2, body[3] - 2)
    rects = [(body, BLACK), (inset, COL), (_clip(hl, inset), LIGHT), (_clip(hr, inset), LIGHT)]
    grid = np.arange(-1.5, 0.51, 0.25)
    print(f"=== scaling={scaling} S={S} body={body} ===")
    rows = []
    for heading in np.linspace(-np.pi, np.pi, n_ang):
        angle = np.rad2deg(-heading)
        rot = pygame.transform.rotate(sp, angle)
        surf = pygame.Surface((WIN, WIN))
        surf.fill(GREY)
        org = blit_rotate_origin(S, S, POS, angle)
        surf.blit(rot, org)
        ref = np.moveaxis(pygame.surfarray.array3d(surf), 0, 1)
        hits, best, bd = [], -1.0, None
        for dx in grid:
            for dy in grid:
                f = float((np.abs(analytic(S, rects, heading, dx, dy).astype(np.int16)
                                  - ref.astype(np.int16)).max(-1) == 0).mean())
                if f > best:
                    best, bd = f, (dx, dy)
                if f == 1.0:
                    hits.append((round(float(dx), 2), round(float(dy), 2)))
        rows.append((heading, rot.get_size()[0], org, best, bd, hits))
    for heading, wr, org, best, bd, hits in rows:
        print(f"  h={heading:+.5f} Wr={wr:3d} dWr={wr - S:3d} orgfrac=({org[0]%1:.4f},{org[1]%1:.4f}) "
              f"exact={best*100:6.2f}% at ({bd[0]:+.2f},{bd[1]:+.2f}) hits={hits}")
    n_perfect = sum(1 for r in rows if r[3] == 1.0)
    print(f"  {n_perfect}/{len(rows)} angles reach 100% with a quarter-pixel offset")


if __name__ == "__main__":
    run(5.5)
