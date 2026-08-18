"""How close is an analytic rotated rectangle to pygame's raster rotate + truncated blit?

highway-env does not draw a rotated rectangle. It draws an axis-aligned rectangle into a small
per-pixel-alpha sprite, calls pygame.transform.rotate (nearest-neighbour, fixed point), and blits the
result at a float origin that pygame truncates to a pixel. Any analytic renderer therefore differs from
it by whatever that pipeline does to the shape. This measures the difference instead of assuming it is
sub-pixel.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np
import pygame

pygame.display.init()
SC = 5.5
LEN, WID, TIRE = 5.0, 2.0, 1.0
pix = lambda v: int(v * SC)
S = pix(LEN + 2 * TIRE)
BODY = (pix(TIRE), pix((LEN + 2 * TIRE) / 2 - WID / 2), pix(LEN), pix(WID))
HL = (pix(TIRE + LEN - 0.72), pix((LEN + 2 * TIRE) / 2 - 1.4 * WID / 3), pix(0.72), pix(0.6))
HR = (pix(TIRE + LEN - 0.72), pix((LEN + 2 * TIRE) / 2 + 0.6 * WID / 5), pix(0.72), pix(0.6))
COL, LIGHT, BLACK, GREY = (100, 200, 255), (147, 255, 255), (60, 60, 60), (100, 100, 100)
W = H = 220


def sprite():
    s = pygame.Surface((S, S), pygame.SRCALPHA)
    pygame.draw.rect(s, COL, BODY, 0)
    pygame.draw.rect(s, LIGHT, HL, 0)
    pygame.draw.rect(s, LIGHT, HR, 0)
    pygame.draw.rect(s, BLACK, BODY, 1)
    return s


def blit_rotate_origin(w, h, pos, angle):
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    br = [p.rotate(angle) for p in box]
    min_box = (min(br, key=lambda p: p[0])[0], min(br, key=lambda p: p[1])[1])
    max_box = (max(br, key=lambda p: p[0])[0], max(br, key=lambda p: p[1])[1])
    pv = pygame.math.Vector2(w / 2, -h / 2)
    pm = pv.rotate(angle) - pv
    return (pos[0] - w / 2 + min_box[0] - pm[0], pos[1] - h / 2 - max_box[1] + pm[1])


def reference(pos, heading):
    surf = pygame.Surface((W, H))
    surf.fill(GREY)
    sp = sprite()
    angle = np.rad2deg(-heading)
    rot = pygame.transform.rotate(sp, angle)
    surf.blit(rot, blit_rotate_origin(S, S, pos, angle))
    return np.moveaxis(pygame.surfarray.array3d(surf), 0, 1)


def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def analytic(pos, heading, pivot_dx=0.0, pivot_dy=0.0):
    """Rects rotated about the sprite centre, sampled at pixel centres (i+0.5, j+0.5)."""
    yy, xx = np.mgrid[0:H, 0:W]
    px, py = xx + 0.5, yy + 0.5
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = GREY
    R = rot2(heading)
    cen = np.array([S / 2.0, S / 2.0])
    inset = (BODY[0] + 1, BODY[1] + 1, BODY[2] - 2, BODY[3] - 2)
    hl = _clip(HL, inset)
    hr = _clip(HR, inset)
    for r, c in ((BODY, BLACK), (inset, COL), (hl, LIGHT), (hr, LIGHT)):
        if r is None:
            continue
        rc = np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]) - cen
        world = np.array([pos[0] + pivot_dx, pos[1] + pivot_dy]) + R @ rc
        d = np.stack([px - world[0], py - world[1]], -1) @ R          # into rect frame
        m = (np.abs(d[..., 0]) <= r[2] / 2.0) & (np.abs(d[..., 1]) <= r[3] / 2.0)
        img[m] = c
    return img


def _clip(r, box):
    x0 = max(r[0], box[0]); y0 = max(r[1], box[1])
    x1 = min(r[0] + r[2], box[0] + box[2]); y1 = min(r[1] + r[3], box[1] + box[3])
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1 - x0, y1 - y0)


def interior(f):
    import scipy.ndimage as ndi
    g = f.astype(np.int16)
    flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
            + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
    return ndi.minimum_filter(flat, size=3)


if __name__ == "__main__":
    print(f"S={S} BODY={BODY} HL={HL} HR={HR}")
    rows = []
    for heading in np.linspace(-np.pi, np.pi, 33):
        pos = (110, 100)
        ref = reference(pos, heading)
        best = None
        for dx in (0.0, -0.5, 0.5, -1.0):
            for dy in (0.0, -0.5, 0.5, -1.0):
                ours = analytic(pos, heading, dx, dy)
                d = np.abs(ours.astype(np.int16) - ref.astype(np.int16)).max(-1)
                m = interior(ref)
                ie = (d[:-1, :-1][m] == 0).mean()
                key = (ie, (d == 0).mean())
                if best is None or key > best[0]:
                    best = (key, dx, dy)
        (ie, ae), dx, dy = best
        rows.append((heading, ie, ae, dx, dy))
        print(f"  heading {heading:+.4f} rad  best pivot ({dx:+.1f},{dy:+.1f})  "
              f"interior_exact {ie*100:6.2f}%  all_exact {ae*100:6.2f}%")
    ie = np.array([r[1] for r in rows])
    print(f"\nmin interior_exact {ie.min()*100:.2f}%  mean {ie.mean()*100:.2f}%")
