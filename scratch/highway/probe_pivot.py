"""Recover the exact sprite->screen affine map of pygame's rotate + truncated blit.

probe_rotate.py established that the raster rotation is exactly an analytic rotated rectangle: the
kernel maps a destination pixel to ``floor`` of an affine function and reads that source pixel, and
"floor(u) == n" is a half-open box test in the rotated frame. What is left is the affine map's
translation, which passes through pygame's fixed-point setup and a blit origin that Rect truncates to
an integer. Measured by lighting one source pixel at a time and finding where it lands.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np
import pygame

pygame.display.init()
GREY = (100, 100, 100)
W = H = 260
POS = (130, 130)


def blit_rotate_origin(w, h, pos, angle):
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    br = [p.rotate(angle) for p in box]
    min_box = (min(br, key=lambda p: p[0])[0], min(br, key=lambda p: p[1])[1])
    max_box = (max(br, key=lambda p: p[0])[0], max(br, key=lambda p: p[1])[1])
    pv = pygame.math.Vector2(w / 2, -h / 2)
    pm = pv.rotate(angle) - pv
    return (pos[0] - w / 2 + min_box[0] - pm[0], pos[1] - h / 2 - max_box[1] + pm[1])


def land(S, a, b, angle):
    """Screen-coordinate centroid of source pixel (a, b) after rotate+blit. None if it vanished."""
    sp = pygame.Surface((S, S), pygame.SRCALPHA)
    sp.set_at((a, b), (255, 0, 0, 255))
    rot = pygame.transform.rotate(sp, angle)
    surf = pygame.Surface((W, H))
    surf.fill(GREY)
    surf.blit(rot, blit_rotate_origin(S, S, POS, angle))
    arr = np.moveaxis(pygame.surfarray.array3d(surf), 0, 1)
    m = (arr[..., 0] == 255) & (arr[..., 1] == 0)
    if not m.any():
        return None, 0
    ys, xs = np.nonzero(m)
    return (xs.mean() + 0.5, ys.mean() + 0.5), int(m.sum())


def fit(S, heading):
    """Least-squares fit of  screen = pos + offset + R(heading) @ (sprite_coord - S/2)."""
    angle = np.rad2deg(-heading)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    src, dst = [], []
    for a in range(2, S - 2, 3):
        for b in range(2, S - 2, 3):
            p, n = land(S, a, b, angle)
            if p is None or n != 1:            # ignore pixels the NN map dropped or duplicated
                continue
            src.append([a + 0.5, b + 0.5])
            dst.append(list(p))
    if len(src) < 4:
        return None
    src = np.array(src)
    dst = np.array(dst)
    pred = np.array(POS) + (R @ (src - S / 2.0).T).T
    off = dst - pred
    return off.mean(0), off.std(0), len(src)


if __name__ == "__main__":
    for S in (38, 14):
        print(f"--- sprite {S}x{S} ---")
        for heading in np.linspace(-np.pi, np.pi, 25):
            r = fit(S, heading)
            if r is None:
                print(f"  h={heading:+.4f}  too few surviving pixels")
                continue
            off, sd, n = r
            org = blit_rotate_origin(S, S, POS, np.rad2deg(-heading))
            rot = pygame.transform.rotate(pygame.Surface((S, S), pygame.SRCALPHA), np.rad2deg(-heading))
            print(f"  h={heading:+.4f} n={n:4d} offset=({off[0]:+.4f},{off[1]:+.4f}) "
                  f"sd=({sd[0]:.4f},{sd[1]:.4f}) rot={rot.get_size()} "
                  f"orgfrac=({org[0]%1:.4f},{org[1]%1:.4f})")
