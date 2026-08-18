"""Refine the pivot offset around the quarter-pixel rule, to see whether the residual is systematic.

probe_pivot2 found the offset on a 0.25 px grid; a handful of angles did not reach 100% there. If the
truth is a nearby non-quarter offset the residual is a calibration error worth fixing; if no offset
reaches 100% the residual is pygame's fixed-point sin/cos and cannot be fixed with a translation.
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
SC = 5.5
pix = lambda v: int(v * SC)
S = pix(7.0)
BODY = (pix(1.0), pix(2.5), pix(5.0), pix(2.0))
HL = (pix(5.28), pix(3.5 - 1.4 * 2 / 3), pix(0.72), pix(0.6))
HR = (pix(5.28), pix(3.5 + 0.6 * 2 / 5), pix(0.72), pix(0.6))
INSET = (BODY[0] + 1, BODY[1] + 1, BODY[2] - 2, BODY[3] - 2)


def _clip(r, b):
    x0, y0 = max(r[0], b[0]), max(r[1], b[1])
    x1, y1 = min(r[0] + r[2], b[0] + b[2]), min(r[1] + r[3], b[1] + b[3])
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1 - x0, y1 - y0)


RECTS = [(BODY, BLACK), (INSET, COL), (_clip(HL, INSET), LIGHT), (_clip(HR, INSET), LIGHT)]
SP = pygame.Surface((S, S), pygame.SRCALPHA)
pygame.draw.rect(SP, COL, BODY, 0)
pygame.draw.rect(SP, LIGHT, HL, 0)
pygame.draw.rect(SP, LIGHT, HR, 0)
pygame.draw.rect(SP, BLACK, BODY, 1)
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


def analytic(heading, dx, dy):
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    img = np.zeros((WIN, WIN, 3), np.uint8)
    img[:] = GREY
    for r, col in RECTS:
        rc = np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]) - S / 2.0
        w = np.array([POS[0] + dx, POS[1] + dy]) + R @ rc
        d = np.stack([PX - w[0], PY - w[1]], -1) @ R
        img[(np.abs(d[..., 0]) <= r[2] / 2.0) & (np.abs(d[..., 1]) <= r[3] / 2.0)] = col
    return img


def rule(heading):
    a = -heading
    wr = int(S * (abs(np.cos(a)) + abs(np.sin(a))))
    d = wr - S
    if d == 0:
        return 0.0, 0.0
    return (-0.5 if d % 2 == 1 else -1.0), -0.5


if __name__ == "__main__":
    worst = []
    for heading in np.linspace(-np.pi, np.pi, 145):
        angle = np.rad2deg(-heading)
        surf = pygame.Surface((WIN, WIN))
        surf.fill(GREY)
        surf.blit(pygame.transform.rotate(SP, angle), blit_origin(S, S, POS, angle))
        ref = np.moveaxis(pygame.surfarray.array3d(surf), 0, 1)
        rx, ry = rule(heading)
        base = int((np.abs(analytic(heading, rx, ry).astype(np.int16)
                           - ref.astype(np.int16)).max(-1) != 0).sum())
        best, bd = base, (rx, ry)
        for dx in np.arange(rx - 0.24, rx + 0.245, 0.04):
            for dy in np.arange(ry - 0.24, ry + 0.245, 0.04):
                n = int((np.abs(analytic(heading, dx, dy).astype(np.int16)
                                - ref.astype(np.int16)).max(-1) != 0).sum())
                if n < best:
                    best, bd = n, (dx, dy)
        worst.append((base, best))
        if base:
            print(f"  h={heading:+.4f} rule=({rx:+.2f},{ry:+.2f}) wrong={base:3d} "
                  f"best={best:3d} at ({bd[0]:+.3f},{bd[1]:+.3f})")
    b = np.array([w[0] for w in worst])
    r = np.array([w[1] for w in worst])
    print(f"\n{ (b==0).sum() }/{len(b)} angles exact under the rule; "
          f"total wrong px {b.sum()} (best achievable by any offset: {r.sum()})")
