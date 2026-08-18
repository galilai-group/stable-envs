"""Where exactly does ``pygame.transform.rotate`` leave its axis-aligned fast path?

``_pivot`` has two branches and they disagree by up to a pixel: an angle that is an exact multiple of
90 degrees offsets by ``-0.5*(S mod 2)``, anything else by the ``Wr``-parity rule. parking-v0 has road
objects at exactly ``np.pi/2`` (a class constant, so exactly 90 degrees) and a goal landmark at
``+-pi/2`` read out of a float32 STATE, which is 90.0000025 degrees. If pygame's fast path has any
tolerance at all, the second of those still takes it and the ``Wr`` rule is the wrong branch for it.

Answered by rotating the same surface at both angles and diffing, then by finding the offset that
reproduces the reference at a sweep of near-90 angles.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()


def sprite(S, rect, colour=(255, 0, 0)):
    s = pygame.Surface((S, S), pygame.SRCALPHA)
    pygame.draw.rect(s, colour, rect, 0)
    return s


def blit_rotate(surf, image, pos, angle):
    """``VehicleGraphics.blit_rotate``, verbatim."""
    w, h = image.get_size()
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
    origin_pos = w / 2, h / 2
    pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
    pivot_move = pivot.rotate(angle) - pivot
    origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0],
              pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
    surf.blit(pygame.transform.rotate(image, angle), origin)


W = H = 120
for S, rect, label in ((14, (0, 0, 14, 14), "landmark 2x2 m at scaling 7"),
                       (49, (7, 17, 35, 14), "vehicle at scaling 7"),
                       (38, (5, 13, 27, 11), "vehicle at scaling 5.5")):
    print(f"\n{label}: S={S} rect={rect}")
    base = sprite(S, rect)
    for deg in (-90.0, -90.0000025, -89.9999975, -90.001, 90.0, 90.0000025):
        r = pygame.transform.rotate(base, deg)
        r0 = pygame.transform.rotate(base, -90.0)
        same = (r.get_size() == r0.get_size()
                and np.array_equal(pygame.surfarray.pixels3d(r), pygame.surfarray.pixels3d(r0)))
        print(f"   rotate({deg!r}) -> size {r.get_size()}, identical to rotate(-90): {same}")

    # the offset that reproduces the reference, per angle: read the drawn rect's bbox and compare
    # against the analytic rectangle's
    for deg in (-90.0, -90.0000025, -89.999, -45.0, -1.0):
        surf = pygame.Surface((W, H))
        surf.fill((0, 0, 0))
        blit_rotate(surf, base, (60, 60), deg)
        a = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))
        ys, xs = np.where((a != 0).any(-1))
        cx = (xs.min() + xs.max() + 1) / 2.0
        cy = (ys.min() + ys.max() + 1) / 2.0
        rad = np.deg2rad(-deg)
        wr = int(S * (abs(np.cos(rad)) + abs(np.sin(rad))))
        axis = np.fmod(deg, 90.0) == 0
        pred = (-0.5 * (S % 2), -0.5 * (S % 2)) if axis else (
            (-0.5, -0.5) if wr % 2 == 1 else (-1.0, -0.5))
        # the rect's own centre inside the sprite, rotated, is what the bbox centre should equal
        off = np.array([rect[0] + rect[2] / 2 - S / 2, rect[1] + rect[3] / 2 - S / 2])
        c, s_ = np.cos(rad), np.sin(rad)
        rot = np.array([[c, -s_], [s_, c]]) @ off
        meas = (cx - 60 - rot[0], cy - 60 - rot[1])
        print(f"   deg {deg!r:>12}: Wr={wr} axis={axis} predicted pivot {pred} "
              f"measured {tuple(round(float(v), 3) for v in meas)}")
