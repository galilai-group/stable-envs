"""Batched SDL_gfx-style integer rasterization with Torch.

Compute geometry in float64 before truncating it, because float32 rounding can move a whole pixel row.
"""

from __future__ import annotations

import math

import numpy as np
import torch

#: Mask for the unsigned 32-bit arithmetic of SDL_gfx blending.
M32 = (1 << 32) - 1
#: Weight that selects an opaque ``pixelColor`` write instead of ``pixelColorWeight``.
SOLID = 256
#: Channel shifts of a 32-bit ``pygame.Surface`` with masks (0xFF0000, 0xFF00, 0xFF).
SHIFT = (16, 8, 0)

#: Circle fringe entries are (dy, dx, weight, pass); weight 256 is opaque.
#: Preserve pass order where the same pixel receives more than one blend.
AACIRCLE_FRINGE: dict[int, tuple[tuple[int, int, int, int], ...]] = {
    1: (
        (-1, -1, 2, 1),
        (-1, -1, 255, 0),
        (-1, 1, 2, 1),
        (-1, 1, 255, 0),
        (1, -1, 2, 1),
        (1, -1, 255, 0),
        (1, 1, 2, 1),
        (1, 1, 255, 0),
    ),
    2: ((-2, -1, 192, 0), (-2, 1, 192, 0), (2, -1, 192, 0), (2, 1, 192, 0)),
    3: (
        (-3, -2, 63, 0),
        (-3, -1, 213, 0),
        (-3, 1, 213, 0),
        (-3, 2, 63, 0),
        (3, -2, 63, 0),
        (3, -1, 213, 0),
        (3, 1, 213, 0),
        (3, 2, 63, 0),
    ),
    4: (
        (-4, -2, 127, 0),
        (-4, -1, 224, 0),
        (-4, 1, 224, 0),
        (-4, 2, 127, 0),
        (-3, -3, 170, 0),
        (-3, 3, 170, 0),
        (-2, -4, 127, 0),
        (-2, 4, 127, 0),
        (2, -4, 127, 0),
        (2, 4, 127, 0),
        (3, -3, 170, 0),
        (3, 3, 170, 0),
        (4, -2, 127, 0),
        (4, -1, 224, 0),
        (4, 1, 224, 0),
        (4, 2, 127, 0),
    ),
    5: (
        (-5, -2, 153, 0),
        (-5, -1, 230, 0),
        (-5, 1, 230, 0),
        (-5, 2, 153, 0),
        (5, -2, 153, 0),
        (5, -1, 230, 0),
        (5, 1, 230, 0),
        (5, 2, 153, 0),
    ),
    6: (
        (-6, -3, 51, 0),
        (-6, -2, 170, 0),
        (-6, -1, 234, 0),
        (-6, 1, 234, 0),
        (-6, 2, 170, 0),
        (-6, 3, 51, 0),
        (-5, -4, 127, 0),
        (-5, 4, 127, 0),
        (-3, -6, 51, 0),
        (-3, 6, 51, 0),
        (3, -6, 51, 0),
        (3, 6, 51, 0),
        (5, -4, 127, 0),
        (5, 4, 127, 0),
        (6, -3, 51, 0),
        (6, -2, 170, 0),
        (6, -1, 234, 0),
        (6, 1, 234, 0),
        (6, 2, 170, 0),
        (6, 3, 51, 0),
    ),
    7: (
        (-7, -3, 85, 0),
        (-7, -2, 183, 0),
        (-7, -1, 237, 0),
        (-7, 1, 237, 0),
        (-7, 2, 183, 0),
        (-7, 3, 85, 0),
        (-6, -4, 192, 0),
        (-6, 4, 192, 0),
        (-3, -7, 85, 0),
        (-3, 7, 85, 0),
        (3, -7, 85, 0),
        (3, 7, 85, 0),
        (6, -4, 192, 0),
        (6, 4, 192, 0),
        (7, -3, 85, 0),
        (7, -2, 183, 0),
        (7, -1, 237, 0),
        (7, 1, 237, 0),
        (7, 2, 183, 0),
        (7, 3, 85, 0),
    ),
    8: (
        (-8, -3, 109, 0),
        (-8, -2, 192, 0),
        (-8, -1, 240, 0),
        (-8, 1, 240, 0),
        (-8, 2, 192, 0),
        (-8, 3, 109, 0),
        (-7, -5, 63, 0),
        (-7, -4, 237, 0),
        (-7, 4, 237, 0),
        (-7, 5, 63, 0),
        (-6, -6, 76, 0),
        (-6, 6, 76, 0),
        (-3, -8, 109, 0),
        (-3, 8, 109, 0),
        (3, -8, 109, 0),
        (3, 8, 109, 0),
        (6, -6, 76, 0),
        (6, 6, 76, 0),
        (7, -5, 63, 0),
        (7, -4, 237, 0),
        (7, 4, 237, 0),
        (7, 5, 63, 0),
        (8, -3, 109, 0),
        (8, -2, 192, 0),
        (8, -1, 240, 0),
        (8, 1, 240, 0),
        (8, 2, 192, 0),
        (8, 3, 109, 0),
    ),
    9: (
        (-9, -4, 15, 0),
        (-9, -3, 127, 0),
        (-9, -2, 199, 0),
        (-9, -1, 241, 0),
        (-9, 1, 241, 0),
        (-9, 2, 199, 0),
        (-9, 3, 127, 0),
        (-9, 4, 15, 0),
        (-8, -5, 127, 0),
        (-8, -4, 240, 0),
        (-8, 4, 240, 0),
        (-8, 5, 127, 0),
        (-7, -6, 183, 0),
        (-7, 6, 183, 0),
        (-5, -8, 127, 0),
        (-5, 8, 127, 0),
        (-4, -9, 15, 0),
        (-4, 9, 15, 0),
        (-3, -9, 127, 0),
        (-3, 9, 127, 0),
        (3, -9, 127, 0),
        (3, 9, 127, 0),
        (4, -9, 15, 0),
        (4, 9, 15, 0),
        (5, -8, 127, 0),
        (5, 8, 127, 0),
        (7, -6, 183, 0),
        (7, 6, 183, 0),
        (8, -5, 127, 0),
        (8, -4, 240, 0),
        (8, 4, 240, 0),
        (8, 5, 127, 0),
        (9, -4, 15, 0),
        (9, -3, 127, 0),
        (9, -2, 199, 0),
        (9, -1, 241, 0),
        (9, 1, 241, 0),
        (9, 2, 199, 0),
        (9, 3, 127, 0),
        (9, 4, 15, 0),
    ),
    10: (
        (-10, -4, 42, 0),
        (-10, -3, 141, 0),
        (-10, -2, 204, 0),
        (-10, -1, 243, 0),
        (-10, 1, 243, 0),
        (-10, 2, 204, 0),
        (-10, 3, 141, 0),
        (-10, 4, 42, 0),
        (-9, -5, 170, 0),
        (-9, 5, 170, 0),
        (-8, -7, 36, 0),
        (-8, 7, 36, 0),
        (-4, -10, 42, 0),
        (-4, 10, 42, 0),
        (4, -10, 42, 0),
        (4, 10, 42, 0),
        (8, -7, 36, 0),
        (8, 7, 36, 0),
        (9, -5, 170, 0),
        (9, 5, 170, 0),
        (10, -4, 42, 0),
        (10, -3, 141, 0),
        (10, -2, 204, 0),
        (10, -1, 243, 0),
        (10, 1, 243, 0),
        (10, 2, 204, 0),
        (10, 3, 141, 0),
        (10, 4, 42, 0),
    ),
    11: (
        (-11, -4, 63, 0),
        (-11, -3, 151, 0),
        (-11, -2, 209, 0),
        (-11, -1, 244, 0),
        (-11, 1, 244, 0),
        (-11, 2, 209, 0),
        (-11, 3, 151, 0),
        (-11, 4, 63, 0),
        (-10, -6, 56, 0),
        (-10, -5, 204, 0),
        (-10, 5, 204, 0),
        (-10, 6, 56, 0),
        (-9, -7, 127, 0),
        (-9, 7, 127, 0),
        (-8, -8, 144, 0),
        (-8, 8, 144, 0),
        (-7, -9, 127, 0),
        (-7, 9, 127, 0),
        (-6, -10, 56, 0),
        (-6, 10, 56, 0),
        (-4, -11, 63, 0),
        (-4, 11, 63, 0),
        (4, -11, 63, 0),
        (4, 11, 63, 0),
        (6, -10, 56, 0),
        (6, 10, 56, 0),
        (7, -9, 127, 0),
        (7, 9, 127, 0),
        (8, -8, 144, 0),
        (8, 8, 144, 0),
        (9, -7, 127, 0),
        (9, 7, 127, 0),
        (10, -6, 56, 0),
        (10, -5, 204, 0),
        (10, 5, 204, 0),
        (10, 6, 56, 0),
        (11, -4, 63, 0),
        (11, -3, 151, 0),
        (11, -2, 209, 0),
        (11, -1, 244, 0),
        (11, 1, 244, 0),
        (11, 2, 209, 0),
        (11, 3, 151, 0),
        (11, 4, 63, 0),
    ),
    12: (
        (-12, -4, 81, 0),
        (-12, -3, 160, 0),
        (-12, -2, 213, 0),
        (-12, -1, 245, 0),
        (-12, 1, 245, 0),
        (-12, 2, 213, 0),
        (-12, 3, 160, 0),
        (-12, 4, 81, 0),
        (-11, -6, 102, 0),
        (-11, -5, 232, 0),
        (-11, 5, 232, 0),
        (-11, 6, 102, 0),
        (-10, -7, 192, 0),
        (-10, 7, 192, 0),
        (-6, -11, 102, 0),
        (-6, 11, 102, 0),
        (-4, -12, 81, 0),
        (-4, 12, 81, 0),
        (4, -12, 81, 0),
        (4, 12, 81, 0),
        (6, -11, 102, 0),
        (6, 11, 102, 0),
        (10, -7, 192, 0),
        (10, 7, 192, 0),
        (11, -6, 102, 0),
        (11, -5, 232, 0),
        (11, 5, 232, 0),
        (11, 6, 102, 0),
        (12, -4, 81, 0),
        (12, -3, 160, 0),
        (12, -2, 213, 0),
        (12, -1, 245, 0),
        (12, 1, 245, 0),
        (12, 2, 213, 0),
        (12, 3, 160, 0),
        (12, 4, 81, 0),
    ),
    13: (
        (-13, -4, 95, 0),
        (-13, -3, 167, 0),
        (-13, -2, 216, 0),
        (-13, -1, 246, 0),
        (-13, 1, 246, 0),
        (-13, 2, 216, 0),
        (-13, 3, 167, 0),
        (-13, 4, 95, 0),
        (-12, -6, 139, 0),
        (-12, -5, 255, 0),
        (-12, 5, 255, 0),
        (-12, 6, 139, 0),
        (-11, -8, 63, 0),
        (-11, 8, 63, 0),
        (-10, -9, 99, 0),
        (-10, 9, 99, 0),
        (-8, -11, 63, 0),
        (-8, 11, 63, 0),
        (-4, -13, 95, 0),
        (-4, 13, 95, 0),
        (4, -13, 95, 0),
        (4, 13, 95, 0),
        (8, -11, 63, 0),
        (8, 11, 63, 0),
        (10, -9, 99, 0),
        (10, 9, 99, 0),
        (11, -8, 63, 0),
        (11, 8, 63, 0),
        (12, -6, 139, 0),
        (12, -5, 255, 0),
        (12, 5, 255, 0),
        (12, 6, 139, 0),
        (13, -4, 95, 0),
        (13, -3, 167, 0),
        (13, -2, 216, 0),
        (13, -1, 246, 0),
        (13, 1, 246, 0),
        (13, 2, 216, 0),
        (13, 3, 167, 0),
        (13, 4, 95, 0),
    ),
    14: (
        (-14, -5, 19, 0),
        (-14, -4, 107, 0),
        (-14, -3, 174, 0),
        (-14, -2, 219, 0),
        (-14, -1, 246, 0),
        (-14, 1, 246, 0),
        (-14, 2, 219, 0),
        (-14, 3, 174, 0),
        (-14, 4, 107, 0),
        (-14, 5, 19, 0),
        (-13, -7, 31, 0),
        (-13, -6, 167, 0),
        (-13, -5, 236, 0),
        (-13, 5, 236, 0),
        (-13, 6, 167, 0),
        (-13, 7, 31, 0),
        (-12, -8, 127, 0),
        (-12, 8, 127, 0),
        (-11, -9, 186, 0),
        (-11, 9, 186, 0),
        (-8, -12, 127, 0),
        (-8, 12, 127, 0),
        (-7, -13, 31, 0),
        (-7, 13, 31, 0),
        (-5, -14, 19, 0),
        (-5, 14, 19, 0),
        (-4, -14, 107, 0),
        (-4, 14, 107, 0),
        (4, -14, 107, 0),
        (4, 14, 107, 0),
        (5, -14, 19, 0),
        (5, 14, 19, 0),
        (7, -13, 31, 0),
        (7, 13, 31, 0),
        (8, -12, 127, 0),
        (8, 12, 127, 0),
        (11, -9, 186, 0),
        (11, 9, 186, 0),
        (12, -8, 127, 0),
        (12, 8, 127, 0),
        (13, -7, 31, 0),
        (13, -6, 167, 0),
        (13, -5, 236, 0),
        (13, 5, 236, 0),
        (13, 6, 167, 0),
        (13, 7, 31, 0),
        (14, -5, 19, 0),
        (14, -4, 107, 0),
        (14, -3, 174, 0),
        (14, -2, 219, 0),
        (14, -1, 246, 0),
        (14, 1, 246, 0),
        (14, 2, 219, 0),
        (14, 3, 174, 0),
        (14, 4, 107, 0),
        (14, 5, 19, 0),
    ),
    15: (
        (-15, -5, 36, 0),
        (-15, -4, 118, 0),
        (-15, -3, 179, 0),
        (-15, -2, 221, 0),
        (-15, -1, 247, 0),
        (-15, 1, 247, 0),
        (-15, 2, 221, 0),
        (-15, 3, 179, 0),
        (-15, 4, 118, 0),
        (-15, 5, 36, 0),
        (-14, -7, 68, 0),
        (-14, -6, 192, 0),
        (-14, -5, 219, 0),
        (-14, 5, 219, 0),
        (-14, 6, 192, 0),
        (-14, 7, 68, 0),
        (-13, -8, 177, 0),
        (-13, 8, 177, 0),
        (-12, -10, 46, 0),
        (-12, 10, 46, 0),
        (-11, -11, 51, 0),
        (-11, 11, 51, 0),
        (-7, -14, 68, 0),
        (-7, 14, 68, 0),
        (-5, -15, 36, 0),
        (-5, 15, 36, 0),
        (-4, -15, 118, 0),
        (-4, 15, 118, 0),
        (4, -15, 118, 0),
        (4, 15, 118, 0),
        (5, -15, 36, 0),
        (5, 15, 36, 0),
        (7, -14, 68, 0),
        (7, 14, 68, 0),
        (11, -11, 51, 0),
        (11, 11, 51, 0),
        (12, -10, 46, 0),
        (12, 10, 46, 0),
        (13, -8, 177, 0),
        (13, 8, 177, 0),
        (14, -7, 68, 0),
        (14, -6, 192, 0),
        (14, -5, 219, 0),
        (14, 5, 219, 0),
        (14, 6, 192, 0),
        (14, 7, 68, 0),
        (15, -5, 36, 0),
        (15, -4, 118, 0),
        (15, -3, 179, 0),
        (15, -2, 221, 0),
        (15, -1, 247, 0),
        (15, 1, 247, 0),
        (15, 2, 221, 0),
        (15, 3, 179, 0),
        (15, 4, 118, 0),
        (15, 5, 36, 0),
    ),
    16: (
        (-16, -5, 51, 0),
        (-16, -4, 127, 0),
        (-16, -3, 184, 0),
        (-16, -2, 224, 0),
        (-16, -1, 248, 0),
        (-16, 1, 248, 0),
        (-16, 2, 224, 0),
        (-16, 3, 184, 0),
        (-16, 4, 127, 0),
        (-16, 5, 51, 0),
        (-15, -7, 100, 0),
        (-15, -6, 213, 0),
        (-15, -5, 204, 0),
        (-15, 5, 204, 0),
        (-15, 6, 213, 0),
        (-15, 7, 100, 0),
        (-14, -9, 58, 0),
        (-14, -8, 219, 0),
        (-14, 8, 219, 0),
        (-14, 9, 58, 0),
        (-13, -10, 127, 0),
        (-13, 10, 127, 0),
        (-12, -11, 160, 0),
        (-12, 11, 160, 0),
        (-10, -13, 127, 0),
        (-10, 13, 127, 0),
        (-9, -14, 58, 0),
        (-9, 14, 58, 0),
        (-7, -15, 100, 0),
        (-7, 15, 100, 0),
        (-5, -16, 51, 0),
        (-5, 16, 51, 0),
        (-4, -16, 127, 0),
        (-4, 16, 127, 0),
        (4, -16, 127, 0),
        (4, 16, 127, 0),
        (5, -16, 51, 0),
        (5, 16, 51, 0),
        (7, -15, 100, 0),
        (7, 15, 100, 0),
        (9, -14, 58, 0),
        (9, 14, 58, 0),
        (10, -13, 127, 0),
        (10, 13, 127, 0),
        (12, -11, 160, 0),
        (12, 11, 160, 0),
        (13, -10, 127, 0),
        (13, 10, 127, 0),
        (14, -9, 58, 0),
        (14, -8, 219, 0),
        (14, 8, 219, 0),
        (14, 9, 58, 0),
        (15, -7, 100, 0),
        (15, -6, 213, 0),
        (15, -5, 204, 0),
        (15, 5, 204, 0),
        (15, 6, 213, 0),
        (15, 7, 100, 0),
        (16, -5, 51, 0),
        (16, -4, 127, 0),
        (16, -3, 184, 0),
        (16, -2, 224, 0),
        (16, -1, 248, 0),
        (16, 1, 248, 0),
        (16, 2, 224, 0),
        (16, 3, 184, 0),
        (16, 4, 127, 0),
        (16, 5, 51, 0),
    ),
    17: (
        (-17, -5, 63, 0),
        (-17, -4, 135, 0),
        (-17, -3, 188, 0),
        (-17, -2, 225, 0),
        (-17, -1, 248, 0),
        (-17, 1, 248, 0),
        (-17, 2, 225, 0),
        (-17, 3, 188, 0),
        (-17, 4, 135, 0),
        (-17, 5, 63, 0),
        (-16, -7, 127, 0),
        (-16, -6, 232, 0),
        (-16, 6, 232, 0),
        (-16, 7, 127, 0),
        (-15, -9, 109, 0),
        (-15, -8, 255, 0),
        (-15, 8, 255, 0),
        (-15, 9, 109, 0),
        (-14, -10, 192, 0),
        (-14, 10, 192, 0),
        (-13, -12, 10, 0),
        (-13, 12, 10, 0),
        (-9, -15, 109, 0),
        (-9, 15, 109, 0),
        (-7, -16, 127, 0),
        (-7, 16, 127, 0),
        (-5, -17, 63, 0),
        (-5, 17, 63, 0),
        (5, -17, 63, 0),
        (5, 17, 63, 0),
        (7, -16, 127, 0),
        (7, 16, 127, 0),
        (9, -15, 109, 0),
        (9, 15, 109, 0),
        (13, -12, 10, 0),
        (13, 12, 10, 0),
        (14, -10, 192, 0),
        (14, 10, 192, 0),
        (15, -9, 109, 0),
        (15, -8, 255, 0),
        (15, 8, 255, 0),
        (15, 9, 109, 0),
        (16, -7, 127, 0),
        (16, -6, 232, 0),
        (16, 6, 232, 0),
        (16, 7, 127, 0),
        (17, -5, 63, 0),
        (17, -4, 135, 0),
        (17, -3, 188, 0),
        (17, -2, 225, 0),
        (17, -1, 248, 0),
        (17, 1, 248, 0),
        (17, 2, 225, 0),
        (17, 3, 188, 0),
        (17, 4, 135, 0),
        (17, 5, 63, 0),
    ),
    18: (
        (-18, -5, 75, 0),
        (-18, -4, 142, 0),
        (-18, -3, 192, 0),
        (-18, -2, 227, 0),
        (-18, -1, 248, 0),
        (-18, 1, 248, 0),
        (-18, 2, 227, 0),
        (-18, 3, 192, 0),
        (-18, 4, 142, 0),
        (-18, 5, 75, 0),
        (-17, -8, 31, 0),
        (-17, -7, 150, 0),
        (-17, -6, 248, 0),
        (-17, 6, 248, 0),
        (-17, 7, 150, 0),
        (-17, 8, 31, 0),
        (-16, -9, 152, 0),
        (-16, 9, 152, 0),
        (-15, -11, 63, 0),
        (-15, 11, 63, 0),
        (-14, -12, 107, 0),
        (-14, 12, 107, 0),
        (-13, -13, 116, 0),
        (-13, 13, 116, 0),
        (-11, -15, 63, 0),
        (-11, 15, 63, 0),
        (-8, -17, 31, 0),
        (-8, 17, 31, 0),
        (-5, -18, 75, 0),
        (-5, 18, 75, 0),
        (5, -18, 75, 0),
        (5, 18, 75, 0),
        (8, -17, 31, 0),
        (8, 17, 31, 0),
        (11, -15, 63, 0),
        (11, 15, 63, 0),
        (13, -13, 116, 0),
        (13, 13, 116, 0),
        (14, -12, 107, 0),
        (14, 12, 107, 0),
        (15, -11, 63, 0),
        (15, 11, 63, 0),
        (16, -9, 152, 0),
        (16, 9, 152, 0),
        (17, -8, 31, 0),
        (17, -7, 150, 0),
        (17, -6, 248, 0),
        (17, 6, 248, 0),
        (17, 7, 150, 0),
        (17, 8, 31, 0),
        (18, -5, 75, 0),
        (18, -4, 142, 0),
        (18, -3, 192, 0),
        (18, -2, 227, 0),
        (18, -1, 248, 0),
        (18, 1, 248, 0),
        (18, 2, 227, 0),
        (18, 3, 192, 0),
        (18, 4, 142, 0),
        (18, 5, 75, 0),
    ),
    19: (
        (-19, -6, 7, 0),
        (-19, -5, 85, 0),
        (-19, -4, 148, 0),
        (-19, -3, 195, 0),
        (-19, -2, 229, 0),
        (-19, -1, 249, 0),
        (-19, 1, 249, 0),
        (-19, 2, 229, 0),
        (-19, 3, 195, 0),
        (-19, 4, 148, 0),
        (-19, 5, 85, 0),
        (-19, 6, 7, 0),
        (-18, -8, 60, 0),
        (-18, -7, 170, 0),
        (-18, -6, 248, 0),
        (-18, 6, 248, 0),
        (-18, 7, 170, 0),
        (-18, 8, 60, 0),
        (-17, -10, 39, 0),
        (-17, -9, 188, 0),
        (-17, 9, 188, 0),
        (-17, 10, 39, 0),
        (-16, -11, 127, 0),
        (-16, 11, 127, 0),
        (-15, -12, 187, 0),
        (-15, 12, 187, 0),
        (-11, -16, 127, 0),
        (-11, 16, 127, 0),
        (-10, -17, 39, 0),
        (-10, 17, 39, 0),
        (-8, -18, 60, 0),
        (-8, 18, 60, 0),
        (-6, -19, 7, 0),
        (-6, 19, 7, 0),
        (-5, -19, 85, 0),
        (-5, 19, 85, 0),
        (5, -19, 85, 0),
        (5, 19, 85, 0),
        (6, -19, 7, 0),
        (6, 19, 7, 0),
        (8, -18, 60, 0),
        (8, 18, 60, 0),
        (10, -17, 39, 0),
        (10, 17, 39, 0),
        (11, -16, 127, 0),
        (11, 16, 127, 0),
        (15, -12, 187, 0),
        (15, 12, 187, 0),
        (16, -11, 127, 0),
        (16, 11, 127, 0),
        (17, -10, 39, 0),
        (17, -9, 188, 0),
        (17, 9, 188, 0),
        (17, 10, 39, 0),
        (18, -8, 60, 0),
        (18, -7, 170, 0),
        (18, -6, 248, 0),
        (18, 6, 248, 0),
        (18, 7, 170, 0),
        (18, 8, 60, 0),
        (19, -6, 7, 0),
        (19, -5, 85, 0),
        (19, -4, 148, 0),
        (19, -3, 195, 0),
        (19, -2, 229, 0),
        (19, -1, 249, 0),
        (19, 1, 249, 0),
        (19, 2, 229, 0),
        (19, 3, 195, 0),
        (19, 4, 148, 0),
        (19, 5, 85, 0),
        (19, 6, 7, 0),
    ),
    20: (
        (-20, -6, 20, 0),
        (-20, -5, 93, 0),
        (-20, -4, 153, 0),
        (-20, -3, 198, 0),
        (-20, -2, 230, 0),
        (-20, -1, 249, 0),
        (-20, 1, 249, 0),
        (-20, 2, 230, 0),
        (-20, 3, 198, 0),
        (-20, 4, 153, 0),
        (-20, 5, 93, 0),
        (-20, 6, 20, 0),
        (-19, -8, 85, 0),
        (-19, -7, 188, 0),
        (-19, -6, 235, 0),
        (-19, 6, 235, 0),
        (-19, 7, 188, 0),
        (-19, 8, 85, 0),
        (-18, -10, 82, 0),
        (-18, -9, 220, 0),
        (-18, 9, 220, 0),
        (-18, 10, 82, 0),
        (-17, -11, 180, 0),
        (-17, 11, 180, 0),
        (-16, -13, 51, 0),
        (-16, 13, 51, 0),
        (-15, -14, 72, 0),
        (-15, 14, 72, 0),
        (-13, -16, 51, 0),
        (-13, 16, 51, 0),
        (-10, -18, 82, 0),
        (-10, 18, 82, 0),
        (-8, -19, 85, 0),
        (-8, 19, 85, 0),
        (-6, -20, 20, 0),
        (-6, 20, 20, 0),
        (-5, -20, 93, 0),
        (-5, 20, 93, 0),
        (5, -20, 93, 0),
        (5, 20, 93, 0),
        (6, -20, 20, 0),
        (6, 20, 20, 0),
        (8, -19, 85, 0),
        (8, 19, 85, 0),
        (10, -18, 82, 0),
        (10, 18, 82, 0),
        (13, -16, 51, 0),
        (13, 16, 51, 0),
        (15, -14, 72, 0),
        (15, 14, 72, 0),
        (16, -13, 51, 0),
        (16, 13, 51, 0),
        (17, -11, 180, 0),
        (17, 11, 180, 0),
        (18, -10, 82, 0),
        (18, -9, 220, 0),
        (18, 9, 220, 0),
        (18, 10, 82, 0),
        (19, -8, 85, 0),
        (19, -7, 188, 0),
        (19, -6, 235, 0),
        (19, 6, 235, 0),
        (19, 7, 188, 0),
        (19, 8, 85, 0),
        (20, -6, 20, 0),
        (20, -5, 93, 0),
        (20, -4, 153, 0),
        (20, -3, 198, 0),
        (20, -2, 230, 0),
        (20, -1, 249, 0),
        (20, 1, 249, 0),
        (20, 2, 230, 0),
        (20, 3, 198, 0),
        (20, 4, 153, 0),
        (20, 5, 93, 0),
        (20, 6, 20, 0),
    ),
}


def circle_hw(r: int) -> list[int]:
    """Return filled-circle row half-widths indexed by absolute vertical offset."""
    hw = [-1] * (r + 1)

    def span(dy, half):
        if half > hw[dy]:
            hw[dy] = half

    cx, cy = 0, r
    ocx = ocy = -1
    df, d_e, d_se = 1 - r, 3, -2 * r + 5
    while True:
        if ocy != cy:
            span(cy, cx)
            ocy = cy
        if ocx != cx:
            if cx != cy:
                span(cx, cy)
            ocx = cx
        if df < 0:
            df, d_e, d_se = df + d_e, d_e + 2, d_se + 2
        else:
            df, d_e, d_se = df + d_se, d_e + 2, d_se + 4
            cy -= 1
        cx += 1
        if cx > cy:
            return hw


def _round16(v: torch.Tensor) -> torch.Tensor:
    """Round a signed 16.16 fixed-point coordinate using SDL_gfx's half-bit rule."""
    return (v >> 16) + ((v & 32768) >> 15)


def polygon_spans(vx: torch.Tensor, vy: torch.Tensor, ys: torch.Tensor, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(B, len(ys))`` inclusive scanline spans ``xa``, ``xb`` of convex integer polygons.

    ``vx`` and ``vy`` are truncated ``(B, n)`` int64 vertices; a row without a span has ``xa > xb``.
    """
    ay, by = vy.roll(1, 1), vy
    ax, bx = vx.roll(1, 1), vx
    swap = ay > by
    ylo = torch.where(swap, by, ay).unsqueeze(-1)  # (B,n,1)
    yhi = torch.where(swap, ay, by).unsqueeze(-1)
    xlo = torch.where(swap, bx, ax).unsqueeze(-1)
    xhi = torch.where(swap, ax, bx).unsqueeze(-1)
    live = (ay != by).unsqueeze(-1)
    maxy = vy.max(1, keepdim=True).values.unsqueeze(-1)  # (B,1,1)

    y = ys.view(1, 1, -1)
    act = live & (((y >= ylo) & (y < yhi)) | ((y == maxy) & (y > ylo) & (y <= yhi)))
    den = (yhi - ylo).clamp_min(1)
    X = torch.div(65536 * (y - ylo), den, rounding_mode="floor") * (xhi - xlo) + 65536 * xlo
    BIG = 1 << 40
    xa = _round16(torch.where(act, X, BIG).amin(1) + 1).clamp_min(0)  # (B,H)
    xb = _round16(torch.where(act, X, -BIG).amax(1) - 1).clamp_max(W - 1)
    # Fewer than two crossings means an empty span; test per row rather than per pixel.
    xa = torch.where(act.sum(1) >= 2, xa, torch.full_like(xa, W))
    return xa, xb


def fill_polygon_mask(vx: torch.Tensor, vy: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor, W: int) -> torch.Tensor:
    """Return a ``(B, H, W)`` scanline mask for convex integer polygons; see ``polygon_spans``.

    ``xs`` selects columns.
    """
    xa, xb = polygon_spans(vx, vy, ys, W)
    x = xs.view(1, 1, -1)
    return (x >= xa.unsqueeze(-1)) & (x <= xb.unsqueeze(-1))


def disc_offsets(r: int) -> list[tuple[int, int]]:
    """Enumerate filled-circle (dy, dx) pixels from the midpoint span table."""
    hw = circle_hw(int(r))
    out = []
    for dy in range(-int(r), int(r) + 1):
        h = hw[abs(dy)]
        for dx in range(-h, h + 1):
            out.append((dy, dx))
    return out


def fill_circle_mask(
    cx: torch.Tensor, cy: torch.Tensor, r: int, ys: torch.Tensor, xs: torch.Tensor, hw_t: torch.Tensor
) -> torch.Tensor:
    """Return a (B, H, W) circle mask for integer (B,) centers and a cached span table."""
    dy = (ys.view(1, -1) - cy.view(-1, 1)).abs()  # (B,H)
    h = torch.where(dy <= r, hw_t[dy.clamp(0, r)], torch.full_like(dy, -1))
    dx = (xs.view(1, 1, -1) - cx.view(-1, 1, 1)).abs()  # ``(B, 1, W)``
    return dx <= h.unsqueeze(-1)


def blend(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
    """Apply SDL_gfx packed-channel alpha blending with unsigned 32-bit wrapping.

    Weight 256 is an opaque write; weight 255 still blends.
    """
    a = (255 * w) >> 8
    S = src << sh
    D = dst << sh
    v = ((((S - D) & M32) * a) & M32) >> 8
    return torch.where(w == SOLID, src.expand_as(dst), ((D + v) & (255 << sh)) >> sh)


def blit_over(dst: torch.Tensor, src_rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Blend a per-pixel-alpha surface onto opaque RGB using SDL_BlitSurface arithmetic.

    This uses alpha directly; SDL_gfx's packed-channel pixel blend uses another rule.
    """
    return dst + (((src_rgb - dst) * alpha) >> 8)


def _clip_encode(x: torch.Tensor, y: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Return Cohen-Sutherland outcodes: left=1, right=2, bottom=4, top=8."""
    z = torch.zeros_like(x)
    c = torch.where(x < 0, z + 1, torch.where(x > W - 1, z + 2, z))
    return c | torch.where(y < 0, z + 8, torch.where(y > H - 1, z + 4, z))


def clip_line(
    x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor, W: int, H: int, iters: int = 6
) -> tuple[torch.Tensor, ...]:
    """Clip batched Sint16 endpoints as SDL_gfx does and return ``(draw, x1, y1, x2, y2)``."""
    draw = torch.zeros_like(x1, dtype=torch.bool)
    done = torch.zeros_like(draw)
    for _ in range(iters):
        c1 = _clip_encode(x1, y1, W, H)
        c2 = _clip_encode(x2, y2, W, H)
        acc = ((c1 | c2) == 0) & ~done
        draw = draw | acc
        done = done | acc | (((c1 & c2) != 0) & ~done)
        act = ~done
        sw = act & (c1 == 0)
        x1, x2 = torch.where(sw, x2, x1), torch.where(sw, x1, x2)
        y1, y2 = torch.where(sw, y2, y1), torch.where(sw, y1, y2)
        c1 = torch.where(sw, c2, c1)

        dxne = x2 != x1
        den = (x2 - x1).to(torch.float32)
        m = torch.where(dxne, (y2 - y1).to(torch.float32) / torch.where(dxne, den, den + 1), torch.ones_like(den))
        msafe = torch.where(m == 0, torch.ones_like(m), m)
        L = act & ((c1 & 1) != 0)
        Rt = act & ~L & ((c1 & 2) != 0)
        Bt = act & ~L & ~Rt & ((c1 & 4) != 0)
        Tp = act & ~L & ~Rt & ~Bt & ((c1 & 8) != 0)
        dyL = torch.trunc((0 - x1).to(torch.float32) * m).to(torch.int64)
        dyR = torch.trunc((W - 1 - x1).to(torch.float32) * m).to(torch.int64)
        dxB = torch.trunc((H - 1 - y1).to(torch.float32) / msafe).to(torch.int64)
        dxT = torch.trunc((0 - y1).to(torch.float32) / msafe).to(torch.int64)
        ny = torch.where(
            L,
            y1 + dyL,
            torch.where(
                Rt, y1 + dyR, torch.where(Bt, torch.full_like(y1, H - 1), torch.where(Tp, torch.zeros_like(y1), y1))
            ),
        )
        nx = torch.where(
            L,
            torch.zeros_like(x1),
            torch.where(
                Rt, torch.full_like(x1, W - 1), torch.where(Bt & dxne, x1 + dxB, torch.where(Tp & dxne, x1 + dxT, x1))
            ),
        )
        x1, y1 = nx, ny
    return draw, x1, y1, x2, y2


def select_(dst: torch.Tensor, mask: torch.Tensor, src: torch.Tensor) -> None:
    """Select broadcast source values into an integer destination in place."""
    if dst.device.type == "cuda":
        torch.where(mask, src, dst, out=dst)
    else:
        dst.bitwise_xor_((dst ^ src) * mask.view(torch.uint8))


class Canvas:
    """Planar ``(3, B, H, W)`` RGB canvas in y-down orientation; ``rgb`` returns the output frames."""

    def __init__(self, B, H, W, device, sh, ys, xs, bg=(255, 255, 255), gfx=None):
        self.B, self.H, self.W = B, H, W
        self.device = device
        self.flat = torch.empty((3, B * H * W + 1), dtype=torch.uint8, device=device)
        self.buf = self.flat[:, :-1].view(3, B, H, W)
        if bg is not None:
            for c in range(3):
                self.buf[c].fill_(int(bg[c]))
        self.ys, self.xs, self.sh = ys, xs, sh
        self.trash = B * H * W
        self.gfx = gfx

    def steps(self, kmax: int) -> torch.Tensor:
        """Return cached arange(kmax+1) as a row, or allocate for an uncached canvas."""
        if self.gfx is not None:
            return self.gfx.steps(kmax)
        return torch.arange(kmax + 1, device=self.device).view(1, -1)

    def solid(self) -> torch.Tensor:
        """Return one opaque-write weight; expand it to match each scatter's write count."""
        if self.gfx is not None:
            return self.gfx.solid
        return torch.tensor([SOLID], dtype=torch.int64, device=self.device)

    def color_u8(self, color) -> torch.Tensor:
        """Return a cached (3,) uint8 color literal; a bare canvas allocates it on demand."""
        if self.gfx is not None:
            return self.gfx.color_u8(color)
        return torch.tensor([int(v) for v in color], dtype=torch.uint8, device=self.device)

    def always(self) -> torch.Tensor:
        """Return one true validity value for expansion to a scatter's write count."""
        if self.gfx is not None:
            return self.gfx.always
        return torch.ones(1, dtype=torch.bool, device=self.device)

    def window(self, win):
        """Return the (y0, y1, x0, x1) canvas view, or the full canvas for None."""
        return self.buf if win is None else self.buf[:, :, win[0] : win[1], win[2] : win[3]]

    def rows(self, win):
        """Return row and column coordinates restricted to an optional pixel window."""
        if win is None:
            return self.ys, self.xs
        return self.ys[win[0] : win[1]], self.xs[win[2] : win[3]]

    def fill_from(self, base: torch.Tensor) -> None:
        """Copy a planar (3, 1, H, W) or (3, B, H, W) background layer."""
        self.buf.copy_(base)

    def paint(self, mask, color, window=None) -> None:
        """Fill masked pixels with an opaque color inside an optional window."""
        buf = self.window(window)
        for c in range(3):
            buf[c].masked_fill_(mask, int(color[c]))

    def _disc_index(self, cy, cx, off, gate=None, lanes=None):
        """Place disc stencil offsets in each frame, redirecting invalid indices to trash.

        ``lanes`` names the environment of each center when they are a subset of the batch.
        """
        B = cy.shape[0]
        y = cy.view(-1, 1) + off[:, 0].view(1, -1)
        x = cx.view(-1, 1) + off[:, 1].view(1, -1)
        ok = (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        if gate is not None:
            ok = ok & (gate if lanes is None else gate[lanes]).view(-1, 1)
        b = (torch.arange(B, device=self.device) if lanes is None else lanes).view(-1, 1)
        return torch.where(
            ok, (b * self.H + y.clamp(0, self.H - 1)) * self.W + x.clamp(0, self.W - 1), self.trash
        ).reshape(-1)

    def paint_disc(self, cy, cx, off, color) -> None:
        """Scatter the filled disc's distinct pixels at each environment's center.

        Only discarded writes may collide at the trash pixel.
        """
        idx = self._disc_index(cy, cx, off)
        n = off.shape[0] * cy.shape[0]
        self.flat[:, idx] = self.color_u8(color).view(3, 1).expand(3, n)

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
        """Blend color and weights at indexed pixels, discarding invalid or out-of-bounds writes."""
        ok = valid & (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        flat = torch.where(ok, (b * self.H + y.clamp(0, self.H - 1)) * self.W + x.clamp(0, self.W - 1), self.trash)
        dst = self.flat[:, flat].to(torch.int64)
        self.flat[:, flat] = blend(color_t, dst, w.reshape(1, -1), self.sh.view(3, 1)).to(torch.uint8)

    def rgb(self, out=None) -> torch.Tensor:
        """Return ``(B, H, W, 3)`` uint8 frames with the reference's final vertical flip, into ``out`` if given."""
        if self.device.type == "cpu":
            # Interleaving first and flipping in a row copy is faster on CPU than flipping each plane.
            planes = self.buf.numpy()
            frames = np.stack((planes[0], planes[1], planes[2]), -1)[:, ::-1]
            if out is None:
                return torch.from_numpy(np.ascontiguousarray(frames))
            np.copyto(out.numpy(), frames)
            return out
        if out is None:
            out = torch.empty((self.B, self.H, self.W, 3), dtype=torch.uint8, device=self.device)
        for c in range(3):
            out[..., c].copy_(self.buf[c].flip(1))
        return out


def to_u8(color: torch.Tensor) -> torch.Tensor:
    """Truncate style colors to uint8 using the reference's integer conversion rule."""
    return color.to(torch.uint8)


class StyleCanvas(Canvas):
    """Canvas with per-environment color, antialiasing and visibility gates.

    ``use`` selects ``(B, 3)`` colors and ``gated`` restricts the primitives that follow.
    """

    def __init__(self, gfx: Gfx, B: int, bg):
        super().__init__(B, gfx.H, gfx.W, gfx.device, gfx.sh, gfx.ys, gfx.xs, None, gfx=gfx)
        self.buf.copy_(to_u8(bg).t().view(3, B, 1, 1))
        self.col = None
        self.aa = None
        self.gate = None

    def use(self, color) -> StyleCanvas:
        """Set (B, 3) colors for subsequent primitives and return this canvas."""
        self.col = color
        return self

    def gated(self, gate) -> StyleCanvas:
        """Set a (B,) boolean visibility gate for subsequent primitives and return this canvas."""
        self.gate = gate
        return self

    def paint(self, mask, color=None, window=None) -> None:
        """Fill a window-sized mask using per-environment colors and the current visibility gate."""
        if self.gate is not None:
            mask = mask & self.gate.view(-1, 1, 1)
        c = to_u8(self.col if color is None else color).t().reshape(3, self.B, 1, 1)
        select_(self.window(window), mask.contiguous().unsqueeze(0), c)

    def paint_spans(self, xa, xb, window=None, color=None) -> None:
        """Fill the inclusive (B, rows) column spans of the window's rows with per-environment colors.

        Only the span pixels are written, so a thin polygon costs its area rather than its window.
        """
        y0, y1, x0, x1 = (0, self.H, 0, self.W) if window is None else window
        lo, hi = xa.clamp_min(x0), xb.clamp_max(x1 - 1)
        n = (hi - lo + 1).clamp_min(0)
        if self.gate is not None:
            n = n * self.gate.view(-1, 1)
        n = n.reshape(-1)
        b = torch.arange(self.B, device=self.device).view(-1, 1).expand(-1, y1 - y0).reshape(-1)
        y = torch.arange(y0, y1, device=self.device).view(1, -1).expand(self.B, -1).reshape(-1)
        start = (b * self.H + y) * self.W + lo.reshape(-1)
        count = int(n.sum())
        if not count:
            return
        first = torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        idx = torch.repeat_interleave(start, n) + torch.arange(count, device=self.device) - first
        c = to_u8(self.col if color is None else color)
        self.flat[:, idx] = c.t()[:, torch.repeat_interleave(b, n)]

    def paint_disc(self, cy, cx, off, color=None, lanes=None) -> None:
        B, S = cy.shape[0], off.shape[0]
        idx = self._disc_index(cy, cx, off, self.gate, lanes)
        c = to_u8(self.col if color is None else color)
        if lanes is not None:
            c = c[lanes]
        self.flat[:, idx] = c.t().reshape(3, B, 1).expand(3, B, S).reshape(3, -1)

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
        if self.aa is not None:
            valid = valid & self.aa[b]
        if self.gate is not None:
            valid = valid & self.gate[b]
        color = self.col.to(torch.int64).t()[:, b] if color_t is None else color_t
        super().blend_at(b, y, x, color, w, valid)


def edge_geometry(
    x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor, W: int, H: int
) -> tuple[torch.Tensor, ...]:
    """Clip a flat batch of edges and compute the constants of their Wu walks.

    Apply the edge writes in order, because shared vertices blend more than once.
    """
    draw, x1, y1, x2, y2 = clip_line(x1, y1, x2, y2, W, H)
    flip = y1 > y2
    xx0 = torch.where(flip, x2, x1)
    yy0 = torch.where(flip, y2, y1)
    dxs = torch.where(flip, x1, x2) - xx0
    dy = torch.where(flip, y1, y2) - yy0
    xdir = torch.where(dxs >= 0, 1, -1)
    dx = dxs.abs()
    vert = dx == 0
    horiz = (dy == 0) & ~vert
    ymaj = dy > dx
    major = torch.where(ymaj, dy, dx)
    minor = torch.where(ymaj, dx, dy)
    E = torch.div(minor << 16, major.clamp_min(1), rounding_mode="floor")
    lo = torch.minimum(xx0, xx0 + xdir * dx)
    return (draw, x1, y1, x2, y2, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo)


def edge_setup(
    x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor, W: int, H: int
) -> tuple[torch.Tensor, ...]:
    """Reorder edge_geometry outputs for aaline_emit, dropping the clipped far endpoint."""
    draw, sx, sy, _x2, _y2, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo = edge_geometry(
        x1, y1, x2, y2, W, H
    )
    return (draw, sx, sy, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo)


def aaline_emit(cv: Canvas, p, color_t, kmax, bidx) -> None:
    """Emit the ordered scatters for one clipped antialiased edge.

    ``kmax`` is a static step bound, so the walk needs no host synchronization.
    """
    draw, sx, sy, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo = p
    B = xx0.shape[0]
    # Keep one weight per write.
    # Expanding a cached scalar preserves that shape without allocating another buffer.
    solid = cv.solid()
    cv.blend_at(bidx, sy, sx, color_t, solid.expand(B), draw)

    k = cv.steps(kmax)
    n = B * (kmax + 1)
    run = solid.expand(n)
    bb = bidx.view(-1, 1).expand(B, kmax + 1).reshape(-1)
    dw = draw.view(-1, 1)
    # SDL_gfx draws axis-aligned edges as solid vlines/hlines without antialiasing.
    cv.blend_at(
        bb,
        (yy0.view(-1, 1) + k).reshape(-1),
        xx0.view(-1, 1).expand(B, kmax + 1).reshape(-1),
        color_t,
        run,
        (dw & vert.view(-1, 1) & (dy > 0).view(-1, 1) & (k <= dy.view(-1, 1))).reshape(-1),
    )
    cv.blend_at(
        bb,
        yy0.view(-1, 1).expand(B, kmax + 1).reshape(-1),
        (lo.view(-1, 1) + k).reshape(-1),
        color_t,
        run,
        (dw & horiz.view(-1, 1) & (k <= dx.view(-1, 1))).reshape(-1),
    )

    gen = ~vert & ~horiz
    kk = k[:, 1:]
    acc = kk * E.view(-1, 1)
    wraps = acc >> 16
    wgt = (acc & 0xFFFF) >> 8
    live = (dw & gen.view(-1, 1) & (kk < major.view(-1, 1))).reshape(-1)
    ym = ymaj.view(-1, 1)
    base_x, base_y, d = xx0.view(-1, 1), yy0.view(-1, 1), xdir.view(-1, 1)
    xa = torch.where(ym, base_x + d * wraps, base_x + d * kk)
    ya = torch.where(ym, base_y + kk, base_y + wraps)
    xb = torch.where(ym, base_x + d * (wraps + 1), xa)
    yb = torch.where(ym, ya, ya + 1)
    bk = bidx.view(-1, 1).expand_as(xa).reshape(-1)
    cv.blend_at(bk, ya.reshape(-1), xa.reshape(-1), color_t, (255 - wgt).reshape(-1), live)
    cv.blend_at(bk, yb.reshape(-1), xb.reshape(-1), color_t, wgt.reshape(-1), live)


def aaline(cv: Canvas, x1, y1, x2, y2, color_t, kmax, bidx) -> None:
    """Clip and draw an antialiased edge with a solid initial pixel and no final endpoint."""
    aaline_emit(cv, edge_setup(x1, y1, x2, y2, cv.W, cv.H), color_t, kmax, bidx)


def polygon(
    cv: Canvas, verts, color, color_t, bidx, kmax, aa: bool = True, fill_first: bool = False, window=None
) -> None:
    """Draw a convex polygon with an antialiased fringe and integer scanline fill.

    ``verts`` should be float64 and ``window`` must contain the polygon footprint. ``fill_first=True`` fills
    before the fringe, which changes edge blending.
    """
    vi = torch.trunc(verts).to(torch.int64)
    vx, vy = vi[..., 0], vi[..., 1]
    n = vx.shape[1]
    km = [kmax] * n if isinstance(kmax, int) else list(kmax)

    def _aa():
        # Clip all edges together, then blend them in order to preserve shared-vertex writes.
        nx, ny = vx.roll(-1, 1), vy.roll(-1, 1)
        pars = edge_setup(vx.t().reshape(-1), vy.t().reshape(-1), nx.t().reshape(-1), ny.t().reshape(-1), cv.W, cv.H)
        B = vx.shape[0]
        for e in range(n):
            aaline_emit(cv, tuple(t[e * B : (e + 1) * B] for t in pars), color_t, km[e], bidx)

    if aa and not fill_first:
        _aa()
    ys, xs = cv.rows(window)
    if isinstance(cv, StyleCanvas) and cv.device.type == "cpu":
        cv.paint_spans(*polygon_spans(vx, vy, ys, cv.W), window, color)
    else:
        cv.paint(fill_polygon_mask(vx, vy, ys, xs, cv.W), color, window)
    if aa and fill_first:
        _aa()


def circle(
    cv: Canvas, cx, cy, r, color, color_t, bidx, hw_t, stencil, aa: bool = True, window=None, disc=None, lanes=None
) -> None:
    """Draw ordered antialiased fringe passes, then the filled circle.

    ``disc`` is a filled-disc stencil (``hw_t`` and ``window`` are used otherwise). With ``lanes``, the
    centers belong to those environments only.
    """
    if aa and stencil is not None:
        B = cx.shape[0]
        for st in stencil:
            S = st.shape[0]
            if not S:
                continue
            bb = bidx.view(-1, 1).expand(B, S).reshape(-1)
            cv.blend_at(
                bb,
                (cy.view(-1, 1) + st[:, 0].view(1, -1)).reshape(-1),
                (cx.view(-1, 1) + st[:, 1].view(1, -1)).reshape(-1),
                color_t,
                st[:, 2].view(1, -1).expand(B, S).reshape(-1),
                cv.always().expand(B * S),
            )
    if lanes is not None:
        cv.paint_disc(cy, cx, disc, color, lanes)
    elif disc is not None:
        cv.paint_disc(cy, cx, disc, color)
    else:
        ys, xs = cv.rows(window)
        cv.paint(fill_circle_mask(cx, cy, r, ys, xs, hw_t), color, window)


#: pygame's default ``Vector2.epsilon``.
_VECTOR_EPSILON = 1e-6


def rotate(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Apply pygame's ``Vector2.rotate_rad`` arithmetic before the canvas's final vertical flip."""
    a = torch.fmod(a, 2 * math.pi)
    a = torch.where(a < 0, a + 2 * math.pi, a)
    quarter = torch.fmod(a + _VECTOR_EPSILON, math.pi / 2) < 2 * _VECTOR_EPSILON
    k = torch.div(a + _VECTOR_EPSILON, math.pi / 2, rounding_mode="trunc").to(torch.int64) % 4
    if a.device.type == "cpu":
        # Vectorized torch trig can differ from C libm by one ulp; there is one angle per lane, so use libm.
        flat = a.detach().reshape(-1).tolist()
        c = torch.tensor([math.cos(x) for x in flat], dtype=a.dtype).reshape(a.shape)
        s = torch.tensor([math.sin(x) for x in flat], dtype=a.dtype).reshape(a.shape)
    else:
        c, s = torch.cos(a), torch.sin(a)
    c = torch.where(quarter, torch.tensor([1.0, 0.0, -1.0, 0.0], dtype=c.dtype, device=c.device)[k], c)
    s = torch.where(quarter, torch.tensor([0.0, 1.0, 0.0, -1.0], dtype=s.dtype, device=s.device)[k], s)
    return torch.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1)


class Gfx:
    """Cache drawing constants for one canvas size and device.

    Warm it up before CUDA graph capture, because creating its tensors during capture is invalid.
    """

    def __init__(self, H: int, W: int, device):
        self.H, self.W = int(H), int(W)
        self.device = torch.device(device)
        self.ys = torch.arange(self.H, device=self.device)
        self.xs = torch.arange(self.W, device=self.device)
        self.sh = torch.tensor(SHIFT, dtype=torch.int64, device=self.device)
        self._hw: dict[int, torch.Tensor] = {}
        self._st: dict[int, list[torch.Tensor]] = {}
        self._disc: dict[int, torch.Tensor] = {}
        self._bi: dict[int, torch.Tensor] = {}
        self._k: dict[tuple, torch.Tensor] = {}
        self._steps: dict[int, torch.Tensor] = {}
        self.solid = torch.tensor([SOLID], dtype=torch.int64, device=self.device)
        self.always = torch.ones(1, dtype=torch.bool, device=self.device)

    def steps(self, kmax: int) -> torch.Tensor:
        k = int(kmax)
        if k not in self._steps:
            self._steps[k] = torch.arange(k + 1, device=self.device).view(1, -1)
        return self._steps[k]

    def batch_index(self, B: int) -> torch.Tensor:
        if B not in self._bi:
            self._bi[B] = torch.arange(B, device=self.device)
        return self._bi[B]

    def hw(self, r: int) -> torch.Tensor:
        if r not in self._hw:
            self._hw[r] = torch.tensor(circle_hw(int(r)), dtype=torch.int64, device=self.device)
        return self._hw[r]

    def disc(self, r: int) -> torch.Tensor:
        """Return cached (S, 2) int64 filled-disc offsets for radius r."""
        if r not in self._disc:
            self._disc[r] = torch.tensor(disc_offsets(int(r)), dtype=torch.int64, device=self.device).view(-1, 2)
        return self._disc[r]

    def stencil(self, r: int) -> list[torch.Tensor]:
        if r not in self._st:
            rows = AACIRCLE_FRINGE.get(int(r))
            if rows is None:
                raise KeyError(
                    f"no aacircle fringe table for radius {r}; regenerate AACIRCLE_FRINGE rather than approximating it"
                )
            self._st[r] = [
                torch.tensor(
                    [[a, b, w] for (a, b, w, p) in rows if p == pas], dtype=torch.int64, device=self.device
                ).view(-1, 3)
                for pas in (0, 1)
            ]
        return self._st[r]

    def verts(self, vals) -> torch.Tensor:
        """Cache float64 vertices so coordinate truncation retains reference precision."""
        key = ("v", tuple(map(tuple, vals)))
        if key not in self._k:
            self._k[key] = torch.tensor(vals, dtype=torch.float64, device=self.device)
        return self._k[key]

    def color_u8(self, c) -> torch.Tensor:
        """Return a cached (3,) uint8 color for opaque fills."""
        key = ("u8", tuple(int(v) for v in c))
        if key not in self._k:
            self._k[key] = torch.tensor(key[1], dtype=torch.uint8, device=self.device)
        return self._k[key]
