# The coefficient table and fixed-point arithmetic below reproduce OpenCV's generic INTER_LINEAR
# path for 8-bit images (modules/imgproc/src/resize.cpp, OpenCV 5.0.0), which is licensed under the
# Apache License, Version 2.0; see LICENSES/Apache-2.0.txt in this repository.
"""``cv2.resize(img, (n, n))`` (INTER_LINEAR, uint8) for batched torch tensors.

Matches OpenCV builds that take the generic fixed-point path, which includes the Linux x86-64
wheels (IPP). Builds that dispatch some sizes to an ARM HAL (KleidiCV, carotene) can differ by
a few levels at those sizes.
"""

from __future__ import annotations

import math

import numpy as np
import torch

_BITS = 11
_ONE = 1 << _BITS


def _axis(src, dst):
    scale = 1.0 / (dst / src)
    ofs = np.zeros(dst, np.int64)
    w = np.zeros((dst, 2), np.int64)
    lo, hi = 0, dst
    for d in range(dst):
        f = np.float32((d + 0.5) * scale - 0.5)
        s = math.floor(f)
        f = np.float32(f - np.float32(s))
        if s < 0:
            lo = d + 1
            f, s = np.float32(0), 0
        if s + 1 >= src:
            hi = min(hi, d)
            if s >= src - 1:
                f, s = np.float32(0), src - 1
        w[d] = [int(np.rint(np.float32(np.float32(1) - f) * np.float32(_ONE))), int(np.rint(f * np.float32(_ONE)))]
        ofs[d] = s
    return ofs, w, lo, hi


class LinearResize:
    """Square ``src`` -> ``dst`` resize of (B, src, src, C) uint8 tensors."""

    def __init__(self, src, dst, device):
        xo, xw, xlo, xhi = _axis(src, dst)
        yo, yw, _, _ = _axis(src, dst)
        inner = (np.arange(dst) >= xlo) & (np.arange(dst) < xhi)

        def t(a):
            return torch.as_tensor(a, device=device)

        self.x0, self.x1 = t(xo), t(np.minimum(xo + 1, src - 1))
        self.xw0, self.xw1 = t(np.where(inner, xw[:, 0], _ONE)), t(np.where(inner, xw[:, 1], 0))
        self.y0, self.y1 = t(yo), t(np.minimum(yo + 1, src - 1))
        self.yw0, self.yw1 = t(yw[:, 0]), t(yw[:, 1])

    def __call__(self, img, chunk=256):
        if len(img) > chunk:
            return torch.cat([self(img[i : i + chunk], chunk) for i in range(0, len(img), chunk)])
        x = img.to(torch.int32)
        h = x[:, :, self.x0] * self.xw0.view(1, 1, -1, 1) + x[:, :, self.x1] * self.xw1.view(1, 1, -1, 1)
        b0, b1 = self.yw0.view(1, -1, 1, 1), self.yw1.view(1, -1, 1, 1)
        v = ((b0 * (h[:, self.y0] >> 4)) >> 16) + ((b1 * (h[:, self.y1] >> 4)) >> 16)
        return ((v + 2) >> 2).clamp(0, 255).to(torch.uint8)
