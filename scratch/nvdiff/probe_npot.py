"""Does nvdiffrast mipmap a non-power-of-two texture, and what does resampling one to POT cost?

MuJoCo does NOT resample: a 300x300 texture painted with its own texel index shows 297 of its 300
distinct rows in a magnified frame, so the base level MuJoCo samples is the one in the model. If
nvdiffrast refuses non-power-of-two mipmapping, resampling is a residual this backend has to declare,
and the size of it is the difference between the resampled base level and the original, per texture.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, save  # noqa: E402


def main():
    import torch

    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext(device="cuda")
    rows = []
    for (h, w) in [(256, 256), (300, 300), (100, 100), (127, 762)]:
        tex = torch.rand(1, h, w, 3, device="cuda")
        uv = torch.rand(1, 64, 64, 2, device="cuda")
        uv_da = torch.full((1, 64, 64, 4), 0.02, device="cuda")
        row = {"h": h, "w": w}
        for mode in ("linear", "linear-mipmap-linear"):
            try:
                out = dr.texture(tex, uv, uv_da if "mipmap" in mode else None, filter_mode=mode)
                row[mode] = f"ok {tuple(out.shape)}"
            except Exception as e:                                    # noqa: BLE001
                row[mode] = f"{type(e).__name__}: {str(e)[:120]}"
        rows.append(row)
        emit("npot", row)
    # cost of the POT resample on the textures this study actually uses
    del ctx
    save("npot.json", rows)


if __name__ == "__main__":
    main()
