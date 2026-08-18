"""What index map does PIL's NEAREST resize actually use? Measured, not read off the source."""
from __future__ import annotations

import numpy as np
from PIL import Image

SRC = 16
probe = np.arange(SRC * SRC, dtype=np.uint8).reshape(SRC, SRC)   # unique-ish per row/col
row = (np.arange(SRC, dtype=np.uint8)[:, None] * np.ones(SRC, np.uint8)).astype(np.uint8)
col = row.T

for dst in list(range(1, 34)):
    im_r = Image.fromarray(row).resize((dst, dst), resample=Image.NEAREST)
    im_c = Image.fromarray(col).resize((dst, dst), resample=Image.NEAREST)
    got_r = np.array(im_r)[:, 0]                                  # which source ROW each dst row took
    got_c = np.array(im_c)[0, :]                                  # which source COL each dst col took
    cand = {
        "floor((i+0.5)*s/d)": ((np.arange(dst) + 0.5) * SRC / dst).astype(np.int64),
        "floor(i*s/d)":       (np.arange(dst) * SRC / dst).astype(np.int64),
        "round(i*(s-1)/(d-1))": (np.round(np.arange(dst) * (SRC - 1) / max(dst - 1, 1))).astype(np.int64),
    }
    hits = [k for k, v in cand.items() if np.array_equal(v, got_r) and np.array_equal(v, got_c)]
    if dst <= 20 or not hits:
        print(f"dst {dst:3d}: rows {got_r.tolist()}")
        print(f"          cols {got_c.tolist()}  matches {hits}")
