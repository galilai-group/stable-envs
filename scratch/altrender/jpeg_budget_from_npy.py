"""The JPEG q95 budget for a cell whose renderer and whose JPEG decoder live in different venvs.

``probe_budget_frontier.py --dump-official`` writes the OFFICIAL frames; this reads them and scores a
q95 round-trip through the repo's own write and read paths (``PIL.Image.save(quality=95)`` from
``examples/bench_storage.py``, ``torchvision.io.decode_jpeg`` from ``examples/bench_lance.py``). No
substitute codec: the number is only a budget if it is the byte stream a worker really decodes.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from probe_budget_frontier import jpeg_roundtrip, metrics

ap = argparse.ArgumentParser()
ap.add_argument("npy", nargs="+")
ap.add_argument("--quality", type=int, default=95)
ap.add_argument("--out", default="")
a = ap.parse_args()

rows = {}
for p in a.npy:
    official = np.load(p)
    cell = p.rsplit("/", 1)[-1].removeprefix("official_").removesuffix(".npy")
    jo, nbytes = jpeg_roundtrip(official, a.quality)
    m = metrics(jo, official)
    m["jpeg_bytes_per_frame"] = nbytes
    m["n_states"] = int(official.shape[0])
    rows[cell] = m
    print(json.dumps({"cell": cell, **m}), flush=True)
if a.out:
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", a.out)
