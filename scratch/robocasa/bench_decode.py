"""The number every "regenerate instead of store" claim has to beat: JPEG decode, per CPU core.

Each family's report quotes its renderer against *its own* official renderer, which answers "is the
GPU path faster than the thing it replaces" but not "is regenerating cheaper than reading". Those are
different questions and the second one is the thesis. Answering it needs one baseline measured at the
frame sizes actually in play, on the same hardware, so the ratios across families are comparable
rather than each being relative to a different CPU renderer.

Reports decode throughput from memory (no disk), which is the *favourable* case for stored datasets
and therefore the honest bar for regeneration to clear.
"""
from __future__ import annotations

import io
import json
import time

import numpy as np


def main():
    from PIL import Image

    rng = np.random.default_rng(0)
    out = {}
    for res in (64, 96, 128, 224, 256, 608):
        # a synthetic frame with realistic structure: flat regions + edges, not white noise
        # (white noise is incompressible and would understate both size and decode speed)
        img = np.zeros((res, res, 3), np.uint8)
        img[:] = rng.integers(30, 80, size=(1, 1, 3))
        for _ in range(12):
            y, x = rng.integers(0, res, 2)
            h, w = rng.integers(res // 8, res // 2, 2)
            img[y:y + h, x:x + w] = rng.integers(0, 255, size=3)

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=90)
        b = buf.getvalue()

        n = max(200, int(4e7 / (res * res)))
        for _ in range(20):
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        t0 = time.perf_counter()
        for _ in range(n):
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        dt = (time.perf_counter() - t0) / n

        out[str(res)] = {
            "jpeg_bytes": len(b),
            "raw_bytes": res * res * 3,
            "decode_fps_per_core": 1.0 / dt,
            "decode_MBps_raw": res * res * 3 / dt / 1e6,
        }
        print(f"{res:4d}^2  jpeg {len(b):7d} B  decode {1.0/dt:10.0f} f/s/core  "
              f"({res*res*3/dt/1e6:7.0f} MB/s of raw frame)")

    print(json.dumps(out, indent=2))
    with open("/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/decode_baseline.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
