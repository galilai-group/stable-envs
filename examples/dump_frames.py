"""Phase 1 of the storage comparison: render real frames and encode them exactly as a corpus would.

Split from the measurement because no interpreter can hold both sides. The env venvs carry
``mujoco_warp``/``jax``/``griddly``; ``lancedb`` + ``torchvision``'s libjpeg-turbo decoder live in the
suite's conda env. So this renders and encodes, and ``bench_lance.py`` reads.

Encoding matches ``stable_worldmodel.data.formats.lance``: JPEG at **quality 95**, one blob per frame,
RGB. Quality is not a detail to round off — it sets both the bytes a dataset occupies and the work its
decoder does, and q90 (this repo's earlier default) is measurably cheaper than the q95 a real corpus
was written at, in the stored path's favour.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

import stable_envs as se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--jpeg-quality", type=int, default=95, help="stable-worldmodel's default")
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/se_frames")
    a = ap.parse_args()

    from PIL import Image

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wanted = [n for n in sorted(se.registry.REGISTRY) if not a.envs or n in a.envs.split(",")]
    print(f"dumping {len(wanted)} envs at {a.size}px q{a.jpeg_quality}", flush=True)

    for name in wanted:
        dest = outdir / f"{name}.npz"
        if dest.exists():
            print(f"[{name}] exists, skipping", flush=True)
            continue
        size, note = a.size, None
        try:
            env = se.make(name, num_envs=1, size=a.size, device="cuda")
        except Exception as e:
            note = f"{type(e).__name__}: {e}"
            try:
                env = se.make(name, num_envs=1, device="cuda")
                size = env.size
            except Exception as e2:
                print(f"[{name}] cannot instantiate: {e2}", flush=True)
                continue
        # Several envs accept ``size=256`` at construction and refuse it at RENDER, because the
        # reference's output size is part of their contract: classic control hardcodes every primitive
        # in pixels, Sokoban needs whole multiples of its 16 px art, MiniHack's grid at 256 px tiles
        # asks for 155 GiB. Falling back only on a construction failure lost 14 envs from the sweep,
        # which read as "these do not work" rather than "these have a fixed size".
        try:
            pool = env.sample_states(a.frames)
            frames = env.render(pool).cpu().numpy()
        except Exception as e:
            note = f"{type(e).__name__}: {e}"
            del env
            torch.cuda.empty_cache()
            try:
                env = se.make(name, num_envs=1, device="cuda")
                size = env.size
                pool = env.sample_states(a.frames)
                frames = env.render(pool).cpu().numpy()
            except Exception as e2:
                print(f"[{name}] cannot render at 256 or native: {e2}", flush=True)
                continue
        finally:
            del env
            torch.cuda.empty_cache()

        # Same guard as bench_storage.py: a frame far larger than the requested size means the env
        # reads ``size`` as something other than the output edge (Craftax: per tile). Re-render at
        # native so the stored-side measurement is of the same picture the GPU side is timed on.
        if note is None and max(frames.shape[1], frames.shape[2]) > 2 * a.size:
            note = (
                f"size={a.size} produced a {frames.shape[1]}x{frames.shape[2]} frame: this env "
                f"reads size per tile; dumped at its native size instead"
            )
            env = se.make(name, num_envs=1, device="cuda")
            size = env.size
            pool = env.sample_states(a.frames)
            frames = env.render(pool).cpu().numpy()
            del env
            torch.cuda.empty_cache()

        blobs = []
        for f in frames:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, format="JPEG", quality=a.jpeg_quality)
            blobs.append(buf.getvalue())
        meta = {
            "env": name,
            "size": size,
            "requested_size": a.size,
            "size_note": note,
            "h": int(frames.shape[1]),
            "w": int(frames.shape[2]),
            "n": len(blobs),
            "jpeg_quality": a.jpeg_quality,
            "state_bytes": int(np.asarray(pool[:1]).nbytes),
            "state_dim": int(np.asarray(pool).shape[1]),
            "jpeg_bytes_total": sum(len(b) for b in blobs),
            "raw_bytes_per_frame": int(frames.shape[1] * frames.shape[2] * 3),
        }
        np.savez(dest, blobs=np.array(blobs, dtype=object), meta=json.dumps(meta))
        print(
            f"[{name}] {meta['h']}x{meta['w']}{'*' if note else ''}  {len(blobs)} frames  "
            f"{meta['jpeg_bytes_total'] / 1e6:.1f} MB jpeg  "
            f"({meta['jpeg_bytes_total'] / len(blobs):,.0f} B/frame vs "
            f"{meta['raw_bytes_per_frame']:,} raw)",
            flush=True,
        )


if __name__ == "__main__":
    main()
