"""Make a LeWM HDF5 dataset with stable-envs frames from the state column of a LeWM dataset.

The input is an HDF5 file in the stable-worldmodel format (one dataset per column, plus ``ep_len`` and
``ep_offset``), for example ``pusht_expert_train.h5`` from the LeWM release. The output keeps every column
except the pixel columns. Without ``--states-only`` it adds a ``pixels`` column that ``Env.render`` draws
from the state rows.

    python examples/lewm/make_dataset.py pusht_expert_train.h5 pusht_snv.h5 --device cuda
    python examples/lewm/make_dataset.py pusht_expert_train.h5 pusht_state.h5 --states-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parent / "config" / "pusht.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", help="input HDF5 dataset with a state column")
    p.add_argument("dst", help="output HDF5 dataset")
    p.add_argument("--config", default=str(CONFIG), help="env and state-row layout (default: config/pusht.yaml)")
    p.add_argument("--states-only", action="store_true", help="copy the columns without pixels and draw nothing")
    p.add_argument("--device", default="cpu", help="device of the stable-envs renderer")
    p.add_argument("--size", type=int, default=224, help="frame size in pixels")
    p.add_argument("--batch", type=int, default=1024, help="frames per render call")
    p.add_argument("--episodes", type=int, default=None, help="keep only the first N episodes")
    p.add_argument("--compression", choices=["none", "lzf", "gzip"], default="lzf", help="pixel compression")
    a = p.parse_args()

    import h5py
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(a.config).env
    t0 = time.perf_counter()
    with h5py.File(a.src, "r") as fi, h5py.File(a.dst, "w") as fo:
        ep_len = fi["ep_len"][:]
        if a.episodes is not None:
            ep_len = ep_len[: a.episodes]
        ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
        rows = int(ep_len.sum())
        fo.create_dataset("ep_len", data=ep_len.astype(np.int32))
        fo.create_dataset("ep_offset", data=ep_offset)
        for key in fi.keys():
            if key in ("ep_len", "ep_offset") or key.startswith("pixels"):
                continue
            src = fi[key]
            dst = fo.create_dataset(key, shape=(rows, *src.shape[1:]), dtype=src.dtype)
            for start in range(0, rows, 1 << 16):
                stop = min(rows, start + (1 << 16))
                dst[start:stop] = src[start:stop]
        if a.states_only:
            print(f"wrote {rows} rows without pixels to {a.dst} in {time.perf_counter() - t0:.1f} s")
            return

        from state_render import from_config

        renderer = from_config(OmegaConf.to_container(cfg), device=a.device, size=a.size)
        compression = None if a.compression == "none" else a.compression
        pixels = fo.create_dataset(
            "pixels",
            shape=(rows, a.size, a.size, 3),
            dtype=np.uint8,
            chunks=(1, a.size, a.size, 3),
            compression=compression,
        )
        states = fo[cfg.state_column]
        t_render = 0.0
        for start in range(0, rows, a.batch):
            stop = min(rows, start + a.batch)
            t = time.perf_counter()
            frames = renderer(torch.as_tensor(states[start:stop], dtype=torch.float32)).cpu().numpy()
            t_render += time.perf_counter() - t
            pixels[start:stop] = frames
            print(f"{stop}/{rows} frames, render {stop / max(t_render, 1e-9):.0f} frames/s", flush=True)
        renderer.close()
    print(f"wrote {rows} frames to {a.dst} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
