"""Regenerating at a fixed 256px/batch-1024 against reading the same frames back off a real disk.

``bench_vs_decode.py`` measures each env at its own native resolution and decodes one frame repeatedly
from memory. That is the right shape for "what does this env cost", and the wrong shape for the
comparison a paper table has to make, in three ways this script fixes:

  **One resolution, one batch.** 256px and batch 1024 for every env, so the column is a like-for-like
  comparison rather than 196 different operating points. An env that cannot render at 256 records why
  and falls back to its native size, flagged, rather than being dropped.

  **Distinct frames.** The decode side reads ``n_frames`` DIFFERENT files rather than one blob
  ``n`` times. Re-decoding one buffer keeps it in L2 and keeps its Huffman tables hot; measured here,
  that flatters the stored path by a factor this script reports as ``mem_over_disk``.

  **A real filesystem.** Frames are written as JPEGs and read back through the page cache from
  node-local disk, and again from the shared filesystem a cluster dataset actually lives on. Both are
  reported, because they differ by more than the renderers do and quoting either alone would be a
  choice about which conclusion to reach.

The storage number is per CPU core, single-threaded, warm. That is deliberately the FAVOURABLE case for
a stored dataset: no cold-cache seek, no decompression of a shard, no network. A pipeline that has to
pull shards over a network will do worse than every number here.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from bench_gpu_vs_cpu import _err, states_for

import stable_envs as se


def write_jpegs(frames: np.ndarray, dest: Path, quality: int) -> tuple[list[Path], int]:
    """One JPEG per frame. Returns the paths and the total bytes on disk."""
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    paths, total = [], 0
    for i, f in enumerate(frames):
        p = dest / f"{i:05d}.jpg"
        Image.fromarray(f).save(p, format="JPEG", quality=quality)
        total += p.stat().st_size
        paths.append(p)
    return paths, total


def read_decode(paths: list[Path], passes: int = 3) -> float:
    """Frames/s for ``open -> decode -> ndarray``, one core, best of ``passes``, page cache warm."""
    from PIL import Image

    for p in paths[:32]:  # warm the cache for the timed set
        np.asarray(Image.open(p).convert("RGB"))
    best = None
    for _ in range(passes):
        t0 = time.perf_counter()
        for p in paths:
            np.asarray(Image.open(p).convert("RGB"))
        dt = (time.perf_counter() - t0) / len(paths)
        best = dt if best is None else min(best, dt)
    return 1.0 / best


def decode_memory(blob: bytes, n: int) -> float:
    from PIL import Image

    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    return n / (time.perf_counter() - t0)


def per_tile_note(hw: tuple[int, int], requested: int) -> str | None:
    """Did the env read ``size`` as something other than the output edge? The message, or None.

    Craftax reads it as a PER-TILE edge, so ``size=256`` means a 1792x2304 frame and a batch of 1024
    asks the allocator for 47 GiB — which reads as "this env OOMs" rather than "this env was asked for
    something other than what the flag means".

    The comparison is on the SHORTEST edge, and that detail is the whole guard. On the longest edge it
    also fires for the aspect-preserving envs, which honour the request exactly and are simply not
    square: highway-env's Highway and Merge render 256x1024 at ``size=256`` and Parking 256x512, all
    with the requested 256 as their short edge. A ``max``-based test re-measures those at native and
    silently rewrites three published rows for envs that did nothing wrong. What actually distinguishes
    Craftax is that NO edge is anywhere near the request: its short edge is 1792 for a request of 256.
    """
    if min(hw) > 2 * int(requested):
        return (
            f"size={requested} produced a {hw[0]}x{hw[1]} frame — its SHORT edge alone is "
            f"{min(hw)}: this env reads size per tile, not as the output edge; measured at its "
            f"native size instead"
        )
    return None


def probe_at(name: str, size: int | None, n_frames: int, pool_max: int):
    """``(pool, frames, hw, size)`` from a batch-1 probe, at ``size`` or at the env's native size.

    The probe env is dropped before returning: it exists to find out the frame shape and to produce
    the frames the storage side writes, and holding it would keep its memory alive across the timed
    render that follows.
    """
    kw = {} if size is None else {"size": int(size)}
    probe = se.make(name, num_envs=1, device="cuda", **kw)
    try:
        pool = probe.sample_states(pool_max)
        frames = probe.render(pool[: min(n_frames, len(pool))]).cpu().numpy()
        return pool, frames, (int(frames.shape[1]), int(frames.shape[2])), int(probe.size)
    finally:
        del probe
        torch.cuda.empty_cache()


def probe_frames(name: str, requested: int, n_frames: int, pool_max: int):
    """``(pool, frames, hw, size, note)`` at ``requested`` px, falling back to native with a reason.

    Three things can go wrong with a requested size and they are all the same kind of fact — "this env
    does not render at 256" — so they take one path and produce one ``note``:

      * the env REFUSES the size at construction (CarRacing's observation is a smoothscale of a fixed
        surface; the swm nav envs have no world-unit description to re-sample);
      * it accepts it and refuses at RENDER (classic control hardcodes primitives in pixels; MiniHack's
        grid at 256 px tiles asks for 155 GiB);
      * it accepts it and means something else by it (``per_tile_note``).

    None of those is an error to record as one: dropping the env from the table reads as "does not
    work" rather than "fixed size", which is a claim about the renderer that is not true. Only a
    failure at the env's OWN native size propagates.
    """
    note = None
    try:
        pool, frames, hw, size = probe_at(name, requested, n_frames, pool_max)
        note = per_tile_note(hw, requested)
        if note is None:
            return pool, frames, hw, size, None
    except Exception as e:  # noqa: BLE001
        note = f"{type(e).__name__}: {e}"
    pool, frames, hw, size = probe_at(name, None, n_frames, pool_max)
    return pool, frames, hw, size, note


def gpu_fps(name: str, batch: int, size: int, hw, pool, reps: int) -> dict:
    env = st = out = None
    try:
        env = se.make(name, num_envs=batch, size=size, device="cuda")
        st = torch.as_tensor(states_for(pool, batch), device="cuda")
        out = torch.empty((batch, hw[0], hw[1], 3), dtype=torch.uint8, device="cuda")
        for _ in range(5):
            env.render(st, out=out)
        torch.cuda.synchronize()
        ts = []
        cpu0, wall0 = time.process_time(), time.perf_counter()
        for _ in range(reps):
            t0 = time.perf_counter()
            env.render(st, out=out)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        # CPU seconds per wall second of the timed loop. Near 0 the device is the limit and the row
        # travels between nodes; at or above 1 the host is busy for the whole render and the number is
        # a property of this node under this load. Measured: DMCBallInCupCatch is at 1.96 and the same
        # configuration has produced 3,664 f/s idle against 271 contended.
        cpu_frac = (time.process_time() - cpu0) / max(time.perf_counter() - wall0, 1e-9)
        dt = float(np.median(ts))
        # The RESOLVED supersample travels with the number. This script never passed one, so it has
        # always measured each env's own default; bench_vs_decode.py forced 1, and the two tables
        # therefore described different pictures for the 44 classes that declare 2. Recording it is
        # what makes the two comparable at all.
        return {
            "fps": batch / dt,
            "seconds_per_batch": dt,
            "batch": batch,
            "supersample": int(env.ss),
            "cpu_frac": cpu_frac,
            "loadavg": os.getloadavg()[0],
        }
    except Exception as e:
        return _err(e)
    finally:
        del env, st, out
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--frames", type=int, default=512, help="distinct frames written and read back")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--local", default="/tmp", help="node-local scratch")
    ap.add_argument(
        "--shared",
        default="/mnt/vast/home/randall.balestriero/tmp_storage_bench",
        help="shared filesystem a cluster dataset would live on; '' to skip",
    )
    ap.add_argument("--envs", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="docs/research/artifacts/results/storage")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="re-measure every env instead of resuming; required after a renderer change, "
        "since resume treats an env already in the output file as done",
    )
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    try:
        cpu_name = (
            subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | head -1"], capture_output=True, text=True)
            .stdout.split(":", 1)[-1]
            .strip()
        )
    except Exception:
        cpu_name = platform.processor()

    res = {
        "device": torch.cuda.get_device_name(0),
        "cpu": cpu_name or "unknown",
        "size": a.size,
        "batch": a.batch,
        "n_frames": a.frames,
        "jpeg_quality": a.jpeg_quality,
        "loadavg": [os.getloadavg()[0]],
        "registry_size": len(se.registry.REGISTRY),
        "envs": {},
    }
    # Resume, so a sweep killed at hour six does not start over. The failure mode this creates is worse
    # than the one it fixes if it is the only option: after a kernel lands, every env in the old file is
    # "done" and the sweep silently republishes the numbers the kernel was written to change. It has
    # happened. ``--fresh`` is the way to say the previous answers are no longer answers.
    prev = outdir / f"storage{tag}.json"
    if prev.exists() and not a.fresh:
        try:
            res["envs"] = json.loads(prev.read_text()).get("envs", {})
        except Exception:
            pass

    def flush():
        res["loadavg"] = res["loadavg"][:1] + [os.getloadavg()[0]]
        (outdir / f"storage{tag}.json").write_text(json.dumps(res, indent=2))

    skip = {s for s in a.skip.split(",") if s}
    wanted = [n for n in sorted(se.registry.REGISTRY) if (not a.envs or n in a.envs.split(",")) and n not in skip]
    print(f"registry {len(se.registry.REGISTRY)}, measuring {len(wanted)} at {a.size}px batch {a.batch}", flush=True)

    for name in wanted:
        if name in res["envs"] and res["envs"][name].get("gpu"):
            print(f"[{name}] done, skipping", flush=True)
            continue
        entry: dict = {}
        try:
            pool, frames, hw, size, note = probe_frames(name, a.size, a.frames, min(a.batch, 4096))
        except Exception as e:
            print(f"[{name}] cannot render at {a.size} or native: {e}", flush=True)
            res["envs"][name] = _err(e)
            flush()
            continue

        if len(frames) < a.frames:  # tile the pool up to the requested count
            frames = np.concatenate([frames] * (a.frames // len(frames) + 1))[: a.frames]

        entry["size"] = size
        entry["requested_size"] = a.size
        entry["size_note"] = note
        entry["frame_shape"] = list(hw)
        entry["state_bytes"] = int(np.asarray(pool[:1]).nbytes)
        entry["gpu"] = gpu_fps(name, a.batch, size, hw, pool, a.reps)

        storage = {}
        for label, root in (("local", a.local), ("shared", a.shared)):
            if not root:
                continue
            dest = Path(root) / f"se_store_{os.getpid()}" / name
            try:
                paths, nbytes = write_jpegs(frames, dest, a.jpeg_quality)
                storage[label] = {
                    "fps": read_decode(paths),
                    "bytes_on_disk": nbytes,
                    "bytes_per_frame": nbytes / len(paths),
                }
            except Exception as e:
                storage[label] = _err(e)
            finally:
                shutil.rmtree(dest, ignore_errors=True)

        buf = io.BytesIO()
        from PIL import Image

        Image.fromarray(frames[0]).save(buf, format="JPEG", quality=a.jpeg_quality)
        storage["memory_one_frame"] = {"fps": decode_memory(buf.getvalue(), max(300, a.frames))}
        entry["storage"] = storage

        res["envs"][name] = entry
        g = entry["gpu"].get("fps")
        loc = storage.get("local", {}).get("fps")
        sh = storage.get("shared", {}).get("fps")
        if g and loc:
            print(
                f"[{name}] {hw[0]}x{hw[1]}{'*' if note else ''}  ours {g:,.0f} f/s  "
                f"local-disk {loc:,.0f}  shared {sh:,.0f}  = {g / loc:,.1f}x local, "
                f"{g / sh:,.1f}x shared"
                if sh
                else f"= {g / loc:,.1f}x local",
                flush=True,
            )
        else:
            print(f"[{name}] incomplete: gpu={entry['gpu']}", flush=True)
        flush()

    flush()


if __name__ == "__main__":
    main()
