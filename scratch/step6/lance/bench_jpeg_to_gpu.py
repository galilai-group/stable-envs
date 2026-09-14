"""The honest GPU-side denominator: stored JPEG -> **frames on the device**.

``bench_lance.py``'s ``local_n`` stops when uint8 CHW tensors exist in HOST memory. A GPU trainer
cannot use them there: it still owes an H2D copy of the whole uint8 batch, and that copy is not
charged anywhere in the committed corpus. Our renderer's frames are already on the device, so
comparing the two without the transfer compares different end states.

Three arms, all over the same warm node-local Lance table, all ending with frames ON THE DEVICE:

``loader8_h2d``  a real ``torch.utils.data.DataLoader``: ``num_workers=8``, a ``batch_sampler`` of
                 ``chunk`` contiguous row ids, ``__getitems__`` doing the Lance take and the single
                 batched ``decode_jpeg`` (``stable_worldmodel``'s own path), ``pin_memory=True``,
                 then ``.to("cuda", non_blocking=True)`` and ``torch.cuda.synchronize()`` in the main
                 process. One untimed epoch first, so worker spawn and the first touch of the file are
                 paid once, as they are in ``bench_lance.time_parallel``.
``nvjpeg_h2d``   the row bytes are moved to the device and decoded there with nvJPEG -- the
                 ``local_gpu`` arm, which already ends on the device. Timed identically.
``h2d_only``     the transfer alone, for a uint8 CHW batch of this env's shape, pinned and unpinned.
                 This is the term ``local_n`` omits, in isolation.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
import bench_lance as BL  # noqa: E402


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


class LanceJPEG(torch.utils.data.Dataset):
    """``stable_worldmodel.data.formats.lance``'s read path, narrowed to the image column.

    ``__getitems__`` is the whole point: torch's fetcher hands the worker the entire index list and
    skips per-item work, which is what makes one batched ``decode_jpeg`` per batch possible.
    """

    def __init__(self, root: str, name: str, n: int):
        self.root, self.name, self.n = root, name, n
        self._ds = None

    def _lazy(self):
        if self._ds is None:
            import lancedb
            self._ds = lancedb.connect(self.root).open_table(self.name).to_lance()
        return self._ds

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # With ``batch_size=None`` torch's fetcher disables auto-collation and hands the WHOLE index
        # list to ``__getitem__``; with a batch_size it hands one index. Serve both, so the loader
        # arm measures the batched Lance take either way.
        if isinstance(i, (list, tuple, range)):
            return self.__getitems__(list(i))
        return self.__getitems__([i])[0]

    def __getitems__(self, idx):
        batch = self._lazy().take(list(idx), columns=["pixels"])
        return BL._decode(batch.column("pixels").to_pylist())


def _identity(x):
    return x


def _worker_init(_):
    torch.set_num_threads(1)


class Chunks(torch.utils.data.Sampler):
    """The exact batch schedule ``bench_lance.time_parallel`` gives its 8 workers.

    A 512-row table holds only two 256-row batches, so a plain sequential epoch would leave six of
    eight workers idle and the "8 worker" number would be a 2-worker number. ``bench_lance`` avoids
    this by giving each of its 8 workers ``reps`` chunks at wrapped offsets, i.e.
    ``workers * reps`` batches over the same table. The same schedule is used here, so the two
    numbers cover the same frames: 8 x 8 x 256 = 16,384.
    """

    def __init__(self, n: int, chunk: int, workers: int, reps: int):
        self.batches = []
        for w in range(workers):
            for i in range(reps):
                s0 = ((i + w) * chunk) % max(n - chunk, 1)
                self.batches.append(list(range(s0, min(s0 + chunk, n))))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def loader_h2d(root, name, n, chunk, reps, workers, pin=True, inner=8):
    from torch.utils.data import DataLoader
    ds = LanceJPEG(root, name, n)
    sampler = Chunks(n, chunk, workers, inner)
    dl = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=workers,
                    collate_fn=_identity, pin_memory=pin, persistent_workers=True,
                    prefetch_factor=2, worker_init_fn=_worker_init,
                    multiprocessing_context="spawn")
    for _ in dl:                                  # one untimed epoch: spawn, imports, first read
        pass
    torch.cuda.synchronize()
    best, tot_best = None, None
    for _ in range(reps):
        frames = 0
        t0 = time.perf_counter()
        for b in dl:
            g = b.to("cuda", non_blocking=True)
            frames += int(g.shape[0])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if best is None or dt / frames < best:
            best, tot_best = dt / frames, frames
    del dl
    return {"fps": 1 / best, "workers": workers, "pin_memory": pin, "frames_per_epoch": tot_best,
            "batches_per_epoch": len(sampler), "epoch_s": best * tot_best,
            "output_location": "DEVICE memory, uint8 CHW"}


def nvjpeg_h2d(tbl, n, chunk, reps):
    r = BL.time_path(tbl, n, chunk, reps, device="cuda")
    r["output_location"] = "DEVICE memory, uint8 CHW (nvJPEG, bytes moved H2D then decoded)"
    return r


def h2d_only(shape, chunk, reps, pin):
    x = torch.empty((chunk, 3, shape[0], shape[1]), dtype=torch.uint8)
    if pin:
        x = x.pin_memory()
    for _ in range(3):
        x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        x.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / chunk
        best = dt if best is None else min(best, dt)
    return {"fps": 1 / best, "pinned": pin, "bytes_per_frame": 3 * shape[0] * shape[1],
            "GBps": 3 * shape[0] * shape[1] / best / 1e9}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this script measures H2D; it needs a GPU")

    import shutil
    root = Path(a.local)
    root.mkdir(parents=True, exist_ok=True)
    res = {"what": "end-to-end stored-JPEG -> frames ON THE DEVICE, the denominator the GPU "
                   "comparison actually needs",
           "cpu": sh("lscpu | grep 'Model name' | head -1").split(":", 1)[-1].strip(),
           "node": platform.node(), "hostname": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(0), "chunk": a.chunk, "reps": a.reps,
           "workers": a.workers, "cpu_count_affinity": len(os.sched_getaffinity(0)),
           "loadavg_start": os.getloadavg(),
           "slurm": {k: v for k, v in os.environ.items()
                     if k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")},
           "python": sys.version.split()[0], "executable": sys.executable, "torch": torch.__version__,
           "local_arm": {"path": str(root), "fstype": sh(f"stat -f -c '%T' {root}")},
           "envs": {}}
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.frames_dir).glob("*.npz"))
    if a.envs:
        keep = set(a.envs.split(","))
        files = [f for f in files if f.stem in keep]

    for fpath in files:
        name = fpath.stem
        z = np.load(fpath, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        blobs = list(z["blobs"])
        n = len(blobs)
        entry = {"meta": {k: v for k, v in meta.items() if k != "provenance"}, "n": n,
                 "jpeg_bytes_per_frame": meta["jpeg_bytes_total"] / n}
        try:
            tbl = BL.build_table(blobs, root, name)
            entry["nvjpeg_h2d"] = nvjpeg_h2d(tbl, n, a.chunk, a.reps)
            del tbl
            entry["loader8_h2d"] = loader_h2d(str(root), name, n, a.chunk, a.reps, a.workers, True)
            entry["loader8_h2d_unpinned"] = loader_h2d(str(root), name, n, a.chunk, a.reps,
                                                       a.workers, False)
            entry["h2d_only_pinned"] = h2d_only((meta["h"], meta["w"]), a.chunk, 20, True)
            entry["h2d_only_unpinned"] = h2d_only((meta["h"], meta["w"]), a.chunk, 20, False)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        finally:
            shutil.rmtree(root / f"{name}.lance", ignore_errors=True)
        entry["loadavg"] = os.getloadavg()
        res["envs"][name] = entry
        bits = " ".join(f"{k}={entry[k]['fps']:,.0f}" for k in
                        ("loader8_h2d", "loader8_h2d_unpinned", "nvjpeg_h2d", "h2d_only_pinned")
                        if isinstance(entry.get(k), dict) and entry[k].get("fps"))
        print(f"[{name}] {meta['h']}x{meta['w']}  {bits}  {entry.get('error','')}", flush=True)
        outp.write_text(json.dumps(res, indent=2))

    res["loadavg_end"] = os.getloadavg()
    outp.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
