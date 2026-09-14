"""Build and run the compiled per-pixel floor for a classic-control frame, and the torch band beside it.

GFXDRAW-2-CPU asks one question: ``cores_cpu >= 1`` needs ~0.85-1.2 ms per frame on the
certification host, and the per-primitive torch path is at 2-4 ms — is a non-torch mechanism
plausibly under the bar, and which one? This measures the candidates on the SAME frame:

  * **torch, as shipped** — ``env.render`` with the planar canvas and bounded primitives.
  * **torch/numpy, interleave only** — the floor of any design that keeps a planar surface: what the
    final ``(3,B,H,W) -> (B,H,W,3)`` pass costs with every primitive free (``interleave_bench.py``).
  * **compiled scalar C++** — ``cost_floor.cpp``, the same op walk as ``gfxdraw_classic`` with
    nothing but the loop, ``-O3 -march=native``.
  * **numba**, where it is installed — the same walk as a ``@njit`` python loop, i.e. the same
    mechanism without a compiled dependency in the wheel.

The op list, the clipped edge table and the disc tables are produced by the repository's own
``gfxdraw_classic.Ops`` and written to a blob, so the C++ measures the per-pixel walk and not a
second implementation of the setup — and its checksum is compared against the torch reference's
frame before any of its numbers are quoted. A cost floor that draws a different picture is not a
floor for this picture.

    python scratch/step6/gfxdraw/cost_floor.py --env CartPole Acrobot --out <path>
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def write_blob(ops, path: Path) -> None:
    """The op list as ``cost_floor.cpp`` reads it: a header, then each table in order."""
    meta = ops.meta_t.cpu().numpy().astype(np.int32)
    vert = ops.vert_t.cpu().numpy().astype(np.int32)
    edge = ops.edge_t.cpu().numpy().astype(np.int32)
    disc = ops.discs.flat.cpu().numpy().astype(np.int32)
    bg = (ops.bg if ops.bg is not None
          else torch.zeros((ops.B, 3), dtype=torch.int32)).cpu().numpy().astype(np.int32)
    aa = ops.aa.cpu().numpy().astype(np.int32)
    plate = (ops.plate.cpu().numpy().astype(np.uint8).reshape(-1) if ops.plate is not None
             else np.zeros(0, np.uint8))
    spr = (ops.sprite.cpu().numpy().astype(np.uint8).reshape(-1) if ops.sprite is not None
           else np.zeros(0, np.uint8))
    hdr = np.array([ops.B, ops.H, ops.W, meta.shape[1], vert.shape[1],
                    int(ops.plate is not None), int(ops.plate.shape[1]) if ops.plate is not None else 1,
                    int(ops.sprite is not None),
                    int(ops.sprite.shape[1]) if ops.sprite is not None else 1,
                    int(ops.sprite.shape[2]) if ops.sprite is not None else 1], np.int32)
    sizes = np.array([disc.size, plate.size, spr.size], np.int32)
    with open(path, "wb") as f:
        for a in (hdr, sizes, meta, vert, edge, disc, bg, aa):
            f.write(np.ascontiguousarray(a).tobytes())
        for a in (plate, spr):
            f.write(np.ascontiguousarray(a).tobytes())


def checksum(frame: np.ndarray) -> int:
    """``cost_floor.cpp``'s checksum, in python, over one frame."""
    h = 0
    for v in frame.reshape(-1).tolist():
        h = (h * 1000003 + v) & 0xFFFFFFFFFFFFFFFF
    return h


def build(cc: str, extra: list[str], out: Path) -> tuple[bool, str]:
    cmd = [cc, "-O3", "-std=c++17", *extra, str(HERE / "cost_floor.cpp"), "-o", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0, (p.stderr or p.stdout)[-600:]


def time_torch(env, states, reps: int) -> float:
    out = torch.empty((env.B, *env.native, 3), dtype=torch.uint8)
    env.render(states, out=out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        env.render(states, out=out)
        ts.append((time.perf_counter() - t0) / env.B)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", nargs="+", default=["CartPole", "Acrobot"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--torch-batches", default="16,64,256",
                    help="batch sizes to time the torch arm at; the compiled walks do not vary")
    ap.add_argument("--cc", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    import stable_envs as se
    from stable_envs.render import gfxdraw_classic as FC

    cc = a.cc or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++") or ""
    doc = {"host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
           "threads": torch.get_num_threads(), "torch": torch.__version__,
           "stable_envs_file": se.__file__, "cc": cc, "envs": {}}
    print(f"{doc['host']}  {doc['cpu']}  {doc['threads']} thread(s)  cc={cc or 'NONE'}")

    variants = {"scalar": ["-march=native"],
                "scalar-no-native": [],
                "scalar-unrolled": ["-march=native", "-funroll-loops"]}
    tmp = Path(tempfile.mkdtemp(prefix="gfx-costfloor-"))
    bins: dict[str, Path] = {}
    for name, extra in variants.items():
        if not cc:
            break
        exe = tmp / f"cost_floor_{name}"
        ok, err = build(cc, extra, exe)
        if ok:
            bins[name] = exe
        else:
            print(f"  build {name}: FAILED {err.splitlines()[-1] if err else ''}")
    if not bins:
        print("  no compiled variant could be built; the C++ floor will be absent from this run")

    for name in a.env:
        if name not in se.REGISTRY:
            print(f"[{name}] not registered")
            continue
        npz = ROOT / f"examples/results/step4/states/{name}.npz"
        pool = np.load(npz)["states"] if npz.is_file() else None
        env = se.make(name, batch=a.batch, device="cpu")
        states = (np.asarray(pool[np.arange(a.batch) % len(pool)], np.float32) if pool is not None
                  else np.asarray(env.sample_states(a.batch), np.float32))
        H, W = env.native
        rec: dict = {"h": H, "w": W, "px": H * W, "batch": a.batch}

        s = torch.as_tensor(states, dtype=torch.float32).reshape(-1, env.state_dim).to(torch.float64)
        sty = env.resolve_style(None, None, batch=a.batch)
        ops = env._ops(s, sty).finish()
        ref = FC.render_classic_torch(ops).cpu().numpy()

        # Torch per FRAME falls with batch (dispatch amortises); the compiled and numba walks are
        # batch-independent by construction, having no dispatch to amortise. Comparing a fused
        # kernel against torch at B=8 would therefore credit the kernel with a win that batching
        # already provides, so the torch arm is swept and the certified batch is the one to read.
        rec["torch_by_batch"] = {}
        for bb in [int(v) for v in a.torch_batches.split(",") if v]:
            e2 = se.make(name, batch=bb, device="cpu")
            st2 = (np.asarray(pool[np.arange(bb) % len(pool)], np.float32) if pool is not None
                   else np.asarray(e2.sample_states(bb), np.float32))
            dt = time_torch(e2, st2, max(2, a.reps // 2))
            rec["torch_by_batch"][str(bb)] = {"ms_per_frame": dt * 1e3,
                                              "ns_per_px": dt / (H * W) * 1e9}
            del e2
        rec["torch_shipped_ms_per_frame"] = time_torch(env, states, a.reps) * 1e3
        rec["torch_shipped_ns_per_px"] = rec["torch_shipped_ms_per_frame"] * 1e6 / (H * W)

        blob = tmp / f"{name}.blob"
        write_blob(ops, blob)
        want = checksum(ref[0])
        for vname, exe in bins.items():
            p = subprocess.run([str(exe), str(blob), str(a.reps)], capture_output=True, text=True)
            if p.returncode != 0:
                rec[f"cpp_{vname}"] = {"error": (p.stderr or p.stdout)[-300:]}
                continue
            r = json.loads(p.stdout.strip().splitlines()[-1])
            # A floor for a DIFFERENT picture is not a floor for this one.
            r["draws_the_same_frame"] = int(r["checksum"]) == want
            rec[f"cpp_{vname}"] = r
        # numba: the same walk without a compiled dependency in the wheel
        try:
            import numba_floor as NB
        except ImportError:
            sys.path.insert(0, str(HERE))
            try:
                import numba_floor as NB
            except ImportError as e:                                      # pragma: no cover
                NB = None
                rec["numba"] = {"absent": str(e)[:200]}
        if NB is not None and NB.HAVE_NUMBA:
            import numba
            doc["numba"] = numba.__version__
            frames = NB.render_batch(ops)                                  # compile + warm
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                NB.render_batch(ops, frames)
                ts.append((time.perf_counter() - t0) / a.batch)
            dt = statistics.median(ts)
            rec["numba"] = {"ms_per_frame": dt * 1e3, "ns_per_px": dt / (H * W) * 1e9,
                            "fps": 1.0 / dt,
                            "draws_the_same_frame": bool(np.array_equal(frames, ref))}
        elif NB is not None:
            rec["numba"] = {"absent": "numba is not installed in this interpreter"}
        doc["envs"][name] = rec
        line = "  ".join(
            f"{k.replace('cpp_', '')} {v['ns_per_px']:6.2f} ns/px {v['ms_per_frame']:6.3f} ms/f"
            f"{'' if v.get('draws_the_same_frame') else ' [DIFFERENT FRAME]'}"
            for k, v in rec.items() if isinstance(v, dict) and "ns_per_px" in v)
        tb = "  ".join(f"B{k} {v['ns_per_px']:5.2f}" for k, v in rec["torch_by_batch"].items())
        print(f"[{name:22s}] torch@B{a.batch} {rec['torch_shipped_ns_per_px']:6.2f} ns/px "
              f"{rec['torch_shipped_ms_per_frame']:6.3f} ms/f  [torch ns/px {tb}]   {line}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print(f"wrote {a.out}")
    bad = [f"{e}:{k}" for e, r in doc["envs"].items() for k, v in r.items()
           if isinstance(v, dict) and v.get("draws_the_same_frame") is False]
    if bad:
        print(f"\nFAILED: {len(bad)} compiled variant(s) drew a different frame: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
