"""One GPU regenerating frames vs one CPU core reading them back: the ratio, per env.

``bench_gpu_vs_cpu.py`` answers "is the GPU path faster than the renderer it replaces". That is the
wrong question for the claim this repo is built on, which is that regenerating a frame from its state
is cheaper than *storing and reading* it. Those are different baselines and only the second one is the
thesis, so this script measures the second one.

The bar is JPEG decode on one CPU core, and it is deliberately set in the dataset's favour:

  ``memory``   decode from a ``BytesIO`` already resident. No filesystem, no page cache, no network.
               This is the headline column, and it is the number regeneration has to beat.
  ``file``     the same decode, reading a real file from node-local disk with the cache warm. Reported
               because the user-facing operation is "load and decode", and omitting the load would be
               quoting a bar nobody actually runs. It is within a few percent of ``memory`` on every
               node measured, which is itself the finding: a stored-frame pipeline is decode-bound, not
               IO-bound, so the favourable column is the honest one.

**The JPEG is encoded from the env's own rendered frame**, at the env's own resolution, rather than
interpolated off a synthetic baseline curve. That matters more than it sounds: decode cost tracks
content, and a tile-atlas env's flat 64px frame and a ray-cast maze's 224px frame differ by 4x in
bytes at the same quality. Quoting one synthetic square for both would flatter whichever env happens to
draw less.

The ratio reported is therefore **CPU cores of decode replaced by one GPU** — divide it into your node's
core count to get the honest statement, which is that a 96-core box reading a dataset is matched by a
fraction of one GPU regenerating it.

Two things this does NOT claim. It does not measure disk or object-store bandwidth, so a pipeline whose
real bottleneck is pulling shards over a network is out of scope and will do worse than the ``file``
column. And it compares one GPU against one core by construction, which is the right comparison for
"should I store frames or regenerate them on the node I have" and the wrong one for a per-watt argument.

    python examples/bench_vs_decode.py --out docs/research/artifacts/results
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from bench_gpu_vs_cpu import _err, cpu_throughput, gpu_throughput, parse_ss, states_for  # noqa: F401

import stable_envs as se


def decode_baseline(frame: np.ndarray, quality: int, min_frames: int, scratch: Path) -> dict:
    """Time ``bytes -> uint8 array`` for one real frame, from memory and from a warm local file.

    Encoded once at ``quality`` and decoded ``n`` times; ``n`` scales inversely with pixel count so a
    64px env and a 608px env both spend a comparable wall clock in the timed loop.
    """
    from PIL import Image

    h, w = int(frame.shape[0]), int(frame.shape[1])
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    blob = buf.getvalue()

    n = max(min_frames, int(4e7 / max(h * w, 1)))

    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    dt_mem = (time.perf_counter() - t0) / n

    dt_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir=str(scratch), delete=False) as fh:
            fh.write(blob)
            path = fh.name
        for _ in range(20):
            np.asarray(Image.open(path).convert("RGB"))
        t0 = time.perf_counter()
        for _ in range(n):
            np.asarray(Image.open(path).convert("RGB"))
        dt_file = (time.perf_counter() - t0) / n
        os.unlink(path)
    except Exception:
        dt_file = None

    return {
        "h": h,
        "w": w,
        "jpeg_quality": quality,
        "jpeg_bytes": len(blob),
        "raw_bytes": h * w * 3,
        "n": n,
        "decode_fps_per_core": 1.0 / dt_mem,
        "decode_MBps_raw": h * w * 3 / dt_mem / 1e6,
        "load_decode_fps_per_core": (1.0 / dt_file) if dt_file else None,
    }


def _cpu_pending(entry: dict) -> bool:
    """Does the official-renderer column still need measuring?

    A stored ``error`` counts as pending, not as done: the first pass recorded 39 of them from a bug in
    the reference bridge rather than from anything about the env, and a rerun that treated any present
    key as an answer would have carried all 39 forward as permanent blanks.
    """
    if "cpu" not in entry:
        return True
    c = entry["cpu"]
    return isinstance(c, dict) and "error" in c


def _cpu_fps(e: dict):
    c = e.get("cpu")
    if not isinstance(c, dict):
        return None
    v = c.get("set_state_render")
    return v.get("fps") if isinstance(v, dict) else None


def markdown(res: dict) -> str:
    rows = []
    for name, e in sorted(res["envs"].items()):
        if "error" in e or not e.get("decode"):
            continue
        d, g = e["decode"], e.get("gpu_best")
        if not g:
            continue
        rows.append(
            {
                "name": name,
                "shape": f"{d['h']}x{d['w']}",
                "batch": g["batch"],
                "gpu": g["fps"],
                "dec": d["decode_fps_per_core"],
                "cpu": _cpu_fps(e),
                "file": d.get("load_decode_fps_per_core"),
                "cores": g["fps"] / d["decode_fps_per_core"],
                "state_b": e.get("state_bytes"),
                "jpeg_b": d["jpeg_bytes"],
                "ss": g.get("supersample"),
                "cpu_frac": g.get("cpu_frac"),
            }
        )
    rows.sort(key=lambda r: -r["cores"])

    L = [
        f"# Three ways to get a frame: {len(rows)} envs",
        "",
        f"Device: {res.get('device')} · CPU: {res.get('cpu')} · "
        f"{res.get('cpu_count')} cores visible · load {res.get('loadavg')}",
        "",
        "All three columns are frames/s for the same transformation, `state -> uint8 frame`.",
        "",
        "* **default** — the env's own official renderer, one frame at a time, which is the only entry",
        "  point it offers. Blank where no reference package is installed in this interpreter.",
        "* **ours** — the GPU path at its best batch.",
        f"* **stored** — one CPU core decoding this env's own frame as JPEG q{res.get('jpeg_quality')} from memory.",
        "",
        "The last column is `ours / stored`: **CPU cores of decode that one GPU replaces.**",
        "",
        "`ss` is the supersample each env RESOLVED to, not the one the flag asked for. It belongs in",
        "the table because 44 of the classes declare `default_supersample = 2` and a sweep that forces",
        "1 on them reports a cheaper picture than `se.make(name)` produces: 1.16x on",
        "DMCBallInCupCatch, 1.26x on AntmazeLargeNavigate and 3.3x on MujocoPusher, measured idle.",
        "",
        "`cpu%` is the process CPU seconds per wall second of the timed loop. Near 0 the device is",
        "the limit and the row travels between nodes; at or above 100% the host is busy for the whole",
        "render, and that row is a measurement of this node's processor under this node's load —",
        "DMCBallInCupCatch measures 196% and has produced 3,664 f/s idle against 271 contended.",
        "",
        "| env | frame | ss | default f/s | ours f/s (batch) | cpu% | stored f/s/core |"
        " **cores replaced** | state B | JPEG B |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cpu = f"{r['cpu']:,.0f}" if r["cpu"] else "—"
        cf = r.get("cpu_frac")
        L.append(
            f"| {r['name']} | {r['shape']} | {r['ss'] or '?'} | {cpu} | "
            f"{r['gpu']:,.0f} ({r['batch']:,}) | "
            f"{'—' if cf is None else format(100 * cf, '.0f') + '%'} | "
            f"{r['dec']:,.0f} | **{r['cores']:,.1f}x** | "
            f"{r['state_b'] or '?'} | {r['jpeg_b']:,} |"
        )
    L += [
        "",
        "## Load-and-decode (warm local file) against decode-from-memory",
        "",
        "| env | memory f/s | file f/s | file/memory |",
        "|---|---|---|---|",
    ]
    for r in rows:
        if r["file"]:
            L.append(f"| {r['name']} | {r['dec']:,.0f} | {r['file']:,.0f} | {r['file'] / r['dec']:.3f} |")
    errs = {k: v["error"] for k, v in res["envs"].items() if "error" in v}
    if errs:
        L += ["", "## Envs that did not measure", ""]
        L += [f"* `{k}` — {v}" for k, v in sorted(errs.items())]
    if res.get("import_errors"):
        L += ["", "## Modules absent in this interpreter", ""]
        L += [f"* `{k}` — {v}" for k, v in sorted(res["import_errors"].items())]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,64,256,1024,4096,16384")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument(
        "--ss",
        default="native",
        help="'native' lets each env use its own default_supersample — the picture "
        "se.make(name) gives. A number forces it on every env and measures something "
        "else; the old default of 1 quietly quartered the rays for 44 classes",
    )
    ap.add_argument("--max-out-gb", type=float, default=8.0)
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--decode-frames", type=int, default=300)
    ap.add_argument(
        "--cpu-frames",
        type=int,
        default=0,
        help="frames through the env's OFFICIAL renderer; 0 leaves that column blank",
    )
    ap.add_argument("--envs", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. the venv name")
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--out", default="docs/research/artifacts/results")
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    batches = [int(b) for b in a.batches.split(",")]
    ss = parse_ss(a.ss)
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
        "cpu_count": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "jpeg_quality": a.jpeg_quality,
        "loadavg": [os.getloadavg()[0]],
        "registry_size": len(se.registry.REGISTRY),
        "registry": sorted(se.registry.REGISTRY),
        "import_errors": se.import_errors(),
        "envs": {},
    }

    prev = outdir / f"bench_vs_decode{tag}.json"
    if prev.exists():
        try:
            res["envs"] = json.loads(prev.read_text()).get("envs", {})
        except Exception:
            pass

    def flush() -> None:
        res["loadavg"] = res["loadavg"][:1] + [os.getloadavg()[0]]
        (outdir / f"bench_vs_decode{tag}.json").write_text(json.dumps(res, indent=2))
        (outdir / f"bench_vs_decode{tag}.md").write_text(markdown(res))

    skip = {s for s in a.skip.split(",") if s}
    wanted = [n for n in sorted(se.registry.REGISTRY) if (not a.envs or n in a.envs.split(",")) and n not in skip]
    print(f"registry {len(se.registry.REGISTRY)}, measuring {len(wanted)}", flush=True)

    for name in wanted:
        done = res["envs"].get(name, {})
        if done.get("gpu_best") and not (a.cpu_frames and _cpu_pending(done)):
            print(f"[{name}] already measured, skipping", flush=True)
            continue
        try:
            probe = se.make(name, num_envs=1, device="cuda")
            size = probe.size
            pool = probe.sample_states(min(max(batches), a.pool))
            frame = probe.render(pool[:1]).cpu().numpy()[0]
            state_bytes = int(np.asarray(pool[:1]).nbytes)
            hw = (int(frame.shape[0]), int(frame.shape[1]))
            del probe
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{name}] cannot instantiate: {e}", flush=True)
            res["envs"][name] = _err(e)
            flush()
            continue

        # A rerun that only adds the official-renderer column keeps the GPU and decode numbers it
        # already has. Re-measuring them would put two different allocations' timings in one row.
        entry = done if done.get("gpu_best") else None
        if entry is None:
            entry = {
                "size": size,
                "state_bytes": state_bytes,
                "decode": decode_baseline(frame, a.jpeg_quality, a.decode_frames, scratch),
                "gpu": [],
            }
            best = None
            for B in batches:
                r = gpu_throughput(name, B, size, ss, a.reps, a.max_out_gb, pool, hw)
                r["batch"] = B
                entry["gpu"].append(r)
                if "fps" in r and (best is None or r["fps"] > best["fps"]):
                    # The resolved supersample travels with the number. Without it a row cannot be
                    # compared against bench_storage.py's, which never forced one — so the two
                    # tables described different pictures for the same 44 envs.
                    best = {
                        "batch": B,
                        "fps": r["fps"],
                        "supersample": r.get("supersample"),
                        "cpu_frac": r.get("cpu_frac"),
                        "loadavg": r.get("loadavg"),
                    }
            entry["gpu_best"] = best
        best = entry["gpu_best"]

        if a.cpu_frames and _cpu_pending(entry):
            # The official renderer is the one column that can take the process down (highway-env's
            # pygame path SIGABRTs), which is why it runs last and the file is written straight after.
            entry["cpu"] = cpu_throughput(se.registry.REGISTRY[name], name, size, a.cpu_frames, pool)
            c = _cpu_fps(entry)
            # "absent" and "raised" are different facts and were printed as the same sentence, which is
            # how a bug in the reference bridge read as 39 envs simply not having a reference package.
            if c:
                why = f"{c:,.1f} f/s"
            elif entry["cpu"] is None:
                why = "no reference bridge on this class"
            else:
                why = f"FAILED — {entry['cpu'].get('error', entry['cpu'])}"
            print(f"[{name}] official renderer: {why}", flush=True)
        res["envs"][name] = entry
        if best:
            ratio = best["fps"] / entry["decode"]["decode_fps_per_core"]
            print(
                f"[{name}] {hw[0]}x{hw[1]}  gpu {best['fps']:,.0f} f/s @b{best['batch']}  "
                f"decode {entry['decode']['decode_fps_per_core']:,.0f} f/s/core  "
                f"= {ratio:,.1f} cores",
                flush=True,
            )
        else:
            print(f"[{name}] no GPU point measured", flush=True)
        flush()

    flush()
    print(markdown(res))


if __name__ == "__main__":
    main()
