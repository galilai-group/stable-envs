"""Step 6 / Oscar re-dump of the JPEG-in-Lance benchmark inputs.

Same contract as ``examples/dump_frames.py`` (render with OUR GPU renderer, encode with PIL at JPEG
quality 95 exactly as ``stable_worldmodel.data.formats.lance._encode_frame`` does), with three
deliberate differences, all of which exist so the storage baseline is comparable to the Step-6
renderer certification rows rather than to a 256 px request:

1. **Size.** The original dump asked for ``size=256`` and fell back to native only when an env
   refused. The Step-6 CPU/GPU rows are certified at each env's OWN size (``env.size`` as
   ``se.make`` gives it), so that is what is dumped here -- no ``size=`` argument at all.
2. **Config set.** Only the 36 ``role == registered_default`` configs of
   ``examples/results/step4/certification.json``.
3. **Seeded states.** ``sample_states`` is given an explicit ``np.random.default_rng(seed)`` so the
   dump is reproducible; the original took the env's default entropy.

Rendering is chunked (``--render-chunk``) purely to bound device memory. Every config here passes
``check_chunked_render_matches`` with observed 0 in the frozen Step-4 certification, so chunking is
pixel-identical to a single call.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import stable_envs as se


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", default="", help="comma list; default = the 36 registered_default rows")
    ap.add_argument("--skip", default="")
    ap.add_argument("--cert", default="examples/results/step4/certification.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from PIL import Image

    root = Path(__file__).resolve().parents[3]
    cert = json.loads(Path(a.cert).read_text()) if Path(a.cert).exists() else \
        json.loads((root / a.cert).read_text())
    default_rows = sorted(k for k, v in cert["configs"].items()
                          if v.get("role") == "registered_default")
    wanted = a.envs.split(",") if a.envs else default_rows
    if a.skip:
        skip = set(a.skip.split(","))
        wanted = [w for w in wanted if w not in skip]

    prov = {
        "renderer": "stable_envs GPU renderer (env.render on device=%s)" % a.device,
        "git_commit": git_commit(root),
        "repo_root": str(root),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "node": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "jpeg_quality": a.jpeg_quality,
        "frames": a.frames,
        "seed": a.seed,
        "size_policy": "env.size (each env's own certified size); no size= argument",
        "state_source": "env.sample_states(frames, rng=np.random.default_rng(seed))",
        "render_chunk": a.render_chunk,
    }
    for m in ("mujoco", "mujoco_warp", "torchvision", "numpy"):
        try:
            import importlib.metadata as md
            prov[m] = md.version(m)
        except Exception:
            prov[m] = None

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"_provenance_{socket.gethostname()}_{os.environ.get('SLURM_JOB_ID','x')}.json").write_text(
        json.dumps(prov, indent=2))
    print(f"dumping {len(wanted)} envs at native size, q{a.jpeg_quality}, {a.frames} frames", flush=True)
    print(json.dumps(prov, indent=2), flush=True)

    for name in wanted:
        dest = outdir / f"{name}.npz"
        if dest.exists():
            print(f"[{name}] exists, skipping", flush=True)
            continue
        try:
            env = se.make(name, batch=1, device=a.device)
        except Exception as e:
            print(f"[{name}] SKIP cannot instantiate: {type(e).__name__}: {e}", flush=True)
            continue
        try:
            size = int(env.size)
            pool = env.sample_states(a.frames, rng=np.random.default_rng(a.seed))
            outs = []
            for i in range(0, a.frames, a.render_chunk):
                outs.append(env.render(pool[i:i + a.render_chunk]).cpu().numpy())
            frames = np.concatenate(outs, axis=0)
        except Exception as e:
            print(f"[{name}] SKIP render failed: {type(e).__name__}: {e}", flush=True)
            del env
            torch.cuda.empty_cache()
            continue
        finally:
            try:
                del env
            except Exception:
                pass
            torch.cuda.empty_cache()

        blobs = []
        for f in frames:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, format="JPEG", quality=a.jpeg_quality)
            blobs.append(buf.getvalue())
        meta = {"env": name, "size": size, "requested_size": None,
                "size_note": "native/certified size (env.size); Step-6 re-dump",
                "h": int(frames.shape[1]), "w": int(frames.shape[2]),
                "n": len(blobs), "jpeg_quality": a.jpeg_quality,
                "state_bytes": int(np.asarray(pool[:1]).nbytes),
                "state_dim": int(np.asarray(pool).reshape(len(pool), -1).shape[1]),
                "jpeg_bytes_total": int(sum(len(b) for b in blobs)),
                "raw_bytes_per_frame": int(frames.shape[1] * frames.shape[2] * 3),
                "provenance": prov}
        # np.savez appends ".npz" to whatever base it is given, so stage on a base name and
        # rename the file it actually wrote.
        tmp_base = dest.parent / f"{name}.part"
        np.savez(tmp_base, blobs=np.array(blobs, dtype=object), meta=json.dumps(meta))
        os.replace(f"{tmp_base}.npz", dest)
        print(f"[{name}] {meta['h']}x{meta['w']}  {len(blobs)} frames  "
              f"{meta['jpeg_bytes_total'] / 1e6:.1f} MB jpeg  "
              f"({meta['jpeg_bytes_total'] / len(blobs):,.0f} B/frame vs "
              f"{meta['raw_bytes_per_frame']:,} raw)", flush=True)


if __name__ == "__main__":
    main()
