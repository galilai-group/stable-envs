"""Collect, per registered_default config, our renderer's certified fps on Oscar hardware.

Two shapes of committed artifact are read:

* the Step-3 sweep shape (``{batches, envs:[{env, device, size, decode, cells:[{batch, fps}]}]}``),
  used by the Step-6 raster2d / raycast3d / tilemap / cpu-3d shards and by
  ``examples/results/step3/merged.json``;
* the gfxdraw paired shape (``{batch, envs:{Name:{before:{...}, after:{se_fps_median}}}}``).

The GPU cell reported is the best batch from the MANDATORY certification list
(1, 16, 64, 256, 1024, 4096) -- certification.md s6.3 takes R and `cores` "at the best certified
batch (i.e. from the mandatory list, not from 65536)". The CPU cell is the best batch at the
**1-thread** budget, which s5.3 fixes as the certification budget; an all-core number is collected
separately and never substituted.

Every value carries the file it came from. Nothing is averaged across files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MANDATORY = [1, 16, 64, 256, 1024, 4096]


def step3_shape(path: Path):
    d = json.loads(path.read_text())
    if not (isinstance(d, dict) and "envs" in d and isinstance(d["envs"], list)):
        return
    prov = d.get("provenance", {})
    for e in d["envs"]:
        if not isinstance(e, dict) or "cells" not in e:
            continue
        cells = [c for c in e["cells"] if c.get("fps") and c.get("batch") in MANDATORY]
        if not cells:
            continue
        best = max(cells, key=lambda c: c["fps"])
        yield {
            "env": e.get("env"), "device": e.get("device"), "size": e.get("size"),
            "shape": e.get("shape"), "supersample": e.get("supersample"),
            "backend": e.get("backend"), "tier": e.get("fidelity_tier"),
            "fps": best["fps"], "batch": best["batch"],
            "all_cells": {c["batch"]: c["fps"] for c in e["cells"] if c.get("fps")},
            "threads": prov.get("torch_threads"), "host": prov.get("host"),
            "device_name": prov.get("device_name"), "cpu": prov.get("cpu"),
            "git_commit": prov.get("git_commit"), "slurm_job_id": prov.get("slurm_job_id"),
            "decode_q90_fps_per_core": (e.get("decode") or {}).get("decode_fps_per_core"),
            "decode_jpeg_bytes_q90": (e.get("decode") or {}).get("jpeg_bytes"),
            "official_cpu_fps": (e.get("official_cpu") or {}).get("fps"),
            "source": str(path),
        }


def paired_shape(path: Path):
    d = json.loads(path.read_text())
    envs = d.get("envs")
    if not (isinstance(envs, dict) and envs and "batch" in d):
        return
    first = next(iter(envs.values()))
    if not (isinstance(first, dict) and "after" in first):
        return
    for name, v in envs.items():
        a = v.get("after") or {}
        if not a.get("se_fps_median"):
            continue
        yield {
            "env": name,
            "device": "cuda" if d.get("device_name") else ("cpu" if "cpu" in path.name else None),
            "size": None, "shape": None, "supersample": None,
            "backend": "gfxdraw", "tier": None,
            "fps": a["se_fps_median"], "batch": d["batch"],
            "all_cells": {d["batch"]: a["se_fps_median"]},
            "threads": d.get("threads"), "host": d.get("host"),
            "device_name": d.get("device_name"), "cpu": d.get("cpu"),
            "git_commit": None, "slurm_job_id": None,
            "decode_q90_fps_per_core": None, "decode_jpeg_bytes_q90": None,
            "official_cpu_fps": a.get("official_fps_median"),
            "source": str(path),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="examples/results/step6/lance/renderer_rows.json")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    cert = json.loads((root / "examples/results/step4/certification.json").read_text())
    wanted = sorted(k for k, v in cert["configs"].items() if v.get("role") == "registered_default")

    files = sorted(set(
        list((root / "examples/results/step6").rglob("*.json"))
        + [root / "examples/results/step3/merged.json"]
        + list((root / "examples/results/step3/gpu").glob("*.json"))
        + list((root / "examples/results/step3/cpu").glob("*.json"))
    ))
    rows = []
    for f in files:
        if "/replay/" in str(f) or "/conformance/" in str(f):
            continue
        try:
            for r in list(step3_shape(f)) + list(paired_shape(f)):
                if r["env"] in wanted:
                    rows.append(r)
        except Exception:
            continue
    out = {"what": "our renderer's fps per registered_default config, harvested with provenance",
           "mandatory_batches": MANDATORY, "n_rows": len(rows), "rows": rows}
    p = root / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"{len(rows)} rows -> {p}")
    seen = {}
    for r in rows:
        seen.setdefault(r["env"], set()).add(r["device"])
    for w in wanted:
        print(f"  {w:34s} {sorted(seen.get(w, [])) or 'MISSING'}")


if __name__ == "__main__":
    main()
