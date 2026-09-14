"""Step 6C ticket 1, deliverable 1 — which frozen Step-4 configurations a raycast3d patch touches.

``examples/results/step4/certification.json`` carries a ``backends`` field, and for the DM Control
shard it says ``['mjwarp', 'raycast3d']`` on all fourteen cells. That field is a MODULE-level
attribution — ``stable_envs/envs/dmcontrol.py`` imports both renderers and serves two families out
of one module — so reading it as a per-configuration replay list over-counts.

This settles it by OBSERVATION rather than by reading: every raycast3d entry point is wrapped, each
configuration is rendered once through the public path, and the configuration is counted as being on
the path only if one of those functions actually ran. The frozen oracle's own device and digest are
carried across from certification.json so the replay obligation is stated with its provenance.

Read-only with respect to everything under examples/results/step4/.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# $PYTHONPATH DECIDES WHICH stable_envs IS MEASURED. A before/after run puts the baseline copy on
# PYTHONPATH for one leg and this branch for the other, so this file's own repo root must NOT be
# inserted ahead of it -- doing that made both legs import the branch and report a 1.00x speed-up as
# a real one. Appended, so the helper imports still resolve when PYTHONPATH is unset.
sys.path.append(str(ROOT))

ENTRY = ("cast_nearest", "cast_nearest_torch", "any_blocking", "any_blocking_torch",
         "composite", "static_depth", "screen_span", "intersect", "shade", "cached_rays")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    import stable_envs as se
    import stable_envs.render.raycast3d as R3

    hits: dict[str, int] = {}

    def wrap(name):
        raw = getattr(R3, name)

        def inner(*a, **k):
            hits[name] = hits.get(name, 0) + 1
            return raw(*a, **k)
        return inner

    for nm in ENTRY:
        setattr(R3, nm, wrap(nm))

    cert = json.loads((ROOT / "examples/results/step4/certification.json").read_text())
    conf = {}
    for shard in ("dmc", "ogb", "raster", "fetch", "swm"):
        p = ROOT / f"examples/results/step4/conformance/{shard}.json"
        if p.is_file():
            j = json.loads(p.read_text())
            conf[shard] = j
    out = {"host": socket.gethostname(), "device": args.device,
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "configs": {}}

    for cfg in sorted(cert["configs"]):
        v = cert["configs"][cfg]
        shard = v.get("shard")
        oc = (conf.get(shard, {}).get("configs", {}).get(cfg, {}) or {}).get("oracle", {})
        rec = {
            "declared_backends": v.get("backends", []),
            "shard": shard, "role": v.get("role"), "tier": v.get("tier"),
            "gate_a": v.get("gate_a"), "failures": v.get("failures", []),
            "siblings": v.get("siblings", []),
            "oracle_device": oc.get("device"), "oracle_batch": oc.get("batch"),
            "oracle_size": oc.get("size"), "oracle_supersample": oc.get("supersample"),
            "digest_ours": oc.get("digest_ours"), "digest_official": oc.get("digest_official"),
            "oracle_host": conf.get(shard, {}).get("provenance", {}).get("host"),
            "oracle_gpu": conf.get(shard, {}).get("provenance", {}).get("gpu"),
        }
        if "raycast3d" not in rec["declared_backends"]:
            rec["raycast3d_on_render_path"] = None       # not claimed; not probed
            out["configs"][cfg] = rec
            continue
        hits.clear()
        try:
            e = se.make(cfg, batch=args.batch, device=args.device)
            npz = ROOT / "examples/results/step4/states" / f"{cfg}.npz"
            st = np.load(npz)["states"][:args.batch] if npz.is_file() \
                else e.sample_states(args.batch)
            e.render(np.asarray(st, np.float32))
            rec["render_class"] = type(e).__mro__[1].__name__
            rec["render_module"] = type(e).__module__
            rec["raycast3d_entry_calls"] = dict(sorted(hits.items()))
            rec["raycast3d_on_render_path"] = bool(hits)
        except Exception as exc:                                        # noqa: BLE001
            rec["error"] = repr(exc)[:400]
            rec["raycast3d_on_render_path"] = "error"
        out["configs"][cfg] = rec
        print(f"{cfg:32s} declared={rec['declared_backends']} "
              f"on_path={rec.get('raycast3d_on_render_path')} "
              f"class={rec.get('render_class')} calls={list(rec.get('raycast3d_entry_calls', {}))}",
              flush=True)

    on = [k for k, v in out["configs"].items() if v.get("raycast3d_on_render_path") is True]
    off = [k for k, v in out["configs"].items()
           if "raycast3d" in v["declared_backends"] and v.get("raycast3d_on_render_path") is False]
    out["summary"] = {"declared_raycast3d": sorted(
        k for k, v in out["configs"].items() if "raycast3d" in v["declared_backends"]),
        "actually_on_render_path": sorted(on), "declared_but_not_on_path": sorted(off)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
