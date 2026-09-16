"""Measure DOWNSAMPLE_EXACT_FRAC for the two cells, under the table's OWN protocol.

The bound is a property of a state set as much as of a renderer, which is why `conformance.py`
carries `downsample_protocol_states` -- the GPU module's `inputs` draw, 8 states at
`render_worker.STATE_SEED`, default appearance -- rather than letting each caller bring its own
states. Measured on any other draw the number is not comparable with the rest of the table, and a
bound applied to the wrong draw fails for the draw instead of for the renderer.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    a = ap.parse_args()

    import numpy as np

    import stable_envs as se
    import stable_envs.testing.conformance as C

    rep: dict = {"job": f"alloc {os.environ.get('SE7_ALLOC_JOB')} on {socket.gethostname()}",
                 "protocol": None}
    for cname in a.cells:
        env = se.make(cname, batch=C.DOWNSAMPLE_PROTOCOL_N, device="cuda")
        states = C.downsample_protocol_states(env)
        r = C.check_resolution_consistency(env, states)
        rep["protocol"] = C.downsample_protocol(states)
        rep[cname] = {"exact_frac": float(r["exact_frac"]), "mean_abs": float(r["mean_abs"]),
                      "size": int(env.size), "full": {k: (float(v) if isinstance(v, (int, float))
                                                          else v) for k, v in r.items()}}
        print(f"{cname}: exact_frac={r['exact_frac']:.6f}  mean {r['mean_abs']:.4f}/255  "
              f"size={env.size}", flush=True)
        print(f"    TABLE LINE:  \"{cname}\":{' ' * max(1, 22 - len(cname))}{r['exact_frac']:.6f},"
              f"   # mean {r['mean_abs']:.4f}/255", flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "downsample_bound.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    print(json.dumps(rep, indent=1, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
