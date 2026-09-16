"""Gate A on both cells, through the REPO'S checks, with the mandatory roundtrip check included.

Written as its own artifact because the gate asks for Gate A explicitly and because the last version
of this list quietly omitted `check_state_roundtrip` -- five checks reported as "Gate A" while the
one that was failing was not among them. The list here is asserted against the repo's own set so a
future omission is a failure rather than a silence.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket

import numpy as np

REQUIRED = ("check_state_roundtrip", "check_determinism", "check_batch_invariance",
            "check_batch_size_invariance", "check_state_dims_matter")
NEEDS_STATES = {"check_batch_size_invariance"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    import stable_envs as se
    import stable_envs.testing.conformance as C

    missing = [f for f in REQUIRED if getattr(C, f, None) is None]
    assert not missing, f"this build has no {missing} -- Gate A cannot be claimed without them"

    report: dict = {"job": f"job {os.environ.get('SLURM_JOB_ID')} on {socket.gethostname()}",
                    "checks": list(REQUIRED)}
    ok = True
    for cname in a.cells:
        env = se.make(cname, batch=a.n, device="cuda")
        states = env.sample_states(env.B, np.random.default_rng(99))
        cell: dict = {}
        for fn in REQUIRED:
            f = getattr(C, fn)
            try:
                cell[fn] = f(env, states) if fn in NEEDS_STATES else f(env)
            except Exception as e:                                           # noqa: BLE001
                cell[fn] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:400]}
                ok = False
            print(f"{cname} {fn}: {json.dumps(cell[fn])[:240]}", flush=True)
        report[cname] = cell
    report["all_checks_ran"] = ok
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_a.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nGATE A ran for {a.cells}; errors: {'none' if ok else 'SEE ABOVE'}", flush=True)
    return 0 if ok else 6


if __name__ == "__main__":
    raise SystemExit(main())
