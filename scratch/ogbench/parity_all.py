"""Parity for all 25 registered cells, on the same 64 states the conformance suite uses.

Reports the whole-frame numbers the suite asserts on AND the agent-pixel numbers it does not, because
the second set is the only part this renderer is responsible for once the plate is baked. Also fixes
the evidence for ``parity_interior_min``: a relaxation is only honest if the measured floor needs it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_pkg import agent_mask, erode  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"


def main():
    os.makedirs(OUT, exist_ok=True)
    import stable_envs as se
    from stable_envs.testing import conformance as C
    log = open(os.path.join(OUT, "parity_all.log"), "w")
    names = [n for n in sorted(se.REGISTRY)
             if any(n.startswith(p) for p in ("Pointmaze", "Antmaze", "Humanoidmaze"))]
    res = []
    for nm in names:
        try:
            e = se.make(nm, batch=16, device="cuda")
            cls = type(e)
            st = e.sample_states(64)
            ours = e.render(st).cpu().numpy()
            ref = cls.reference_env(size=e.size)
            off = cls.reference_render(ref, st)
            pa = C.pixel_agreement(ours, off)
            am = agent_mask(ref, st, e._scene().agent_ids, e.size)
            d = np.abs(ours.astype(np.int16) - off.astype(np.int16)).max(-1)
            ai = erode(am)
            r = dict(env=nm, loco=cls.LOCO, maze=cls.MAZE,
                     interior_exact=round(float(pa["interior_exact_frac"]), 6),
                     exact=round(float(pa["exact_frac"]), 6), mean=round(float(pa["mean_abs"]), 5),
                     agent_px_per_frame=round(float(am.sum()) / len(st), 1),
                     agent_interior_px=int(ai.sum()),
                     agent_int_le1=round(float((d[ai] <= 1).mean()), 4) if ai.sum() else None,
                     agent_int_mean=round(float(d[ai].mean()), 3) if ai.sum() else None,
                     agent_all_le8=round(float((d[am] <= 8).mean()), 4) if am.sum() else None,
                     non_agent_exact=round(float((d[~am] == 0).mean()), 6))
        except Exception:  # noqa: BLE001
            import traceback
            log.write(f"FAIL {nm}\n{traceback.format_exc()}\n")
            log.flush()
            continue
        res.append(r)
        log.write(json.dumps(r) + "\n")
        log.flush()
    log.close()
    with open(os.path.join(OUT, "parity_all.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
