"""Does any real bundle fail ``_unit_rays``, and how close does the closest one come?

RAYCAST-5 shipped ``_UNIT_D_TOL = 1e-4``. Independent review showed the identity is FALSE inside that
window -- at ``|d|^2 = 1 + 9e-5`` and 100 m a probe lost 113,008 of 126,184 unbounded hits -- so the
window was tightened to 1e-6. Tightening an admission check has a failure mode of its own: if a real
bundle now falls outside it, that bundle silently takes the unbounded path, the pixels stay identical
(the unbounded path is the definition) and the bound quietly stops applying where it was measured to
pay. Nothing downstream would notice.

So this instruments ``_unit_rays`` itself across every live config and reports two things: how many
bundles were REJECTED, which must be zero, and the worst ``||d|^2 - 1|`` actually seen, which is the
number the tolerance should be sized against.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

CONFIGS = """AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard""".split()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    import stable_envs as se
    from stable_envs.render import raycast3d as R3

    note = {"tool": "r6_unit_ray_probe", "host": socket.gethostname(), "device": a.device,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "stable_envs_from": se.__file__,
            "unit_d_tol": R3._UNIT_D_TOL, "bound_slack_noise": R3._BOUND_SLACK_NOISE,
            "primitive_bounds": R3.PRIMITIVE_BOUNDS, "configs": {}}

    real = R3._unit_rays
    state = {"calls": 0, "rejected": 0, "worst": 0.0}

    def spy(dd):
        state["calls"] += 1
        w = float((dd.detach() - 1.0).abs().max()) if dd.numel() else 0.0
        state["worst"] = max(state["worst"], w)
        ok = real(dd)
        if not ok:
            state["rejected"] += 1
        return ok

    R3._unit_rays = spy
    try:
        for name in a.configs:
            state.update(calls=0, rejected=0, worst=0.0)
            try:
                env = se.make(name, batch=a.batch, device=a.device)
                st = np.asarray(env.sample_states(a.batch), np.float32)
                env.render(st)
                torch.manual_seed(0)
                env.render(np.asarray(env.sample_states(a.batch), np.float32))
                note["configs"][name] = {"calls": state["calls"], "rejected": state["rejected"],
                                         "worst_abs_dd_minus_1": state["worst"]}
            except Exception as e:                                         # noqa: BLE001
                note["configs"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{name:28s} {note['configs'][name]}", flush=True)
    finally:
        R3._unit_rays = real

    ok = [v for v in note["configs"].values() if "calls" in v]
    note["summary"] = {
        "configs_measured": len(ok),
        "total_calls": sum(v["calls"] for v in ok),
        "total_rejected": sum(v["rejected"] for v in ok),
        "worst_abs_dd_minus_1": max((v["worst_abs_dd_minus_1"] for v in ok), default=None),
    }
    w, tol = note["summary"]["worst_abs_dd_minus_1"], R3._UNIT_D_TOL
    note["summary"]["headroom_x"] = (tol / w) if w else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(note, indent=1))
    print(f"\nSUMMARY {note['summary']}")
    print(f"REJECTED_BUNDLES={note['summary']['total_rejected']} "
          f"(must be 0; anything else means a real bundle now takes the unbounded path)")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
