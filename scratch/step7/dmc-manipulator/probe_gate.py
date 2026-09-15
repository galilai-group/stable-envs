"""Phase-0: run the Gate-A mandatory conformance checks that a local CPU can reach, on the
prototype cell, plus a CPU throughput number (NOT a certification number -- Mac, not Oscar batch).

Same patch as ``probe_prototype``: the ``*_quat`` half of ``_episode_slots``'s guard is removed in
this process only. Nothing under ``stable_envs/`` is edited.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch

from probe_prototype import patched_episode_slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/phase0")
    a = ap.parse_args()

    import mujoco
    from importlib.metadata import version as _v
    from stable_envs.envs import dmcontrol as D
    from stable_envs.testing import conformance as C

    patched_episode_slots(D)
    Cell = type("DMCManipulatorBringBall", (D.DMControlSuite,), dict(
        DOMAIN="manipulator", TASK="bring_ball", CAMERA=0, default_size=a.size,
        name="dmc-manipulator-bring-ball",
        reference_id="dm_control/manipulator-bring_ball", __doc__="phase-0 prototype"))

    res = dict(size=a.size, device="cpu", mujoco=mujoco.__version__,
               dm_control=_v("dm_control"), host="Apple M5 Mac (local); NOT a certification host")

    env = Cell(batch=8, size=a.size, device="cpu", seed=0)
    checks = {}
    for nm, fn in [
        ("check_state_roundtrip", lambda: C.check_state_roundtrip(env, n=8)),
        ("check_state_dims_matter", lambda: C.check_state_dims_matter(env, n=8)),
        ("check_batch_invariance", lambda: C.check_batch_invariance(env, n=8)),
        ("check_determinism", lambda: C.check_determinism(env)),
        ("check_set_state_copies", lambda: C.check_set_state_copies(env)),
        ("check_style_is_state", lambda: C.check_style_is_state(env)),
        ("check_default_style_is_official", lambda: C.check_default_style_is_official(env)),
        ("check_per_env_variation", lambda: C.check_per_env_variation(env)),
        ("check_variations_do_something", lambda: C.check_variations_do_something(env)),
        ("check_colors_apply_exactly", lambda: C.check_colors_apply_exactly(env)),
    ]:
        try:
            checks[nm] = fn()
        except Exception as e:                                   # noqa: BLE001
            checks[nm] = {"ERROR": f"{type(e).__name__}: {e}"}
        print(nm, json.dumps(checks[nm], default=str)[:400])

    states = env.sample_states(8, np.random.default_rng(5))
    for nm, fn in [
        ("check_batch_size_invariance",
         lambda: C.check_batch_size_invariance(env, states, sizes=(1, 3, 8))),
        ("check_order_independence", lambda: C.check_order_independence(env, states)),
        ("check_render_is_pure", lambda: C.check_render_is_pure(env, states)),
        ("check_chunked_render_matches",
         lambda: C.check_chunked_render_matches(env, states, chunks=(1, 3, 8))),
        ("check_input_form_invariance", lambda: C.check_input_form_invariance(env, states)),
        ("check_resolution_consistency",
         lambda: C.check_resolution_consistency(env, C.downsample_protocol_states(env))),
    ]:
        try:
            checks[nm] = fn()
        except Exception as e:                                   # noqa: BLE001
            checks[nm] = {"ERROR": f"{type(e).__name__}: {e}"}
        print(nm, json.dumps(checks[nm], default=str)[:400])
    res["checks"] = checks

    # throughput, indicative only
    for B in (1, 64, 256):
        e = Cell(batch=B, size=a.size, device="cpu", seed=0)
        st = e.sample_states(B, np.random.default_rng(11))
        t = torch.as_tensor(st)
        e.render(t)
        t0 = time.perf_counter()
        for _ in range(3):
            e.render(t)
        dt = (time.perf_counter() - t0) / 3
        res[f"cpu_fps_B{B}"] = float(B / dt)
        print(f"B={B}: {B / dt:.1f} f/s (indicative, local Mac CPU)")

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"gate_{a.size}.json")
    json.dump(res, open(p, "w"), indent=1, default=str)
    print("wrote", p)


if __name__ == "__main__":
    main()
