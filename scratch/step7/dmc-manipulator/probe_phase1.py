"""Phase-1 acceptance on the REGISTERED ``DMCManipulatorBringBall``: Gate A + parity + the leaves.

No patching of anything: this drives ``se.REGISTRY`` exactly as the conformance suite does.

  * the Gate-A mandatory checks a local CPU can reach, at 224 (the certified size) and 84;
  * parity against ``physics.render`` per draw at both sizes;
  * the three leaves that carry ``stable_worldmodel``'s registered default -- ``visibility.target_ball``
    (render_target = 0), ``target_ball.shape`` (its .015 sphere) and ``light.diffuse_scale`` (0.7) --
    each measured for the pixels it moves, and the SWM-default frame assembled from all three;
  * whether the two ``target_ball`` quaternion dimensions become LIVE under the BOX alternative, which
    is what bounds the declared ``check_state_dims_matter`` failure.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch

CELL = "DMCManipulatorBringBall"


def diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return dict(px=int((d > 0).sum()), frac=float((d > 0).mean()), max=int(d.max()),
                mean=float(d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 84])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/phase1")
    a = ap.parse_args()

    import mujoco
    from importlib.metadata import version as _v
    import stable_envs as se
    from stable_envs.testing import conformance as C

    cls = se.REGISTRY[CELL]
    res = dict(cell=CELL, mujoco=mujoco.__version__, dm_control=_v("dm_control"),
               device="cpu", default_size=int(cls.default_size),
               host="Apple M5 Mac (local); NOT a certification host",
               reference_pair_note=("measured on mujoco 3.12.0 + dm_control 1.0.45; the declared DMC "
                                    "reference pair is mujoco 3.11.0 + dm_control 1.0.44 on Oscar"),
               state_dims_note_declared=bool(getattr(cls, "state_dims_note", None)))

    for S in a.sizes:
        env = cls(batch=8, size=S, device="cpu", seed=0)
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
            ("check_task_object_visibility", lambda: C.check_task_object_visibility(env, n=16)),
        ]:
            try:
                checks[nm] = fn()
            except Exception as e:                                   # noqa: BLE001
                checks[nm] = {"ERROR": f"{type(e).__name__}: {e}"}
            print(f"[{S}]", nm, json.dumps(checks[nm], default=str)[:300])
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
            print(f"[{S}]", nm, json.dumps(checks[nm], default=str)[:300])

        ref = C.reference_env(cls, S)
        per_draw = []
        for k in range(a.draws):
            st = env.sample_states(a.n, np.random.default_rng(1000 + k))
            got = env.render(torch.as_tensor(st)).cpu().numpy()
            want = cls.reference_render(ref, st)
            pa = C.pixel_agreement(got, want)
            pa["ssim"] = float(C.ssim(got, want))
            per_draw.append(pa)
            print(f"[{S}] draw {k}: exact {pa['exact_frac']:.5f} interior "
                  f"{pa['interior_exact_frac']} bad {pa['interior_bad_px']}/{pa['interior_px']} "
                  f"mean {pa['mean_abs']:.4f} max {pa['max_abs']} ssim {pa['ssim']:.5f}")
        res[f"size_{S}"] = dict(checks=checks, parity_per_draw=per_draw)

    # ---- the SWM registered default, reached through leaves ----------------
    S = int(cls.default_size)
    env = cls(batch=8, size=S, device="cpu", seed=0)
    st = env.sample_states(16, np.random.default_rng(77))
    t = torch.as_tensor(st)
    base = env.render(t).cpu().numpy()
    leaves = {}
    for nm, over in [
        ("visibility.target_ball=0", {"visibility.target_ball": [0]}),
        ("target_ball.shape=0 (swm box .015)", {"target_ball.shape": [0]}),
        ("target_ball.shape=1 (swm sphere .015)", {"target_ball.shape": [1]}),
        ("light.diffuse_scale=0.7/0.6 (swm intensity 0.7)", {"light.diffuse_scale": [0.7 / 0.6]}),
        ("swm registered default (all three)",
         {"visibility.target_ball": [0], "target_ball.shape": [1],
          "light.diffuse_scale": [0.7 / 0.6]}),
    ]:
        leaves[nm] = diff(env.render(t, variation=over).cpu().numpy(), base)
        print("leaf", nm, leaves[nm])
    res["swm_default_through_leaves"] = leaves

    # ---- do the quaternion dims come alive under the BOX alternative? ------
    names = list(env.state_names)
    iw, iy = names.index("target_ball_quat_qw"), names.index("target_ball_quat_qy")
    rot = st.copy()
    ang = 2.0 * np.arctan2(rot[:, iy], rot[:, iw]) + 0.6
    rot[:, iw], rot[:, iy] = np.cos(ang / 2), np.sin(ang / 2)
    box = {"target_ball.shape": [0]}
    res["quat_dims_under_shapes"] = {
        "default_sphere_.02": diff(env.render(torch.as_tensor(rot)).cpu().numpy(), base),
        "box_.015": diff(env.render(torch.as_tensor(rot), variation=box).cpu().numpy(),
                         env.render(t, variation=box).cpu().numpy()),
    }
    print("quat under shapes", json.dumps(res["quat_dims_under_shapes"], default=float))

    for B in (1, 64, 256):
        e = cls(batch=B, size=S, device="cpu", seed=0)
        s = torch.as_tensor(e.sample_states(B, np.random.default_rng(11)))
        e.render(s)
        t0 = time.perf_counter()
        for _ in range(3):
            e.render(s)
        res[f"cpu_fps_B{B}"] = float(B / ((time.perf_counter() - t0) / 3))
        print(f"B={B}: {res[f'cpu_fps_B{B}']:.1f} f/s (indicative, local Mac CPU)")

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "phase1_acceptance.json")
    json.dump(res, open(p, "w"), indent=1, default=str)
    print("wrote", p)


if __name__ == "__main__":
    main()
