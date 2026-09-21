"""Before any downstream number: do the two Reacher renderers draw the same picture at all?

The PushT arm had this probe (``docs/research/artifacts/step5_lewm_pusht/parity_probe.py``) and it is what turned a
null result into a diagnosis. The Reacher arm needs it more, because ``stable_worldmodel``'s reacher
is not ``suite.load('reacher', 'hard')``: it recompiles the model through ``mjcf`` after editing
materials, geom types and densities, and its variation space defaults include
``rendering.render_target = 0``, which sets the target material's ALPHA to zero.

That last one matters more than a colour would. ``stable_envs``' DM Control appearance space carries
material colour, light diffuse and camera fovy -- it has no alpha, because MuJoCo's own visibility
filter is applied when the scene is baked. So if the reference declines to draw its target and we
draw it, every frame differs by a red dot in a place that moves per episode, and a downstream
comparison would be measuring that dot rather than the renderer.

This probe therefore answers three questions with pixels rather than with reasoning:

  1. Does the reference draw the target at all? Asked by rendering the reference twice with the
     target geom moved and comparing: an invisible geom cannot change the frame.
  2. How close is our frame to the reference's on identical states, with the target where the env
     put it?
  3. How close is it with the target moved out of the camera's view -- the only way this renderer
     can express "do not draw that geom"?

    python docs/research/artifacts/step5_downstream/reacher/parity_probe.py --n 32 --out tests/fixtures/regression/join/step5/reacher
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

#: Where the harness puts a geom the reference refuses to draw. Far outside a scene whose arm
#: reaches 0.22 m and whose camera sees roughly +/-0.3 m, and finite so MuJoCo still computes a pose.
OFFSCREEN = (10.0, 10.0)

#: Step 6's replacement for it: the style leaf ``DMCReacher*`` now declares. Zeroing it zeroes the
#: target geom's rgba alpha, which is what ``stable_worldmodel``'s ``render_target = 0`` does to the
#: material and what MuJoCo's own renderer honours -- so the two sides now express the same thing the
#: same way instead of one of them faking it with a position.
VISIBILITY = "visibility.target"

#: Fixed so the probe is reproducible and so the states are the env's own, not a box someone chose.
PROBE_SEED = 5150


def rollout(swm_env, n: int, seed: int, steps_between: int = 5) -> list[dict]:
    """``n`` states the official env genuinely reaches, with its own reset and random actions."""
    rng = np.random.default_rng(seed)
    lo, hi = swm_env.action_space.low, swm_env.action_space.high
    out = []
    swm_env.reset(seed=seed)
    for i in range(n):
        # A fresh episode every few states, so the probe sees more than one target draw: the target
        # position is the half of the state that only changes at initialize_episode.
        if i % 8 == 0:
            swm_env.reset(seed=seed + i)
        for _ in range(int(rng.integers(1, steps_between + 1))):
            swm_env.step(rng.uniform(lo, hi).astype(swm_env.action_spec_dtype))
        p = swm_env.env.physics
        out.append({"qpos": np.array(p.data.qpos, np.float64), "qvel": np.array(p.data.qvel, np.float64)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=32, help="states to compare")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--task", default="qpos_match", help="the swm reacher task the eval config names")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "tests" / "fixtures" / "regression" / "join" / "step5" / "reacher"
    )
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import sys

    import mujoco
    import torch  # noqa: F401  -- imported for the side effect of loading before dm_control's GL
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

    import stable_envs as se
    from stable_envs.testing import conformance as C

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from state_bridge import read_state, slot_names

    swm_env = ReacherDMControlWrapper(task=args.task, seed=PROBE_SEED)
    swm_env.reset(seed=PROBE_SEED)
    ours = se.make("DMCReacherHard", num_envs=1, size=args.size, device=args.device)
    slots = slot_names(ours)
    print("our state layout:", ours.state_names, "-> slots", slots)

    p = swm_env.env.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    tgt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")
    if tgt < 0:
        raise SystemExit("the official reacher has no geom named 'target'")

    # (1) Does the reference draw its target? Move it and see whether the frame notices.
    frames_here, frames_moved = [], []
    for _ in range(4):
        swm_env.reset(seed=PROBE_SEED)
        frames_here.append(swm_env.render(args.size, args.size).copy())
        keep = np.array(m.geom_pos[tgt], np.float64)
        m.geom_pos[tgt, :2] = OFFSCREEN
        p.forward()
        frames_moved.append(swm_env.render(args.size, args.size).copy())
        m.geom_pos[tgt] = keep
        p.forward()
    moved_delta = int(np.abs(np.stack(frames_here).astype(int) - np.stack(frames_moved).astype(int)).max())
    target_drawn = moved_delta > 0
    print(
        f"reference target visibility: moving the target changes at most {moved_delta}/255 -> "
        f"{'DRAWN' if target_drawn else 'NOT DRAWN'}"
    )

    states = rollout(swm_env, args.n, PROBE_SEED)
    # ``(state override, style override)`` per mode. ``hidden`` is Step 6's representation -- the
    # state keeps the target position the episode drew and the env's own ``visibility.target`` leaf
    # says it is not painted; ``offscreen`` is Step 5's mechanism, kept so the recorded arm stays
    # reproducible and so the two can be compared on the same states in one run.
    modes = {
        "as_placed": (None, None),
        "hidden": (None, {VISIBILITY: 0}),
        "offscreen": ({"target": (*OFFSCREEN, 0.01)}, None),
    }
    res = {
        "task": args.task,
        "size": args.size,
        "n": args.n,
        "reference_target_drawn": bool(target_drawn),
        "reference_target_move_delta": moved_delta,
        "our_state_names": list(ours.state_names),
        "slots": [list(s) for s in slots],
        "modes": {},
    }
    keep_frames = {}

    for mode, (override, variation) in modes.items():
        if variation and VISIBILITY not in ours.style_names:
            raise SystemExit(f"DMCReacherHard declares no {VISIBILITY!r}: {ours.style_names}")
        off_all, our_all = [], []
        for s in states:
            with p.reset_context():
                p.data.qpos[:] = s["qpos"]
                p.data.qvel[:] = s["qvel"]
            off_all.append(swm_env.render(args.size, args.size).copy())
            st = read_state(ours, swm_env, slots=slots, target_override=override)
            our_all.append(ours.render(st.reshape(1, -1), variation=variation)[0].cpu().numpy())
        off_a, our_a = np.stack(off_all), np.stack(our_all)
        keep_frames[mode] = (off_a, our_a)
        agree = C.pixel_agreement(our_a, off_a)
        ssim = C.ssim(our_a, off_a)
        res["modes"][mode] = {
            "exact_frac": float(agree["exact_frac"]),
            "interior_exact_frac": (
                None if agree["interior_exact_frac"] is None else float(agree["interior_exact_frac"])
            ),
            "interior_px": int(agree["interior_px"]),
            "mean_abs": float(agree["mean_abs"]),
            "max_abs": int(agree["max_abs"]),
            "within8_frac": float(agree["within8_frac"]),
            "ssim": float(ssim),
        }
        print(
            f"{mode:>10}: exact {agree['exact_frac']:.4f}  mean|d| {agree['mean_abs']:.3f}  "
            f"max {agree['max_abs']}  ssim {ssim:.4f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    best = min(res["modes"], key=lambda k: res["modes"][k]["mean_abs"])
    res["closest_mode"] = best
    (args.out / "parity_probe.json").write_text(json.dumps(res, indent=2))

    try:
        import imageio.v2 as imageio

        off_a, our_a = keep_frames[best]
        k = min(4, len(off_a))
        rows = [
            np.concatenate(
                [
                    off_a[i],
                    our_a[i],
                    np.clip(4 * np.abs(off_a[i].astype(int) - our_a[i].astype(int)), 0, 255).astype(np.uint8),
                ],
                axis=1,
            )
            for i in range(k)
        ]
        imageio.imwrite(args.out / "parity_probe.png", np.concatenate(rows, axis=0))
        print(f"wrote {args.out / 'parity_probe.png'} (official | ours | 4x|diff|, mode={best})")
    except Exception as e:  # a missing imageio is not a reason to lose the numbers
        print(f"(no contact sheet: {e})")
    print(f"wrote {args.out / 'parity_probe.json'}")


if __name__ == "__main__":
    main()
