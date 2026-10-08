"""Stage A — record one RoboCasa trajectory and the frames the OFFICIAL renderer produced for it.

Runs in the ``robocasa`` conda env (robosuite 1.5.2, robocasa 1.0.1, mujoco 3.3.1). It writes
everything stage B needs and nothing that requires robosuite to read back: the expanded MJCF carries
absolute asset paths and compiles standalone, so the MJWarp half runs in a different interpreter with
a different MuJoCo.

Three things are recorded that a naive dump would omit, each of which silently invalidates the
comparison if it is missing:

* **The joint LAYOUT, not just ``qpos``.** Stage B recompiles the same XML under a different MuJoCo,
  and two compilations of one scene need not order joints identically. A ``qpos`` copied by index is
  then a mis-POSED scene scored as a mis-BUILT one -- the single most expensive mistake in this
  family's history (``pre-repository-cleanup:docs/research/robocasa.md`` Section 9, the retracted 94.95 %). The names and
  ``jnt_qposadr`` travel with the array so stage B can map by name and assert it did.
* **The official frame comes from ``env.sim.render``**, i.e. robosuite's own offscreen context with
  ``vopt.geomgroup[1] = True`` and everything else off. A bare ``mujoco.Renderer`` defaults to geom
  groups 0-2 and paints robosuite's ~730 COLLISION primitives over the scene, producing a frame
  RoboCasa never shows. The frame is flipped vertically because ``sim.render`` returns bottom-up.
* **A steady-state throughput number for that same renderer**, measured after the model is compiled
  and the context is live, so the ratio stage C prints is loop-against-loop rather than
  loop-against-setup.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

DEFAULT_OUT = "/mnt/vast/home/randall.balestriero/robocasa_visual_work"
#: The two kitchens the fidelity table brackets: the best and the worst of the four measured
#: (``pre-repository-cleanup:docs/research/robocasa.md`` Section 11 -- within-8 0.992 against 0.188). Showing only the first
#: would be a demo; showing both is the measurement.
KITCHENS = {"L52S03": (52, 3), "L45S33": (45, 33)}
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToSink")
    ap.add_argument("--kitchen", default="L52S03", choices=sorted(KITCHENS))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    layout, style = KITCHENS[args.kitchen]
    d = os.path.join(args.out, args.kitchen)
    os.makedirs(d, exist_ok=True)

    import mujoco
    from robocasa.utils.env_utils import create_env

    env = create_env(
        env_name=args.task,
        robots="PandaOmron",
        camera_names=list(CAMERAS),
        camera_widths=args.res,
        camera_heights=args.res,
        seed=args.seed,
        render_onscreen=False,
        layout_ids=layout,
        style_ids=style,
    )
    t0 = time.perf_counter()
    env.reset()
    t_reset = time.perf_counter() - t0

    sim = env.sim
    m = sim.model._model
    lo, hi = env.action_spec
    nu = len(lo)
    # The arm and gripper only. PandaOmron's tail dimensions drive the MOBILE BASE, and a moving base
    # carries every robot-mounted camera with it -- the comparison would then also be measuring
    # whether two renderers agree about a camera that moved, which is a different question.
    n_arm = min(7, nu)
    rng = np.random.default_rng(args.seed)

    qpos = np.empty((args.steps, int(m.nq)), np.float64)
    frames = {c: np.empty((args.steps, args.res, args.res, 3), np.uint8) for c in CAMERAS}
    a = np.zeros(nu)
    for t in range(args.steps):
        # ``sim.forward()`` BEFORE reading qpos and BEFORE rendering, and it is not a formality.
        # ``mj_step`` computes the forward dynamics at the OLD qpos and then integrates, so when it
        # returns, ``data.qpos`` is the new state and ``data.xpos`` -- which is what
        # ``mjv_updateScene`` reads -- still describes the old one. Render there and the frame is one
        # substep behind its own qpos. Measured on this scene: re-rendering the recorded qpos through
        # ``mj_forward`` on the SAME live model then disagrees with the recorded frame on 7.7 % of
        # pixels (39.0 % on the wrist camera), which is a state mismatch that would be scored as a
        # renderer difference and would inflate every fidelity number below.
        sim.forward()
        qpos[t] = sim.data.qpos
        for c in CAMERAS:
            frames[c][t] = sim.render(width=args.res, height=args.res, camera_name=c)[::-1]
        a[:n_arm] = np.clip(0.85 * a[:n_arm] + 0.5 * rng.normal(size=n_arm), lo[:n_arm], hi[:n_arm])
        env.step(a)

    # The official renderer's throughput is NOT measured here. It is measured in
    # ``check_reference_bridge.py``, which warms each variant separately and takes the best of two
    # passes -- an unwarmed single pass measured the raster at 345 f/s and the strictly larger
    # restore-and-raster at 479, which is impossible, and one bad throughput number is worse than
    # none. Keeping one measurement path also keeps the two halves of a ratio on one clock.

    # The claim the whole comparison rests on, asserted here rather than assumed downstream: with the
    # compiled model held fixed, the recorded frame is a pure function of the recorded qpos. Checked
    # by restoring into a FRESH ``MjData`` and re-rendering through a bare ``mujoco.Renderer`` at geom
    # group 1 -- which is separately shown to
    # reproduce ``sim.render`` bit-identically on the live model, and is therefore the official
    # renderer without robosuite in the process.
    vopt = mujoco.MjvOption()
    vopt.geomgroup[:] = 0
    vopt.geomgroup[1] = 1
    chk = mujoco.Renderer(m, width=args.res, height=args.res)  # never closed; EGL contexts persist
    d_chk = mujoco.MjData(m)
    worst = {}
    for c in CAMERAS:
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
        w = 0
        for t in np.linspace(0, args.steps - 1, 8).astype(int):
            d_chk.qpos[:] = qpos[t]
            d_chk.qvel[:] = 0.0
            mujoco.mj_forward(m, d_chk)
            chk.update_scene(d_chk, camera=cid, scene_option=vopt)
            w = max(w, int(np.abs(chk.render().astype(np.int16) - frames[c][t].astype(np.int16)).max()))
        worst[c] = w
    if max(worst.values()) != 0:
        raise AssertionError(
            f"the recorded frames are not a pure function of the recorded qpos (max abs diff per "
            f"camera {worst}). Every fidelity number computed from this dump would be scoring a state "
            f"mismatch as a renderer difference"
        )

    xml = sim.model.get_xml()
    with open(os.path.join(d, "scene.xml"), "w") as f:
        f.write(xml)
    np.save(os.path.join(d, "qpos.npy"), qpos)
    for c in CAMERAS:
        np.save(os.path.join(d, f"official_{c}.npy"), frames[c])

    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    layout_info = {
        "joint_names": jnames,
        "jnt_qposadr": np.asarray(m.jnt_qposadr, int).tolist(),
        "jnt_type": np.asarray(m.jnt_type, int).tolist(),
        "nq": int(m.nq),
        "ngeom": int(m.ngeom),
        "nmesh": int(m.nmesh),
        "ntex": int(m.ntex),
        "nmeshface": int(m.nmeshface),
    }
    meta = {
        "task": args.task,
        "kitchen": args.kitchen,
        "layout": layout,
        "style": style,
        "steps": args.steps,
        "res": args.res,
        "seed": args.seed,
        "cameras": list(CAMERAS),
        "mujoco": mujoco.__version__,
        "reset_s": t_reset,
        "xml_bytes": len(xml),
        "state_sufficiency_max_abs": worst,
        "ep_meta": env.get_ep_meta(),
        "model": layout_info,
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("ep_meta",)}, indent=1, default=str), flush=True)
    print(f"wrote {d}", flush=True)


if __name__ == "__main__":
    main()
