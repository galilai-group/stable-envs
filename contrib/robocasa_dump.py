"""Record one RoboCasa trajectory and the frames the official renderer produced for it.

``contrib/robocasa.py`` renders from the cache this script writes. Run it in an interpreter with
RoboCasa installed (robosuite 1.5.2, robocasa 1.0.1, mujoco 3.3.1). The output needs no robosuite to
read back: the expanded MJCF has absolute asset paths and compiles on its own, so the renderer can run
in a different interpreter with a different MuJoCo.

Three things are recorded with the trajectory:

* **The joint layout, not just ``qpos``.** The reader recompiles the same XML under a different
  MuJoCo, and two compilations of one scene need not order joints the same way. The joint names and
  ``jnt_qposadr`` are saved so the reader can map ``qpos`` by name.
* **The official frame comes from ``env.sim.render``**, robosuite's own offscreen context with
  ``vopt.geomgroup[1] = True`` and everything else off. A bare ``mujoco.Renderer`` defaults to geom
  groups 0-2 and draws robosuite's collision primitives over the scene. The frame is flipped
  vertically because ``sim.render`` returns it bottom-up.
* **A check that the frames are a pure function of ``qpos``**, done by re-rendering saved states
  into a fresh ``MjData``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

DEFAULT_OUT = os.path.expanduser(os.environ.get("STABLE_ENVS_ROBOCASA_CACHE", "~/.cache/stable-envs/robocasa"))
#: The kitchens ``contrib/robocasa.py`` registers.
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
