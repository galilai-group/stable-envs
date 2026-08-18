"""Why is a bare ``mujoco.Renderer`` at geom group 1 NOT bit-identical to ``env.sim.render``?

``docs/research/robocasa.md`` Section 3 records the two as bit-identical, and on a 200-step trajectory
re-rendered from a recompiled MJCF they are not: 94.0 % of pixels exact, mean 0.30, max 120 on
``robot0_agentview_left`` and 62.7 % on the wrist camera. Something between the two claims is different,
and the difference matters because a visual-tier RoboCasa env can only be conformance-tested in one
process if the official frame can be produced WITHOUT robosuite (robocasa pins mujoco 3.3.1;
mujoco_warp needs >= 3.11, so no interpreter holds both).

Three candidates, separated rather than guessed:

  A  renderer configuration   same live model, same live MjData: robosuite's offscreen context vs a
                              bare Renderer at group 1. Any difference here is flags, not state.
  B  state restore            the frame is a function of qpos ONLY if nothing else moves geometry.
                              PandaOmron carries MOCAP bodies whose pose lives in ``data.mocap_pos``.
  C  recompile                the same MJCF compiled twice.

Run in the ``robocasa`` conda env on a GPU node.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/robocasa_visual_work/visual_reference.json"
CAMS = ("robot0_agentview_left", "robot0_eye_in_hand")


def stats(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return {"max": int(d.max()), "mean": float(d.mean()), "exact": float((d == 0).mean())}


def main() -> None:
    import mujoco
    from robocasa.utils.env_utils import create_env

    res = 256
    env = create_env(env_name="PickPlaceCounterToSink", robots="PandaOmron", camera_names=list(CAMS),
                     camera_widths=res, camera_heights=res, seed=7, render_onscreen=False,
                     layout_ids=52, style_ids=3)
    env.reset()
    lo, hi = env.action_spec
    a = np.zeros(len(lo))
    rng = np.random.default_rng(7)
    for _ in range(20):
        a[:7] = np.clip(0.85 * a[:7] + 0.5 * rng.normal(size=7), lo[:7], hi[:7])
        env.step(a)

    sim = env.sim
    m = sim.model._model
    d = sim.data._data
    qpos = np.array(d.qpos, np.float64)
    mocap_pos = np.array(d.mocap_pos, np.float64)
    mocap_quat = np.array(d.mocap_quat, np.float64)
    xml = sim.model.get_xml()
    rec = {"nmocap": int(m.nmocap), "nq": int(m.nq), "res": res, "mujoco": mujoco.__version__,
           "mocap_body_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                                for b in range(m.nbody) if int(m.body_mocapid[b]) >= 0],
           "A_live_renderer": {}, "B_state_restore": {}, "C_recompile": {}}

    vopt = mujoco.MjvOption()
    vopt.geomgroup[:] = 0
    vopt.geomgroup[1] = 1
    live = mujoco.Renderer(m, width=res, height=res)
    for cam in CAMS:
        ref = sim.render(width=res, height=res, camera_name=cam)[::-1].copy()
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        live.update_scene(d, camera=cid, scene_option=vopt)
        rec["A_live_renderer"][cam] = stats(ref, live.render())

    # B and C: recompile the SAME xml, restore qpos, then qpos + mocap.
    p = "/mnt/vast/home/randall.balestriero/robocasa_visual_work/probe_scene.xml"
    with open(p, "w") as f:
        f.write(xml)
    m2 = mujoco.MjModel.from_xml_path(p)
    d2 = mujoco.MjData(m2)
    r2 = mujoco.Renderer(m2, width=res, height=res)
    for label, use_mocap in (("qpos_only", False), ("qpos_plus_mocap", True)):
        d2.qpos[:] = qpos
        d2.qvel[:] = 0.0
        if use_mocap:
            d2.mocap_pos[:] = mocap_pos
            d2.mocap_quat[:] = mocap_quat
        mujoco.mj_forward(m2, d2)
        for cam in CAMS:
            ref = sim.render(width=res, height=res, camera_name=cam)[::-1].copy()
            cid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            r2.update_scene(d2, camera=cid, scene_option=vopt)
            got = r2.render()
            rec["B_state_restore"].setdefault(label, {})[cam] = stats(ref, got)
            if label == "qpos_plus_mocap" and cam == CAMS[0]:
                # Localise the residual by geom, the way probe_alpha did: MuJoCo's segmentation
                # buffer is (objid, objtype), NOT the other way round.
                r2.enable_segmentation_rendering()
                r2.update_scene(d2, camera=cid, scene_option=vopt)
                seg = r2.render()[..., 0]
                r2.disable_segmentation_rendering()
                bad = np.abs(ref.astype(np.int16) - got.astype(np.int16)).max(-1) > 0
                ids, cnt = np.unique(seg[bad], return_counts=True)
                order = np.argsort(-cnt)[:12]
                rec["residual_by_geom"] = [
                    {"geom": mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_GEOM, int(ids[i]))
                     or f"geom{int(ids[i])}",
                     "type": int(m2.geom_type[int(ids[i])]) if int(ids[i]) >= 0 else -1,
                     "px": int(cnt[i]), "frac_of_residual": float(cnt[i] / max(1, bad.sum()))}
                    for i in order if int(ids[i]) >= 0]

    # C: two compilations of the same XML, same state, same renderer.
    m3 = mujoco.MjModel.from_xml_path(p)
    d3 = mujoco.MjData(m3)
    r3 = mujoco.Renderer(m3, width=res, height=res)
    d3.qpos[:] = qpos
    d3.qvel[:] = 0.0
    d3.mocap_pos[:] = mocap_pos
    d3.mocap_quat[:] = mocap_quat
    mujoco.mj_forward(m3, d3)
    for cam in CAMS:
        cid2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        cid3 = mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        r2.update_scene(d2, camera=cid2, scene_option=vopt)
        r3.update_scene(d3, camera=cid3, scene_option=vopt)
        rec["C_recompile"][cam] = stats(r2.render(), r3.render())

    # D: restore into a fresh MjData on the LIVE model. If this is exact, the state restore is fine
    # and the residual belongs to the XML round trip, not to "qpos is not the state".
    d4 = mujoco.MjData(m)
    d4.qpos[:] = qpos
    d4.qvel[:] = 0.0
    d4.mocap_pos[:] = mocap_pos
    d4.mocap_quat[:] = mocap_quat
    mujoco.mj_forward(m, d4)
    rec["D_restore_on_live_model"] = {}
    for cam in CAMS:
        ref = sim.render(width=res, height=res, camera_name=cam)[::-1].copy()
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        live.update_scene(d4, camera=cid, scene_option=vopt)
        rec["D_restore_on_live_model"][cam] = stats(ref, live.render())

    # E: how much did writing the model out as text and reading it back move it?
    rec["E_xml_roundtrip"] = {
        f: float(np.abs(np.asarray(getattr(m, f), np.float64)
                        - np.asarray(getattr(m2, f), np.float64)).max())
        for f in ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size", "mesh_vert",
                  "cam_pos", "cam_quat", "jnt_pos", "jnt_axis")}
    rec["E_xml_roundtrip"]["xpos_max_abs"] = float(
        np.abs(np.asarray(d4.xpos, np.float64) - np.asarray(d2.xpos, np.float64)).max())

    with open(OUT, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
