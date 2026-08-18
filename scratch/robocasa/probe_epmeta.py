"""Is ``ep_meta`` (6.4 kB) a sufficient scene descriptor after all? Localise every residual pixel.

The recorded answer is no -- 94.95 % of pixels, excursions to 208/255 -- and it is load-bearing,
because it is the difference between storing a 385 kB MJCF per episode and storing 1.7 kB gzipped.
But ``env_utils.create_obj`` explicitly honours ``cfg["info"]["mjcf_path"]`` "to do deterministic
playback", and ``ObjCat.get_mjcf_kwargs`` returns a fixed scale per category, so the object half
ought to be exact. ``probe_bucket.py`` confirms it is (object geom signature matches) while the
KITCHEN half does not -- and it also found the replay is not even self-consistent (two replays of
the same ep_meta differ from each other), which no missing-descriptor-field story explains.

The obvious suspect is the comparison rather than the descriptor: the earlier probes restore
``qpos`` only. RoboCasa's model carries two MOCAP bodies (``left_eef_target``, ``right_eef_target``)
whose pose lives in ``mjSTATE_FULLPHYSICS``, not in ``qpos``. This restores the full state, and if
anything is still left it renders each geom class separately to say WHICH geoms disagree.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out"


def sha(s):
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def names_of(m, kind):
    adr = {"body": m.name_bodyadr, "geom": m.name_geomadr, "jnt": m.name_jntadr}[kind]
    raw = bytes(m.names)
    return [raw[a:raw.index(b"\x00", a)].decode() for a in adr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToSink")
    ap.add_argument("--layout", type=int, default=52)
    ap.add_argument("--style", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--res", type=int, default=128)
    args = ap.parse_args()

    import mujoco
    from robocasa.utils.env_utils import create_env

    cam = "robot0_agentview_left"
    env = create_env(env_name=args.task, robots="PandaOmron", camera_names=[cam],
                     camera_widths=args.res, camera_heights=args.res, seed=7,
                     render_onscreen=False, layout_ids=args.layout, style_ids=args.style)

    def render():
        return env.sim.render(width=args.res, height=args.res, camera_name=cam)[::-1].copy()

    def full_state():
        m = env.sim.model._model
        s = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, env.sim.data._data, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return s

    env.reset()
    m0 = env.sim.model._model
    ep = json.loads(json.dumps(env.get_ep_meta(), default=str))
    s0 = full_state()
    q0 = np.array(env.sim.data.qpos).copy()
    mocap0 = (np.array(env.sim.data._data.mocap_pos).copy(),
              np.array(env.sim.data._data.mocap_quat).copy())
    xml0 = env.sim.model.get_xml()
    ref = render()
    rec = {"task": args.task, "layout": args.layout, "style": args.style, "res": args.res,
           "nmocap": int(m0.nmocap), "mocap_bodies": [n for n, b in zip(names_of(m0, "body"),
                                                                       m0.body_mocapid) if b >= 0],
           "ep_meta_bytes": len(json.dumps(ep)),
           "ep_meta_gz_bytes": len(gzip.compress(json.dumps(ep).encode())),
           "xml_bytes": len(xml0), "xml_gz_bytes": len(gzip.compress(xml0.encode())),
           "state_floats": int(s0.size), "reps": []}
    print("mocap bodies:", rec["mocap_bodies"], flush=True)

    ejn = names_of(m0, "jnt")
    jadr = {n: (int(m0.jnt_qposadr[i]), int(m0.jnt_type[i])) for i, n in enumerate(ejn)}
    width = {0: 7, 1: 4, 2: 1, 3: 1}

    def score(tag, frame):
        d = np.abs(frame.astype(np.int16) - ref.astype(np.int16))
        return dict(tag=tag, px_exact=float((d.max(-1) == 0).mean()), px_mean=float(d.mean()),
                    px_max=int(d.max()), px_within8=float((d.max(-1) <= 8).mean()))

    for r in range(args.reps):
        env.set_ep_meta(ep)
        env.reset()
        m = env.sim.model._model
        row = {"rep": r, "xml_matches": sha(env.sim.model.get_xml()) == sha(xml0),
               "nq": int(m.nq), "ngeom": int(m.ngeom), "nmocap": int(m.nmocap)}

        # (a) qpos only, mapped by joint name -- what the earlier probes did
        q = np.array(m.qpos0).copy()
        for i, n in enumerate(names_of(m, "jnt")):
            if n in jadr:
                a, _ = jadr[n]
                w = width[int(m.jnt_type[i])]
                b = int(m.jnt_qposadr[i])
                q[b:b + w] = q0[a:a + w]
        env.sim.data.qpos[:] = q
        env.sim.forward()
        row["qpos_only"] = score("qpos_only", render())

        # (b) plus mocap
        if m.nmocap == m0.nmocap:
            env.sim.data._data.mocap_pos[:] = mocap0[0]
            env.sim.data._data.mocap_quat[:] = mocap0[1]
            env.sim.forward()
            row["qpos_plus_mocap"] = score("qpos_plus_mocap", render())

        # (c) full physics state, if the state vector is the same width
        if s0.size == mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS):
            mujoco.mj_setState(m, env.sim.data._data, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            env.sim.forward()
            row["full_state"] = score("full_state", render())

        rec["reps"].append(row)
        print(json.dumps(row), flush=True)

    # ---- if anything is left, say WHICH geoms ------------------------------------------------
    last = rec["reps"][-1]
    best = last.get("full_state") or last.get("qpos_plus_mocap") or last["qpos_only"]
    rec["best"] = best
    if best["px_exact"] < 1.0:
        m = env.sim.model._model
        gn = names_of(m, "geom")
        seg = env.sim.render(width=args.res, height=args.res, camera_name=cam,
                             segmentation=True)[::-1]
        objid = seg[..., 0]
        f = render()
        d = np.abs(f.astype(np.int16) - ref.astype(np.int16)).max(-1)
        bad = d > 0
        ids, cnt = np.unique(objid[bad], return_counts=True)
        top = sorted(zip(cnt.tolist(), ids.tolist()), reverse=True)[:15]
        rec["disagreeing_geoms"] = [{"geom": gn[i] if 0 <= i < len(gn) else str(i),
                                     "px": c, "frac_of_frame": c / d.size} for c, i in top]
        print("top disagreeing geoms:", json.dumps(rec["disagreeing_geoms"], indent=1), flush=True)

    with open(os.path.join(OUT, "epmeta.json"), "w") as fo:
        json.dump(rec, fo, indent=1, default=str)
    print("wrote", os.path.join(OUT, "epmeta.json"))


if __name__ == "__main__":
    main()
