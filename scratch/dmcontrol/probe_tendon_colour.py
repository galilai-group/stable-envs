"""The one piece of ``ball_in_cup``'s tendon that is not geometry: what colour MuJoCo paints it.

``probe_catenary.py`` established the shape bit-for-bit. What is left is that the tendon has THREE
drawing branches and they are not the same colour:

  * slack: 29 catenary points -> 28 capsules, painted the raw ``tendon_rgba`` (0.5, 0.5, 0.5, 1);
  * near-vertical: 3 points -> 2 capsules, same colour;
  * TAUT: ``mjv_isCatenary`` returns 0 and the non-catenary branch draws ONE capsule between the two
    wrap points, recoloured by the limit constraint's IMPEDANCE, measured spanning
    (0.860, 0.025, 0.025) to (0.880, 0.050, 0.050) -- i.e. red, not grey.

and ball_in_cup reaches the taut branch in 35-44% of the states a random policy visits, so the third
branch is not an edge case that a renderer can decline.

The hypothesis to test is that the colour is exactly

    rgb = (1 - imp) * tendon_rgba[:3] + imp * m.vis.rgba.constraint[:3]

with ``imp`` the impedance of the tendon-limit row of the constraint set, i.e. something a renderer
can compute from the state. Two candidate sources for ``imp`` are tried, because they cost different
amounts per frame and only one of them may exist:

  A. ``d.efc_KBIP[row, 3]`` for the row with ``efc_type == mjCNSTR_LIMIT_TENDON`` -- needs the full
     ``mj_forward``, which is what a renderer would have to run per state instead of ``mj_kinematics``;
  B. MuJoCo's own impedance law evaluated on the limit margin, from the tendon's ``solimp`` and
     ``ten_length`` -- needs only ``mj_kinematics`` + ``mj_comPos`` + ``mj_tendon``.

Whichever reproduces MuJoCo's own scene colours to float32 decides how expensive the mover is.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


def main() -> None:
    import mujoco
    from dm_control import suite

    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/dmc2")
    a = ap.parse_args()

    env = suite.load("ball_in_cup", "catch", task_kwargs={"random": 0})
    p = env.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    d = p.data.ptr if hasattr(p.data, "ptr") else p.data
    out: dict = {
        "has_efc_KBIP": hasattr(d, "efc_KBIP"),
        "tendon_rgba": [float(x) for x in np.asarray(m.tendon_rgba).reshape(-1)],
        "vis_rgba_constraint": [float(x) for x in np.asarray(m.vis.rgba.constraint).reshape(-1)],
        "tendon_range": np.asarray(m.tendon_range, np.float64).tolist(),
        "tendon_limited": [int(x) for x in np.asarray(m.tendon_limited).reshape(-1)],
        "numslices": int(m.vis.quality.numslices),
    }

    scn = mujoco.MjvScene(m, maxgeom=max(2000, m.ngeom * 4))
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type = int(mujoco.mjtCamera.mjCAMERA_FIXED)
    cam.fixedcamid = 0

    rng = np.random.default_rng(0)
    spec = env.action_spec()
    rows = []
    env.reset()
    for k in range(int(a.states)):
        if k % 32 == 0:
            env.reset()
        env.step(rng.uniform(spec.minimum, spec.maximum))
        mujoco.mjv_updateScene(m, d, opt, None, cam, int(mujoco.mjtCatBit.mjCAT_ALL), scn)
        seg = [scn.geoms[i] for i in range(int(scn.ngeom))
               if int(scn.geoms[i].objtype) == int(mujoco.mjtObj.mjOBJ_TENDON)]
        if not seg:
            continue
        rgba = np.asarray(seg[0].rgba, np.float64)
        ten_len = float(d.ten_length[0])
        upper = float(m.tendon_range[0][1])
        # A: the impedance MuJoCo itself put in the constraint set
        impA = None
        if hasattr(d, "efc_KBIP") and int(d.nefc):
            et = np.asarray(d.efc_type, np.int64)
            ei = np.asarray(d.efc_id, np.int64)
            want = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
            hits = np.flatnonzero((et == want) & (ei == 0))
            if len(hits):
                # all four columns, because "KBIP" names them and the order is not documented in the
                # bindings: one of these is the impedance and the others are the stiffness, the
                # damping and the impedance's derivative
                impA = [float(x) for x in np.asarray(d.efc_KBIP)[hits[0]]]
        rows.append({"n": len(seg), "rgba": rgba.tolist(), "ten_length": ten_len,
                     "slack": ten_len < upper, "impA": impA,
                     "nefc": int(d.nefc)})
    out["n_states"] = len(rows)
    taut = [r for r in rows if not r["slack"]]
    slack = [r for r in rows if r["slack"]]
    out["counts"] = {"taut": len(taut), "slack": len(slack),
                     "segments_slack": sorted({r["n"] for r in slack}),
                     "segments_taut": sorted({r["n"] for r in taut})}
    grey = np.asarray(m.tendon_rgba, np.float64).reshape(-1)[:3]
    con = np.asarray(m.vis.rgba.constraint, np.float64).reshape(-1)[:3]
    out["slack_colour_exact"] = bool(
        all(np.allclose(np.asarray(r["rgba"])[:3], grey, atol=1e-6) for r in slack))
    # invert the blend for every taut state, and check the implied imp is CONSISTENT across channels
    err_from_impA, implied = [], []
    for r in taut:
        rgb = np.asarray(r["rgba"], np.float64)[:3]
        denom = con - grey
        use = np.flatnonzero(np.abs(denom) > 1e-9)
        imp = float(np.mean((rgb - grey)[use] / denom[use]))
        implied.append(imp)
        if r["impA"] is not None:
            err_from_impA.append([float(np.abs((1 - c) * grey + c * con - rgb).max())
                                  for c in r["impA"]])
    out["implied_imp"] = {"min": float(np.min(implied)) if implied else None,
                          "max": float(np.max(implied)) if implied else None}
    out["efc_KBIP_max_colour_err_per_column"] = (
        np.max(np.asarray(err_from_impA), axis=0).tolist() if err_from_impA else None)
    out["efc_KBIP_n"] = len(err_from_impA)
    out["sample_taut"] = taut[:4]
    out["sample_slack"] = slack[:2]

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "tendon_colour.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("sample_taut", "sample_slack")},
                     indent=1, default=float))
    print("sample taut:", json.dumps(out["sample_taut"], default=float))
    print(f"@@wrote {path}")


if __name__ == "__main__":
    main()
