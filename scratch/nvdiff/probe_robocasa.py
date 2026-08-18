"""The second blocker: RoboCasa's textured BOX and CYLINDER geoms, which MJWarp cannot UV at all.

``docs/research/robocasa.md`` Section 7: ``sample_texture`` derives UVs for ``PLANE`` and ``MESH``
only, so a textured ``BOX``/``CYLINDER``/``SPHERE`` samples texel (0,0) and the whole surface comes
out one flat colour -- 178-401 geoms per kitchen, 30.6 % of the frame, and the minimal repro is a
checkered box on which MuJoCo paints 167 distinct colours and MJWarp paints 2. In nvdiffrast the UVs
are ours to supply, so the question is not whether they can be supplied but whether the ones this
module generates are MuJoCo's.

Scored per GEOM CLASS off MuJoCo's own segmentation buffer, because a whole-frame number cannot tell
a fixed box from a broken mesh. The class that decides it is ``BOX + texture``.

Reads the expanded MJCFs under ``scratch/robocasa/out/corpus`` -- absolute asset paths, compiles
standalone, no robosuite needed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from common import emit, save, score  # noqa: E402

CORPUS = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/corpus"
TYPE_NAME = {0: "PLANE", 1: "HFIELD", 2: "SPHERE", 3: "CAPSULE", 4: "ELLIPSOID", 5: "CYLINDER",
             6: "BOX", 7: "MESH"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitchen", default="L52S03")
    ap.add_argument("--ep", default="ep000")
    ap.add_argument("--camera", default="robot0_agentview_left")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--shadow0", action="store_true", help="mjRND_SHADOW off on the reference")
    ap.add_argument("--refl0", action="store_true", help="zero mat_reflectance on BOTH sides")
    ap.add_argument("--opaque", action="store_true",
                    help="hide PARTIALLY transparent geoms on BOTH sides, so the textured-box "
                         "question is not answered by a blending pass this renderer does not have")
    args = ap.parse_args()

    import mujoco
    import torch

    from stable_envs.render import nvdiff as ND

    xml = os.path.join(CORPUS, args.kitchen, f"{args.ep}.xml")
    # One qpos per EPISODE, and each episode is a different compiled kitchen -- so the states for one
    # model come from that model's own episode. Perturbing the robot's joints gives the rest.
    qpos = np.load(os.path.join(CORPUS, args.kitchen, f"{args.ep}_qpos.npy")).reshape(1, -1)
    m = mujoco.MjModel.from_xml_path(xml)
    m.vis.quality.offsamples = 0
    if args.refl0:
        m.mat_reflectance[:] = 0.0
    d = mujoco.MjData(m)
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    assert cam_id >= 0, args.camera

    alpha = np.array([ND.effective_rgba(m, g)[3] for g in range(m.ngeom)])
    if args.opaque:
        # MuJoCo draws transparent geoms in a second, blended pass that always lands on top of the
        # opaque ones; this renderer depth-tests them like everything else. On this kitchen that is
        # visible as a translucent placement box painted over a counter -- MuJoCo shows the green box,
        # we show the white counter. Removing them from BOTH sides is what isolates the texgen
        # question from the blending one.
        m.geom_group[(alpha > 0) & (alpha < 1)] = 7
    tex = np.array([ND._tex_of(m, int(m.geom_matid[g])) for g in range(m.ngeom)])
    gtype = np.asarray(m.geom_type, np.int64)
    vis = (alpha > 0) & np.isin(m.geom_group, ND.DEFAULT_GROUPS)
    census = {}
    for g in range(m.ngeom):
        if not vis[g]:
            continue
        k = f"{TYPE_NAME.get(int(gtype[g]), gtype[g])}{'+tex' if tex[g] >= 0 else ''}"
        census[k] = census.get(k, 0) + 1
    emit("census", {"kitchen": args.kitchen, "ngeom": int(m.ngeom), "visible": int(vis.sum()),
                    "hidden_zero_alpha": int(((alpha <= 0) & np.isin(m.geom_group,
                                                                    ND.DEFAULT_GROUPS)).sum()),
                    "ntex": int(m.ntex), "nmesh": int(m.nmesh), "by_class": census})

    base = np.asarray(qpos, np.float64)[0]
    if base.shape[0] < m.nq:
        base = np.concatenate([base, np.zeros(m.nq - base.shape[0])])
    base = base[:m.nq]
    rng = np.random.default_rng(0)
    states = [base]
    nj = min(m.nq, 40)
    for _ in range(args.states - 1):
        q = base.copy()
        q[:nj] = q[:nj] + rng.uniform(-0.05, 0.05, nj)
        states.append(q)
    states = np.stack(states)

    ren = mujoco.Renderer(m, height=args.size, width=args.size)
    ref, seg = [], []
    for q in states:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=int(cam_id))
        if args.shadow0:
            ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        if args.refl0:
            ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        ref.append(ren.render().copy())
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=int(cam_id))
        seg.append(ren.render()[..., 0].copy())
        ren.disable_segmentation_rendering()
    ref, seg = np.stack(ref), np.stack(seg)

    cls = {}
    for name in ("BOX+tex", "MESH+tex", "MESH", "BOX", "CYLINDER+tex", "CYLINDER", "PLANE+tex"):
        ids = [g for g in range(m.ngeom) if vis[g]
               and f"{TYPE_NAME.get(int(gtype[g]), gtype[g])}{'+tex' if tex[g] >= 0 else ''}" == name]
        if ids:
            cls[name] = np.isin(seg, np.array(ids))
    cls["frame"] = np.ones_like(seg, bool)
    cls["sky"] = seg < 0

    qt = torch.as_tensor(states, dtype=torch.float32)
    out = {}
    for name, kw in [("uv", dict(tex=True)), ("notex", dict(tex=False))]:
        br = ND.BatchRenderer(m, res=args.size, device="cuda", tex_policy="pot")
        br.resolve_from_mujoco(m, d, camera=None)
        br.use_tex = kw["tex"]
        if name == "uv":
            emit("scene", {"geoms_drawn": br.scene.G, "verts": br.scene.V,
                           "tris": int(br.scene.face.shape[0]),
                           "ntex_loaded": len(br.scene.textures),
                           "resampled": sum(1 for v in br.scene.textures.values()
                                            if v["resampled"]),
                           "mipmap_ok": sum(1 for v in br.scene.textures.values()
                                            if v["mipmap_ok"])})
        ours = br.render_qpos(qt, camera_id=int(cam_id)).cpu().numpy()
        row = {k: score(ours, ref, v) for k, v in cls.items()}
        out[name] = row
        emit(f"score.{name}", {k: {kk: v.get(kk) for kk in
                                   ("n", "mean_abs", "exact_frac", "within8_frac",
                                    "interior_exact")} for k, v in row.items()})
        if name == "uv":
            np.save("/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/"
                    f"rc_{args.kitchen}_ours.npy", ours[:2])
            np.save("/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/"
                    f"rc_{args.kitchen}_ref.npy", ref[:2])
        del br
    save(f"robocasa_{args.kitchen}{'_s0' if args.shadow0 else ''}.json", {"args": vars(args), "census": census, "scores": out})


if __name__ == "__main__":
    main()
