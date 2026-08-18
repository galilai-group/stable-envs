"""The ceiling of the BIT-EXACT path: how fast can MuJoCo's own renderer be made to go?

This is the measurement the whole RoboCasa question turns on, and neither prior investigation made
it. Every environment shipped in this repo -- including the DM Control family landed this week --
declares ``parity_interior_min = 1.0``: 100 % of interior pixels bit-identical to the official
renderer, with no env yet using the relaxation mechanism. For RoboCasa exactly one renderer can
meet that by construction, MuJoCo's own EGL rasterizer, because it IS the official renderer.

So the question "should stable-envs ship RoboCasa" reduces to: with the compiled model already
built and resident -- i.e. with G3 assumed away and setup cost assumed to be zero -- can MuJoCo's
rasterizer regenerate frames faster than a CPU core decodes them (5,137 f/s on real RoboCasa JPEGs)?

Measured here at the resolution RoboCasa datasets use, with every model-side speedup applied:

  * hide the 154 zero-alpha marker geoms (they are rasterized by MuJoCo too, and discarded);
  * delete every geom outside the rendered group, and every mesh only they referenced, so the
    rasterizer is handed the smallest model that draws the same picture;
  * scale across PROCESSES, which is the configuration with no GIL in it at all -- so the answer
    does not depend on free-threading landing.

Each variant is checked pixel-identical against the unmodified model before its fps is believed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def make_opt(mujoco, groups=(1,)):
    o = mujoco.MjvOption()
    o.geomgroup[:] = 0
    for g in groups:
        o.geomgroup[g] = 1
    return o


def hide_zero_alpha(mjm, hidden_group=4):
    a = mjm.geom_rgba[:, 3].copy()
    hm = mjm.geom_matid >= 0
    a[hm] = mjm.mat_rgba[mjm.geom_matid[hm], 3]
    n = int((a <= 0).sum() & 0xFFFFFFF)
    mjm.geom_group[a <= 0] = hidden_group
    return int((a <= 0).sum())


def strip_to_visible(xml_path, out_path, groups=(1,)):
    """Delete every geom outside the rendered groups, and the assets only they used.

    MuJoCo's rasterizer walks the geom list to build ``MjvScene`` on every ``update_scene``; a
    model carrying 730 collision primitives and 185 collision meshes pays for them in scene
    construction whether or not they are drawn.
    """
    import xml.etree.ElementTree as ET
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    raw = bytes(m.names)
    gname = [raw[a:raw.index(b"\x00", a)].decode() for a in m.name_geomadr]
    keep = {gname[i] for i in range(m.ngeom) if int(m.geom_group[i]) in groups}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Deleting a body's only geom leaves MuJoCo with no mass for a moving body, which is a compile
    # error. Write every body's inertial explicitly from the ALREADY COMPILED model first, so the
    # stripped model keeps exactly the inertias it had -- the renderer never reads them, but the
    # compiler insists they exist.
    raw_n = bytes(m.names)
    bname = {raw_n[a:raw_n.index(b"\x00", a)].decode(): i for i, a in enumerate(m.name_bodyadr)}
    for b in root.iter("body"):
        i = bname.get(b.get("name"))
        if i is None or b.find("inertial") is not None:
            continue
        el = ET.SubElement(b, "inertial")
        el.set("pos", " ".join("%.9g" % v for v in m.body_ipos[i]))
        el.set("quat", " ".join("%.9g" % v for v in m.body_iquat[i]))
        el.set("mass", "%.9g" % max(float(m.body_mass[i]), 1e-6))
        el.set("diaginertia", " ".join("%.9g" % max(v, 1e-9) for v in m.body_inertia[i]))
    n_del = 0
    parents = list(root.iter())                      # snapshot: removing while iterating is unsafe
    for parent in parents:
        for g in list(parent.findall("geom")):
            if g.get("name") not in keep:
                parent.remove(g); n_del += 1
    # drop assets nothing references any more
    used = set()
    for g in root.iter("geom"):
        for a in ("mesh", "material"):
            if g.get(a):
                used.add(g.get(a))
    asset = root.find("asset")
    for mat in asset.findall("material"):
        if mat.get("name") in used and mat.get("texture"):
            used.add(mat.get("texture"))
    n_asset = 0
    for el in list(asset):
        if el.tag in ("mesh", "material", "texture") and el.get("name") not in used:
            if el.tag == "texture" and el.get("type") == "skybox":
                continue
            asset.remove(el); n_asset += 1
    tree.write(out_path)
    return n_del, n_asset


def bench(xml, qpos, res, cam, iters, groups=(1,), q=None):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos[:m.nq]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=res, width=res)
    opt = make_opt(mujoco, groups)
    cid = max(0, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam))
    r.update_scene(d, camera=cid, scene_option=opt)
    img = r.render().copy()
    t0 = time.perf_counter()
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        r.update_scene(d, camera=cid, scene_option=opt)
        r.render()
    fps = iters / (time.perf_counter() - t0)
    if q is not None:
        q.put(fps)
    return fps, img


def _worker(xml, qpos, res, cam, iters, q):
    os.environ["MUJOCO_GL"] = "egl"
    bench(xml, qpos, res, cam, iters, q=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(OUT, "corpus", "L52S03", "ep000.xml"))
    ap.add_argument("--qpos", default=None)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", default=os.path.join(OUT, "exact_ceiling.json"))
    args = ap.parse_args()

    import mujoco
    qp = args.qpos or args.xml.replace(".xml", "_qpos.npy")
    qpos = np.load(qp) if os.path.exists(qp) else None
    if qpos is None:
        qpos = mujoco.MjModel.from_xml_path(args.xml).qpos0.copy()

    rec = {"xml": args.xml, "res": args.res, "mujoco": mujoco.__version__,
           "decode_fps_per_core_real_robocasa": 5137.0}

    fps0, img0 = bench(args.xml, qpos, args.res, args.cam, args.iters)
    rec["full_model"] = {"fps": fps0}
    print("full model            %.0f fps" % fps0, flush=True)

    # variant: strip everything outside the rendered group
    stripped = os.path.join(OUT, "stripped_scene.xml")
    n_geom, n_asset = strip_to_visible(args.xml, stripped)
    fps1, img1 = bench(stripped, qpos, args.res, args.cam, args.iters)
    d1 = np.abs(img1.astype(np.int16) - img0.astype(np.int16))
    rec["stripped"] = {"fps": fps1, "geoms_deleted": n_geom, "assets_deleted": n_asset,
                       "px_exact": float((d1.max(-1) == 0).mean()), "px_max": int(d1.max())}
    print("stripped to group 1   %.0f fps  (%d geoms, %d assets removed, exact=%.5f)"
          % (fps1, n_geom, n_asset, rec["stripped"]["px_exact"]), flush=True)

    best = stripped if rec["stripped"]["px_exact"] == 1.0 else args.xml
    rec["scaling_xml"] = best
    rows = []
    ctx = mp.get_context("spawn")
    for n in args.procs:
        q = ctx.Queue()
        ps = [ctx.Process(target=_worker, args=(best, qpos, args.res, args.cam,
                                                max(200, args.iters // n), q)) for _ in range(n)]
        t0 = time.perf_counter()
        for p in ps:
            p.start()
        vals = [q.get() for _ in ps]
        for p in ps:
            p.join()
        rows.append({"procs": n, "per_proc_fps": float(np.mean(vals)),
                     "aggregate_fps": float(np.sum(vals)), "wall_s": time.perf_counter() - t0})
        print("procs=%2d  per-proc %.0f  aggregate %.0f fps" %
              (n, rows[-1]["per_proc_fps"], rows[-1]["aggregate_fps"]), flush=True)
    rec["multiprocess"] = rows
    rec["best_exact_fps"] = max([fps0, fps1] + [r["aggregate_fps"] for r in rows])
    rec["cores_replaced_exact_path"] = rec["best_exact_fps"] / 5137.0
    print("\nBEST EXACT-PATH THROUGHPUT %.0f fps  ->  %.2f cores of JPEG decode"
          % (rec["best_exact_fps"], rec["cores_replaced_exact_path"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
