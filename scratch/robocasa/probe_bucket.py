"""Episodes per COMPILED MODEL, and what actually differs between two episodes of one kitchen.

The prior investigation measured K = 1 (one compiled kitchen per episode) and concluded RoboCasa
fails G3. K = 1 was measured on a *stream of resets*, which is one access pattern; a dataset is
written once and read many times, and the reader picks the order. This probe measures the three
things that decide whether re-ordering can raise K:

  1. How much of the compiled model is the KITCHEN (a function of layout/style/gen_textures) and
     how much is the per-episode OBJECT set. If the kitchen half is identical across resets, the
     per-episode delta is small and a union model is possible.
  2. Whether ``ep_meta`` reproduces the compiled model EXACTLY. ``env_utils.create_obj`` honours
     ``cfg["info"]["mjcf_path"]`` explicitly "to do deterministic playback", so the earlier
     94.95 %-of-pixels result deserves re-measurement with the state mapped by JOINT NAME rather
     than by index, and with the ep_meta JSON round-tripped the way a stored corpus would.
  3. How many distinct object instances a single (task, layout, style) draws over many episodes —
     the size of the union model a bucketed pipeline would have to compile.

Everything is written to out/bucket.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out"


def sha(s) -> str:
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def model_facts(m) -> dict:
    return dict(nq=int(m.nq), nv=int(m.nv), nbody=int(m.nbody), ngeom=int(m.ngeom),
                nmesh=int(m.nmesh), nmeshface=int(m.nmeshface), ntex=int(m.ntex),
                nmat=int(m.nmat), njnt=int(m.njnt), texdata_MB=float(m.tex_data.size / 1e6))


def names_of(m, kind: str) -> list[str]:
    import mujoco
    adr = {"body": m.name_bodyadr, "geom": m.name_geomadr, "jnt": m.name_jntadr,
           "mesh": m.name_meshadr, "tex": m.name_texadr, "mat": m.name_matadr}[kind]
    raw = bytes(m.names)
    out = []
    for a in adr:
        e = raw.index(b"\x00", a)
        out.append(raw[a:e].decode())
    return out


def object_bodies(env) -> list[str]:
    """Root body name of every sampled OBJECT (not fixture, not robot)."""
    out = []
    for name, obj in env.objects.items():
        rb = getattr(obj, "root_body", None)
        out.append(rb if rb else f"{name}_main")
    return out


def subtree_geoms(m, root_names: list[str]) -> np.ndarray:
    """Boolean mask over geoms belonging to the subtree of any named root body."""
    bnames = names_of(m, "body")
    idx = {n: i for i, n in enumerate(bnames)}
    roots = {idx[n] for n in root_names if n in idx}
    inside = np.zeros(m.nbody, bool)
    for b in range(m.nbody):
        p, seen = b, 0
        while p != 0 and seen < 64:
            if p in roots:
                inside[b] = True
                break
            p = int(m.body_parentid[p]); seen += 1
    return inside[m.geom_bodyid]


def geom_signature(m, mask=None) -> str:
    """A hash of everything that decides how the geoms in `mask` are DRAWN."""
    gn = np.array(names_of(m, "geom"))
    mn = np.array(names_of(m, "mesh") or [""])
    sel = np.ones(m.ngeom, bool) if mask is None else mask
    rows = []
    for g in np.nonzero(sel)[0]:
        mesh = mn[m.geom_dataid[g]] if (m.geom_type[g] == 7 and m.geom_dataid[g] >= 0) else ""
        rows.append("|".join([
            gn[g], str(int(m.geom_type[g])), str(int(m.geom_group[g])), mesh,
            np.array2string(m.geom_size[g], precision=6),
            np.array2string(m.geom_pos[g], precision=6),
            np.array2string(m.geom_quat[g], precision=6),
            np.array2string(m.geom_rgba[g], precision=6),
        ]))
    return sha("\n".join(sorted(rows)))


def qpos_by_name(src_m, src_qpos, dst_m):
    """Map a qpos vector from one compiled model to another by JOINT NAME.

    Two compilations of the same scene do not have to order joints the same way, and comparing
    frames after an index-wise copy scores a *mis-posed* scene rather than a mis-built one.
    """
    import mujoco
    sn, dn = names_of(src_m, "jnt"), names_of(dst_m, "jnt")
    sadr = {n: (int(src_m.jnt_qposadr[i]), int(src_m.jnt_type[i])) for i, n in enumerate(sn)}
    out = np.array(dst_m.qpos0, dtype=np.float64).copy()
    width = {0: 7, 1: 4, 2: 1, 3: 1}
    hit = miss = 0
    for i, n in enumerate(dn):
        if n in sadr:
            a, t = sadr[n]
            w = width[int(dst_m.jnt_type[i])]
            b = int(dst_m.jnt_qposadr[i])
            out[b:b + w] = src_qpos[a:a + w]
            hit += 1
        else:
            miss += 1
    return out, hit, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToSink")
    ap.add_argument("--layout", type=int, default=52)
    ap.add_argument("--style", type=int, default=3)
    ap.add_argument("--n", type=int, default=24, help="resets at the pinned kitchen")
    ap.add_argument("--res", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import mujoco
    from robocasa.utils.env_utils import create_env

    rec: dict = {"mujoco": mujoco.__version__, "task": args.task,
                 "layout": args.layout, "style": args.style, "n_resets": args.n}
    print("mujoco", mujoco.__version__, flush=True)
    print("MjSpec available:", hasattr(mujoco, "MjSpec"), flush=True)
    rec["has_mjspec"] = hasattr(mujoco, "MjSpec")
    if rec["has_mjspec"]:
        rec["mjspec_api"] = sorted(a for a in dir(mujoco.MjSpec) if not a.startswith("_"))
        rec["mjsbody_api"] = sorted(a for a in dir(mujoco.MjsBody) if not a.startswith("_"))

    cam = "robot0_agentview_left"
    env = create_env(env_name=args.task, robots="PandaOmron", camera_names=[cam],
                     camera_widths=args.res, camera_heights=args.res, seed=0,
                     render_onscreen=False, layout_ids=args.layout, style_ids=args.style)

    def render():
        return env.sim.render(width=args.res, height=args.res, camera_name=cam)[::-1].copy()

    eps = []
    per_slot = defaultdict(Counter)
    t_reset = []
    for i in range(args.n):
        t0 = time.perf_counter()
        env.reset()
        t_reset.append(time.perf_counter() - t0)
        m = env.sim.model._model
        xml = env.sim.model.get_xml()
        em = env.get_ep_meta()
        objs = object_bodies(env)
        omask = subtree_geoms(m, objs)
        paths = {c["name"]: c.get("info", {}).get("mjcf_path", "") for c in em["object_cfgs"]}
        for k, v in paths.items():
            per_slot[k][os.path.relpath(v, "/").split("objects/")[-1]] += 1
        e = dict(model_facts(m), xml_sha=sha(xml), xml_bytes=len(xml),
                 xml_gz_bytes=len(gzip.compress(xml.encode())),
                 obj_roots=objs, n_obj_geoms=int(omask.sum()),
                 obj_sig=geom_signature(m, omask), kitchen_sig=geom_signature(m, ~omask),
                 n_kitchen_geoms=int((~omask).sum()),
                 obj_paths=paths,
                 ep_meta_bytes=len(json.dumps(em, default=str)),
                 gen_tex_sha=sha(json.dumps(em.get("gen_textures", {}), sort_keys=True, default=str)),
                 cam_sha=sha(json.dumps(em.get("cam_configs", {}), sort_keys=True, default=str)),
                 robot_base=[list(np.asarray(em["init_robot_base_pos"], float)),
                             list(np.asarray(em["init_robot_base_ori"], float))])
        eps.append(e)
        print(f"[{i}] ngeom={e['ngeom']} nq={e['nq']} nmesh={e['nmesh']} "
              f"kitchen_geoms={e['n_kitchen_geoms']} obj_geoms={e['n_obj_geoms']} "
              f"xml={e['xml_sha']} kitchen={e['kitchen_sig']} obj={e['obj_sig']}", flush=True)

    rec["reset_seconds"] = dict(mean=float(np.mean(t_reset)), min=float(np.min(t_reset)),
                                max=float(np.max(t_reset)))
    rec["episodes"] = eps
    rec["distinct_full_xml"] = len({e["xml_sha"] for e in eps})
    rec["distinct_kitchen_sig"] = len({e["kitchen_sig"] for e in eps})
    rec["distinct_obj_sig"] = len({e["obj_sig"] for e in eps})
    rec["distinct_gen_tex"] = len({e["gen_tex_sha"] for e in eps})
    rec["distinct_cam"] = len({e["cam_sha"] for e in eps})
    rec["distinct_robot_base"] = len({json.dumps(e["robot_base"]) for e in eps})
    rec["kitchen_geoms"] = sorted({e["n_kitchen_geoms"] for e in eps})
    rec["obj_geoms"] = sorted({e["n_obj_geoms"] for e in eps})
    rec["obj_instances_per_slot"] = {k: dict(v) for k, v in per_slot.items()}
    rec["distinct_instances_per_slot"] = {k: len(v) for k, v in per_slot.items()}
    print("\n== distinct over", args.n, "resets at one kitchen ==")
    print("full XML:", rec["distinct_full_xml"], " kitchen sig:", rec["distinct_kitchen_sig"],
          " object sig:", rec["distinct_obj_sig"], " gen_textures:", rec["distinct_gen_tex"],
          " cam_configs:", rec["distinct_cam"], " robot base:", rec["distinct_robot_base"])
    print("distinct instances per slot:", rec["distinct_instances_per_slot"], flush=True)

    # ---------------------------------------------------------------- ep_meta sufficiency, redone
    # Reference episode, then replay from a JSON round-tripped ep_meta the way a corpus would store
    # it, and compare with qpos mapped by JOINT NAME.
    env.reset()
    ref_m = env.sim.model._model
    ref_xml = env.sim.model.get_xml()
    ref_em = json.loads(json.dumps(env.get_ep_meta(), default=str))
    ref_q = np.array(env.sim.data.qpos, dtype=np.float64).copy()
    ref_objs = object_bodies(env)
    ref_kit = geom_signature(ref_m, ~subtree_geoms(ref_m, ref_objs))
    ref_obj = geom_signature(ref_m, subtree_geoms(ref_m, ref_objs))
    ref_frame = render()

    def replay(tag, use_state=True):
        m = env.sim.model._model
        r = dict(model_facts(m), xml_sha=sha(env.sim.model.get_xml()),
                 xml_matches=sha(env.sim.model.get_xml()) == sha(ref_xml))
        objs = object_bodies(env)
        r["kitchen_sig_matches"] = geom_signature(m, ~subtree_geoms(m, objs)) == ref_kit
        r["obj_sig_matches"] = geom_signature(m, subtree_geoms(m, objs)) == ref_obj
        if use_state:
            q, hit, miss = qpos_by_name(ref_m, ref_q, m)
            r["joints_mapped"], r["joints_unmapped"] = hit, miss
            env.sim.data.qpos[:] = q
            env.sim.forward()
        f = render()
        d = np.abs(f.astype(np.int16) - ref_frame.astype(np.int16))
        r.update(px_exact=float((d.max(-1) == 0).mean()), px_mean=float(d.mean()),
                 px_max=int(d.max()), px_within8=float((d.max(-1) <= 8).mean()))
        rec[tag] = r
        print(f"{tag}: {json.dumps({k: v for k, v in r.items() if k != 'texdata_MB'})}", flush=True)
        return f

    env.set_ep_meta(ref_em)
    env.reset()
    fa = replay("A_ep_meta_json_roundtrip")

    # same, twice more, to separate "ep_meta is insufficient" from "replay is nondeterministic"
    env.set_ep_meta(ref_em)
    env.reset()
    replay("A2_ep_meta_again")
    env.unset_ep_meta()

    try:
        import imageio.v2 as imageio
        imageio.imwrite(os.path.join(OUT, "bucket_ref.png"), ref_frame)
        imageio.imwrite(os.path.join(OUT, "bucket_epmeta.png"), fa)
        imageio.imwrite(os.path.join(OUT, "bucket_diff_x8.png"),
                        np.clip(np.abs(fa.astype(np.int16) - ref_frame.astype(np.int16)) * 8,
                                0, 255).astype(np.uint8))
    except Exception as exc:                                # imageio is a convenience, not the result
        rec["imageio_error"] = repr(exc)

    with open(os.path.join(OUT, "bucket.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", os.path.join(OUT, "bucket.json"))


if __name__ == "__main__":
    main()
