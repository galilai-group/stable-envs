"""Write a small RoboCasa corpus the way a DATASET is written: bucketed by kitchen.

A dataset is written once and read many times, and the reader chooses the order. RoboCasa's own
corpora are already bucketed this way -- the target split is 50 tasks x 500 demos over TEN kitchens
-- so the access pattern the earlier G4 number assumed (one compiled kitchen per episode, in the
order the collector happened to reset) is a property of the collector, not of the data.

This dumps, per (layout, style) kitchen, E episodes: the expanded MJCF, the ep_meta, and a bank of
real qpos vectors taken from the env after reset and a few steps. Everything downstream (union
model, MJWarp setup timings, throughput) reads these files and never needs robosuite again -- the
expanded MJCF carries absolute asset paths and compiles standalone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ROOT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/corpus"


def sha(s):
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToSink")
    ap.add_argument("--kitchens", type=int, nargs="+", default=[52, 3, 24, 54, 45, 33, 10, 59],
                    help="flat list of layout style layout style ...")
    ap.add_argument("--eps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6, help="env steps before the qpos snapshot")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    pairs = [(args.kitchens[i], args.kitchens[i + 1]) for i in range(0, len(args.kitchens), 2)]
    os.makedirs(args.root, exist_ok=True)

    from robocasa.utils.env_utils import create_env

    cam = "robot0_agentview_left"
    summary = {"task": args.task, "eps_per_kitchen": args.eps, "kitchens": [], "res": args.res}

    for (layout, style) in pairs:
        tag = f"L{layout:02d}S{style:02d}"
        d = os.path.join(args.root, tag)
        os.makedirs(d, exist_ok=True)
        env = create_env(env_name=args.task, robots="PandaOmron", camera_names=[cam],
                         camera_widths=args.res, camera_heights=args.res, seed=1234 + layout,
                         render_onscreen=False, layout_ids=layout, style_ids=style)
        info = {"layout": layout, "style": style, "tag": tag, "episodes": []}
        for j in range(args.eps):
            t0 = time.perf_counter()
            env.reset()
            t_reset = time.perf_counter() - t0
            lo, hi = env.action_spec
            for _ in range(args.steps):
                env.step(np.zeros_like(lo))
            xml = env.sim.model.get_xml()
            q = np.array(env.sim.data.qpos, dtype=np.float64).copy()
            m = env.sim.model._model
            with open(os.path.join(d, f"ep{j:03d}.xml"), "w") as f:
                f.write(xml)
            np.save(os.path.join(d, f"ep{j:03d}_qpos.npy"), q)
            with open(os.path.join(d, f"ep{j:03d}_meta.json"), "w") as f:
                json.dump(env.get_ep_meta(), f, default=str)
            frame = env.sim.render(width=args.res, height=args.res, camera_name=cam)[::-1].copy()
            np.save(os.path.join(d, f"ep{j:03d}_frame.npy"), frame)
            e = {"xml_sha": sha(xml), "nq": int(m.nq), "ngeom": int(m.ngeom), "nmesh": int(m.nmesh),
                 "nmeshface": int(m.nmeshface), "ntex": int(m.ntex),
                 "xml_bytes": len(xml), "reset_s": t_reset,
                 "obj_roots": [getattr(o, "root_body", f"{n}_main") for n, o in env.objects.items()],
                 "obj_paths": {c["name"]: c.get("info", {}).get("mjcf_path", "")
                               for c in env.get_ep_meta()["object_cfgs"]}}
            info["episodes"].append(e)
            print(f"{tag} ep{j:03d} ngeom={e['ngeom']} nq={e['nq']} sha={e['xml_sha']} "
                  f"reset={t_reset:.2f}s", flush=True)
        info["distinct_xml"] = len({e["xml_sha"] for e in info["episodes"]})
        summary["kitchens"].append(info)
        del env

    with open(os.path.join(args.root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", os.path.join(args.root, "summary.json"))


if __name__ == "__main__":
    main()
