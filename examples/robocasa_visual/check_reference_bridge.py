"""Is a plain ``mujoco.Renderer`` at geom group 1 the SAME renderer robosuite calls? Asked, not assumed.

This decides whether a RoboCasa env can be conformance-tested in one process at all, and the reason it
is a real question is a hard version pin:

    robocasa/__init__.py:1006   assert mujoco.__version__ == "3.3.1"
    mujoco_warp 3.11.0          requires mujoco >= 3.11

so no interpreter can hold both robocasa and MJWarp. If the official frame can only be produced by
``env.sim.render``, a visual-tier env cannot check itself. ``docs/research/robocasa.md`` Section 3
reports that restricting a bare ``mujoco.Renderer`` to geom group 1 reproduces ``env.sim.render``
bit-identically -- but it reports it *within one MuJoCo version*, and the bridge this repo needs also
has to cross 3.3.1 -> 3.11.0.

So both hops are measured here, against frames ``dump_trajectory.py`` recorded from robosuite itself:

    hop 1   same MuJoCo, robosuite's offscreen context vs a bare Renderer at group 1
    hop 2   3.3.1 (robosuite's) vs 3.11.0 (the one MJWarp can share a process with)

Run the first in the ``robocasa`` conda env and the second in ``~/venvs/se3d``; each prints the max
absolute difference over the whole trajectory, and only ``0`` licenses the bridge.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

DEFAULT_OUT = "/mnt/vast/home/randall.balestriero/robocasa_visual_work"


def group1_option():
    """The ``MjvOption`` ``robosuite/environments/base.py`` builds for its offscreen context.

    ``vopt.geomgroup[0] = render_collision_mesh`` (default ``False``) and ``vopt.geomgroup[1] =
    render_visual_mesh`` (default ``True``); everything else is MuJoCo's default. A bare
    ``mujoco.Renderer`` shows groups 0-2 and would paint robosuite's ~730 collision primitives over
    the visual meshes -- a frame RoboCasa never produces.
    """
    import mujoco

    vopt = mujoco.MjvOption()
    vopt.geomgroup[:] = 0
    vopt.geomgroup[1] = 1
    return vopt


def render_group1(xml: str, qpos: np.ndarray, cam: str, res: int, n: int) -> np.ndarray:
    import mujoco

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, width=res, height=res)  # never closed; see conformance.py's rule
    vopt = group1_option()
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    assert cid >= 0, f"camera {cam!r} not in the model"
    out = np.empty((n, res, res, 3), np.uint8)
    for i in range(n):
        d.qpos[:] = qpos[i]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cid, scene_option=vopt)
        out[i] = r.render()
    return out


def bench_official(xml: str, qpos: np.ndarray, cam: str, res: int, reps: int = 300) -> dict:
    """The official rasterizer's steady-state throughput, TWO ways, each after its own warm-up.

    ``raster`` is the loop-vs-loop number: ``update_scene`` + ``render`` with the pose already in
    ``data``. ``restore`` is what a dataset consumer performs -- write ``qpos``, ``mj_forward``,
    render -- and on a settled kitchen the constraint solver is comparable to the raster.

    Each variant gets a full warm-up pass of its own and is then measured twice, best taken. A single
    pass measured this at 345 f/s for the raster and 479 for the strictly larger restore-and-raster,
    which is impossible and is what an unwarmed first loop looks like.
    """
    import time

    import mujoco

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, width=res, height=res)
    vopt = group1_option()
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    d.qpos[:] = qpos[0]
    mujoco.mj_forward(m, d)

    def raster(n):
        for i in range(n):
            r.update_scene(d, camera=cid, scene_option=vopt)
            r.render()

    def restore(n):
        for i in range(n):
            d.qpos[:] = qpos[i % len(qpos)]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cid, scene_option=vopt)
            r.render()

    out = {}
    for name, fn in (("raster", raster), ("restore", restore)):
        fn(reps // 3)  # warm-up, discarded
        best = 0.0
        for _ in range(2):
            t0 = time.perf_counter()
            fn(reps)
            best = max(best, reps / (time.perf_counter() - t0))
        out[name] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitchen", default="L52S03")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tag", default="unknown")
    ap.add_argument("--bench-res", type=int, nargs="+", default=[128, 224, 256])
    args = ap.parse_args()

    import mujoco

    d = os.path.join(args.out, args.kitchen)
    meta = json.load(open(os.path.join(d, "meta.json")))
    qpos = np.load(os.path.join(d, "qpos.npy"))
    res = int(meta["res"])
    n = min(args.n, len(qpos))
    rec = {
        "kitchen": args.kitchen,
        "mujoco": mujoco.__version__,
        "tag": args.tag,
        "recorded_under_mujoco": meta["mujoco"],
        "n": n,
        "res": res,
        "cameras": {},
    }
    for cam in meta["cameras"]:
        official = np.load(os.path.join(d, f"official_{cam}.npy"))[:n]
        ours = render_group1(os.path.join(d, "scene.xml"), qpos, cam, res, n)
        diff = np.abs(official.astype(np.int16) - ours.astype(np.int16))
        rec["cameras"][cam] = {
            "max_abs": int(diff.max()),
            "mean_abs": float(diff.mean()),
            "exact_frac": float((diff.max(-1) == 0).mean()),
        }
        print(f"{cam}: max {diff.max()}  mean {diff.mean():.5f}  exact {(diff.max(-1) == 0).mean():.6f}", flush=True)
    rec["official_fps"] = {
        str(r): bench_official(os.path.join(d, "scene.xml"), qpos, meta["cameras"][0], int(r)) for r in args.bench_res
    }
    print("official fps:", json.dumps(rec["official_fps"]), flush=True)
    p = os.path.join(args.out, f"refbridge_{args.tag}_{args.kitchen}.json")
    with open(p, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
