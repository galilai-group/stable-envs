"""Stage B — re-render stage A's trajectory with MJWarp, and price the two things a user pays.

Runs in ``~/venvs/se3d`` (mujoco 3.11.0 + mujoco-warp 3.11.0). It never imports robosuite: the MJCF
stage A wrote carries absolute asset paths and compiles standalone, which is what lets the two halves
live in different interpreters.

Every known correction is applied, and each of them is silent when omitted:

  ``prepare_model``   zero-alpha GEOMS **and** SITES out of the enabled groups. MJWarp's shader reads
                      ``color[0..2]`` and packs a constant 255, so RoboCasa's 150-300 invisible
                      placement markers render as opaque saturated green over ~40 % of the frame.
  ``gl_rounding()``   MJWarp truncates float->uint8; the GL pipeline MuJoCo renders through rounds.
  BGRA                ``pack_rgba_to_uint32`` writes ``(a<<24)|(r<<16)|(g<<8)|b``.
  shadows ON          MuJoCo's default. (RoboCasa's one light has ``castshadow=0``, so this changes
                      nothing here -- it is set because matching the reference's flags is the rule,
                      not because it pays.)
  geom group 1        what robosuite actually draws. Groups 0-2 would add ~730 collision primitives.

Setup is decomposed rather than totalled, because a live-generation user pays it once per compiled
kitchen and pays the loop every frame, and the break-even between them is the whole economic question.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_OUT = os.environ.get("ROBOCASA_VISUAL_OUT", "/mnt/vast/home/randall.balestriero/robocasa_visual_work")
CORPUS = os.environ.get("ROBOCASA_CORPUS", os.path.join(REPO, "examples", "robocasa_visual", "corpus"))


def load_mjwarp_helpers():
    """``stable_envs.render.mjwarp`` by PATH.

    Importing it as a package member would execute ``stable_envs/__init__.py``, which pulls in
    gymnasium and torch-side env dependencies this venv deliberately does not have. The module itself
    has no package-internal imports, so loading the file is exact rather than a copy that can drift.
    """
    spec = importlib.util.spec_from_file_location(
        "se_mjwarp", os.path.join(REPO, "stable_envs/render/mjwarp/__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["se_mjwarp"] = mod  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


MW = load_mjwarp_helpers()


def qpos_by_name(mjm, meta, qpos):
    """Map stage A's ``qpos`` onto THIS compilation's joint layout, by name, and assert it is total.

    Copying by index across two compilations is what produced this family's retracted 94.95 %: it
    scores a mis-posed scene and reads as a mis-built one. If the layouts happen to agree the mapping
    is the identity and costs nothing; if they do not, this is the difference between a fidelity
    measurement and a nonsense one.
    """
    import mujoco

    src_names = meta["model"]["joint_names"]
    src_adr = meta["model"]["jnt_qposadr"]
    src_type = meta["model"]["jnt_type"]
    width = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
    out = np.zeros((len(qpos), int(mjm.nq)), np.float64)
    hit = 0
    identical = int(mjm.nq) == int(meta["model"]["nq"])
    for j in range(mjm.njnt):
        nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm not in src_names:
            raise AssertionError(f"joint {nm!r} is in this compilation and not in stage A's")
        k = src_names.index(nm)
        w = width.get(int(mjm.jnt_type[j]), 1)
        assert w == width.get(int(src_type[k]), 1), f"joint {nm!r} changed type between compilations"
        dst, src = int(mjm.jnt_qposadr[j]), int(src_adr[k])
        out[:, dst : dst + w] = qpos[:, src : src + w]
        identical &= dst == src
        hit += w
    assert hit == int(mjm.nq), f"mapped {hit} of {mjm.nq} qpos entries"
    return out, bool(identical)


class Ctx:
    """A live MJWarp render context for one compiled kitchen, plus how long each stage of it took."""

    def __init__(self, xml, res, nworld, cam_name, batched_fields=(), groups=(1,), bake=False):
        import mujoco
        import mujoco_warp as mjw
        import torch
        import warp as wp

        self.wp, self.mjw, self.torch = wp, mjw, torch
        self.nworld, self.res = int(nworld), int(res)
        t = {}
        t0 = time.perf_counter()
        self.mjm = mujoco.MjModel.from_xml_path(xml)
        t["compile_s"] = time.perf_counter() - t0

        self.prepared = MW.prepare_model(self.mjm, groups=groups)
        self.gaps = MW.texture_gaps(self.mjm, groups=groups)
        t["bake"] = bool(bake)
        if bake:
            b = MW.bake_primitive_textures(self.mjm, groups=groups)
            t["baked_materials"] = len(b["baked"])
            t["mixed_materials"] = len(b["mixed"])
        self.cam = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        assert self.cam >= 0, f"camera {cam_name!r} not in the model"

        d0 = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, d0)
        t0 = time.perf_counter()
        bs = {k: int(nworld) for k in batched_fields}
        self.wm = mjw.put_model(self.mjm, batch_sizes=bs) if bs else mjw.put_model(self.mjm)
        wp.synchronize()
        t["put_model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.wd = mjw.put_data(self.mjm, d0, nworld=int(nworld))
        wp.synchronize()
        t["put_data_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # The rounding patch has to be live while the megakernel's source is GENERATED (here) and
        # while it is COMPILED (first launch), so the warm-up frame is inside the block.
        with MW.gl_rounding():
            self.rc = MW.create_render_context(
                self.mjm,
                nworld=int(nworld),
                cam_res=(int(res), int(res)),
                render_rgb=True,
                cam_active=[i == self.cam for i in range(self.mjm.ncam)],
                enabled_geom_groups=list(groups),
                use_textures=True,
                use_shadows=True,
                render_skybox=True,
            )
            wp.synchronize()
            t["render_context_s"] = time.perf_counter() - t0
            self._frame()
            wp.synchronize()
        self.bgra = wp.to_torch(self.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
        t0 = time.perf_counter()
        with wp.ScopedCapture() as cap:
            self._frame()
        self.graph = cap.graph
        t["graph_capture_s"] = time.perf_counter() - t0
        t["setup_total_s"] = sum(t.values())
        t["n_hidden_geoms"] = len(self.prepared.hidden_geoms)
        t["n_hidden_sites"] = len(self.prepared.hidden_sites)
        t["n_texgen_gap_geoms"] = len(self.gaps)
        t["ngeom"], t["nmesh"], t["ntex"] = int(self.mjm.ngeom), int(self.mjm.nmesh), int(self.mjm.ntex)
        self.timings = t

    def _frame(self):
        self.mjw.kinematics(self.wm, self.wd)
        self.mjw.camlight(self.wm, self.wd)
        self.mjw.refit_bvh(self.wm, self.wd, self.rc)
        self.mjw.render(self.wm, self.wd, self.rc)

    def render(self, qpos):
        """``(n, nq)`` -> ``(n, res, res, 3)`` uint8 RGB on device. ``n <= nworld``."""
        n = len(qpos)
        q = self.wp.to_torch(self.wd.qpos)
        q[:n] = self.torch.as_tensor(qpos, dtype=self.torch.float32, device=q.device)
        self.torch.cuda.synchronize()
        self.wp.capture_launch(self.graph)
        self.wp.synchronize()
        return self.bgra[:n][..., [2, 1, 0]].contiguous()


def bench(ctx, qpos, reps=20):
    """Steady-state frames/s for a full batch, graph-replayed, including the channel swap."""
    q = np.tile(qpos[:1], (ctx.nworld, 1)) if len(qpos) < ctx.nworld else qpos[: ctx.nworld]
    ctx.render(q)
    t0 = time.perf_counter()
    for _ in range(reps):
        ctx.render(q)
    dt = time.perf_counter() - t0
    return ctx.nworld * reps / dt


def do_trajectory(args, rec):
    d = os.path.join(args.out, args.kitchen)
    meta = json.load(open(os.path.join(d, "meta.json")))
    qpos = np.load(os.path.join(d, "qpos.npy"))
    res = int(meta["res"])
    nworld = min(args.max_worlds, len(qpos))

    # Both variants, always, because the bake is an APPROXIMATION and an approximation that is not
    # scored against the thing it approximates is a decoration. Stage C reports the pair.
    for cam in meta["cameras"]:
        for bake, tag in ((False, "mjwarp"), (True, "mjwarpbake")):
            ctx = Ctx(
                os.path.join(d, "scene.xml"),
                res,
                nworld,
                cam,
                batched_fields=("cam_pos", "cam_quat", "light_diffuse"),
                bake=bake,
            )
            q, identical = qpos_by_name(ctx.mjm, meta, qpos)
            frames = np.empty((len(q), res, res, 3), np.uint8)
            for lo in range(0, len(q), nworld):
                hi = min(lo + nworld, len(q))
                frames[lo:hi] = ctx.render(q[lo:hi]).cpu().numpy()
            np.save(os.path.join(d, f"{tag}_{cam}.npy"), frames)
            entry = dict(
                ctx.timings,
                camera=cam,
                kitchen=args.kitchen,
                res=res,
                nworld=nworld,
                qpos_layout_identical=identical,
                fps_batch=bench(ctx, q),
            )
            rec["trajectory"].append(entry)
            print(json.dumps(entry, indent=1), flush=True)
            del ctx


def do_throughput(args, rec):
    """Throughput as a (ratio, batch, RESOLUTION) surface, never as a peak.

    Resolution is in the sweep because it moves the answer more than the batch does and in the
    opposite direction to intuition: a raytraced frame costs 4x as many rays at 256 px as at 128,
    while a stored JPEG costs only 1.4x as much to decode (7,483 f/s/core at 128 against 5,486 at
    256). So "how many CPU cores of decode does this GPU replace" falls by roughly 3x between the two,
    and quoting one resolution's number as the family's number would be off by that factor.
    """
    d = os.path.join(args.out, args.kitchen)
    meta = json.load(open(os.path.join(d, "meta.json")))
    qpos = np.load(os.path.join(d, "qpos.npy"))
    for res in args.throughput_res or [int(meta["res"])]:
        for nworld in args.batches:
            ctx = Ctx(os.path.join(d, "scene.xml"), int(res), nworld, meta["cameras"][0])
            q, _ = qpos_by_name(ctx.mjm, meta, qpos)
            e = dict(ctx.timings, batch=nworld, res=int(res), fps=bench(ctx, q))
            rec["throughput"].append(e)
            print(json.dumps(e, indent=1), flush=True)
            del ctx


def do_grids(args, rec):
    """Two grids, and the difference between them is the honest part.

    ``dr_one_batch``   ONE compiled kitchen, ONE render context, ONE graph launch: the camera pose and
                       the light are per-WORLD model fields, so N appearances cost one launch. This is
                       the capability the visual tier exists for.
    ``kitchen_grid``   N DIFFERENT kitchens. These are different compiled models with different
                       geometry, so they are N contexts and N launches -- MJWarp batches worlds of one
                       model, not models. Recorded separately so the first claim is not read onto it.
    """
    d = os.path.join(args.out, args.kitchen)
    meta = json.load(open(os.path.join(d, "meta.json")))
    qpos = np.load(os.path.join(d, "qpos.npy"))
    res = int(meta["res"])
    n = int(args.grid)
    cam = meta["cameras"][0]

    ctx = Ctx(
        os.path.join(d, "scene.xml"), res, n, cam, batched_fields=("cam_pos", "cam_quat", "light_diffuse", "mat_rgba")
    )
    q, _ = qpos_by_name(ctx.mjm, meta, qpos)
    import mujoco

    rng = np.random.default_rng(0)
    wp = ctx.wp
    base_pos = np.asarray(ctx.mjm.cam_pos[ctx.cam], np.float64)
    base_quat = np.asarray(ctx.mjm.cam_quat[ctx.cam], np.float64)
    base_light = np.asarray(ctx.mjm.light_diffuse, np.float64)
    cp, cq = wp.to_torch(ctx.wm.cam_pos), wp.to_torch(ctx.wm.cam_quat)
    ld = wp.to_torch(ctx.wm.light_diffuse)
    torch = ctx.torch
    for w in range(n):
        jitter = rng.normal(0, 0.06, 3) if w else np.zeros(3)
        yaw = float(rng.uniform(-0.10, 0.10)) if w else 0.0
        dq = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        out = np.empty(4)
        mujoco.mju_mulQuat(out, dq, base_quat)
        cp[w, ctx.cam] = torch.as_tensor(base_pos + jitter, dtype=torch.float32, device=cp.device)
        cq[w, ctx.cam] = torch.as_tensor(out, dtype=torch.float32, device=cq.device)
        scale = 1.0 if w == 0 else float(rng.uniform(0.55, 1.45))
        tint = np.ones(3) if w == 0 else rng.uniform(0.7, 1.3, 3)
        ld[w] = torch.as_tensor(base_light * scale * tint, dtype=torch.float32, device=ld.device)
    idx = np.linspace(0, len(q) - 1, n).astype(int)
    t0 = time.perf_counter()
    grid = ctx.render(q[idx]).cpu().numpy()
    dt = time.perf_counter() - t0
    np.save(os.path.join(args.out, "dr_one_batch.npy"), grid)
    rec["grids"]["dr_one_batch"] = {
        "kitchen": args.kitchen,
        "worlds": n,
        "res": res,
        "launches": 1,
        "contexts": 1,
        "seconds": dt,
        "varied": ["camera pose", "light diffuse (scale + tint)"],
    }
    del ctx

    panels, labels, tim = [], [], []
    xmls = sorted(glob.glob(os.path.join(CORPUS, "*", "ep00[0-2].xml")))
    for xml in xmls[: int(args.kitchens)]:
        tag = f"{os.path.basename(os.path.dirname(xml))}/{os.path.basename(xml)[:-4]}"
        c = Ctx(xml, res, 1, cam)
        qk = np.load(xml.replace(".xml", "_qpos.npy"))[None]
        if qk.shape[1] != int(c.mjm.nq):
            del c
            continue
        panels.append(c.render(qk).cpu().numpy()[0])
        labels.append(tag)
        tim.append(c.timings)
        del c
    if panels:
        np.save(os.path.join(args.out, "kitchen_grid.npy"), np.stack(panels))
        rec["grids"]["kitchen_grid"] = {
            "panels": labels,
            "contexts": len(panels),
            "launches": len(panels),
            "res": res,
            "setup_s_each": [t["setup_total_s"] for t in tim],
            "note": "N different compiled models = N contexts; MJWarp batches worlds of ONE model",
        }
    with open(os.path.join(args.out, "grid_labels.json"), "w") as f:
        json.dump({"dr_one_batch": int(args.grid), "kitchen_grid": labels}, f, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitchen", default="L52S03")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-worlds", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--throughput-res", type=int, nargs="+", default=[128, 224, 256])
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--kitchens", type=int, default=12)
    ap.add_argument("--stages", nargs="+", default=["trajectory"], choices=["trajectory", "throughput", "grids"])
    args = ap.parse_args()

    import mujoco
    import mujoco_warp as mjw

    rec = {
        "mujoco": mujoco.__version__,
        "mujoco_warp": getattr(mjw, "__version__", "?"),
        "kitchen": args.kitchen,
        "trajectory": [],
        "throughput": [],
        "grids": {},
    }
    for s in args.stages:
        {"trajectory": do_trajectory, "throughput": do_throughput, "grids": do_grids}[s](args, rec)
    path = os.path.join(args.out, f"mjwarp_{args.kitchen}_{'_'.join(args.stages)}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
