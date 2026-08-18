"""What is a DM Control frame made of, and can MJWarp reproduce it?

Answers, per domain, in one run:

  * **composition** -- how many pixels are floor (textured checker), skybox (textured gradient),
    and geoms, from MuJoCo's own segmentation buffer. This is the number that decides whether
    "textures are the blocker" is a big claim or a small one.
  * **which render effects are load-bearing** -- per-flag ablation of ``mjRND_SHADOW`` /
    ``mjRND_REFLECTION`` / ``mjRND_SKYBOX``, with the baseline RESTORED each time, because
    ``mjv_updateScene`` does not reset ``scene.flags`` and an unrestored ablation is silently
    cumulative. DM Control differs from OGBench here on both counts: its floor material has
    ``reflectance=.2`` (so the scene is mirrored in the ground) and its light casts shadows from
    MOVING geometry (so neither effect can be baked into a static plate).
  * **MJWarp agreement**, with all three known traps closed: the byte view is BGRA and both
    readings are scored; geom groups are set explicitly on both sides; and the render flags are
    set on ``MjvScene.flags`` after ``update_scene``, not on ``MjvOption.flags``.

Run under SLURM with MUJOCO_GL=egl.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")

import numpy as np

GROUPS = (0, 1, 2)


def load(domain: str, task: str):
    from dm_control import suite
    env = suite.load(domain, task, task_kwargs={"random": 0})
    p = env.physics
    return env, p, (p.model.ptr if hasattr(p.model, "ptr") else p.model)


def states(env, n: int, seed: int = 0) -> np.ndarray:
    """qpos the task actually reaches."""
    rng = np.random.default_rng(seed)
    p = env.physics
    env.reset()
    qs = []
    for i in range(n * 10):
        env.step(rng.uniform(env.action_spec().minimum, env.action_spec().maximum))
        if i % 10 == 0:
            qs.append(np.array(p.data.qpos, np.float64))
    return np.stack(qs[:n])


class Ref:
    """One ``mujoco.Renderer`` per (model, size), never closed -- see conformance.py's rule."""

    def __init__(self, mjm, res: int, groups=GROUPS):
        import mujoco
        mjm.vis.global_.offwidth = max(int(mjm.vis.global_.offwidth), res)
        mjm.vis.global_.offheight = max(int(mjm.vis.global_.offheight), res)
        self.mjm = mjm
        self.r = mujoco.Renderer(mjm, height=res, width=res)
        self.opt = mujoco.MjvOption()
        for g in range(len(self.opt.geomgroup)):
            self.opt.geomgroup[g] = 1 if g in groups else 0

    def render(self, mjd, cam: int, shadow=True, reflection=True, skybox=True, seg=False):
        import mujoco
        self.r.disable_segmentation_rendering()
        if seg:
            self.r.enable_segmentation_rendering()
        self.r.update_scene(mjd, camera=cam, scene_option=self.opt)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadow)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(reflection)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = int(skybox)
        img = self.r.render().copy()
        self.r.disable_segmentation_rendering()
        return img


def agreement(ref: np.ndarray, got: np.ndarray) -> dict:
    a, b = ref.astype(np.int32), got.astype(np.int32)
    d = np.abs(a - b).max(-1)
    mse = float(((a - b) ** 2).mean())
    return dict(exact=float((d == 0).mean()), mae=float(np.abs(a - b).mean()),
                within2=float((d <= 2).mean()), within8=float((d <= 8).mean()),
                psnr=float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)))


def composition(mjm, mjd, ref: Ref, cam: int) -> dict:
    """Fraction of the frame that is floor / skybox / other geoms, from the segmentation buffer."""
    import mujoco
    seg = ref.render(mjd, cam, seg=True)
    gid = seg[..., 0].astype(np.int64)
    gtype = seg[..., 1].astype(np.int64)
    n = gid.size
    out = {"skybox_frac": float((gid < 0).mean())}
    per = {}
    tex_frac = 0.0
    for g in np.unique(gid[gid >= 0]):
        if int(gtype[gid == g][0]) != int(mujoco.mjtObj.mjOBJ_GEOM):
            continue
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"
        f = float((gid == g).sum()) / n
        per[name] = round(f, 6)
        mid = int(mjm.geom_matid[int(g)])
        if mid >= 0:
            tid = mjm.mat_texid[mid]
            tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
            if tid >= 0:
                tex_frac += f
    out["per_geom"] = dict(sorted(per.items(), key=lambda kv: -kv[1])[:8])
    out["textured_geom_frac"] = round(tex_frac, 6)
    out["textured_total_frac"] = round(tex_frac + out["skybox_frac"], 6)
    return out


def mjwarp_ctx(mjm, res, cam, nworld=1, groups=GROUPS, **kw):
    import mujoco_warp as mjw
    sig = inspect.signature(mjw.create_render_context).parameters
    opts = dict(nworld=nworld, cam_res=(res, res), render_rgb=True,
                cam_active=[i == cam for i in range(mjm.ncam)])
    if "enabled_geom_groups" in sig:
        opts["enabled_geom_groups"] = list(groups)
    for k, v in kw.items():
        if k in sig:
            opts[k] = v
    return mjw.create_render_context(mjm, **opts), sorted(sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=84)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--domains", default="cartpole:swingup,pendulum:swingup,acrobot:swingup,"
                                         "point_mass:easy,ball_in_cup:catch,cheetah:run,"
                                         "walker:walk,hopper:hop,reacher:easy,finger:spin")
    ap.add_argument("--out", default="")
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    out = {"mujoco": mujoco.__version__, "res": args.res, "domains": {}}
    for spec in args.domains.split(","):
        dom, task = spec.split(":")
        key = f"{dom}-{task}"
        rec: dict = {}
        try:
            env, p, mjm = load(dom, task)
            qs = states(env, args.n)
            mjd = mujoco.MjData(mjm)
            ref = Ref(mjm, args.res)

            def put(q):
                mjd.qpos[:] = q
                mjd.qvel[:] = 0
                mujoco.mj_forward(mjm, mjd)

            put(qs[0])
            rec["composition"] = composition(mjm, mjd, ref, args.cam)

            # per-flag ablation, baseline restored each time
            base = np.stack([(put(q), ref.render(mjd, args.cam))[1] for q in qs])
            abl = {}
            for flag in ("shadow", "reflection", "skybox"):
                kw = {flag: False}
                alt = np.stack([(put(q), ref.render(mjd, args.cam, **kw))[1] for q in qs])
                d = np.abs(base.astype(np.int32) - alt.astype(np.int32)).max(-1)
                abl[flag] = {"pixels_changed": float((d > 0).mean()), "mae": float(d.mean())}
            rec["flag_ablation"] = abl

            # MJWarp, both channel readings, a few configs
            rec["mjwarp"] = {}
            for cfgname, cfg in (("shadows_on", dict(use_textures=True, use_shadows=True,
                                                     render_skybox=True)),
                                 ("shadows_off", dict(use_textures=True, use_shadows=False,
                                                      render_skybox=True)),
                                 ("no_tex", dict(use_textures=False, use_shadows=True,
                                                 render_skybox=True))):
                md = mujoco.MjData(mjm)
                md.qpos[:] = qs[0]
                mujoco.mj_forward(mjm, md)
                m_w = mjw.put_model(mjm)
                d_w = mjw.put_data(mjm, md, nworld=1)
                rc, sig = mjwarp_ctx(mjm, args.res, args.cam, **cfg)
                out["create_render_context_params"] = sig
                got = []
                for q in qs:
                    wp.to_torch(d_w.qpos).copy_(
                        __import__("torch").as_tensor(q[None].astype(np.float32), device="cuda"))
                    mjw.kinematics(m_w, d_w)
                    mjw.camlight(m_w, d_w)
                    mjw.refit_bvh(m_w, d_w, rc)
                    mjw.render(m_w, d_w, rc)
                    wp.synchronize()
                    got.append(rc.rgb_data.numpy().view(np.uint8).reshape(args.res, args.res, 4)
                               .copy())
                got = np.stack(got)
                rec["mjwarp"][cfgname] = {
                    "bgra_swapped": agreement(base, got[..., [2, 1, 0]]),
                    "raw_as_rgb": agreement(base, got[..., :3]),
                }
                if args.png and cfgname == "shadows_on":
                    import imageio.v2 as iio
                    os.makedirs(args.png, exist_ok=True)
                    strip = np.concatenate([base[0], got[0][..., [2, 1, 0]],
                                            np.clip(8 * np.abs(base[0].astype(np.int16)
                                                               - got[0][..., [2, 1, 0]].astype(
                                                                   np.int16)), 0, 255).astype(
                                                np.uint8)], 1)
                    iio.imwrite(os.path.join(args.png, f"{key}_strip.png"), strip)
            out["domains"][key] = rec
            print(f"{key}: comp={rec['composition']['textured_total_frac']:.3f} textured  "
                  f"abl={ {k: round(v['pixels_changed'], 4) for k, v in abl.items()} }  "
                  f"mjw={ {k: round(v['bgra_swapped']['exact'], 4) for k, v in rec['mjwarp'].items()} }",
                  flush=True)
        except Exception as e:                                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            out["domains"][key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{key}: FAIL {type(e).__name__}: {e}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
