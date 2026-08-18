"""Decompose a DM Control frame into what a baked plate gives for free and what must be drawn.

The OGBench renderer works because the camera is fixed, the scene is static, and the agent's
only effect on the frame is the pixels it covers. DM Control breaks the last of those in two
ways that a static plate cannot absorb, and this measures both rather than assuming either:

  * the ground material has ``reflectance=.2``, so a moving geom is MIRRORED in the floor;
  * every domain's light has ``castshadow=1`` under ``<quality shadowsize="2048"/>``, so a moving
    geom DARKENS floor pixels it does not cover.

Both ablations are done as MODEL edits (``light_castshadow``, ``mat_reflectance``), not as
``mjRND_SHADOW`` on the scene. Toggling the render flag puts MuJoCo on a different lighting path
entirely -- the OGBench study measured 43-66% of pixels changing for an effect whose genuine
occlusion footprint was 0.05% -- so a flag ablation answers a different question than the one asked.

"Plate" here means the frame with every MOVING geom assigned to a disabled geom group, so the
movers are absent from ``mjvScene`` entirely and cast neither shadow nor reflection. That is the
picture a bake would produce, and its constancy across states is checked rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HIDDEN_GROUP = 5


def load(domain: str, task: str):
    from dm_control import suite
    env = suite.load(domain, task, task_kwargs={"random": 0})
    p = env.physics
    return env, p, (p.model.ptr if hasattr(p.model, "ptr") else p.model)


def rollout_qpos(env, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = env.physics
    env.reset()
    qs = []
    for i in range(n * 10):
        env.step(rng.uniform(env.action_spec().minimum, env.action_spec().maximum))
        if i % 10 == 0:
            qs.append(np.array(p.data.qpos, np.float64))
    return np.stack(qs[:n])


def moving_geoms(mjm) -> list[int]:
    """Geoms whose world pose depends on qpos: anything not attached to the world body."""
    return [g for g in range(mjm.ngeom) if int(mjm.geom_bodyid[g]) != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=84)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--domains", default="cartpole:swingup,pendulum:swingup,acrobot:swingup,"
                                         "point_mass:easy,ball_in_cup:catch,reacher:easy,"
                                         "finger:spin,cheetah:run,walker:walk,hopper:hop,"
                                         "swimmer:swimmer6,fish:swim,humanoid:stand")
    args = ap.parse_args()

    import mujoco

    out = {"mujoco": mujoco.__version__, "res": args.res, "domains": {}}
    for spec in args.domains.split(","):
        dom, task = spec.split(":")
        key = f"{dom}-{task}"
        try:
            env, p, mjm = load(dom, task)
            qs = rollout_qpos(env, args.n)
            mjd = mujoco.MjData(mjm)
            mjm.vis.global_.offwidth = max(int(mjm.vis.global_.offwidth), args.res)
            mjm.vis.global_.offheight = max(int(mjm.vis.global_.offheight), args.res)
            r = mujoco.Renderer(mjm, height=args.res, width=args.res)
            opt = mujoco.MjvOption()
            movers = moving_geoms(mjm)
            g0 = np.array(mjm.geom_group, copy=True)
            cast0 = np.array(mjm.light_castshadow, copy=True)
            refl0 = np.array(mjm.mat_reflectance, copy=True)

            def shot(q, hide=False, shadow=True, reflect=True, cam=0):
                mjm.geom_group[:] = g0
                if hide:
                    mjm.geom_group[movers] = HIDDEN_GROUP
                mjm.light_castshadow[:] = cast0 if shadow else 0
                mjm.mat_reflectance[:] = refl0 if reflect else 0.0
                for g in range(len(opt.geomgroup)):
                    opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
                mjd.qpos[:] = q
                mjd.qvel[:] = 0
                mujoco.mj_forward(mjm, mjd)
                r.update_scene(mjd, camera=cam, scene_option=opt)
                img = r.render().copy()
                mjm.geom_group[:] = g0
                mjm.light_castshadow[:] = cast0
                mjm.mat_reflectance[:] = refl0
                return img

            def frac(a, b):
                d = np.abs(a.astype(np.int32) - b.astype(np.int32)).max(-1)
                return {"pix": float((d > 0).mean()), "mae": float(d.mean()),
                        "max": int(d.max())}

            full = np.stack([shot(q) for q in qs])
            plate = np.stack([shot(q, hide=True) for q in qs])
            no_sh = np.stack([shot(q, shadow=False) for q in qs])
            no_rf = np.stack([shot(q, reflect=False) for q in qs])
            no_both = np.stack([shot(q, shadow=False, reflect=False) for q in qs])
            plate_nb = np.stack([shot(q, hide=True, shadow=False, reflect=False) for q in qs])

            rec = {
                "n_movers": len(movers), "ngeom": int(mjm.ngeom),
                # is the plate genuinely constant?
                "plate_constant_max": int(np.abs(plate.astype(np.int32)
                                                 - plate[0].astype(np.int32)).max()),
                # everything the mover pass must produce, at the official appearance
                "mover_pass": frac(full, plate),
                # the same with shadows and reflection removed from the MODEL: the mover's own pixels
                "mover_pixels_only": frac(no_both, plate_nb),
                # what each effect is worth
                "shadows_worth": frac(full, no_sh),
                "reflection_worth": frac(full, no_rf),
            }
            out["domains"][key] = rec
            print(f"{key:22s} movers={rec['n_movers']:2d} plate_const={rec['plate_constant_max']:3d} "
                  f"mover_pass={rec['mover_pass']['pix']:.4f} "
                  f"mover_px={rec['mover_pixels_only']['pix']:.4f} "
                  f"shadow={rec['shadows_worth']['pix']:.4f}/{rec['shadows_worth']['mae']:.2f} "
                  f"reflect={rec['reflection_worth']['pix']:.4f}/{rec['reflection_worth']['mae']:.2f}",
                  flush=True)
        except Exception as e:                                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            out["domains"][key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{key}: FAIL {e}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
