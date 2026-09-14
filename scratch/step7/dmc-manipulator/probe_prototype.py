"""Phase-0 design validation: what a ``DMControlSuite`` cell for manipulator-bring_ball WOULD score.

Nothing in ``stable_envs/`` is edited. The shipped ``DMControlSuite`` is subclassed here and ONE
guard is patched out in this process only -- ``_episode_slots`` refuses a per-episode ``*_quat``
slot, and manipulator's episode draws ``body_quat['target_ball']``. Everything else (the three
plate bakes, the mover census, the ray-cast pass, the shadow and reflection passes, the composite)
is the shipped code, so the numbers below are the DESIGN's numbers and not a reimplementation's.

Reported per draw:
  * ``pixel_agreement`` against ``physics.render`` (all-pixel exact, interior exact, mean |diff|);
  * the plate/mover SPLIT of every disagreeing pixel, i.e. how much of the residual the ray-caster
    owns, computed from MuJoCo's own mover-hidden plate;
  * whether the walls' reflections (the shipped bake keeps ONE reflective plane) are where the
    residual lives.

Run:  MUJOCO_GL=glfw python scratch/step7/dmc-manipulator/probe_prototype.py --size 224 --n 64
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch


def patched_episode_slots(mod):
    """``_episode_slots`` with the ``*_quat`` half of its guard removed, and nothing else.

    The guard's own reason is a SIZE in the state: ``_prims`` reads each mover's size once at bake
    time and shares it across the batch, so a per-row size would have to thread through every
    intersector. That argument does NOT extend to orientation: ``_fk`` writes the drawn slot into the
    model, runs ``mj_kinematics``, and reads ``d.geom_xmat`` PER ROW into an already per-row
    ``(B,P,3,3)`` tensor which ``cast_nearest`` / ``any_blocking`` consume as such. This patch is the
    experiment that tests that reading.
    """
    orig = mod._episode_slots

    def slots(env, m):
        import numpy as _np
        have = [k for k in mod.EPISODE_ARRAYS if hasattr(m, k)]
        draws = []
        for _ in range(mod.N_DRAWS):
            env.reset()
            draws.append({k: _np.array(getattr(m, k), copy=True).reshape(-1) for k in have})
        out = []
        for k in have:
            st = _np.stack([d[k] for d in draws]).astype(_np.float64)
            for i in _np.flatnonzero(st.max(0) - st.min(0) > 0):
                out.append((k, int(i)))
        bad = [s for s in out if s[0].endswith(("_size", "_dir"))]   # _quat removed
        if bad:
            raise RuntimeError(f"unsupported episode draw: {bad}")
        return tuple(out)

    mod._episode_slots = slots
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--ss", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/phase0")
    a = ap.parse_args()

    import mujoco
    from importlib.metadata import version as _v
    from stable_envs.envs import dmcontrol as D
    from stable_envs.testing import conformance as C

    patched_episode_slots(D)

    Cell = type("DMCManipulatorBringBall", (D.DMControlSuite,), dict(
        DOMAIN="manipulator", TASK="bring_ball", CAMERA=0,
        default_size=a.size,
        name="dmc-manipulator-bring-ball",
        reference_id="dm_control/manipulator-bring_ball",
        __doc__="phase-0 prototype only; not registered"))

    res = dict(size=a.size, n=a.n, draws=a.draws, supersample=a.ss, device=a.device,
               mujoco=mujoco.__version__, dm_control=_v("dm_control"),
               reference_pair_note=("measured on mujoco 3.12.0 + dm_control 1.0.45 (local se6-venv); "
                                    "the declared DMC reference pair is mujoco 3.11.0 + "
                                    "dm_control 1.0.44 on Oscar's se-vis311/se-tex311"))

    env = Cell(batch=1, size=a.size, supersample=a.ss, device=a.device, backend="auto", seed=0)
    res["state_dim"] = int(env.state_dim)
    res["state_names"] = list(env.state_names)
    res["nq"] = int(env.nq)
    res["episode_slots"] = [f"{arr}[{i}]" for arr, i in Cell._slots()]
    res["variation_space"] = [p for p, _, _ in env.variation_space.flat_layout()]
    sc = env._scene()
    res["movers"] = dict(
        geoms=[mujoco.mj_id2name(sc.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in sc.movers[0]],
        sites=[mujoco.mj_id2name(sc.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in sc.movers[1]])
    res["statics"] = dict(
        geoms=[mujoco.mj_id2name(sc.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in sc.statics[0]],
        sites=[mujoco.mj_id2name(sc.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in sc.statics[1]])
    res["shadow_light_index_in_scene"] = int(sc.shadow_light)
    res["light_kinds"] = [int(k) for k in sc.lights.kind]
    res["light_castshadow"] = [int(k) for k in sc.lights.castshadow]
    res["reflectance_used"] = float(sc.reflectance)
    res["tendon"] = None if sc.tendon is None else "spatial"
    res["plate_shadow_differs"] = bool(
        (sc.plate.to(torch.int16) != sc.plate_shadow.to(torch.int16)).any().item())
    res["plate_norefl_differs"] = bool(
        (sc.plate.to(torch.int16) != sc.plate_norefl.to(torch.int16)).any().item())

    # how many reflective planes there are, against the ONE the bake keeps
    m = sc.model
    refl = []
    for i in sc.statics[0]:
        mid = int(m.geom_matid[i])
        if int(m.geom_type[i]) == 0 and mid >= 0 and float(m.mat_reflectance[mid]) > 0:
            refl.append(dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                             reflectance=float(m.mat_reflectance[mid])))
    res["reflective_static_planes"] = refl

    ref = Cell.reference_env(a.size)
    per_draw = []
    rng = np.random.default_rng(0)
    for k in range(a.draws):
        states = env.sample_states(a.n, np.random.default_rng(1000 + k))
        got = env.render(torch.as_tensor(states, device=a.device)).cpu().numpy()
        want = Cell.reference_render(ref, states)
        pa = C.pixel_agreement(got, want)
        pa["ssim"] = float(C.ssim(got, want))

        # the plate/mover split: MuJoCo's own frame with every mover hidden gives the pixels the
        # plate OWNS; a disagreement there is the plate's, and one elsewhere is the ray-caster's.
        d = np.abs(got.astype(np.int16) - want.astype(np.int16)).max(-1)
        plate_np = sc.plate.cpu().numpy().astype(np.int16)
        mover_mask = (np.abs(want.astype(np.int16) - plate_np[None]).max(-1) > 0)
        bad = d > 0
        nbad = int(bad.sum())
        pa["bad_px"] = nbad
        pa["bad_inside_mover_region_frac"] = float((bad & mover_mask).sum() / max(1, nbad))
        pa["plate_region_px"] = int((~mover_mask).sum())
        pa["plate_region_exact_frac"] = float((d[~mover_mask] == 0).mean())
        pa["mover_region_px"] = int(mover_mask.sum())
        pa["mover_region_exact_frac"] = float((d[mover_mask] == 0).mean())
        per_draw.append(pa)
        print(f"draw {k}: exact {pa['exact_frac']:.4f} interior "
              f"{pa['interior_exact_frac']} (frac {pa['interior_frac']:.4f}) "
              f"mean {pa['mean_abs']:.3f} max {pa['max_abs']} ssim {pa['ssim']:.4f} "
              f"bad {nbad} of which in-mover-region {pa['bad_inside_mover_region_frac']:.3f}")
    res["per_draw"] = per_draw

    def agg(key):
        v = [p[key] for p in per_draw if p[key] is not None]
        return None if not v else dict(min=float(np.min(v)), median=float(np.median(v)),
                                       max=float(np.max(v)))
    res["summary"] = {k: agg(k) for k in
                      ("exact_frac", "interior_exact_frac", "interior_frac", "mean_abs", "max_abs",
                       "ssim", "plate_region_exact_frac", "mover_region_exact_frac",
                       "bad_inside_mover_region_frac")}

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"prototype_{a.size}_ss{a.ss}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res["summary"], indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
