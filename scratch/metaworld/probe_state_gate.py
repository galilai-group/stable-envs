"""WHERE Meta-World's per-episode information actually lives, and whether a fixed-width state
reaches it.

``probe_model_mutation.py`` widened the state to ``qpos`` + the model arrays a ``reset`` mutates and
closed 2 of 5 tasks exactly. This asks why the other 3 resisted, by walking a LADDER of
progressively larger restores and, at the top of the ladder, diffing every array on ``MjModel`` and
``MjData`` to name what is left.

The rungs, in order:

  R1  data state only        qpos/qvel/act/ctrl/mocap + ``mj_forward``
  R2  R1 + the 14 arrays     what ``probe_model_mutation.py`` restored
  R3  R1 + EVERY model array the largest restore that is still "a state plus the model"
  R4  R3 + derived data      the arrays MuJoCo's renderer actually READS, written AFTER
                             ``mj_forward``, which is the only way to reproduce a direct
                             ``data.*`` write that ``mj_forward`` would otherwise clobber
  R5  R1 + ``site_xpos``     R4 restricted to sites, to test the ``_set_pos_site`` hypothesis alone

Everything renders through ``mujoco.Renderer``, never through the env's own viewer: two GL contexts
in one process is the trap recorded in ``docs/research/adroit.md`` §2.1, and ``mujoco.Renderer`` is a
pure function of ``(model, data)``, which is exactly the property under test.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scratch.metaworld.probe_metaworld import make                          # noqa: E402

OLD14 = ("body_pos", "body_quat", "site_pos", "site_quat", "geom_pos", "geom_quat", "geom_size",
         "geom_rgba", "site_rgba", "mat_rgba", "light_pos", "cam_pos", "geom_matid", "site_matid")

# what mjv_updateScene reads off MjData to place the scene. Restoring these AFTER mj_forward is the
# only way to reproduce a value some env wrote straight into data and never into the model.
DERIVED = ("xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos",
           "site_xmat", "cam_xpos", "cam_xmat", "light_xpos", "light_xdir", "subtree_com")

DATA_STATE = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart")

TASKS = ["reach-v3", "pick-place-v3", "door-open-v3", "button-press-v3", "drawer-open-v3"]


def model_arrays(m) -> dict[str, np.ndarray]:
    """Every writable numpy array on MjModel. Both episodes share one MjModel object, so the
    structural (integer, address) arrays are identical and restoring them is a no-op; taking them
    all is what makes R3 an upper bound rather than a guess."""
    out = {}
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            v = getattr(m, name)
        except Exception:                                                   # noqa: BLE001
            continue
        if isinstance(v, np.ndarray) and v.size and v.dtype != object:
            out[name] = v.copy()
    return out


def data_arrays(d) -> dict[str, np.ndarray]:
    out = {}
    for name in dir(d):
        if name.startswith("_"):
            continue
        try:
            v = getattr(d, name)
        except Exception:                                                   # noqa: BLE001
            continue
        if isinstance(v, np.ndarray) and v.size and v.dtype != object:
            out[name] = v.copy()
    return out


def restore(obj, snap: dict[str, np.ndarray], keys) -> None:
    for k in keys:
        if k not in snap:
            continue
        try:
            getattr(obj, k)[:] = snap[k]
        except Exception:                                                   # noqa: BLE001
            pass


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    e = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max": int(e.max()), "mean": round(float(e.mean()), 4),
            "frac_diff": round(float((e.max(-1) > 0).mean()), 6)}


def probe(task: str, size: int = 128, steps: int = 12, seed: int = 0) -> dict:
    env = make(task, size)
    u = env.unwrapped
    m, d = u.model, u.data
    rng = np.random.default_rng(seed)

    ren = mujoco.Renderer(m, height=size, width=size)
    opt_all = mujoco.MjvOption()
    opt_nosite = mujoco.MjvOption()
    opt_nosite.sitegroup[:] = 0

    def shot(opt=None):
        ren.update_scene(d, camera="corner", scene_option=opt if opt is not None else opt_all)
        return ren.render().copy()

    def rollout():
        env.reset()
        for _ in range(steps):
            env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))

    # ---- episode A: the reference -------------------------------------------------------------
    rollout()
    snap_m = model_arrays(m)
    snap_d = data_arrays(d)
    fa = shot()
    fa_nosite = shot(opt_nosite)
    tgt_a = np.array(getattr(u, "_target_pos", [np.nan] * 3), dtype=float).tolist()

    # ---- episode B: a different draw, then the ladder ------------------------------------------
    for _ in range(8):                       # two draws that coincide would make the test vacuous
        rollout()
        tgt_b = np.array(getattr(u, "_target_pos", [np.nan] * 3), dtype=float).tolist()
        if not np.allclose(tgt_a, tgt_b, equal_nan=True):
            break
    snap_mB = model_arrays(m)                # episode B's model, restored before every rung so the
    snap_dB = data_arrays(d)                 # rungs are independent rather than cumulative
    rec = {"task": task, "steps": steps, "target_pos_A": tgt_a, "target_pos_B": tgt_b,
           "draw_differs": bool(not np.allclose(tgt_a, tgt_b, equal_nan=True)),
           "rungs": {}, "rungs_nosite": {}}

    rec["rungs"]["R0_nothing"] = stats(shot(), fa)
    rec["rungs_nosite"]["R0_nothing"] = stats(shot(opt_nosite), fa_nosite)

    def rung(name, model_keys=(), post_data_keys=()):
        restore(m, snap_mB, snap_mB.keys())            # back to episode B before every rung
        restore(d, snap_dB, snap_dB.keys())
        restore(m, snap_m, model_keys)
        restore(d, snap_d, DATA_STATE)
        mujoco.mj_forward(m, d)
        restore(d, snap_d, post_data_keys)
        rec["rungs"][name] = stats(shot(), fa)
        rec["rungs_nosite"][name] = stats(shot(opt_nosite), fa_nosite)

    rung("R1_data_only")
    rung("R2_old14", OLD14)
    rung("R3_all_model", snap_m.keys())
    rung("R4_all_model_plus_derived", snap_m.keys(), DERIVED)
    rung("R5_data_plus_site_xpos", (), ("site_xpos", "site_xmat"))
    rung("R6_old14_plus_site_xpos", OLD14, ("site_xpos", "site_xmat"))

    # ---- back to R3 (the honest "fixed-width state" rung): what is left, and who owns it? -------
    restore(m, snap_mB, snap_mB.keys())
    restore(d, snap_dB, snap_dB.keys())
    restore(m, snap_m, snap_m.keys())
    restore(d, snap_d, DATA_STATE)
    mujoco.mj_forward(m, d)
    f3 = shot()

    now_m, now_d = model_arrays(m), data_arrays(d)
    rec["model_still_differs"] = sorted(
        k for k in snap_m if k in now_m and snap_m[k].shape == now_m[k].shape
        and not np.array_equal(snap_m[k], now_m[k]))
    rec["data_still_differs"] = sorted(
        k for k in snap_d if k in now_d and snap_d[k].shape == now_d[k].shape
        and not np.array_equal(snap_d[k], now_d[k]))

    bad = (np.abs(f3.astype(np.int16) - fa.astype(np.int16)).max(-1) > 0)
    ren.enable_segmentation_rendering()
    ren.update_scene(d, camera="corner", scene_option=opt_all)
    seg = ren.render().copy()
    ren.disable_segmentation_rendering()
    ids, types = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
    owners: dict[str, int] = {}
    for oid, otype in zip(ids[bad], types[bad]):
        if otype == int(mujoco.mjtObj.mjOBJ_GEOM):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(oid)) or f"geom{oid}"
            key = f"geom:{nm}"
        elif otype == int(mujoco.mjtObj.mjOBJ_SITE):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, int(oid)) or f"site{oid}"
            key = f"site:{nm}"
        else:
            key = f"other:{otype}"
        owners[key] = owners.get(key, 0) + 1
    rec["R3_residual_owners"] = dict(sorted(owners.items(), key=lambda kv: -kv[1])[:12])
    rec["R3_residual_pixels"] = int(bad.sum())

    # the frame at R3 minus the goal site, as a share of the frame
    rec["site_share_of_frame"] = round(
        float((types == int(mujoco.mjtObj.mjOBJ_SITE)).mean()), 6)
    env.close()
    return rec


def reset_staleness(task: str, size: int = 128) -> dict:
    """Is the frame RIGHT AFTER ``reset()`` the frame the state implies?

    Several ``reset_model`` implementations write ``model.body_pos`` / ``model.site_pos`` AFTER
    their last ``mj_forward`` (``_set_obj_xyz`` is what forwards), so ``data.site_xpos`` still holds
    the PREVIOUS draw's placement. Nothing in a stored state can reproduce that frame, because it is
    a function of the write order inside ``reset``, not of the state. One ``mj_forward`` — or one
    ``step``, which forwards — removes it.
    """
    env = make(task, size)
    u = env.unwrapped
    m, d = u.model, u.data
    ren = mujoco.Renderer(m, height=size, width=size)

    def shot():
        ren.update_scene(d, camera="corner")
        return ren.render().copy()

    env.reset()
    env.reset()                                     # so a stale value is a PREVIOUS draw, not init
    f_reset = shot()
    sx_before = np.array(d.site_xpos, copy=True)
    xp_before = np.array(d.xpos, copy=True)
    mujoco.mj_forward(m, d)
    f_fwd = shot()
    rec = {"task": task, "reset_vs_forward": stats(f_reset, f_fwd),
           "site_xpos_stale_max_m": round(float(np.abs(sx_before - d.site_xpos).max()), 6),
           "xpos_stale_max_m": round(float(np.abs(xp_before - d.xpos).max()), 6)}

    env.step(np.zeros(u.action_space.shape, dtype=np.float32))
    f_step = shot()
    mujoco.mj_forward(m, d)
    rec["step_vs_forward"] = stats(f_step, shot())
    env.close()
    return rec


def seed_ladder(task: str, size: int = 128) -> dict:
    """Can an episode be reproduced at all? Three routes, one process each would be cleaner but the
    draw is a property of the env object, not of the GL context, so one process is sound here."""
    out = {"task": task}

    def draws(fn, n=3):
        env = make(task, size)
        u = env.unwrapped
        got = []
        for i in range(n):
            fn(env, u, i)
            got.append(np.array(u._last_rand_vec, dtype=float).copy())
        env.close()
        return got

    def spread(v):
        v = np.stack(v)
        return round(float(np.abs(v - v[0]).max()), 8)

    out["reset_seed_1"] = spread(draws(lambda e, u, i: e.reset(seed=1)))
    out["env_seed_then_reset"] = spread(
        draws(lambda e, u, i: (e.unwrapped.seed(1) if hasattr(e.unwrapped, "seed") else None,
                               e.reset())))

    def seeded(e, u, i):
        u.seeded_rand_vec = True
        u._np_random = np.random.default_rng(1)
        u._np_random_seed = 1
        e.reset()

    out["seeded_rand_vec_plus_rng_reseed"] = spread(draws(seeded))

    def frozen(e, u, i):
        if i == 0:
            e.reset()
            u._pinned = np.array(u._last_rand_vec, copy=True)
        u._freeze_rand_vec = True
        u._last_rand_vec = u._pinned
        e.reset()

    out["freeze_rand_vec"] = spread(draws(frozen))
    return out


if __name__ == "__main__":
    tasks = sys.argv[1:] or TASKS
    if tasks == ["ALL"]:
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS
        tasks = sorted(ALL_V3_ENVIRONMENTS)
    out = {"state_gate": [], "seed_gate": [], "staleness": []}
    for t in tasks:
        for key, fn in (("state_gate", probe), ("staleness", reset_staleness),
                        ("seed_gate", seed_ladder)):
            try:
                r = fn(t)
            except Exception as e:                                          # noqa: BLE001
                import traceback
                traceback.print_exc()
                r = {"task": t, "error": f"{type(e).__name__}: {e}"}
            out[key].append(r)
            print(key, json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_state_gate.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
