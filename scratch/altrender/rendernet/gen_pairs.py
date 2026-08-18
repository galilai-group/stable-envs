"""Training pairs: ``(cheap render at r px, OFFICIAL frame at 256 px)``, on DISJOINT state sets.

The data for this problem is free, which is exactly why the split has to be designed rather than
taken. Both families draw their states from a small fixed pool -- RoboCasa from 200 recorded
trajectory states, doubled and yaw-randomised into 400 rows; HumanoidBench from 512 states the
official env visited -- and if a corrector is trained on frames drawn from that pool and evaluated on
frames drawn from the same pool, the evaluation measures a lookup table.

So the split is made at the level of the UNDERLYING RECORDED STATE, before any augmentation:

* **RoboCasa.** ``_yawed_pool`` row ``2i`` and row ``2i+1`` are the same recorded state ``i`` at two
  random object yaws, so splitting the 400 rows would put the same arm pose on both sides. The split
  is on ``i``: recorded states ``0..n_train-1`` are training, ``n_train..199`` are held out, and a
  training draw re-randomises the yaws with a different generator -- the same generative process the
  env uses, restricted to the training half. A fraction of training draws additionally interpolate
  the non-quaternion ``qpos`` entries between two TRAINING rows, which reaches arm poses between the
  recorded ones without leaving their convex hull.
* **HumanoidBench.** The 512-state pool is split 384/128 by index, and training draws interpolate
  between two training rows with the root quaternion renormalised (``mj_forward`` renormalises it
  anyway, and ``reference_render`` asserts the round-trip, so an unnormalised quaternion is a hard
  error rather than a silent one).

A third split, ``decl``, is the env's own ``sample_states`` at seeds 0-3 -- the exact draw the
repository measures a ``visual_fidelity`` declaration on. It OVERLAPS the training states by
construction and is generated anyway, because the declared numbers are what a reader will compare
against and a comparison against a different state draw is not a comparison. It is labelled
in-distribution everywhere it is reported.

Out-of-distribution splits are made by ``--kitchen``/``--camera`` overrides and by ``--style``, which
draws the env's own declared appearance variation (camera offset/pan/tilt and light scale/tint on
RoboCasa) per sample and applies it to BOTH the cheap render and the official one -- the reference
renderer does not take a style, so this module writes the pose into the reference model itself.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from probe_budget_frontier import ctx_cache, env_module, family_of, patched

ROOT = "/mnt/vast/home/randall.balestriero/data/rendernet"

#: RoboCasa recorded states 0..149 train, 150..199 held out. 75/25 at the level that matters.
N_TRAIN_REC_RC = 150
#: HumanoidBench pool rows 0..383 train, 384..511 held out.
N_TRAIN_HB = 384


# --------------------------------------------------------------------------------- state sampling

def _free_joint_addrs(m):
    """``[(pos_adr, quat_adr), ...]`` for every free joint, so yaw can be redrawn the way the env
    draws it and interpolation can avoid walking a quaternion off the unit sphere."""
    import mujoco
    out = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            adr = int(m.jnt_qposadr[j])
            out.append((adr, adr + 3))
    return out


def robocasa_states(cls, split: str, n: int, seed: int, p_interp: float = 0.3):
    """``(n, state_dim)`` float32 in the env's live-dim space, from the declared split."""
    import stable_envs.envs.robocasa as RC
    s = RC._scene(cls.KITCHEN)
    live = RC._live_dims(cls.KITCHEN, cls.CAMERA, 256)
    pool = RC._yawed_pool(cls.KITCHEN)                 # (400, nq); row 2i, 2i+1 <- recorded i
    n_rec = len(pool) // 2
    rec = np.arange(n_rec)
    train_rec = rec[rec < N_TRAIN_REC_RC]
    held_rec = rec[rec >= N_TRAIN_REC_RC]
    rng = np.random.default_rng(seed)
    free = _free_joint_addrs(s.model)

    def rows_of(recs):
        return np.concatenate([2 * recs, 2 * recs + 1])

    def redraw_yaw(q):
        for _, qa in free:
            yaw = rng.uniform(-np.pi, np.pi, len(q))
            q[:, qa:qa + 4] = np.stack([np.cos(yaw / 2), np.zeros_like(yaw),
                                        np.zeros_like(yaw), np.sin(yaw / 2)], 1)
        return q

    if split == "train":
        base = pool[rows_of(train_rec)]
        q = base[rng.integers(0, len(base), n)].copy()
        if p_interp > 0:
            k = int(p_interp * n)
            other = base[rng.integers(0, len(base), k)]
            t = rng.uniform(0.0, 1.0, (k, 1))
            mix = q[:k] * (1 - t) + other * t
            for _, qa in free:                          # quaternions do not lerp; keep an endpoint
                mix[:, qa:qa + 4] = q[:k, qa:qa + 4]
            q[:k] = mix
        q = redraw_yaw(q)
    elif split == "heldout":
        base = pool[rows_of(held_rec)]                  # 100 rows, arm poses never seen in training
        q = base[rng.integers(0, len(base), n)].copy()
        q = redraw_yaw(q)
    else:
        raise ValueError(split)
    return q[:, live].astype(np.float32)


def humanoidbench_states(cls, split: str, n: int, seed: int, p_interp: float = 0.5):
    import stable_envs.envs.humanoidbench as HB
    pool = HB._state_pool(cls.ROBOT, cls.TASK, HB._build_bundle(cls.ROBOT, cls.TASK).nq)
    rng = np.random.default_rng(seed)
    base = pool[:N_TRAIN_HB] if split == "train" else pool[N_TRAIN_HB:]
    q = base[rng.integers(0, len(base), n)].astype(np.float64).copy()
    if split == "train" and p_interp > 0:
        k = int(p_interp * n)
        other = base[rng.integers(0, len(base), k)].astype(np.float64)
        t = rng.uniform(0.0, 1.0, (k, 1))
        q[:k] = q[:k] * (1 - t) + other * t
    # The root free joint's quaternion is qpos[3:7] for every HumanoidBench robot. Renormalise it:
    # ``reference_render`` asserts the qpos round-trip to 1e-5 and ``mj_forward`` renormalises in
    # place, so an unnormalised quaternion is caught rather than quietly rendered from a different
    # pose than the one recorded.
    nq = q.shape[1]
    if nq >= 7:
        w = q[:, 3:7]
        nrm = np.linalg.norm(w, axis=1, keepdims=True)
        nrm[nrm < 1e-9] = 1.0
        q[:, 3:7] = w / nrm
    return q.astype(np.float32)


# --------------------------------------------------------------------------------- style sampling

def sample_style(env, n: int, seed: int, frac_default: float = 0.25,
                 only: tuple[str, ...] | None = None) -> np.ndarray:
    """``(n, style_dim)`` from the env's OWN ``sample_style``, a fraction pinned at the default.

    Pinned rows matter: a ``visual_fidelity`` declaration is measured at the default appearance, so a
    corrector that never saw the default would be scored on a picture it was never trained for.

    ``only`` restricts which LEAVES vary; every other leaf is held at its default. It exists because a
    knob can only be exercised here if the OFFICIAL renderer can be given the same value, and on
    HumanoidBench two of them cannot -- see ``reference_frames_hb``. Pinning them is the honest form
    of that: the split then declares which part of the variation space it covers instead of covering
    a knob on one side of the comparison only.
    """
    rng = np.random.default_rng(seed)
    v = env.sample_style(n, rng=rng)
    dflt = env.default_style()
    if only is not None:
        keep = set(only)
        for path, off, w in env.variation_space.flat_layout():
            if path not in keep:
                v[:, off:off + w] = np.asarray(dflt).reshape(1, -1)[:, off:off + w]
    pin = rng.random(n) < frac_default
    v[pin] = dflt
    return v.astype(np.float32)


def style_leaf(env, packed: np.ndarray, path: str) -> np.ndarray:
    """``(n, w)`` slice of a packed style array, addressed by the space's own flat layout."""
    for p, off, w in env.variation_space.flat_layout():
        if p == path:
            return np.asarray(packed[:, off:off + w], np.float64)
    raise KeyError(f"{path} not in {[p for p, _, _ in env.variation_space.flat_layout()]}")


# ------------------------------------------------------------------------------ official renderer

def reference_frames_robocasa(cls, env, states, style=None, size=256, log_every=1024):
    """Official frames, with a per-state camera pose and light when a style is given.

    ``RoboCasaBase.reference_render`` renders at the model's own camera and light, which is right for
    a declaration and wrong for training under the env's declared variation. The pose arithmetic is
    the env's own ``_camera_pose``, called here rather than reimplemented, so the two renderers are
    given the SAME camera by construction rather than by two implementations agreeing.
    """
    import mujoco
    import stable_envs.envs.robocasa as RC
    m, d, r, vopt = RC._reference(cls.KITCHEN, int(size))
    s = RC._scene(cls.KITCHEN)
    live = RC._live_dims(cls.KITCHEN, cls.CAMERA, int(size))
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA)
    base_light = np.asarray(m.light_diffuse, np.float64).copy()
    base_pos = np.asarray(m.cam_pos[cid], np.float64).copy()
    base_quat = np.asarray(m.cam_quat[cid], np.float64).copy()
    st = np.asarray(states, np.float64)
    out = np.empty((len(st), int(size), int(size), 3), np.uint8)
    if style is not None:
        cpos, cquat = env._camera_pose(style_leaf(env, style, "camera.offset"),
                                       style_leaf(env, style, "camera.pan"),
                                       style_leaf(env, style, "camera.tilt"))
        lsc = style_leaf(env, style, "light.scale").reshape(-1, 1, 1)
        ltn = style_leaf(env, style, "light.tint").reshape(-1, 1, 3)
        lights = base_light[None] * lsc * ltn
    t0 = time.perf_counter()
    for i, row in enumerate(st):
        if style is not None:
            m.cam_pos[cid] = cpos[i]
            m.cam_quat[cid] = cquat[i]
            m.light_diffuse[:] = lights[i]
        d.qpos[:] = s.pinned
        d.qpos[live] = row
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cid, scene_option=vopt)
        out[i] = r.render()
        if log_every and (i + 1) % log_every == 0:
            print(f"  official {i + 1}/{len(st)}  {(i + 1) / (time.perf_counter() - t0):.1f} f/s",
                  flush=True)
    m.cam_pos[cid] = base_pos
    m.cam_quat[cid] = base_quat
    m.light_diffuse[:] = base_light
    return out


#: The leaves of HumanoidBench's variation space that the OFFICIAL renderer can be given here.
#:
#: ``camera.*`` and ``light.diffuse`` are ordinary ``MjModel`` fields with the same meaning on both
#: sides -- ``cam_poscom0`` / ``cam_mat0`` for a ``TRACKCOM`` camera and ``light_diffuse`` -- so the
#: env's own camera arithmetic can be written straight into the reference model and the default
#: round-trips BIT-EXACTLY (asserted below).
#:
#: ``floor.color`` and ``robot.color`` are NOT here and that is deliberate. On the MJWarp side they
#: are a per-world ``mat_rgba`` on a rebuilt one-material-per-geom table whose floor entry has had its
#: TEXTURE REMOVED and replaced by that texture's dominant texel (``humanoidbench._build_bundle``).
#: The reference floor still wears the checker, so "the same colour" is not the same write on the two
#: sides: setting the reference's ``geom_rgba`` to the style colour multiplies MuJoCo's texture by it
#: and the default stops reproducing itself. Guessing a ratio or a delta would produce a number about
#: this harness rather than about the renderer, which is the second bug §8 of the README records. The
#: two colour knobs are therefore held at their defaults and the split says so in its metadata.
HB_STYLE_LEAVES = ("camera.azimuth_delta", "camera.distance_scale", "camera.elevation_delta",
                   "light.diffuse")


def _hb_reference_style(cls, ref, env, style):
    """Write a batch of styles into the reference ``MjModel``, and check the default is a no-op.

    Returns ``(apply(i), restore())``. ``apply`` writes row ``i``'s camera and light into the model;
    ``restore`` puts the pristine values back, because ``reference_env`` deliberately never closes an
    env and a later caller would otherwise inherit the last row's appearance.
    """
    import mujoco
    import torch
    u = ref.unwrapped
    m = u.model
    b = env._b
    cam = int(b.info["cam"])
    mode = int(m.cam_mode[cam])
    if mode != int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM):
        raise SystemExit(f"{cls.__name__}: camera mode {mode} is not TRACKCOM; the env's camera "
                         f"arithmetic writes cam_poscom0/cam_mat0 and would not reach the frame")
    base_pos = np.array(m.cam_poscom0[cam], copy=True)
    base_mat = np.array(m.cam_mat0[cam], copy=True)
    base_light = np.array(m.light_diffuse, copy=True)
    sty = env.resolve_style(np.asarray(style, np.float32), None, batch=len(style))
    campos, cammat = env._camera(sty, 0, len(style))
    campos = campos.detach().cpu().numpy().astype(np.float64)
    cammat = cammat.detach().cpu().numpy().astype(np.float64).reshape(len(style), 9)
    light = sty["light.diffuse"].detach().cpu().numpy().reshape(-1, 1, 1).astype(np.float64)
    dflt = np.asarray(env.default_style(), np.float32)

    def apply(i):
        m.cam_poscom0[cam] = campos[i]
        m.cam_mat0[cam] = cammat[i]
        m.light_diffuse[:] = base_light[None] * light[i]

    def restore():
        m.cam_poscom0[cam] = base_pos
        m.cam_mat0[cam] = base_mat
        m.light_diffuse[:] = base_light

    # THE GUARD. A default row must leave the model where it was, or every frame this function
    # produces is a frame of a scene the corrector's input is not a render of.
    #
    # The tolerance is FLOAT32 EPSILON and not zero, and the reason is not slack. ``_camera`` returns
    # torch float32 -- those are the exact numbers MJWarp is given -- so writing them into a float64
    # ``MjModel`` field reproduces the default to ~3e-8 rather than to the bit. Rounding the
    # REFERENCE to the same float32 the cheap render already uses moves the two renderers together,
    # not apart, which is why the check is that the residual is at float32 resolution and not that it
    # is absent. Anything larger is a real disagreement about where the camera is.
    j = int(np.argmax(np.abs(np.asarray(style, np.float32) - dflt[None]).max(1) == 0))
    if float(np.abs(np.asarray(style[j], np.float32) - dflt).max()) == 0.0:
        apply(j)
        err = max(float(np.abs(m.cam_poscom0[cam] - base_pos).max()),
                  float(np.abs(m.cam_mat0[cam] - base_mat).max()),
                  float(np.abs(np.asarray(m.light_diffuse) - base_light).max()))
        restore()
        if err > 1e-6:
            raise SystemExit(f"{cls.__name__}: the DEFAULT style moved the reference model by "
                             f"{err:.3e}; every styled frame would be of a different scene than the "
                             f"cheap render it is paired with")
        print(f"  default-style round-trip on the reference model: {err:.2e} (float32 resolution), "
              f"row {j}", flush=True)
    del torch
    return apply, restore


def reference_frames_hb(cls, states, style=None, env=None, size=256, log_every=512):
    ref = cls.reference_env(int(size))
    out = np.empty((len(states), int(size), int(size), 3), np.uint8)
    t0 = time.perf_counter()
    if style is None:
        step = 256
        for lo in range(0, len(states), step):
            hi = min(lo + step, len(states))
            out[lo:hi] = cls.reference_render(ref, states[lo:hi])
            if log_every:
                print(f"  official {hi}/{len(states)}  {hi / (time.perf_counter() - t0):.1f} f/s",
                      flush=True)
        return out

    import mujoco
    apply, restore = _hb_reference_style(cls, ref, env, style)
    u = ref.unwrapped
    m, d = u.model, u.data
    nq = int(m.nq)
    try:
        for i, st in enumerate(np.asarray(states, np.float64)):
            apply(i)
            d.qpos[:] = st[:nq]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            err = float(np.abs(np.asarray(d.qpos) - st[:nq]).max())
            if err > 1e-5:
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state (max "
                    f"{err:.3e}); every number computed from it would compare two different scenes")
            u.mujoco_renderer.viewer.make_context_current()
            out[i] = np.ascontiguousarray(u.render())
            if log_every and (i + 1) % log_every == 0:
                print(f"  official {i + 1}/{len(states)}  "
                      f"{(i + 1) / (time.perf_counter() - t0):.1f} f/s", flush=True)
    finally:
        restore()
    return out


# ----------------------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--split", required=True, choices=("train", "heldout", "decl"))
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--render-size", type=int, default=112)
    ap.add_argument("--tex-lod", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--style", action="store_true",
                    help="draw the env's declared appearance variation per sample")
    ap.add_argument("--frac-default", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk", type=int, default=512, help="states per MJWarp render call")
    a = ap.parse_args()

    import stable_envs as se
    cls = se.REGISTRY[a.env]
    mod = env_module(a.env)
    family = family_of(a.env)
    lod = a.tex_lod
    ov = dict(enable_backface_culling=False)

    tag = a.tag or f"r{a.render_size}_tex{lod}" + ("_sty" if a.style else "")
    outdir = os.path.join(ROOT, a.env, f"{a.split}_{tag}")
    os.makedirs(outdir, exist_ok=True)

    # ---- states -----------------------------------------------------------------------------
    if a.split == "decl":
        allst = []
        for k in range(4):
            p = se.make(a.env, batch=1, size=256, device="cuda", seed=k)
            allst.append(p.sample_states(64))
            del p
        states = np.concatenate(allst).astype(np.float32)
        ctx_cache(mod).clear()
        torch.cuda.empty_cache()
    elif family == "RoboCasa":
        states = robocasa_states(cls, a.split, a.n, a.seed + 1000 * (a.split == "heldout"))
    else:
        states = humanoidbench_states(cls, a.split, a.n, a.seed + 1000 * (a.split == "heldout"))
    lo = len(states) * a.shard // a.nshards
    hi = len(states) * (a.shard + 1) // a.nshards
    states = states[lo:hi]
    print(f"{a.env} {a.split} shard {a.shard}/{a.nshards}: {len(states)} states, "
          f"render {a.render_size} px tex_lod {lod} style={a.style}", flush=True)

    # ---- style ------------------------------------------------------------------------------
    style = None
    probe = se.make(a.env, batch=1, size=256, device="cuda", seed=a.seed)
    only = None if family == "RoboCasa" else HB_STYLE_LEAVES
    all_leaves = [p for p, _, _ in probe.variation_space.flat_layout()]
    if a.style:
        style = sample_style(probe, len(states), a.seed + 7 + 100 * a.shard, a.frac_default, only)

    # ---- official frames (the target) --------------------------------------------------------
    t0 = time.perf_counter()
    if family == "RoboCasa":
        hr = reference_frames_robocasa(cls, probe, states, style)
    else:
        hr = reference_frames_hb(cls, states, style, probe)
    t_off = time.perf_counter() - t0
    print(f"official frames: {len(hr)} in {t_off:.1f}s ({len(hr) / t_off:.1f} f/s)", flush=True)
    del probe
    ctx_cache(mod).clear()
    torch.cuda.empty_cache()

    # ---- cheap render (the input) ------------------------------------------------------------
    orig = patched(mod, ov, lod)
    try:
        env = se.make(a.env, batch=1, size=256, device="cuda", seed=a.seed)
        lr = np.empty((len(states), a.render_size, a.render_size, 3), np.uint8)
        t0 = time.perf_counter()
        for i in range(0, len(states), a.chunk):
            j = min(i + a.chunk, len(states))
            sty = None if style is None else style[i:j]
            lr[i:j] = env.render(states[i:j], size=a.render_size, style=sty).cpu().numpy()
        print(f"cheap render: {len(lr)} in {time.perf_counter() - t0:.1f}s", flush=True)
        del env
    finally:
        mod.MW.create_render_context = orig
        ctx_cache(mod).clear()
        torch.cuda.empty_cache()

    np.save(os.path.join(outdir, f"states_{a.shard}.npy"), states)
    np.save(os.path.join(outdir, f"lr_{a.shard}.npy"), lr)
    np.save(os.path.join(outdir, f"hr_{a.shard}.npy"), hr)
    if style is not None:
        np.save(os.path.join(outdir, f"style_{a.shard}.npy"), style)
    meta = {"env": a.env, "split": a.split, "n": len(states), "render_size": a.render_size,
            "tex_lod": lod, "overrides": {k: str(v) for k, v in ov.items()}, "style": a.style,
            "style_leaves": (None if not a.style else (list(only) if only is not None
                                                       else all_leaves)),
            "frac_default": a.frac_default if a.style else None,
            "shard": a.shard, "nshards": a.nshards, "seed": a.seed,
            "split_rule": ("RoboCasa: recorded states 0..149 train / 150..199 heldout, yaw redrawn "
                           "per draw; 30%% of training draws interpolate two training rows"
                           if family == "RoboCasa" else
                           "HumanoidBench: pool rows 0..383 train / 384..511 heldout, 50%% of "
                           "training draws interpolate two training rows")}
    with open(os.path.join(outdir, f"meta_{a.shard}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", outdir, flush=True)


if __name__ == "__main__":
    main()
