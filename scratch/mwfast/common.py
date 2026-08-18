"""Shared pieces for the Meta-World opt-in fast path: states, official frames, metrics.

Meta-World is a PARITY family and its frame is a COMPOSITE, which changes what a "cheap render"
even means here. Every frame is

    MuJoCo's own raster of the episode's static scene (the plate, ~64 % of the pixels, exact by
    construction) + mujoco_warp's ray-cast movers, selected by the segmentation buffer,

so lowering the internal resolution lowers BOTH halves at once: the plate is baked by
``mujoco.Renderer`` at ``res`` and the ray-cast runs at ``res``. That is why the lever is applied
through ``env.render(states, size=r)`` and nothing in ``stable_envs/envs/`` is touched -- the env
already takes the resolution as an argument and already bakes a plate per (episode, appearance,
RESOLUTION).

THE SPLIT. Meta-World's own state pool is 64 episodes x 4 states, drawn with ``_seeded_reset`` at
seeds 10_000..10_063, and ``sample_states`` draws rows out of it. An episode is the unit that
matters: Meta-World writes the episode's placement into ``model.body_pos``/``model.site_pos``, so two
states of one episode share a plate and share every static pixel. Splitting rows would therefore put
the same scene on both sides of the split. So the split here is on the SEED, and the training seeds,
the held-out seeds and the shipped pool's seeds are three disjoint ranges:

    train    seeds 20_000 +          held out   seeds 30_000 +          shipped pool 10_000..10_063

which makes ``decl`` -- the draw a declaration is measured on, i.e. ``sample_states`` -- a THIRD
disjoint set rather than an in-distribution one, unlike the RoboCasa study where ``decl`` overlapped
training by construction.
"""
from __future__ import annotations

import io
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch
import torch.nn.functional as F

LOGS = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/mwfast"
DATA = "/mnt/vast/home/randall.balestriero/data/mwfast"

#: Seed bases for the three disjoint episode sets. 10_000 is the shipped pool's (``_state_pool``).
SEED_TRAIN, SEED_HELD = 20_000, 30_000


def mw():
    import stable_envs.envs.metaworld as M
    return M


def extend_zoo():
    """Register the (architecture, input resolution) pairs this study needs, AT RUNTIME.

    ``rendernet/nets.py`` is another agent's file and is not edited here. Its ``ZOO`` is a plain
    dict, so the entries Meta-World needs and RoboCasa did not -- chiefly the small UNet at a 96 px
    input, which is the only architecture that fits the budget of the family's slowest ordinary tasks
    -- are added to it from here. Same tuple shape: ``(class, trunk size, width, depth, unshuffle,
    render size)``.
    """
    import nets
    extra = {
        "unet_96t128_w48_d2_u2":   (nets.UNetTiny, 128, 48, 2, 2, 96),
        "unet_96t128_w32_d2_u2":   (nets.UNetTiny, 128, 32, 2, 2, 96),
        "unet_128_w48_d2_u2b":     (nets.UNetTiny, 128, 48, 2, 2, 128),
        "espcn_96t128_w48_d4_u2":  (nets.ESPCN, 128, 48, 4, 2, 96),
        "unet_112t128_w32_d2_u2":  (nets.UNetTiny, 128, 32, 2, 2, 112),
    }
    for k, v in extra.items():
        nets.ZOO.setdefault(k, v)
    return nets


def task_of(env_name: str) -> str:
    import stable_envs as se
    return se.REGISTRY[env_name].TASK


def episode_states(task: str, n_episodes: int, seed_base: int, per_episode: int = 4,
                   rng_seed: int = 0) -> np.ndarray:
    """``(n_episodes * per_episode, state_dim)`` from the OFFICIAL env's own reset distribution.

    A copy of ``metaworld._state_pool``'s generative process with the seed base and the episode count
    made arguments -- same ``_seeded_reset`` (the only route that reproduces a draw on 50/50 tasks),
    same 0-25 random actions per state, same random yaw on every free joint. It is a copy rather than
    a call because ``_state_pool`` hard-codes ``10_000 + ep`` and 64 episodes, and the whole point of
    this module is that the training episodes and the shipped pool's episodes must not intersect.
    """
    import mujoco
    M = mw()
    b = M._build_bundle(task)
    env, u, m, d = b.env, b.env.unwrapped, b.model, b.data
    free = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    rng = np.random.default_rng(rng_seed)
    dim = b.nq + len(b.slots)
    out = np.empty((n_episodes * per_episode, dim), np.float32)
    i = 0
    for ep in range(n_episodes):
        M._seeded_reset(env, seed_base + ep)
        slice_vals = np.array([float(getattr(m, a).reshape(-1)[f]) for a, f in b.slots])
        for k in range(per_episode):
            for _ in range(int(rng.integers(0, 26)) if k else 0):
                env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))
            q = np.array(d.qpos, np.float64)
            for adr in free:
                yaw = float(rng.uniform(-np.pi, np.pi))
                q[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            out[i, :b.nq] = q
            out[i, b.nq:] = slice_vals
            i += 1
    for a in M.MODEL_ARRAYS:                        # leave the shared model as it was found
        getattr(m, a)[:] = b.pristine[a]
    return out


def episode_index(n_episodes: int, per_episode: int = 4) -> np.ndarray:
    """Which episode each row of ``episode_states`` came from. Needed to group by plate."""
    return np.repeat(np.arange(n_episodes), per_episode)


# ------------------------------------------------------------------------------ official renderer

def official_frames(cls, states, size: int = 256, style=None, probe=None, log_every: int = 512):
    """``mujoco.Renderer`` frames at ``size``, optionally at a non-default appearance.

    Without a style this is ``cls.reference_render`` verbatim (it is called). With one, the style has
    to be written into the REFERENCE model, because ``reference_render`` renders the model's own
    camera, colours and light -- which is right for a declaration and wrong for a corrector trained
    under the env's declared variation. The writes are exactly the ones ``MetaWorldBase._bake``
    makes, in the same order, and the arithmetic is the env's OWN ``_camera``: the two renderers are
    then given the same camera by construction rather than by two implementations agreeing.

    The default row is asserted to be a no-op on the reference model, because a style path that
    moves the model at its own default is a path where every frame is of a different scene than the
    cheap render it is paired with.
    """
    import mujoco
    M = mw()
    ref = cls.reference_env(int(size))
    if style is None:
        out = np.empty((len(states), int(size), int(size), 3), np.uint8)
        step = 256
        for lo in range(0, len(states), step):
            hi = min(lo + step, len(states))
            out[lo:hi] = cls.reference_render(ref, states[lo:hi])
        return out

    b = M._build_bundle(cls.TASK)
    u = ref.unwrapped
    m, d = u.model, u.data
    dflt = np.asarray(probe.default_style(), np.float32)
    sty = np.asarray(style, np.float32)
    paths = probe._paths()
    layout = {p: probe.variation_space.flat_layout() for p in ()}          # unused, kept explicit
    del layout

    def over_of(row):
        o = {}
        off = 0
        for p, a, w in probe.variation_space.flat_layout():
            o[p] = np.asarray(row[a:a + w], np.float64)
            off += w
        return o

    light0 = np.array(b.light0, np.float64)
    base_cam_pos = np.array(m.cam_pos[b.cam], copy=True)
    base_cam_quat = np.array(m.cam_quat[b.cam], copy=True)
    base_light = np.array(m.light_diffuse, copy=True)

    def apply(over):
        for name, k in b.knobs.items():
            col = probe._var(over, f"{name}.color")
            m.geom_rgba[k["geoms"], :3] = col
            if len(k["mats"]):
                m.mat_rgba[k["mats"], :3] = col
        m.light_diffuse[:] = light0 * float(probe._var(over, "light.diffuse")[0])
        pos, quat = probe._camera(over)
        m.cam_pos[b.cam] = pos
        m.cam_quat[b.cam] = quat

    # THE GUARD: a default row must leave the camera and the light where they were.
    j = int(np.argmax((np.abs(sty - dflt[None]).max(1) == 0)))
    if float(np.abs(sty[j] - dflt).max()) == 0.0:
        M._apply_state(b, m, d, np.asarray(states[j], np.float64))
        apply(over_of(sty[j]))
        err = max(float(np.abs(m.cam_pos[b.cam] - base_cam_pos).max()),
                  float(np.abs(m.cam_quat[b.cam] - base_cam_quat).max()),
                  float(np.abs(np.asarray(m.light_diffuse) - base_light).max()))
        if err > 1e-9:
            raise SystemExit(f"{cls.__name__}: the DEFAULT style moved the reference model by {err:.3e}")
        print(f"  default-style round-trip on the reference model: {err:.2e} (row {j})", flush=True)

    out = np.empty((len(states), int(size), int(size), 3), np.uint8)
    for i, st in enumerate(np.asarray(states, np.float64)):
        M._apply_state(b, m, d, st, forward=False)       # restores the pristine 14 arrays, writes st
        apply(over_of(sty[i]))
        mujoco.mj_forward(m, d)
        got = M._read_state(b, m, d)
        err = float(np.abs(got - st).max())
        if err > 1e-5:
            raise AssertionError(f"{cls.__name__}: reference did not reach the state ({err:.3e})")
        out[i] = np.ascontiguousarray(u.render())
        if log_every and (i + 1) % log_every == 0:
            print(f"  official {i + 1}/{len(states)}", flush=True)
    for a in M.MODEL_ARRAYS:
        getattr(m, a)[:] = b.pristine[a]
    m.cam_quat[b.cam] = base_cam_quat
    m.light_diffuse[:] = base_light
    return out


# ------------------------------------------------------------------------------------ upsampling

def upsample(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """``(B,r,r,3)`` uint8 -> ``(B,size,size,3)`` uint8, a no-op when already at ``size``."""
    if x.shape[1] == size:
        return x
    y = x.permute(0, 3, 1, 2).to(torch.float32)
    kw = {} if mode == "nearest" else {"align_corners": False}
    y = F.interpolate(y, size=(size, size), mode=mode, **kw)
    return y.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


# -------------------------------------------------------------------------------------- metrics

#: Frames per SSIM block. ``conformance.ssim`` runs a 3-D uniform filter over the whole (N,H,W)
#: stack, which smooths along the FRAME axis, so SSIM on 512 frames is a different quantity from
#: SSIM on the 64 every declaration in this repo is measured at. Everything here is blocked at 64.
SSIM_BLOCK = 64


def metrics(ours: np.ndarray, official: np.ndarray, block: int = SSIM_BLOCK) -> dict:
    """The four declared quantities plus the error tail, averaged over blocks of ``block`` frames."""
    from stable_envs.testing.conformance import pixel_agreement, ssim
    keys = ("mean_abs", "within8_frac", "exact_frac", "interior_exact_frac", "interior_mean_abs",
            "interior_frac")
    acc: dict = {k: [] for k in keys}
    acc["ssim"] = []
    mx, tail = 0, []
    for lo in range(0, len(ours), block):
        hi = min(lo + block, len(ours))
        pa = pixel_agreement(ours[lo:hi], official[lo:hi])
        for k in keys:
            v = pa.get(k)
            if v is not None:
                acc[k].append(float(v))
        acc["ssim"].append(ssim(ours[lo:hi], official[lo:hi]))
        mx = max(mx, int(pa["max_abs"]))
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    out = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
    out["max_abs"] = mx
    out["p999"] = float(np.percentile(d, 99.9))
    out["p9999"] = float(np.percentile(d, 99.99))
    out["n"] = int(len(ours))
    return out


def jpeg_roundtrip(frames: np.ndarray, quality: int = 95):
    """The repo's own write and read paths: PIL encode, torchvision decode."""
    from PIL import Image
    from torchvision.io import ImageReadMode, decode_jpeg
    blobs = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=quality)
        blobs.append(buf.getvalue())
    ts = [torch.frombuffer(b, dtype=torch.uint8) for b in blobs]
    out = torch.stack(decode_jpeg(ts, mode=ImageReadMode.RGB))
    return out.permute(0, 2, 3, 1).contiguous().numpy(), sum(len(b) for b in blobs) / len(blobs)


def clear_caches():
    M = mw()
    M._CTX_CACHE.clear()
    M._PLATE_CACHE.clear()
    torch.cuda.empty_cache()
