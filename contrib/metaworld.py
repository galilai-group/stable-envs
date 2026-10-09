from __future__ import annotations

import numpy as np
import torch

from stable_envs._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

# Check optional dependencies without importing the simulator during discovery.
import importlib.util as _ilu

import mujoco as _mj

_VENV = "/mnt/vast/home/randall.balestriero/venvs/se-fetch"
_missing = [_m for _m in ("metaworld", "warp", "mujoco_warp") if _ilu.find_spec(_m) is None]
_mjver = tuple(int(x) for x in str(_mj.__version__).split(".")[:2])
if _missing or _mjver < (3, 11):  # pragma: no cover
    raise ImportError(
        "the Meta-World family ray-casts the Sawyer's meshes with mujoco_warp's batch raytracer, "
        "which this interpreter cannot run: "
        + (f"missing {', '.join(_missing)}; " if _missing else "")
        + f"mujoco is {_mj.__version__} and mujoco_warp requires >= 3.11 (metaworld itself pins "
        f"mujoco==3.3.0, which is why it is installed with --no-deps). The envs were measured in "
        f"{_VENV}; see contrib/requirements/metaworld.txt"
    )

from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.render import _resources as _RES
from stable_envs.render import mjwarp as MW
from stable_envs.spaces import Box, Dict

# Validate this task table against the source package when constructing a bundle.
TASKS = (
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "hand-insert-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "plate-slide-side-v3",
    "plate-slide-v3",
    "push-back-v3",
    "push-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3",
)

#: The model camera every task ships, ``mode="fixed"``, from ``xyz_base.xml``.
CAMERA = "corner"
# Reset-varying, pixel-visible elements of these arrays can enter the render state.
MODEL_ARRAYS = (
    "body_pos",
    "body_quat",
    "site_pos",
    "site_quat",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rgba",
    "site_rgba",
    "mat_rgba",
    "light_pos",
    "cam_pos",
    "geom_matid",
    "site_matid",
)
# Model fields that MJWarp supports per world; sites are handled by the static bake.
BATCHABLE = (
    "body_pos",
    "body_quat",
    "site_pos",
    "site_quat",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rgba",
    "mat_rgba",
    "light_pos",
    "cam_pos",
    "geom_matid",
)
# Always batch episode placement and fields changed by appearance controls.
ALWAYS_BATCHED = ("body_pos", "body_quat", "geom_rgba", "mat_rgba", "light_diffuse", "cam_pos", "cam_quat")

HIDDEN_GROUP = MW.HIDDEN_GROUP
VISIBLE_GROUPS = MW.DEFAULT_GROUPS
# Bound per-context image buffers and amortize per-chunk synchronization.
MJW_CHUNK = 256
N_DRAWS = 6
# Keep multiple poses per episode for plate reuse and constancy checks.
POOL_EPISODES, POOL_PER_EPISODE = 64, 4

_MODEL_CACHE: dict = {}
_RASTER_CACHE: dict = {}
_CTX_CACHE = _RES.BoundedCache(3, name="metaworld.ctx", on_evict=MW.release_cached_ctx)
_CTX_CACHE_MAX = 3
_PLATE_CACHE_MAX = 96
_PLATE_CACHE = _RES.BoundedCache(_PLATE_CACHE_MAX, name="metaworld.plate")
_POOL_CACHE: dict = {}


def _make_task_env(task: str, size: int | None = None):
    """Construct an observable-goal task with unfrozen reset placement.

    Only supplying a size requests rgb_array rendering.
    """
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS

    kw: dict = {"camera_name": CAMERA}
    if size is not None:
        kw.update(render_mode="rgb_array", width=int(size), height=int(size))
    env = ALL_V3_ENVIRONMENTS[task](**kw)
    u = env.unwrapped
    u._partially_observable = False
    u._freeze_rand_vec = False
    u._set_task_called = True
    return env


def _seeded_reset(env, seed: int) -> None:
    """Seed the task-local placement generator before reset_model can consume it.

    The source reset path draws twice; preserve that ordering.
    """
    u = env.unwrapped
    u.seeded_rand_vec = True
    u._np_random = np.random.default_rng(int(seed))
    u._np_random_seed = int(seed)
    env.reset()


def _raster(task: str, res: int, m):
    """Reuse a process-lifetime MuJoCo renderer for this model and resolution."""
    import mujoco

    # A renderer binds to one MjModel; task and resolution alone do not identify that model.
    key = (task, int(res), id(m))
    hit = _RASTER_CACHE.get(key)
    if hit is None:
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(res))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(res))
        hit = _RASTER_CACHE[key] = mujoco.Renderer(m, width=int(res), height=int(res))
    return hit


class _Bundle:
    __slots__ = (
        "task",
        "env",
        "model",
        "data",
        "cam",
        "nq",
        "qpos0",
        "pristine",
        "slots",
        "state_names",
        "movers",
        "mover_sites",
        "static_geoms",
        "plate_static",
        "casting",
        "knobs",
        "pivot",
        "cam_pos0",
        "cam_quat0",
        "light0",
        "batched",
        "info",
        "mat_tex_mean",
        "alpha",
        "translucent",
    )


def _effective_rgb(m, g) -> np.ndarray:
    """Resolve geom color, falling back to the material for default geom rgba."""
    rgba = np.asarray(m.geom_rgba[g], np.float64)
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[mid], np.float64)[:3]
    return rgba[:3]


def _subtree(m, root: int) -> set[int]:
    out = {int(root)}
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x in out:
                out.add(b)
                break
            x = int(m.body_parentid[x])
    return out


_COLS = {"pos": "xyz", "quat": "wxyz", "size": "xyz", "rgba": "rgba", "matid": ("",)}


def _slot_name(m, arr: str, f: int) -> str:
    import mujoco

    kind, field = arr.split("_", 1)
    a = np.asarray(getattr(m, arr))
    w = int(a.reshape(a.shape[0], -1).shape[1])
    row, col = divmod(int(f), w)
    obj = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "mat": mujoco.mjtObj.mjOBJ_MATERIAL,
        "light": mujoco.mjtObj.mjOBJ_LIGHT,
        "cam": mujoco.mjtObj.mjOBJ_CAMERA,
    }[kind]
    nm = mujoco.mj_id2name(m, obj, row) or f"{kind}{row}"
    labels = _COLS.get(field, "")
    tag = labels[col] if col < len(labels) and labels[col] else str(col)
    return f"{arr}.{nm}.{tag}" if tag else f"{arr}.{nm}"


def _build_bundle(task: str) -> _Bundle:
    """Discover episode-dependent model fields, moving geometry, and color controls."""
    import mujoco

    hit = _MODEL_CACHE.get(task)
    if hit is not None:
        return hit

    from metaworld.env_dict import ALL_V3_ENVIRONMENTS

    if set(ALL_V3_ENVIRONMENTS) != set(TASKS):
        extra = sorted(set(ALL_V3_ENVIRONMENTS) - set(TASKS))
        gone = sorted(set(TASKS) - set(ALL_V3_ENVIRONMENTS))
        raise RuntimeError(
            f"metaworld's ALL_V3_ENVIRONMENTS no longer matches this module's TASKS table "
            f"(added {extra}, removed {gone}); the registry would silently be measuring a different "
            f"benchmark from the one that was measured"
        )

    env = _make_task_env(task)
    u = env.unwrapped
    m, d = u.model, u.data
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    if cam < 0:
        raise RuntimeError(
            f"{task}: no model camera {CAMERA!r}; the free camera is history-dependent and is not a reference"
        )

    b = _Bundle()
    b.task, b.env, b.model, b.data, b.cam = task, env, m, d, cam
    b.nq = int(m.nq)

    draws = []
    for k in range(N_DRAWS):
        _seeded_reset(env, k)
        mujoco.mj_forward(m, d)  # Recompute derived transforms after reset changes model fields.
        draws.append(
            {a: np.array(getattr(m, a), copy=True) for a in MODEL_ARRAYS} | {"qpos": np.array(d.qpos, copy=True)}
        )
    b.pristine = {a: np.array(draws[-1][a], copy=True) for a in MODEL_ARRAYS}
    b.qpos0 = np.array(draws[-1]["qpos"], np.float64)

    cand = []
    for a in MODEL_ARRAYS:
        stack = np.stack([dr[a].reshape(-1) for dr in draws])
        for f in np.flatnonzero((stack != stack[0]).any(0)):
            cand.append((a, int(f)))

    b.slots = _live_slots(task, b, draws, cand)

    weld = np.asarray(m.body_weldid, np.int64)
    mocapid = np.asarray(m.body_mocapid, np.int64)
    root = np.asarray(m.body_rootid, np.int64)
    gbody = np.asarray(m.geom_bodyid, np.int64)
    is_mover = (weld[gbody] != 0) | (mocapid[gbody] >= 0)
    b.movers = np.flatnonzero(is_mover).astype(np.int64)
    sbody = np.asarray(m.site_bodyid, np.int64)
    b.mover_sites = np.flatnonzero((weld[sbody] != 0) | (mocapid[sbody] >= 0)).astype(np.int64)
    # MJWarp ignores alpha; translucent movers need a separate compositing pass.
    b.alpha = MW.effective_alpha(m, "geom")
    b.translucent = np.array([g for g in b.movers if 0.0 < float(b.alpha[g]) < 1.0], np.int64)
    b.static_geoms = np.flatnonzero(~is_mover).astype(np.int64)
    # Upload transforms for welded geoms that MJWarp skips in local-to-global conversion.
    b.plate_static = np.array(
        [g for g in range(m.ngeom) if weld[gbody[g]] == 0 and mocapid[root[gbody[g]]] == -1], np.int64
    )

    names = list(_qpos_names(m))
    if len(names) != b.nq:
        raise RuntimeError(f"{task}: named {len(names)} qpos entries for nq={b.nq}")
    names += [_slot_name(m, a, f) for a, f in b.slots]
    if len(set(names)) != len(names):
        raise RuntimeError(f"{task}: duplicate state names in {names}")
    b.state_names = tuple(names)

    b.cam_pos0 = np.array(m.cam_pos[cam], np.float64)
    b.cam_quat0 = np.array(m.cam_quat[cam], np.float64)
    b.light0 = np.array(m.light_diffuse, np.float64)
    b.casting = np.flatnonzero(np.asarray(m.light_castshadow, np.int64) != 0).astype(np.int64)
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tablelink")
    b.pivot = np.array(m.body_pos[tb], np.float64) if tb >= 0 else np.array([0.0, 0.6, 0.0], np.float64)
    b.knobs = _colour_knobs(task, b)
    b.mat_tex_mean = _mover_texture_means(m, b)

    slot_arrays = {a for a, _ in b.slots}
    unbatchable = {a for a in slot_arrays if a not in BATCHABLE}
    for a in sorted(unbatchable):
        bad = [f for x, f in b.slots if x == a and _slot_touches_a_mover(m, a, f)]
        if bad:
            raise RuntimeError(
                f"{task}: the episode writes {a}[{bad}] on MOVING geometry, and mujoco_warp has no "
                f"per-world {a}; the ray-cast pass would draw every episode with the first one's value"
            )
    b.batched = tuple(sorted(set(ALWAYS_BATCHED) | (slot_arrays & set(BATCHABLE))))
    # Record excluded candidates: pinning both renderers can hide a mistaken exclusion.
    b.info = {
        "n_movers": int(len(b.movers)),
        "n_static": int(len(b.static_geoms)),
        "n_slots": len(b.slots),
        "candidates": [f"{a}[{f}]" for a, f in cand],
        "dropped_as_unseen": [f"{a}[{f}]" for a, f in cand if (a, f) not in set(b.slots)],
    }
    _MODEL_CACHE[task] = b
    return b


def _qpos_names(m) -> list[str]:
    import mujoco

    out: list[str] = []
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        t = int(m.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            out += [f"{nm}.{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        elif t == int(mujoco.mjtJoint.mjJNT_BALL):
            out += [f"{nm}.{c}" for c in ("qw", "qx", "qy", "qz")]
        else:
            out.append(nm)
    return out


def _slot_touches_a_mover(m, arr: str, f: int) -> bool:
    weld = np.asarray(m.body_weldid, np.int64)
    gbody = np.asarray(m.geom_bodyid, np.int64)
    kind = arr.split("_")[0]
    w = int(np.asarray(getattr(m, arr)).reshape(np.asarray(getattr(m, arr)).shape[0], -1).shape[1])
    row = f // w
    if kind == "site":
        return False  # MJWarp draws no sites at all
    if kind == "geom":
        return bool(weld[gbody[row]] != 0)
    if kind == "body":
        return any(weld[bb] != 0 for bb in _subtree(m, row))
    if kind == "mat":
        return any(weld[gbody[g]] != 0 for g in range(m.ngeom) if int(m.geom_matid[g]) == row)
    return True  # lights and cameras reach every mover pixel


def _live_slots(task: str, b: _Bundle, draws, cand) -> tuple[tuple[str, int], ...]:
    """Select reset-varying fields whose individual substitution changes source pixels.

    Fields excluded from state must be pinned to the pristine snapshot in both
    the plate bake and source renderer; otherwise rendering depends on history.
    """
    import mujoco

    if not cand:
        return ()
    m, d = b.model, b.data
    r = _raster(task, MetaWorldBase.default_size, m)
    base_q = np.asarray(draws[0]["qpos"], np.float64)

    def shot() -> np.ndarray:
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=b.cam)
        return r.render()

    def load(dr) -> None:
        for a in MODEL_ARRAYS:
            getattr(m, a)[:] = dr[a]
        d.qpos[:] = base_q
        d.qvel[:] = 0.0

    load(draws[0])
    ref = shot().copy()
    live = []
    for a, f in cand:
        arr = getattr(m, a)
        keep = arr.reshape(-1)[f]
        for dr in draws[1:]:
            v = dr[a].reshape(-1)[f]
            if v == keep:
                continue
            arr.reshape(-1)[f] = v
            moved = not np.array_equal(shot(), ref)
            arr.reshape(-1)[f] = keep
            if moved:
                live.append((a, f))
                break
    for a in MODEL_ARRAYS:  # leave the model on the snapshot
        getattr(m, a)[:] = b.pristine[a]
    return tuple(live)


#: Whether a MOVER whose texture MJWarp cannot show is drawn in ``mat_rgba`` alone or in
#: ``mat_rgba x`` the texture's mean. See ``_mover_texture_means``; measured, not chosen.
MOVER_TEXTURE_MEAN = True


def _mover_texture_means(m, b: _Bundle) -> np.ndarray:
    """Return per-material texture means for the flat mover approximation.

    Multiply source material colors by these means when textures cannot be
    sampled. Static geometry keeps its source textures in the baked plate.
    """
    out = np.ones((int(m.nmat), 3), np.float64)
    if not MOVER_TEXTURE_MEAN:
        return out
    for mid in sorted({int(m.geom_matid[g]) for g in b.movers if int(m.geom_matid[g]) >= 0}):
        t = m.mat_texid[mid]
        tid = int(np.max(t)) if np.ndim(t) else int(t)
        if tid >= 0:
            out[mid] = MW.texture_mean_rgb(m, tid)
    return out


def _colour_knobs(task: str, b: _Bundle) -> dict:
    """Select a visible effective-color group for each appearance control.

    Exclude materials shared outside the group, so recoloring cannot affect
    unrelated geometry.
    """
    import mujoco

    m, d = b.model, b.data
    for a in MODEL_ARRAYS:
        getattr(m, a)[:] = b.pristine[a]
    d.qpos[:] = b.qpos0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    r = _raster(task, MetaWorldBase.default_size, m)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=b.cam)
    seg = r.render().copy()
    r.disable_segmentation_rendering()
    ids = np.where(seg[..., 1].astype(np.int64) == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0].astype(np.int64), -1)
    px = np.bincount(ids[ids >= 0].reshape(-1), minlength=m.ngeom)

    weld = np.asarray(m.body_weldid, np.int64)
    gbody = np.asarray(m.geom_bodyid, np.int64)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    robot_bodies = _subtree(m, base) if base >= 0 else set()
    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in VISIBLE_GROUPS]
    groups = {
        "robot": [g for g in vis if int(gbody[g]) in robot_bodies],
        "object": [g for g in vis if weld[gbody[g]] != 0 and int(gbody[g]) not in robot_bodies],
    }

    wearers: dict[int, list[int]] = {}
    for g in range(m.ngeom):
        if int(m.geom_matid[g]) >= 0:
            wearers.setdefault(int(m.geom_matid[g]), []).append(g)

    knobs: dict = {}
    for name, gs in groups.items():
        if not gs:
            continue
        classes: dict[tuple, list[int]] = {}
        for g in gs:
            classes.setdefault(tuple(np.round(_effective_rgb(m, g), 6)), []).append(g)
        col, members = max(classes.items(), key=lambda kv: (int(px[kv[1]].sum()), len(kv[1])))
        geoms, mats = [], []
        for g in members:
            mid = int(m.geom_matid[g])
            if mid < 0:
                geoms.append(g)
            elif all(x in members for x in wearers[mid]):
                geoms.append(g)
                mats.append(mid)
        if geoms:
            knobs[name] = {
                "geoms": np.array(sorted(geoms), np.int64),
                "mats": np.array(sorted(set(mats)), np.int64),
                "init": np.asarray(col, np.float64),
            }
    # Match table and floor by material name, independent of their visible area.
    for name, mat in (("table", "table_wood"), ("floor", "basic_floor")):
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
        if mid < 0:
            continue
        gs = [g for g in wearers.get(mid, []) if int(m.geom_group[g]) in VISIBLE_GROUPS]
        if not gs:
            continue
        knobs[name] = {
            "geoms": np.array(sorted(gs), np.int64),
            "mats": np.array([mid], np.int64),
            "init": np.asarray(m.mat_rgba[mid], np.float64)[:3],
        }
    return knobs


# --------------------------------------------------------------------------- the MJWarp context


class _Ctx:
    __slots__ = (
        "wm",
        "wd",
        "rc",
        "bgra",
        "seg",
        "graph",
        "nworld",
        "res",
        "wp",
        "mjw",
        "bvh_ids",
        "bvh_ids0",
        "bvh_mover_slots",
        "bvh_tr_slots",
        "bvh_substitute",
    )


def _context(task: str, res: int, nworld: int) -> _Ctx:
    """Cache a captured MJWarp graph by task, resolution, and world count."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    key = (task, int(res), int(nworld))
    hit = _CTX_CACHE.get(key)
    if hit is not None:
        return hit
    b = _build_bundle(task)
    m, d = b.model, b.data
    for a in MODEL_ARRAYS:  # a canonical model, so the context is a
        getattr(m, a)[:] = b.pristine[a]  # function of the task and nothing else
    d.qpos[:] = b.qpos0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    if MW.needs_com_pos(m):  # ``corner`` is mode="fixed"; assert it
        raise RuntimeError(
            f"{task}: a camera or light tracks subtree_com, which kinematics does not "
            f"write; the frame loop here would render every state from one pose"
        )

    margin0 = np.array(m.geom_margin, copy=True), np.array(m.pair_margin, copy=True)
    tex0 = np.array(m.mat_texid, copy=True)
    m.geom_margin[:] = 0.0
    m.pair_margin[:] = 0.0
    _neutralise_cube_textures(m)
    prep = MW.prepare_model(m, VISIBLE_GROUPS)
    try:
        c = _Ctx()
        c.wp, c.mjw, c.nworld, c.res = wp, mjw, int(nworld), int(res)
        d0 = mujoco.MjData(m)
        d0.qpos[:] = b.qpos0
        mujoco.mj_forward(m, d0)
        c.wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in b.batched})
        c.wd = mjw.put_data(m, d0, nworld=int(nworld))
        # Capture inside gl_rounding; patching only warmup leaves a truncating replay kernel.
        # Keep light specialization disabled here because it can change rounding.
        from contrib.render import lightspec as LS

        with MW.gl_rounding(), LS.specialised_lights(enabled=False):
            c.rc = MW.create_render_context(
                m,
                nworld=int(nworld),
                cam_res=(int(res), int(res)),
                render_rgb=True,
                render_seg=True,
                cam_active=[i == b.cam for i in range(m.ncam)],
                enabled_geom_groups=list(VISIBLE_GROUPS),
                use_textures=True,
                use_shadows=True,
                render_skybox=True,
                enable_specular=True,
            )
            _frame(c)
            wp.synchronize()
            # Keep buffer views; advanced channel indexing would retain a frozen copy.
            c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
            c.seg = wp.to_torch(c.rc.seg_data).reshape(int(nworld), int(res), int(res), 2)[..., 0]
            with wp.ScopedCapture() as cap:
                _frame(c)
            c.graph = cap.graph
        # Repoint mover slots to static geometry for extra passes without rebuilding the context.
        c.bvh_ids = wp.to_torch(c.rc.enabled_geom_ids)
        c.bvh_ids0 = c.bvh_ids.clone()
        mover_set = set(int(g) for g in b.movers)
        c.bvh_mover_slots = torch.as_tensor(
            [i for i, g in enumerate(c.bvh_ids0.tolist()) if int(g) in mover_set],
            dtype=torch.long,
            device=c.bvh_ids.device,
        )
        tr_set = set(int(g) for g in b.translucent)
        c.bvh_tr_slots = np.array([i for i, g in enumerate(c.bvh_ids0.tolist()) if int(g) in tr_set], np.int64)
        rest = [int(g) for g in c.bvh_ids0.tolist() if int(g) not in mover_set]
        c.bvh_substitute = rest[0] if rest else -1
    finally:
        MW.restore_model(m, prep)
        m.mat_texid[:] = tex0
        m.geom_margin[:], m.pair_margin[:] = margin0
    _CTX_CACHE.put(key, c)
    return c


def _neutralise_cube_textures(m) -> int:
    """Remove material cube-texture bindings that fault in the supported MJWarp path.

    Keep the skybox. Static colors come from the source bake; movers use the
    texture-mean approximation.
    """
    import mujoco

    ttype = np.asarray(m.tex_type, np.int64)
    flat = m.mat_texid.reshape(m.nmat, -1) if m.mat_texid.ndim > 1 else m.mat_texid.reshape(-1, 1)
    n = 0
    for i in range(flat.shape[0]):
        for j in range(flat.shape[1]):
            t = int(flat[i, j])
            if t >= 0 and ttype[t] == int(mujoco.mjtTexture.mjTEXTURE_CUBE):
                flat[i, j] = -1
                n += 1
    return n


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


class _Plate:
    """Static source images under bright and shadowed lighting, plus model transforms.

    ``off`` caches the corresponding MJWarp frame without movers; comparing it
    to the live frame identifies shadows cast onto static geometry.
    """

    __slots__ = ("img", "dark", "off", "body_pos", "body_quat", "gxpos", "gxmat", "model")


class MetaWorldBase(Env):
    """Composite source-rendered scenery with MJWarp-rendered moving geometry.

    The render state stores qpos and visible model fields changed at reset.
    Plates are keyed by episode placement and appearance. This is sufficient
    for rendering, not for resuming the simulator.
    """

    family = "metaworld"

    TASK: str = ""
    default_size = 480  # metaworld's own SawyerXYZEnv default
    default_supersample = 1
    # Include every appearance leaf because any task may bake it into static geometry.
    PLATE_PATHS = (
        "table.color",
        "floor.color",
        "robot.color",
        "object.color",
        "light.diffuse",
        "camera.distance_scale",
        "camera.azimuth_delta",
        "camera.elevation_delta",
    )
    parity_interior_min = 0.978
    parity_all_exact_min = 0.80
    parity_note = (
        "FIVE residuals, measured separately, in the order of how much they cost. (1) MESH SHADING IS "
        "PER-VERTEX IN MuJoCo AND PER-PIXEL HERE, and it is the whole of what is left. MuJoCo draws "
        "the Sawyer's and the mechanisms' meshes as triangles with Gouraud interpolation of vertex "
        "colours; an analytic Blinn-Phong evaluation lands within a level or two of it and not on it. "
        "Looked at SIGNED on the movers' interior, the difference is {0: 27.8%, +1: 22.7%, -1: 8.4%, "
        "+2: 7.3%, -2: 4.0%, ...} -- a broad spread, which is what interpolation-versus-evaluation "
        "looks like and is NOT what a quantisation rule looks like (that signature is Fetch's "
        "{0: 86.8%, -1: 13.1%}, and it was found and fixed here too: mujoco_warp truncates float->uint8 "
        "where GL rounds, worth interior 0.791 -> 0.9988 on its own). It is the same limit "
        "the OGBench investigation refuted three ways, and it bites in proportion to the mover share: "
        "window-open (0.980766) has 16.7% of the frame moving, dial-turn (0.999830) has almost none. "
        "(2) THE SHADOW BOUNDARY. A mover's shadow on a static surface IS reproduced -- two bakes per "
        "episode, bright and with the casting light's diffuse and specular removed, selected by "
        "MJWarp's own movers-off render -- but MuJoCo filters a 4096 shadow map where MJWarp casts one "
        "ray, so the two masks agree on 94.5% (IoU) and disagree on a one-pixel rim. Uncorrected this "
        "was 8.6% of the frame at a mean of 18.9/255. (3) TEXTURES ON MOVERS. mujoco_warp hard-faults "
        "with CUDA 700 on a cube texture and its sample_texture derives UVs for PLANE and MESH only, "
        "so a mover's textured surface is drawn FLAT. The constant is mat_rgba x the texture's mean, "
        "because MuJoCo paints texture x rgba and mat_rgba alone is the modulation factor without the "
        "thing it modulates -- on door-close-v3 that correction is worth mean 2.96 -> 1.10/255 and "
        "interior 0.98768 -> 0.99275. What is left is the texture's DETAIL, which is closed by "
        "non-mesh texture UVs in mujoco_warp -- the same upstream gap Adroit needs -- "
        "and by nothing this repo can do locally. (4) TRANSLUCENT MOVERS. mujoco_warp's shader reads "
        "only color[0..2] and packs a constant 255, so window_glass (rgba='0 .3 .4 .1') would be "
        "painted opaque; it is blended against the layer behind it instead, which needs that layer "
        "rendered (an extra replay on the 2 tasks that have such a mover, and nothing on the other "
        "48). Worth mean 2.33 -> 1.11/255 on window-open-v3. (5) SITES ON MOVING BODIES ARE NOT DRAWN "
        "AT ALL. MuJoCo draws sites in groups 0-2 and mujoco_warp draws none; a site on a STATIC body "
        "is baked into the plate and is therefore exact -- which is what puts reach's and pick-place's "
        "goal marker in the picture -- while the Sawyer's own headsite is 0.0004-0.0014 of the frame "
        "and is missing. The static ~64% of the frame (table, wall, floor, skybox, the Sawyer's "
        "pedestal, and every per-episode placement) is exact BY CONSTRUCTION, because it is baked "
        "through MuJoCo itself."
    )

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        b = _build_bundle(self.TASK)
        self.state_dim = len(b.state_names)
        self.state_names = b.state_names
        # Episode fields are model placements, distinct from joint coordinates in qpos.
        k = self.state_dim - b.nq
        self.state_layout = ((("qpos",), b.nq, 0),) + ((((("episode",), k, b.nq)),) if k else ())
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        self._b = b
        self._check_plate_is_constant()

    def _check_plate_is_constant(self) -> None:
        """Check that two states from one episode bake the same image with movers hidden."""
        pool = _state_pool(self.TASK)
        st = pool[:2] if POOL_PER_EPISODE > 1 else pool[:1]  # consecutive rows share an episode
        if len(st) < 2:
            return
        a = self._bake(int(self.size), None, st[0])
        c = self._bake(int(self.size), None, st[1])
        dd = int((a.img.to(torch.int16) - c.img.to(torch.int16)).abs().max())
        assert dd == 0, (
            f"{type(self).__name__}: two states of the SAME episode bake plates that differ by "
            f"{dd}/255 with every mover hidden, so something that moves with qpos is being treated as "
            f"scenery and the composite would draw it twice.\n"
            f"SUSPECT THE NODE FIRST if this fires on some tasks and not on their near-identical "
            f"siblings. Both frames here are MuJoCo renders of the SAME static scene, so this is also "
            f"the cheapest possible test of whether the machine's offscreen rasterizer is "
            f"deterministic at all — and it has caught two hosts doing exactly that during this "
            f"family's calibration (6 of 50 tasks on one, 5 of 50 on another, every one of them "
            f"passing elsewhere within minutes). See the EGL note in "
            f"stable_envs/testing/conformance.py."
        )

    def make_variation_space(self) -> Dict:
        """Expose float material colors, light intensity, and camera offsets.

        Float colors preserve source defaults. Skybox textures and the compiled
        headlight cannot vary per lane.
        """
        b = _build_bundle(self.TASK)
        space: dict = {}
        for name in ("table", "floor", "robot", "object"):
            k = b.knobs.get(name)
            if k is not None:
                space[name] = Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=k["init"].tolist())})
        space["light"] = Dict({"diffuse": Box(low=0.0, high=2.0, shape=(1,), init_value=[1.0])})
        space["camera"] = Dict(
            {
                "distance_scale": Box(low=0.6, high=1.6, shape=(1,), init_value=[1.0]),
                "azimuth_delta": Box(low=-25.0, high=25.0, shape=(1,), init_value=[0.0]),
                "elevation_delta": Box(low=-20.0, high=20.0, shape=(1,), init_value=[0.0]),
            }
        )
        return Dict(space)

    def _var(self, variation, path) -> np.ndarray:
        if variation is not None and path in variation:
            return np.asarray(variation[path], np.float64).reshape(-1)
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        return np.asarray(node.value, np.float64).reshape(-1)

    def _paths(self) -> tuple[str, ...]:
        have = set(self.variation_space.names())
        return tuple(p for p in self.PLATE_PATHS if p in have)

    def _camera(self, variation):
        """Orbit and scale the source camera around the table; zero deltas preserve its pose."""
        import mujoco

        b = self._b if hasattr(self, "_b") else _build_bundle(self.TASK)
        s = float(self._var(variation, "camera.distance_scale")[0])
        az = np.deg2rad(float(self._var(variation, "camera.azimuth_delta")[0]))
        el = np.deg2rad(float(self._var(variation, "camera.elevation_delta")[0]))
        v = b.cam_pos0 - b.pivot
        ca, sa = np.cos(az), np.sin(az)
        rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        axis = np.cross(v / max(np.linalg.norm(v), 1e-12), [0.0, 0.0, 1.0])
        n = np.linalg.norm(axis)
        if n < 1e-9:
            re = np.eye(3)
        else:
            axis = axis / n
            k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            re = np.eye(3) + np.sin(el) * k + (1.0 - np.cos(el)) * (k @ k)
        r = rz @ re
        pos = b.pivot + s * (r @ v)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(r).reshape(-1))
        out = np.empty(4)
        mujoco.mju_mulQuat(out, q, b.cam_quat0)
        return pos, out

    def _plate(self, res, over, state_row) -> _Plate:
        """Cache a static bake by resolution, device, appearance, and episode fields.

        Use exact float32 episode values: rounding would make cache contents depend
        on which state arrived first.
        """
        key = (
            self.TASK,
            int(res),
            str(self.device),
            tuple(np.asarray(state_row, np.float32)[self._b.nq :].tolist()),
            tuple(float(v) for p in self._paths() for v in self._var(over, p)),
        )
        hit = _PLATE_CACHE.get(key)
        if hit is not None:
            return hit
        return _PLATE_CACHE.put(key, self._bake(res, over, state_row))

    def _bake(self, res, variation, state_row) -> _Plate:
        """Bake static lighting and transforms after restoring the pristine model.

        Apply variations by absolute assignment so repeated camera scaling and
        material changes do not accumulate.
        """
        import mujoco

        b = self._b
        m, d = b.model, b.data
        _apply_state(b, m, d, np.asarray(state_row, np.float64), forward=False)
        for name, k in b.knobs.items():
            col = self._var(variation, f"{name}.color")
            m.geom_rgba[k["geoms"], :3] = col
            if len(k["mats"]):
                m.mat_rgba[k["mats"], :3] = col
        m.light_diffuse[:] = b.light0 * float(self._var(variation, "light.diffuse")[0])
        pos, quat = self._camera(variation)
        m.cam_pos[b.cam] = pos
        m.cam_quat[b.cam] = quat

        gkeep = np.array(m.geom_group, copy=True)
        skeep = np.array(m.site_group, copy=True)
        m.geom_group[b.movers] = HIDDEN_GROUP
        if len(b.mover_sites):
            m.site_group[b.mover_sites] = HIDDEN_GROUP
        mujoco.mj_forward(m, d)
        r = _raster(self.TASK, int(res), m)
        r.update_scene(d, camera=b.cam)
        img = r.render().copy()
        # Remove diffuse/specular from casting lights to bake fully shadowed static pixels.
        ld = np.array(m.light_diffuse, copy=True)
        ls = np.array(m.light_specular, copy=True)
        if len(b.casting):
            m.light_diffuse[b.casting] = 0.0
            m.light_specular[b.casting] = 0.0
            r.update_scene(d, camera=b.cam)
            dark = r.render().copy()
            m.light_diffuse[:], m.light_specular[:] = ld, ls
        else:
            dark = img
        m.geom_group[:] = gkeep
        m.site_group[:] = skeep

        p = _Plate()
        p.img = torch.as_tensor(img, device=self.device)
        p.dark = p.img if dark is img else torch.as_tensor(dark, device=self.device)
        p.off = None
        p.body_pos = torch.as_tensor(np.array(m.body_pos, np.float32), device=self.device)
        p.body_quat = torch.as_tensor(np.array(m.body_quat, np.float32), device=self.device)
        p.gxpos = torch.as_tensor(np.array(d.geom_xpos[b.plate_static], np.float32), device=self.device)
        p.gxmat = torch.as_tensor(
            np.array(d.geom_xmat[b.plate_static], np.float32).reshape(-1, 3, 3), device=self.device
        )
        p.model = {
            a: torch.as_tensor(np.array(getattr(m, a), np.float32), device=self.device)
            for a in b.batched
            if a in MODEL_ARRAYS and a not in ("body_pos", "body_quat")
        }
        # Apply texture means only to the uploaded mover materials, after the textured source bake.
        if "mat_rgba" in p.model:
            p.model["mat_rgba"][:, :3] *= torch.as_tensor(b.mat_tex_mean, dtype=torch.float32, device=self.device)
        for a in MODEL_ARRAYS:
            getattr(m, a)[:] = b.pristine[a]
        return p

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        res = size * ss
        sty = self.resolve_style(style, variation, batch=B)
        b = self._b
        paths = self._paths()

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        # Group by exact appearance and episode fields to avoid order-dependent plate reuse.
        skey = torch.cat([sty[p].reshape(B, -1) for p in paths] + [s[:, b.nq :]], 1).cpu().numpy()
        uniq, inv = np.unique(skey, axis=0, return_inverse=True)
        offs, o = [], 0
        for p in paths:
            w = int(sty.layout[p][1])
            offs.append((o, w))
            o += w
        plates, plates1 = [], []
        for row in uniq:
            over = {p: row[a : a + w] for p, (a, w) in zip(paths, offs)}
            # With movers hidden, the plate must be independent of qpos.
            st_row = np.concatenate([b.qpos0[: b.nq], row[o:]]).astype(np.float32)
            plates.append(self._plate(res, over, st_row))
            plates1.append(plates[-1] if ss == 1 else self._plate(size, over, st_row))
        img = torch.stack([p.img for p in plates])
        dark = torch.stack([p.dark for p in plates])
        img1 = img if ss == 1 else torch.stack([p.img for p in plates1])
        dark1 = dark if ss == 1 else torch.stack([p.dark for p in plates1])
        grp = torch.as_tensor(inv, device=self.device, dtype=torch.long)
        campos, camquat = self._camera_batch(sty, B)
        stacked = self._plate_stacks(plates)
        off_all = self._movers_off_stack(plates)
        lut = self._mover_lut()
        for lo in range(0, B, MJW_CHUNK):
            hi = min(lo + MJW_CHUNK, B)
            n = hi - lo
            nw = 1 if n <= 1 else min(MJW_CHUNK, 1 << (n - 1).bit_length())
            c = _context(self.TASK, res, nw)
            self._upload(c, s[lo:hi], sty, lo, hi, stacked, grp[lo:hi], campos, camquat)
            # Synchronize Torch uploads and Warp replay before reading or reusing shared buffers.
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            # Snapshot RGB and derive masks before extra passes overwrite the context buffers.
            rgb = c.bgra[:n][..., [2, 1, 0]]  # advanced indexing: a COPY
            seg = c.seg[:n]
            segp = (seg + 1).reshape(-1)  # -1 (a miss) becomes row 0 of both tables
            mv = lut.index_select(0, segp).view(seg.shape)
            alpha = (
                self._blend_lut().index_select(0, segp).view(seg.shape).unsqueeze(-1) if len(b.translucent) else None
            )
            g = grp[lo:hi]
            off = off_all[g] if off_all is not None else self._movers_off(c, n, plates, g)
            pl = torch.where((rgb != off).any(-1, keepdim=True) & ~mv.unsqueeze(-1), dark[g], img[g])
            comp = torch.where(mv.unsqueeze(-1), self._blend(c, rgb, pl, n, alpha), pl)
            if ss == 1:
                buf[lo:hi] = comp
            else:
                small = (
                    comp.to(torch.float32)
                    .reshape(n, size, ss, size, ss, 3)
                    .mean((2, 4))
                    .round_()
                    .clamp_(0, 255)
                    .to(torch.uint8)
                )
                any_mv = mv.reshape(n, size, ss, size, ss).any(4).any(2)
                # Use the dark plate only where every subsample is shadowed; retain bright boundary pixels.
                sh1 = ((rgb != off).any(-1) & ~mv).reshape(n, size, ss, size, ss).all(4).all(2)
                pl1 = torch.where(sh1.unsqueeze(-1), dark1[g], img1[g])
                buf[lo:hi] = torch.where(any_mv.unsqueeze(-1), small, pl1)
        return buf.view(*lead, size, size, 3) if lead else buf

    def _blend(
        self, c: _Ctx, rgb: torch.Tensor, plate: torch.Tensor, n: int, alpha: torch.Tensor | None
    ) -> torch.Tensor:
        """Blend translucent movers over a second render with those movers hidden.

        The background can contain other moving geometry, so the static plate alone
        is insufficient. This extra pass runs only for tasks with translucent movers.
        """
        if alpha is None or not len(c.bvh_tr_slots):
            return rgb
        w = alpha
        behind, seg_behind = self._layer_off(c, n, self._idx("bvh_translucent", c.bvh_tr_slots))
        mv_behind = torch.isin(seg_behind, self._idx("movers", self._b.movers).to(seg_behind.dtype))
        base = torch.where(mv_behind.unsqueeze(-1), behind, plate).to(torch.float32)
        out = w * rgb.to(torch.float32) + (1.0 - w) * base
        return out.round_().clamp_(0, 255).to(torch.uint8)

    def _blend_lut(self) -> torch.Tensor:
        """Alpha indexed by segmentation ID + 1; row zero represents a missed ray."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("alpha_lut")
        if hit is None:
            a = np.concatenate([[1.0], np.asarray(self._b.alpha, np.float32)])
            hit = cache["alpha_lut"] = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        return hit

    def _mover_lut(self) -> torch.Tensor:
        """Mover membership indexed by segmentation ID + 1, with misses mapped to false."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("mover_lut")
        if hit is None:
            lut = torch.zeros(int(self._b.model.ngeom) + 1, dtype=torch.bool, device=self.device)
            lut[self._idx("movers+1", np.asarray(self._b.movers, np.int64) + 1)] = True
            hit = cache["mover_lut"] = lut
        return hit

    def _camera_batch(self, sty, B: int):
        keys = ("camera.distance_scale", "camera.azimuth_delta", "camera.elevation_delta")
        camkey = torch.cat([sty[p].reshape(-1, 1) for p in keys], 1)[:B].cpu().numpy()
        uniq, inv = np.unique(np.round(camkey.astype(np.float64), 6), axis=0, return_inverse=True)
        poses = [self._camera({keys[0]: r[0:1], keys[1]: r[1:2], keys[2]: r[2:3]}) for r in uniq]
        pos = torch.as_tensor(np.stack([p for p, _ in poses])[inv], device=self.device, dtype=torch.float32)
        quat = torch.as_tensor(np.stack([q for _, q in poses])[inv], device=self.device, dtype=torch.float32)
        return pos, quat

    def _plate_stacks(self, plates) -> dict:
        out = {
            "body_pos": torch.stack([p.body_pos for p in plates]),
            "body_quat": torch.stack([p.body_quat for p in plates]),
            "gxpos": torch.stack([p.gxpos for p in plates]),
            "gxmat": torch.stack([p.gxmat for p in plates]),
        }
        out["model"] = {a: torch.stack([p.model[a] for p in plates]) for a in plates[0].model}
        return out

    def _movers_off_stack(self, plates):
        """Stack cached movers-off frames, or return None if any required frame is missing."""
        if not len(self._b.casting) or any(p.off is None for p in plates):
            return None
        return torch.stack([p.off for p in plates])

    def _layer_off(self, c: _Ctx, n: int, slots: torch.Tensor):
        """Render with selected BVH slots replaced by a static geom; return owned RGB/seg copies.

        Always restore the shared slot table, including after a failed launch.
        """
        try:
            c.bvh_ids[slots] = c.bvh_substitute
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            return c.bgra[:n][..., [2, 1, 0]], c.seg[:n].clone()
        finally:
            c.bvh_ids.copy_(c.bvh_ids0)
            torch.cuda.synchronize()

    def _movers_off(self, c: _Ctx, n: int, plates, g) -> torch.Tensor:
        """Cache the no-mover shadow reference for each episode/appearance plate."""
        if c.bvh_substitute < 0 or not len(self._b.casting):
            return c.bgra[:n][..., [2, 1, 0]]  # nothing casts: no pixel can be shadowed
        # Gather only plates present in this chunk; others may not have cached frames yet.
        here = sorted({int(k) for k in g.tolist()})
        want = [k for k in here if plates[k].off is None]
        if want:
            frames, _ = self._layer_off(c, n, c.bvh_mover_slots)
            first = {int(k): int(i) for i, k in reversed(list(enumerate(g.tolist())))}
            for k in want:
                plates[k].off = frames[first[k]].clone()
        pos = torch.full((len(plates),), -1, dtype=torch.long, device=g.device)
        pos[torch.as_tensor(here, dtype=torch.long, device=g.device)] = torch.arange(len(here), device=g.device)
        return torch.stack([plates[k].off for k in here])[pos[g]]

    def _scale(self, name: str, mats) -> torch.Tensor:
        cache = self.__dict__.setdefault("_scale_cache", {})
        hit = cache.get(name)
        if hit is None:
            hit = cache[name] = torch.as_tensor(
                self._b.mat_tex_mean[np.asarray(mats, np.int64)], dtype=torch.float32, device=self.device
            ).reshape(1, -1, 3)
        return hit

    def _idx(self, key: str, values) -> torch.Tensor:
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = torch.as_tensor(np.asarray(values, np.int64), device=self.device)
        return hit

    def _upload(
        self,
        c: _Ctx,
        s: torch.Tensor,
        sty,
        lo: int,
        hi: int,
        stacked: dict,
        grp,
        campos: torch.Tensor,
        camquat: torch.Tensor,
    ) -> None:
        b = self._b
        wp = c.wp
        n = hi - lo
        wp.to_torch(c.wd.qpos)[:n] = s[:, : b.nq]

        # Welded geom transforms come from CPU forward; MJWarp does not recompute them.
        wp.to_torch(c.wm.body_pos)[:n] = stacked["body_pos"][grp]
        wp.to_torch(c.wm.body_quat)[:n] = stacked["body_quat"][grp]
        if len(b.plate_static):
            st = self._idx("plate_static", b.plate_static)
            wp.to_torch(c.wd.geom_xpos)[:n, st] = stacked["gxpos"][grp]
            wp.to_torch(c.wd.geom_xmat)[:n, st] = stacked["gxmat"][grp]
        for a, t in stacked["model"].items():
            if a == "cam_pos":  # written from the camera knobs below
                continue
            wp.to_torch(getattr(c.wm, a))[:n] = t[grp]

        gr = wp.to_torch(c.wm.geom_rgba)
        mr = wp.to_torch(c.wm.mat_rgba)
        for name, k in b.knobs.items():
            col = sty[f"{name}.color"][lo:hi]
            gr[:n, self._idx(f"{name}.geoms", k["geoms"]), :3] = col.unsqueeze(1)
            if len(k["mats"]):
                # Modulate material colors by texture means consistently with the initial bake.
                mr[:n, self._idx(f"{name}.mats", k["mats"]), :3] = col.unsqueeze(1) * self._scale(name, k["mats"])
        wp.to_torch(c.wm.light_diffuse)[:n] = self._light0().unsqueeze(0) * sty["light.diffuse"][lo:hi].reshape(
            n, 1, 1
        )
        wp.to_torch(c.wm.cam_pos)[:n, b.cam] = campos[lo:hi]
        wp.to_torch(c.wm.cam_quat)[:n, b.cam] = camquat[lo:hi]

    def _light0(self) -> torch.Tensor:
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("light0")
        if hit is None:
            hit = cache["light0"] = torch.as_tensor(self._b.light0, device=self.device, dtype=torch.float32)
        return hit

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a cached source pool with multiple poses per seeded episode.

        Free-joint object orientations are additionally varied by random yaw.
        """
        rng = rng or self.rng
        pool = _state_pool(self.TASK)
        return pool[rng.integers(0, len(pool), int(batch))].copy()

    @classmethod
    def reference_env(cls, size: int = 480):
        """Construct a source task configured for rgb_array output at the requested size."""
        return _make_task_env(cls.TASK, size=int(size))

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore model fields and qpos, run forward, and render source frames.

        Require the reconstructed render state to agree within 1e-5 before recording
        a frame. Fields outside state are pinned to the same snapshot as the plate.
        """
        b = _build_bundle(cls.TASK)
        u = env.unwrapped
        m, d = u.model, u.data
        out = []
        for st in np.asarray(states, np.float64):
            _apply_state(b, m, d, st)
            got = _read_state(b, m, d)
            err = float(np.abs(got - st).max())
            if err > 1e-5:
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state (max "
                    f"{err:.3e}); every parity number computed from it would be comparing two "
                    f"different scenes"
                )
            out.append(np.ascontiguousarray(u.render()))
        return np.stack(out)


def _apply_state(b: _Bundle, m, d, st, forward: bool = True) -> None:
    """Restore the pristine model, apply qpos and episode fields, and optionally run forward."""
    import mujoco

    for a in MODEL_ARRAYS:
        getattr(m, a)[:] = b.pristine[a]
    for i, (a, f) in enumerate(b.slots):
        arr = getattr(m, a)
        v = st[b.nq + i]
        arr.reshape(-1)[f] = np.rint(v) if arr.dtype.kind in "iu" else v
    d.qpos[:] = st[: b.nq]
    d.qvel[:] = 0.0
    if forward:
        mujoco.mj_forward(m, d)


def _read_state(b: _Bundle, m, d) -> np.ndarray:
    return np.concatenate(
        [np.asarray(d.qpos, np.float64), np.array([float(getattr(m, a).reshape(-1)[f]) for a, f in b.slots])]
    )


def _state_pool(task: str) -> np.ndarray:
    """Build seeded source rollouts and vary free-joint yaw for render coverage."""
    hit = _POOL_CACHE.get(task)
    if hit is not None:
        return hit
    import mujoco

    b = _build_bundle(task)
    env = b.env
    u = env.unwrapped
    m, d = b.model, b.data
    free = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    rng = np.random.default_rng(0)
    dim = b.nq + len(b.slots)
    out = np.empty((POOL_EPISODES * POOL_PER_EPISODE, dim), np.float32)
    i = 0
    for ep in range(POOL_EPISODES):
        _seeded_reset(env, 10_000 + ep)
        slice_vals = np.array([float(getattr(m, a).reshape(-1)[f]) for a, f in b.slots])
        for k in range(POOL_PER_EPISODE):
            for _ in range(int(rng.integers(0, 26)) if k else 0):
                env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))
            q = np.array(d.qpos, np.float64)
            for adr in free:
                yaw = float(rng.uniform(-np.pi, np.pi))
                q[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            out[i, : b.nq] = q
            out[i, b.nq :] = slice_vals
            i += 1
    for a in MODEL_ARRAYS:
        getattr(m, a)[:] = b.pristine[a]
    _POOL_CACHE[task] = out
    return out


def _class_name(task: str) -> str:
    return "MetaWorld" + "".join(p.capitalize() for p in task[:-3].strip("-").split("-"))


def _make_classes():
    made = {}
    for task in TASKS:
        cname = _class_name(task)
        made[cname] = register(
            type(
                cname,
                (MetaWorldBase,),
                dict(
                    TASK=task,
                    name=f"metaworld-{task}",
                    reference_id=f"metaworld/{task}",
                    __doc__=(
                        f"Meta-World {task} through the model camera 'corner'. State is qpos plus the "
                        f"episode's model slice (the placement Meta-World writes into body_pos/site_pos "
                        f"at reset); the picture is MuJoCo's bake of that episode's static scene with "
                        f"mujoco_warp's ray-cast movers composited over it by the segmentation buffer. "
                        f"The reference is ALL_V3_ENVIRONMENTS[{task!r}](render_mode='rgb_array', "
                        f"camera_name='corner'), which is what Meta-World's own benchmark wrappers "
                        f"instantiate — its gymnasium ids are vector envs over task SETS."
                    ),
                ),
            )
        )
    return made


globals().update(_make_classes())
