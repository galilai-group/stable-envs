"""Meta-World v3 — 50 registered ids, one per task, each a PER-EPISODE baked plate over MJWarp movers.

Reference: ``metaworld.env_dict.ALL_V3_ENVIRONMENTS[task](render_mode="rgb_array", width=480,
height=480, camera_name="corner")`` at its native 480x480, i.e. MuJoCo's EGL raster of a lit 3-D
scene through the MODEL camera ``corner``. The gymnasium ids Meta-World registers are BENCHMARK
VECTOR envs (``Meta-World/MT1``, ``Meta-World/ML1-train``), and its two single-task ids take
``(env_name, seed)`` and nothing else, so there is no id that constructs one task at a chosen
resolution with a chosen camera — the class table is what the benchmark wrappers instantiate anyway,
and it is what ``reference_env`` uses.

WHY THE CAMERA IS A MODEL CAMERA AND NOT A CHOICE
--------------------------------------------------

``camera_name`` defaults to ``None``, which lands on gymnasium's FREE camera — the one whose pose is
frozen at the first render from whatever state the viewer happened to be constructed in, which is the
history dependence ``docs/research/adroit.md`` records. ``corner`` is declared ``mode="fixed"`` in
``xyz_base.xml``, so its pose is a constant of the model and the same state renders the same frame in
any process. That is a precondition for this family existing at all, not a stylistic preference.

THE STATE IS ``qpos`` PLUS A HANDFUL OF MODEL FLOATS, AND THAT IS NOT OPTIONAL
------------------------------------------------------------------------------

Meta-World writes the episode's placement into the **model** at ``reset`` — ``model.body_pos`` for the
door/drawer/box, ``model.site_pos`` for the goal marker — not into ``qpos``. So ``qpos`` alone is not
a state: the same joint angles draw two different pictures in two episodes (measured: 63-71/255).
``docs/research/metaworld.md`` §1 walks a ladder of restores and finds ``qpos`` plus the 14 model
arrays **bit-exact on 50/50 tasks** against a control of 239/255.

This env carries that slice at ELEMENT granularity rather than by the array. ``_build_bundle`` resets
the env several times with a pinned RNG, takes the elements of those 14 arrays that actually MOVE
between draws, and keeps the ones that reach a pixel — measured, one MuJoCo render per candidate, not
assumed. Everything else in those arrays is pinned to a snapshot on BOTH sides (``reference_render``
restores it too), so an element that varies per episode and reaches no pixel cannot make the frame
history-dependent and cannot show up as a dead state dimension either. In practice the live slice is
3-6 floats.

NEVER TAKE A FRAME STRAIGHT OUT OF ``reset()``
-----------------------------------------------

``_set_obj_xyz`` is what forwards the simulation inside ``reset_model``, and several tasks write
``model.body_pos``/``model.site_pos`` AFTER their last call to it, so ``data.site_xpos`` still holds
the PREVIOUS draw's placement when ``reset()`` returns. Measured: stale on **28/50 tasks**, worst
169/255, ``site_xpos`` off by up to 0.124 m — and **0/50 after one ``step()``**, because ``step``
forwards. This is a probe protocol rule, not a defect a port models: ``reference_render`` calls
``mj_forward`` after writing the state and before rendering, and the state pool is read after a
rollout rather than out of a bare reset. The earlier NO-GO for this family was measured against those
stale frames and is withdrawn.

THE ONE DEPARTURE FROM ``fetch.py``: THE PLATE IS PER-EPISODE
--------------------------------------------------------------

Fetch caches its baked plate by STYLE, because its static scene is a function of the appearance alone.
Here it is not: the per-episode placement is written into the model and moves scenery. The plate is
bit-identical across 12 states WITHIN an episode on 50/50 tasks, and closing the mover set under the
kinematic subtree narrows the cross-episode drift from 255/255 over 10.7% of pixels to 54/255 over
1.46% — it does not remove it (``probe_backend4.py``). So the cache key here is
``(task, res, device, every style leaf, THE STATE'S MODEL SLICE)``: the plate is keyed on the episode,
and one MuJoCo render (3.2 ms) is paid per distinct episode rather than per distinct appearance, about
6% of a 500-step episode's batched rendering. It is declared rather than discovered, and it is why
``PLATE_PATHS`` is every leaf rather than the static ones — with the episode already in the key, an
extra style column costs nothing and removes a whole class of "which knob is static" mistakes.

MJWarp CANNOT MOVE A WELDED BODY'S GEOMS, WHICH IS WHY THE STATIC KINEMATICS ARE UPLOADED
------------------------------------------------------------------------------------------

``mujoco_warp``'s ``_geom_local_to_global`` RETURNS EARLY for a geom whose body is welded to the world
(``body_weldid == 0``): world geometry is static, so ``geom_xpos``/``geom_xmat`` are computed once, in
``put_data``, and no per-world ``body_pos`` ever reaches them. Meta-World places the table-mounted
scenery per episode, so left alone MJWarp would ray-cast every episode's movers against episode
zero's furniture and occlude them in the wrong places. ``kinematics`` DOES walk static bodies (a
branch is an ancestor chain, so every body is in one), so ``d.xpos`` is right and only the geom
transform is missing — and the plate bake has already computed exactly that on the CPU. So each
distinct episode's ``d.geom_xpos``/``d.geom_xmat`` for the welded geoms ride along with its plate and
are uploaded per world. It costs one gather; getting it wrong costs silhouettes.

THE SHADOW THE PLATE CANNOT HOLD, AND THE THIRD BAKE THAT PUTS IT BACK
-----------------------------------------------------------------------

Fetch's scene has ``castshadow="false"`` on its one light, so its plate design never had to answer
this. Meta-World has three lights and **one of them casts**, and hiding a geom to bake the plate hides
its shadow with it — so a naive composite is missing the robot's shadow on the table. Measured on
``reach-v3``, 8 states at 480 px: it is **8.6% of the frame at a mean of 18.9/255**, and 100% of the
non-mover disagreement. That is a visibly missing shadow, not a quantisation residual, and every
pixel of it lands where the interior mask does not look (the table wears a wood texture, so it has no
flat 3x3 neighbourhood) — which is exactly how a defect this size hides behind an interior number.

It is put back with MuJoCo's own arithmetic on both branches rather than approximated:

* a MuJoCo pixel is ``ambient + shadow * (diffuse + specular)`` summed over lights, so a SECOND bake
  with the casting light's diffuse and specular zeroed is precisely what MuJoCo draws where that light
  is fully occluded. Both bakes are per-episode, so both are cached with the plate.
* which static pixels the movers shadow comes from MJWarp, by rendering the same scene twice: once as
  usual, and once with the movers removed from ``rc.enabled_geom_ids`` — each mover's BVH slot
  re-pointed at a static geom, so the tree keeps its shape and one static surface is simply listed
  twice. The movers-off frame is a function of the model, the lights and the camera and NOT of qpos,
  so it too is cached on the plate: a 500-step rollout through one episode pays one extra graph replay
  in total.

Measured, before and after, on the same 8 states: all-pixel exactness **0.8017 -> 0.8826**, mean
**2.396 -> 0.920/255**, within-8 **0.898 -> 0.978**, interior **0.99923 -> 0.99978**, at a throughput
cost of 2% (2,454 f/s against 2,391 at batch 256). What is left is the shadow BOUNDARY — MuJoCo
filters a 4096 shadow map, MJWarp casts one ray per light — which is 0.5% of the frame; the two masks
agree on **94.5%** of the shadow (IoU) and the disagreement is a one-pixel rim.

WHAT THE PLATE HOLDS, AND THE FOUR MODEL SURGERIES THE MJWarp SIDE NEEDS
-------------------------------------------------------------------------

A geom is a MOVER iff its body is not welded to the world (or is a mocap body); everything else —
table, wall, floor, skybox, the Sawyer's pedestal and controller box, and every geom whose placement
is a constant of the episode — is baked through MuJoCo itself and is exact by construction. Sites are
baked too when they sit on a static body, which is what puts ``reach``/``pick-place``'s goal marker
in the picture; sites on moving bodies are not drawn by anything (MJWarp draws no sites at all) and
are declared in ``parity_note``.

The MJWarp context is built from the env's OWN ``MjModel``, with four changes made inside a window and
undone afterwards:

* **``geom_margin``/``pair_margin`` zeroed.** ``put_model`` validates the model's COLLISION
  configuration even when the caller only wants a render context, and refuses *every* Meta-World model:
  the Sawyer gripper pads carry ``geom_margin`` 0.001 on BOX/MESH pairs, which its MULTICCD path
  rejects. A margin reaches no pixel. ``mjENBL_MULTICCD`` no longer exists on ``mjtEnableBit`` in 3.11,
  so disabling the flag is not the route; zeroing the margin is.
* **Cube textures neutralised.** ``mujoco_warp`` hard-faults with **CUDA error 700** on
  ``mjTEXTURE_CUBE``, and every task carries 2-4 of them (the table's wood, the wall's metal). In the
  composite a static pixel comes from MuJoCo's bake, so MJWarp has to get static geometry's DEPTH
  right and never its COLOUR — pointing those materials away from their cube slot turns the fault into
  a working render at no cost to any composited pixel. Measured: the 24 tasks with at least one MOVER
  wearing a cube texture score a median composite interior 0.999870 against 0.999577 for the 26
  without, i.e. no penalty at all.
* **``prepare_model``**, which parks zero-alpha geometry (the invisible mocap sphere, the collision
  proxies) out of the visible groups, because MJWarp's shader never reads alpha and would paint them
  opaque.
* **``gl_rounding()``**, because MJWarp truncates float->uint8 where OpenGL rounds. On Fetch that one
  rule is worth interior 0.791 -> 0.9988; it is applied here for the same reason.

Two more rules apply to the MOVERS' COLOURS, and both are measured rather than reasoned:

* **A mover whose texture MJWarp cannot show is drawn in ``mat_rgba`` TIMES THE TEXTURE'S MEAN.**
  MuJoCo paints a textured surface as ``texture x rgba``, so the material's colour is a modulation
  factor and not the surface's colour; drawing the flat surface in ``mat_rgba`` alone is the factor
  with the thing it modulates thrown away, and on ``door-close-v3``'s door panel that is 1.7x too
  bright over 4.7% of the frame. Correcting the constant is worth mean **2.96 -> 1.10/255** and
  interior **0.98768 -> 0.99275** there, and it never costs anything: the plate is drawn by MuJoCo
  with its textures intact, so only mover pixels can move.
* **A TRANSLUCENT mover is blended against the layer behind it**, which has to be rendered — see
  ``_blend``. Blending it against the plate instead, which costs no launch, was tried and measured
  and moves the wrong way.

Neither is ``mjwarp.bake_primitive_textures``, whose docstring forbids it on a parity path: that tool
answers "should a surface the backend COULD texture be flattened?", and its answer here is still no.
The question here is which constant stands in for a flat draw that is already forced.

A CUDA 700 poisons the process — one real fault produced seven fabricated "Failed to allocate 4 bytes"
afterwards — so the calibration script runs one task per process, and this module raises rather than
retrying.

SEEDING: BOTH DOCUMENTED ROUTES FAIL, AND ``sample_states`` USES A THIRD
------------------------------------------------------------------------

``reset(seed=k)`` reproduces an episode on **0/50** tasks and so does the ``seed()`` the docstring
recommends. ``_get_state_rand_vec`` falls back to the process-global ``np.random`` unless
``seeded_rand_vec`` is set, and ``SawyerXYZEnv.reset`` calls ``reset_model()`` BEFORE
``super().reset()`` applies a seed — and calls it twice per reset, so each reset consumes two draws.
Setting ``seeded_rand_vec = True`` and reseeding ``np_random`` before every reset moves the draw onto
the env's own generator and reproduces it on **50/50**, which is what makes ``sample_states(rng=)``
regenerate the same corpus in a fresh process.

MEASURED (480 px, 64 states per task from the env's own pool, H200; ``scratch/metaworld/calibrate.py``)
-------------------------------------------------------------------------------------------------------

All 50 tasks, one process each:

=========================  =========  =========  =========
quantity                     median        min        max
=========================  =========  =========  =========
interior-exact              0.999700   0.980766   0.999830
all-pixel exact             0.879925   0.815260   0.883819
mean |diff| /255            0.7971     0.4816     1.1091
within 8/255                0.978300   0.961600   0.980700
f/s at batch 256               2,313      1,027      2,542
=========================  =========  =========  =========

against the official renderer's **251 f/s** warm render-only and **155 f/s** for the operation a
dataset consumer actually performs (restore the state, then render). So **9.2x at batch 256** on the
renderer alone and ~15x end to end; 1.6x at batch 1, 5.9x at 8, 8.8x at 64, and there is no crossover
— the composite is ahead of MuJoCo at every batch size including one. At 224 px the same code
measures interior min 0.988731 / median 0.999936 and **21.4x** at batch 256, because MuJoCo's raster
is nearly resolution-independent at these sizes (302 f/s at 224 against 251 at 480) while a
ray-caster is linear in pixels. The per-task table is in ``docs/research/metaworld.md`` §9.
"""

from __future__ import annotations

import numpy as np
import torch

from .._mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

# THE DEPENDENCY CHECK, at module scope, probing the FAILING PATH rather than the family's name.
# ``metaworld`` pins ``mujoco==3.3.0``; ``mujoco_warp`` needs ``mujoco >= 3.11``; and the shared
# interpreter this repo's suite runs in carries 3.9 with neither ``mujoco_warp`` nor ``warp``. So on
# that interpreter this family CANNOT render, and the only question is whether it fails LOUDLY HERE or
# quietly at the first frame — a lazy import inside ``render`` chose the second for Fetch and produced
# 300 failures for one missing wheel. Raising here removes exactly these 50 envs and puts the reason in
# ``stable_envs.envs.import_errors()``. The checks are ``find_spec`` and a version read rather than
# imports, because importing ``warp`` costs ~2 s and importing ``metaworld`` ~1 s on every
# ``import stable_envs``.
import importlib.util as _ilu

import mujoco as _mj

#: Where the 50-task measurement was taken. Built with ``python -m venv --system-site-packages`` so
#: that only ``mujoco`` is shadowed (3.11.0 over the shared 3.9) and ``metaworld``, ``gymnasium``,
#: ``torch``, ``triton`` and ``scipy`` stay shared. Meta-World runs unmodified against 3.11 —
#: its ``mujoco==3.3.0`` pin is a pin, not a requirement, and installing it with ``--no-deps`` into
#: this venv is what keeps the two families on one interpreter.
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
        f"{_VENV}; see docs/research/metaworld.md and the `metaworld` extra in pyproject.toml"
    )

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import mjwarp as MW
from ..spaces import Box, Dict

#: The 50 v3 tasks, in ``sorted(ALL_V3_ENVIRONMENTS)`` order. Written out rather than imported so that
#: ``import stable_envs`` does not import ``metaworld``; ``_build_bundle`` asserts the table against
#: the package's own dict the first time any task is built, so a Meta-World release that adds or
#: renames a task is an error here rather than a silently missing env.
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
#: The 14 model arrays R2 restores (``docs/research/metaworld.md`` §1). The state's model slice is a
#: subset of their ELEMENTS — the ones a reset moves AND that reach a pixel.
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
#: Of those, the ones ``mujoco_warp`` exposes as per-world batched ``Model`` fields (spec ``("*", ...)``).
#: ``site_rgba``/``site_matid`` are absent from its Model entirely, which costs nothing: MJWarp draws
#: no sites, so a site array is a plate-only quantity by construction.
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
#: Always given a per-world row, whether or not the episode writes them: ``body_pos``/``body_quat``
#: carry the per-episode placement of the MOVING scenery (a door's hinge body, a drawer's slide body),
#: and the rest are what a style leaf writes.
ALWAYS_BATCHED = ("body_pos", "body_quat", "geom_rgba", "mat_rgba", "light_diffuse", "cam_pos", "cam_quat")

HIDDEN_GROUP = MW.HIDDEN_GROUP
VISIBLE_GROUPS = MW.DEFAULT_GROUPS
#: Worlds per MJWarp render context. A context owns ``nworld * H * W * 12`` bytes, so this is the knob
#: that bounds memory; it is NOT the knob that saturates the ray-cast, which is done by 32 worlds
#: (``scratch/perf/prof_render.py --mode nworld``: 8,215 f/s at 32, 8,414 at 64, 8,600 at 256, 8,647 at
#: 1024). What it buys instead is fewer CHUNKS: every chunk pays two device synchronisations and a
#: pass of Python issuing the per-world writes, and at batch 1024 that was 16 of each. Measured end to
#: end at 256 px / batch 1024: **64 -> 7,300 f/s, 128 -> 7,661, 256 -> 7,782, 512 -> 7,850**, and on
#: Fetch 18,323 / 19,851 / 20,540 / 20,936. 256 rather than 512 because the last step is 1-2% and the
#: buffers double: at ``supersample=2`` a 256-world context already owns 805 MB and three are cached.
#: THE FRAME DOES NOT MOVE WITH IT, which is what makes this a free knob rather than a contract
#: change -- ``scratch/perf/chunk_invariance.py`` renders 512 states at 64, 128, 256 and 512 and gets
#: max 0/255 on 0 pixels, on Meta-World, Fetch and RoboCasa alike. (HumanoidBench is deliberately not
#: raised: there the chunk IS the pinned world count, and a batch of 1 would launch 256 worlds.)
MJW_CHUNK = 256
#: Resets used to discover which model elements the episode draw moves. Six draws is what
#: ``probe_state_gate.py`` needed to see every task's placement vary; more only costs reset time.
N_DRAWS = 6
#: Episodes in the state pool, and states taken from each. Several states per episode is deliberate —
#: it is what makes the plate cache hit, and what lets ``_check_plate_is_constant`` ask its question.
POOL_EPISODES, POOL_PER_EPISODE = 64, 4

_MODEL_CACHE: dict = {}
_RASTER_CACHE: dict = {}
_CTX_CACHE = _RES.BoundedCache(3, name="metaworld.ctx", on_evict=MW.release_cached_ctx)
_CTX_CACHE_MAX = 3
_PLATE_CACHE_MAX = 96
_PLATE_CACHE = _RES.BoundedCache(_PLATE_CACHE_MAX, name="metaworld.plate")
_POOL_CACHE: dict = {}


# --------------------------------------------------------------------------- constructing a task


def _make_task_env(task: str, size: int | None = None):
    """One Meta-World task, configured the way its benchmark wrapper configures it.

    ``_set_task_called`` because the env refuses to reset without a task; ``_freeze_rand_vec = False``
    because a frozen draw is the goal-hidden variant and not the env a dataset comes from;
    ``_partially_observable = False`` because that is what ``_create_observable_goal_envs`` does and it
    changes the observation, never a pixel. ``render_mode=None`` builds NO viewer and therefore no EGL
    context — the only construction that costs one is ``reference_env``.
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
    """A reset whose draw is REPRODUCIBLE — the third route in ``docs/research/metaworld.md`` §2.

    ``reset(seed=k)`` and ``env.seed(k)`` both fail on 50/50 tasks, because ``_get_state_rand_vec``
    reads the process-global ``np.random`` unless ``seeded_rand_vec`` is set and because
    ``reset_model()`` runs before ``super().reset()`` applies a seed. Moving the draw onto the env's own
    generator and reseeding that generator here reproduces the draw on 50/50. Note the reset consumes
    TWO draws (``reset_model`` is called once by ``SawyerXYZEnv.reset`` and once by
    ``MujocoEnv.reset``), which is deterministic and therefore not a problem — only surprising.
    """
    u = env.unwrapped
    u.seeded_rand_vec = True
    u._np_random = np.random.default_rng(int(seed))
    u._np_random_seed = int(seed)
    env.reset()


def _raster(task: str, res: int, m):
    """One ``mujoco.Renderer`` per (task, resolution), built once and NEVER closed.

    Releasing an EGL context corrupts rendering in every other live context in the process, silently;
    the rule and the measurement behind it are in ``stable_envs/testing/conformance.py``. The model's
    offscreen framebuffer is widened to the requested resolution first, because Meta-World's own is
    640x480 and ``Renderer`` validates against it.

    The model is passed in rather than looked up: this is called from inside ``_build_bundle``, before
    the bundle exists.
    """
    import mujoco

    # The model's identity is in the key. A ``mujoco.Renderer`` binds to ONE ``MjModel``, and this
    # cache outlives ``_MODEL_CACHE``; keyed on the task alone, anything that rebuilt a bundle (a probe
    # clearing the caches, a second interpreter-level import) would silently keep rendering the old
    # model's data through the new model's states. It presents as the plate-constancy assertion failing
    # at 255/255, which reads like a mover misclassification and is not one.
    key = (task, int(res), id(m))
    hit = _RASTER_CACHE.get(key)
    if hit is None:
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(res))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(res))
        hit = _RASTER_CACHE[key] = mujoco.Renderer(m, width=int(res), height=int(res))
    return hit


# --------------------------------------------------------------------------- the model bundle


class _Bundle:
    """Everything a task needs that depends on neither the state, the resolution nor the appearance."""

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
    """MuJoCo's own resolution order for a geom's colour: its ``rgba`` wins unless it is the default."""
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


#: How each of the 14 arrays' columns are labelled, so a state dimension's name says what it IS.
_COLS = {"pos": "xyz", "quat": "wxyz", "size": "xyz", "rgba": "rgba", "matid": ("",)}


def _slot_name(m, arr: str, f: int) -> str:
    """``body_pos.drawer.y`` — the array, the element it belongs to, and which number of it."""
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
    """Compile the task, find its per-episode state slice, its movers and its colour knobs.

    Everything here is MEASURED off the model and the env's own resets rather than tabulated: 50 tasks
    cannot be hand-checked, and a table that drifts from the package is worse than no table.
    """
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
            f"benchmark from the one docs/research/metaworld.md reports"
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

    # ---- the episode draw: which model elements move, and which of those reach a pixel ----------
    draws = []
    for k in range(N_DRAWS):
        _seeded_reset(env, k)
        mujoco.mj_forward(m, d)  # never read a frame, or a model, straight out of reset
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

    # ---- movers, static geometry, and the geoms MJWarp will not transform -----------------------
    weld = np.asarray(m.body_weldid, np.int64)
    mocapid = np.asarray(m.body_mocapid, np.int64)
    root = np.asarray(m.body_rootid, np.int64)
    gbody = np.asarray(m.geom_bodyid, np.int64)
    is_mover = (weld[gbody] != 0) | (mocapid[gbody] >= 0)
    b.movers = np.flatnonzero(is_mover).astype(np.int64)
    sbody = np.asarray(m.site_bodyid, np.int64)
    b.mover_sites = np.flatnonzero((weld[sbody] != 0) | (mocapid[sbody] >= 0)).astype(np.int64)
    # A geom's EFFECTIVE alpha, by MuJoCo's own resolution order. MJWarp's shader reads only
    # color[0..2] and packs a constant 255, so a translucent mover would be painted opaque; the
    # composite blends it instead (see ``render``). Meta-World's case is ``window_glass``,
    # rgba="0 .3 .4 .1", which is 8% of the frame on the two window tasks.
    b.alpha = MW.effective_alpha(m, "geom")
    b.translucent = np.array([g for g in b.movers if 0.0 < float(b.alpha[g]) < 1.0], np.int64)
    b.static_geoms = np.flatnonzero(~is_mover).astype(np.int64)
    # EXACTLY mujoco_warp's own early-out in ``_geom_local_to_global``: these are the geoms whose
    # world transform it never recomputes, so these are the rows that have to be uploaded per world.
    b.plate_static = np.array(
        [g for g in range(m.ngeom) if weld[gbody[g]] == 0 and mocapid[root[gbody[g]]] == -1], np.int64
    )

    # ---- the state's names ----------------------------------------------------------------------
    names = list(_qpos_names(m))
    if len(names) != b.nq:
        raise RuntimeError(f"{task}: named {len(names)} qpos entries for nq={b.nq}")
    names += [_slot_name(m, a, f) for a, f in b.slots]
    if len(set(names)) != len(names):
        raise RuntimeError(f"{task}: duplicate state names in {names}")
    b.state_names = tuple(names)

    # ---- appearance knobs, camera and lights ----------------------------------------------------
    b.cam_pos0 = np.array(m.cam_pos[cam], np.float64)
    b.cam_quat0 = np.array(m.cam_quat[cam], np.float64)
    b.light0 = np.array(m.light_diffuse, np.float64)
    b.casting = np.flatnonzero(np.asarray(m.light_castshadow, np.int64) != 0).astype(np.int64)
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tablelink")
    b.pivot = np.array(m.body_pos[tb], np.float64) if tb >= 0 else np.array([0.0, 0.6, 0.0], np.float64)
    b.knobs = _colour_knobs(task, b)
    b.mat_tex_mean = _mover_texture_means(m, b)

    # ---- which arrays the MJWarp model has to carry per world ------------------------------------
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
    # The candidates and what the liveness filter dropped, kept because they are the one claim in this
    # env that no downstream test can check: both sides pin the dropped elements to the same snapshot,
    # so an element wrongly called dead would be consistently wrong on OUR side and on the REFERENCE's
    # and would show up in neither the parity number nor the state-sufficiency check. It shows up here.
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
    """One label per ``qpos`` entry, from the model's own joint table."""
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
    """Does model element ``arr[f]`` change anything MJWarp has to draw for a moving body?"""
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
    """Of the elements a reset MOVES, the ones that reach a pixel. One MuJoCo render per candidate.

    An element that varies per episode and reaches no pixel must not be a state dimension —
    ``check_state_dims_matter`` would correctly call it dead — but it also must not be left free, or
    the frame would depend on the reset history. Both sides pin it: this bundle's ``pristine``
    snapshot is restored before every bake AND inside ``reference_render``, so the excluded elements
    are a constant of the env rather than of the episode.

    The probe is the same substitution ``check_state_dims_matter`` performs, so the two cannot
    disagree: take draw 0's model, substitute ONE element from another draw, and ask whether MuJoCo's
    own frame moved.
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
    """``(nmat, 3)`` multiplier for the materials a MOVER wears and MJWarp cannot texture.

    MuJoCo paints a textured surface as ``texture x rgba``: the material's colour is a MODULATION
    FACTOR, not the surface's colour. MJWarp is shown no cube texture at all here (§4, CUDA 700), so
    it draws that surface in ``mat_rgba`` alone — the factor with the thing it modulates thrown away.
    On ``door-close-v3`` that is (0.30, 0.32, 0.35) standing in for a texture whose mean is
    (0.577, 0.589, 0.624), i.e. **1.7x too bright over 4.7% of the frame**, which is most of that
    task's mean error.

    The flat draw is forced; the only choice is the constant, and the product is the constant that
    preserves the surface's mean. This is deliberately NOT ``mjwarp.bake_primitive_textures``, whose
    docstring forbids it on a parity path — that tool answers a different question (should a surface
    the backend COULD texture be flattened?) and its answer here would still be no.

    Static geometry is untouched by construction: it is baked through MuJoCo, textures and all.
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
    """The colour leaves, their targets and their OFFICIAL default, chosen by visible pixel count.

    Two rules make ``default == the official appearance`` an identity rather than a hope:

    * a knob owns one EFFECTIVE COLOUR, not a body. The Sawyer wears (0.5, 0.1, 0.1) on its arm and
      (0.2, 0.2, 0.2) on its pedestal, so a single ``robot.color`` applied to both would flatten one of
      them at its own default. The knob takes the colour class with the most pixels in a segmentation
      render of the scene, which also guarantees it is not inert.
    * a geom whose MATERIAL is shared with a geom outside the class is left out, because MuJoCo reads
      the geom's ``rgba`` and MJWarp reads the material's — recolouring through a shared material would
      repaint something the knob does not own, and recolouring only the geom would move the plate and
      not the ray-cast.
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
    # The two named materials every scene file shares (basic_scene{,_b,_c}.xml). Named rather than
    # inferred because "the biggest static colour class" is the table in 50/50 tasks and saying
    # ``table`` is only honest if it IS the table.
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
    """An MJWarp render context plus the device views a render writes through."""

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
    """One captured-graph MJWarp context per (task, resolution, worlds).

    GRAPH CAPTURE IS THE WHOLE GAME. Un-captured, these four kernels run at ~430 f/s — 1.4x the
    official renderer, not 30x — because at this scene size the frame is launch-bound. A throughput
    number near 1.4x means the capture is not happening, not that the GPU is slow.
    """
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
        # THE CAPTURE IS INSIDE THE PATCH, and that is the whole of it: ``mujoco_warp.render`` defines
        # its megakernel inside itself, so the source is regenerated per call from whatever
        # ``pack_rgba_to_uint32`` is bound then. Captured outside, the graph every later render replays
        # is the TRUNCATING kernel and this block reaches only the warm-up frame — measured at 21.2% of
        # this family's channels one level low (``scratch/perf/rounding.py``). See ``MW.gl_rounding``.
        #
        # ``specialised_lights`` IS OFF HERE, and it is off for the one reason that matters on a
        # parity family. It pays more on this scene than on any other in the repo -- three lights and
        # a headlight, ONE of which casts (``light_castshadow = [false, true, false]``), so stock
        # ``mujoco_warp`` compiles four any-hit BVH traversals and executes one, and deleting the
        # three dead ones measures **1.04-1.09x across 13 tasks** at 256 px / batch 1024 (e.g. reach
        # 7,777 -> 8,422, sweep-into 8,001 -> 8,694, assembly 7,633 -> 8,249). It is bit-identical on
        # 11 of those 13 and NOT on ``window-open`` and ``window-close``, where the raw megakernel
        # differs by 1/255 on **2 of 4,194,304 pixels** -- reproducibly, on three independent jobs,
        # against a stock-vs-stock control of 0 (``scratch/perf/lightspec_kernel.py``). The
        # segmentation buffer is identical, so it is not a different hit; it is ``fast_math``
        # reassociating the surviving light's arithmetic in a new inlining context.
        #
        # Two pixels in four million cannot move ``parity_interior_min``. That is not the test: this
        # family's claim is that its frame IS the frame, so it does not take a speedup that changes
        # one. ``STABLE_ENVS_LIGHTSPEC=1`` turns it on for a caller who has read the number above and
        # wants the 1.08x; RoboCasa and HumanoidBench, where the same elision is bit-identical on
        # every cell measured, take it by default.
        from ..render import mjwarp_lightspec as LS

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
            # VIEWS of the warp buffers. The channel swap is applied per render rather than here: doing
            # ``[..., [2,1,0]]`` at construction is advanced indexing, which COPIES, and the object
            # would then hold one frozen frame while every later render wrote into the buffer behind it.
            c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
            c.seg = wp.to_torch(c.rc.seg_data).reshape(int(nworld), int(res), int(res), 2)[..., 0]
            with wp.ScopedCapture() as cap:
                _frame(c)
            c.graph = cap.graph
        # THE MOVERS-OFF SWITCH, for the shadow pass. ``rc.enabled_geom_ids`` is the BVH's slot ->
        # geom table, read on every launch (including a graph replay, which reads device memory at
        # replay time rather than closing over its contents). Re-pointing each MOVER slot at a STATIC
        # geom removes the movers from the ray-cast and from every shadow ray while leaving the tree's
        # shape alone: one static surface is simply listed twice, which changes no hit and no colour.
        # There is no per-geom enable flag in mujoco_warp to use instead, and building a second context
        # would double the megakernel compile and the texture upload for a frame that differs from
        # this one only in which geoms exist.
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
    """Point every material away from a ``mjTEXTURE_CUBE`` slot, keeping ``mat_rgba`` and the skybox.

    ``mujoco_warp`` hard-faults with CUDA error 700 on a cube texture, reproduced on two unrelated
    model families (Franka Kitchen and all 50 Meta-World tasks) on 3.11.0 and on ``main``, with no
    upstream issue. It is only survivable for a design that DISCARDS those pixels: in the composite a
    static pixel comes from MuJoCo's bake, so MJWarp needs static geometry's depth and never its
    colour. Measured cost to the composite: none (§5).
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
    """MuJoCo's bake of one episode's static scene, in both lightings, plus its static kinematics.

    ``img`` is the scene as MuJoCo draws it. ``dark`` is the same scene with the SHADOW-CASTING lights'
    diffuse and specular zeroed, which is exactly what MuJoCo draws where those lights are fully
    occluded — a MuJoCo pixel is ``ambient + shadow * (diffuse + specular)`` per light — so the two
    bracket every static pixel and neither is an approximation of the other. ``off`` is MJWarp's own
    render of the same scene with the movers removed; a static pixel where MJWarp's movers-on frame
    differs from it is a pixel the movers shadow. All three are functions of the episode and the
    appearance and of no joint angle, so all three are cached together.
    """

    __slots__ = ("img", "dark", "off", "body_pos", "body_quat", "gxpos", "gxmat", "model")


# --------------------------------------------------------------------------- the env


class MetaWorldBase(Env):
    """Base for the 50 ids. ``TASK`` is set by the generated subclasses."""

    TASK: str = ""
    default_size = 480  # metaworld's own SawyerXYZEnv default
    default_supersample = 1
    #: EVERY variation leaf, not just the static ones. The plate is already keyed on the episode (see
    #: the module docstring), so an extra column in the key costs a cache slot and removes the whole
    #: "is this knob static?" question — which for a family of 50 differently-built scenes is a
    #: question with 50 different answers.
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
    #: MEASURED over all 50 tasks, 64 states each from the env's own pool, 480 px: interior-exact
    #: median 0.999700, min 0.980766 (``window-open-v3``), 48/50 above 0.99 and 50/50 above 0.98. The
    #: floor sits just below the WORST TASK rather than at a round number, so a state draw cannot flake
    #: the gate and a regression on any one task fails it. It is resolution-dependent and that is a
    #: property of the interior MASK, not of the renderer: at 224 px the same code measures min
    #: 0.988731 / median 0.999936, because a flat 3x3 neighbourhood is rarer there and less of the
    #: mesh-shading residual is counted. See ``parity_note``.
    parity_interior_min = 0.978
    #: MEASURED: min 0.815260 (``door-close-v3``), median 0.879925. Below the shared 0.90 bar for a
    #: reason that is about the SCENE and not the renderer: all-pixel exactness is the per-mover-pixel
    #: residual weighted by how much of the frame moves, and a Sawyer plus its manipulandum covers
    #: 12.5-16.7% of this frame against Fetch's 8.9%. Per mover pixel this port is the BETTER of the
    #: two (27.8% of the arm's interior pixels are bit-exact here; Fetch's arm is at ~0). The interior
    #: floor, which is the number that describes the renderer, is not relaxed by this.
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
        "docs/research/ogbench.md refuted three ways, and it bites in proportion to the mover share: "
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
        "non-mesh texture UVs in mujoco_warp -- the same upstream gap docs/research/adroit.md needs -- "
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
        # Public state: two keys, and they are what the two halves ARE rather than an
        # invention. ``qpos`` is the simulator's configuration; ``episode`` is what the task wrote
        # into the MODEL when the episode began (``_live_slots``), which is a different KIND of
        # quantity and is the whole subject of the widening this family performs. Assigned per
        # instance because both widths come from the compiled model. Every column is covered by one
        # of the two, so no offset is ``None``, ``_hidden_state`` is never consulted and
        # ``pack``/``unpack`` round-trip the frozen corpus bit for bit.
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

    # ---- the assumption the design rests on, checked rather than assumed --------------------

    def _check_plate_is_constant(self) -> None:
        """Within ONE episode, the frame with every mover hidden must not depend on ``qpos``.

        This is the weaker half of Fetch's check, and it is the half that is true here: the plate is
        keyed on the episode precisely BECAUSE the cross-episode version is false (0/50 tasks, drift
        54/255 over 1.46% of pixels even with the mover set closed under the kinematic subtree). What
        must hold, and is measured on 50/50 tasks in ``probe_backend2.py``, is that two states of the
        SAME episode bake the same plate — otherwise a mover was misclassified as scenery and the
        composite would carry it twice.
        """
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

    # ---- variation --------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS are the official picture.

        Colours are float Boxes in [0,1] rather than ``RGBBox`` for the reason the OGBench and Fetch
        ports give: MuJoCo's material colours are floats and a uint8 channel cannot hold one, so a
        uint8 default would shift a surface by a level and make "our defaults reproduce the reference"
        false.

        Two things are deliberately NOT knobs. The **skybox** is a texture and ``mujoco_warp`` uploads
        textures when the render context is built, so a per-value skybox means a render context per
        value — which is exactly the "do not scale context creation" rule this repo enforces. The
        **headlight** is baked into the megakernel as a compile-time static for the same reason.
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
        """The style leaves this task actually has — a task with no separable object has no knob."""
        have = set(self.variation_space.names())
        return tuple(p for p in self.PLATE_PATHS if p in have)

    def _camera(self, variation):
        """The model camera, rigidly orbited about the table. Zero deltas give the model's own pose.

        A model camera has a pose, not a (lookat, distance, azimuth, elevation), so the knobs are a
        rotation and a scale applied to it rather than a re-derivation: at the defaults the transform
        is the identity and ``cam_pos``/``cam_quat`` are the compiled model's own numbers to the last
        bit, which is what makes "the default appearance IS the official frame" an identity.
        """
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

    # ---- the plate --------------------------------------------------------------------------

    def _plate(self, res, over, state_row) -> _Plate:
        """The bake for one (appearance, EPISODE), cached.

        THE CACHE KEY IS NOT ``fetch.py``'s. There the plate is a function of the style alone, so the
        key is the style; here Meta-World writes the episode's placement into the model, the plate
        moves with it, and closing the mover set under the kinematic subtree narrows the cross-episode
        drift to 54/255 over 1.46% of pixels rather than removing it. So the episode — that is, the
        model slice of the STATE — is part of the key, and one MuJoCo render is paid per distinct
        episode. It is keyed on the exact float32 values rather than a rounded copy: two states whose
        placements differ in the last bit must not share a bake, or which of them baked it would depend
        on the order the batch arrived in.
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
        """MuJoCo's own raster of ONE episode's static scene, and its static kinematics.

        Every mutation is an absolute assignment from the pristine snapshot rather than an increment,
        so baking appearance B after A gives the same model as baking B first — ``camera.distance_scale``
        is a MULTIPLIER and would otherwise compound.
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
        # The same bake with the casting lights' diffuse and specular removed: the value MuJoCo gives
        # a static pixel that a mover fully shadows. Meta-World declares ONE casting light of three
        # (``basic_scene.xml``), which is what makes this a two-way choice rather than 2^k bakes — and
        # it is the same fact that makes the plate design work here and fail on Franka Kitchen.
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
        # THE MOVER COLOURS MJWarp READS COME FROM HERE, not from the context's model: the render
        # uploads this snapshot per world. So the texture-mean correction is applied here, where it
        # reaches the ray-cast and cannot reach the plate — the plate was drawn by MuJoCo above, with
        # its textures intact.
        if "mat_rgba" in p.model:
            p.model["mat_rgba"][:, :3] *= torch.as_tensor(b.mat_tex_mean, dtype=torch.float32, device=self.device)
        for a in MODEL_ARRAYS:
            getattr(m, a)[:] = b.pristine[a]
        return p

    # ---- render -----------------------------------------------------------------------------

    @torch.no_grad()
    def render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device. Style may differ per environment.

        The MJWarp pass is ONE call per chunk — colours, light, camera and the per-episode placement
        are all per-world fields, so nothing about a per-environment appearance or a per-environment
        EPISODE splits it. Only the baked plate is shared per (appearance, episode), and it is gathered
        per row afterwards.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
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

        # THE PLATE KEY IS THE APPEARANCE AND THE EPISODE. The model slice is part of the state, so
        # grouping on it is grouping on the episode; states from one episode share a bake. Exact
        # float32 equality, not a rounded key — see ``_plate``.
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
            # The plate is baked at a CANONICAL qpos, which is the invariant ``_check_plate_is_constant``
            # asserts: with every mover hidden the frame is a function of the episode and the
            # appearance, and of no joint angle.
            st_row = np.concatenate([b.qpos0[: b.nq], row[o:]]).astype(np.float32)
            plates.append(self._plate(res, over, st_row))
            plates1.append(plates[-1] if ss == 1 else self._plate(size, over, st_row))
        img = torch.stack([p.img for p in plates])
        dark = torch.stack([p.dark for p in plates])
        img1 = img if ss == 1 else torch.stack([p.img for p in plates1])
        dark1 = dark if ss == 1 else torch.stack([p.dark for p in plates1])
        grp = torch.as_tensor(inv, device=self.device, dtype=torch.long)
        # HOISTED OUT OF THE CHUNK LOOP, all four for the same reason: they are functions of the
        # PLATES and the STYLE, and the chunk boundary is a property of the render context. Inside the
        # loop each one was paid per chunk -- a device synchronisation and a host sort for the camera,
        # a rebuild of the (G, res, res, 3) stack for the movers-off gather, and five stacks of the
        # plates' model arrays. Measured with ``scratch/perf/prof_render.py`` at 256 px / batch 1024,
        # when the batch was still cut into sixteen chunks: 32 ms of a 162 ms render.
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
            # torch wrote the per-world arrays on its stream; warp replays the graph on its own. The
            # two syncs bracket the launch so a frame can never be rendered from half-written state,
            # which at these rates costs microseconds and is the difference between deterministic and
            # nearly deterministic.
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            # EVERYTHING DERIVED FROM THIS LAUNCH IS TAKEN NOW. Both passes below replay the graph and
            # overwrite ``c.bgra`` and ``c.seg`` in place, so a mask read afterwards describes a frame
            # in which the very geometry it is asking about has been removed. That is not a crash: the
            # alpha lookup silently returns 1.0 everywhere and the blend becomes a bit-identical no-op,
            # which is exactly how it read on first measurement.
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
                # A supersampled output pixel is shadowed only where EVERY subsample is: a partly
                # shadowed pixel is a shadow boundary, which is the one place the two shadow models
                # are entitled to disagree, and the bright plate is the better guess there.
                sh1 = ((rgb != off).any(-1) & ~mv).reshape(n, size, ss, size, ss).all(4).all(2)
                pl1 = torch.where(sh1.unsqueeze(-1), dark1[g], img1[g])
                buf[lo:hi] = torch.where(any_mv.unsqueeze(-1), small, pl1)
        return buf.view(*lead, size, size, 3) if lead else buf

    def _blend(
        self, c: _Ctx, rgb: torch.Tensor, plate: torch.Tensor, n: int, alpha: torch.Tensor | None
    ) -> torch.Tensor:
        """MuJoCo's own blend for a TRANSLUCENT mover: ``a * pane + (1-a) * WHAT IS BEHIND THE PANE``.

        ``mujoco_warp``'s shader takes ``base_color = vec3(color[0], color[1], color[2])`` and packs a
        constant 255, so it has no alpha path at all and paints a translucent geom opaque. Fetch's
        answer was to hide such geoms, because its four were 2 m crosshair bars at alpha 0.1 reaching
        0.000002 of interior pixels. Meta-World's is a WINDOW PANE — ``window_glass``,
        ``rgba="0 .3 .4 .1"`` on a MESH — and attributing the residual per geom makes it **56.9% of
        window-open-v3's total error on its own**: 1.56% of the frame at a mean of 88.1/255.

        THE SECOND LAYER HAS TO BE RENDERED, and that is measured rather than assumed. Blending the
        pane against the PLATE — the cheap version, no extra launch — was tried first and moves the
        wrong way (window-open mean 2.334 -> 2.418), because what is behind a window pane in these
        scenes is the rest of the window, which is a MOVER and is not in the plate. So the pane's own
        BVH slots are re-pointed at a static geom for one extra replay, which gives both the colour
        behind it and the segmentation that says whether that colour is a mover or the plate.

        Unlike the movers-off pass this one CANNOT be cached: what is behind the glass moves with
        qpos. It is therefore gated on the task having a translucent mover at all — 48 of the 50 take
        the ``return`` on the first line and pay nothing.
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
        """``(ngeom+1,)`` alpha per segmentation id, SHIFTED so that row 0 is the ray that hit nothing.

        Shifted rather than clamped so that one ``seg + 1`` serves this table and ``_mover_lut``'s. The
        miss row is opaque, which is unobservable: the alpha is read only where the mover mask is true.
        """
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("alpha_lut")
        if hit is None:
            a = np.concatenate([[1.0], np.asarray(self._b.alpha, np.float32)])
            hit = cache["alpha_lut"] = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        return hit

    def _mover_lut(self) -> torch.Tensor:
        """``(ngeom+1,)`` bool: is this segmentation id a mover? Indexed, not searched.

        ``torch.isin(seg, movers)`` on a (64, 256, 256) segmentation broadcasts the frame against the
        mover list, i.e. it touches the frame once per mover, and Meta-World's tasks carry 22-39 of
        them. A table lookup touches it once.
        """
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("mover_lut")
        if hit is None:
            lut = torch.zeros(int(self._b.model.ngeom) + 1, dtype=torch.bool, device=self.device)
            lut[self._idx("movers+1", np.asarray(self._b.movers, np.int64) + 1)] = True
            hit = cache["mover_lut"] = lut
        return hit

    def _camera_batch(self, sty, B: int):
        """``(B,3)``/``(B,4)`` camera pose for the WHOLE batch, on device, in ONE host round trip."""
        keys = ("camera.distance_scale", "camera.azimuth_delta", "camera.elevation_delta")
        camkey = torch.cat([sty[p].reshape(-1, 1) for p in keys], 1)[:B].cpu().numpy()
        uniq, inv = np.unique(np.round(camkey.astype(np.float64), 6), axis=0, return_inverse=True)
        poses = [self._camera({keys[0]: r[0:1], keys[1]: r[1:2], keys[2]: r[2:3]}) for r in uniq]
        pos = torch.as_tensor(np.stack([p for p, _ in poses])[inv], device=self.device, dtype=torch.float32)
        quat = torch.as_tensor(np.stack([q for _, q in poses])[inv], device=self.device, dtype=torch.float32)
        return pos, quat

    def _plate_stacks(self, plates) -> dict:
        """``(G, ...)`` stacks of everything ``_upload`` gathers per row, built once per render."""
        out = {
            "body_pos": torch.stack([p.body_pos for p in plates]),
            "body_quat": torch.stack([p.body_quat for p in plates]),
            "gxpos": torch.stack([p.gxpos for p in plates]),
            "gxmat": torch.stack([p.gxmat for p in plates]),
        }
        out["model"] = {a: torch.stack([p.model[a] for p in plates]) for a in plates[0].model}
        return out

    def _movers_off_stack(self, plates):
        """``(G, res, res, 3)`` of every plate's movers-off frame, or ``None`` if one is not baked yet.

        ``None`` is the FIRST render of a given episode set, where the frames do not exist and have to
        come out of an extra graph replay per chunk (``_movers_off``); from the second render on they
        are all on their plates and the gather is one stack and one index.
        """
        if not len(self._b.casting) or any(p.off is None for p in plates):
            return None
        return torch.stack([p.off for p in plates])

    def _layer_off(self, c: _Ctx, n: int, slots: torch.Tensor):
        """Replay the frame with ``slots`` removed from the BVH; returns ``(rgb, seg)`` copies.

        The one primitive behind both the shadow pass and the translucent pass. ``enabled_geom_ids``
        is the BVH's slot -> geom table and is read at replay time, so re-pointing a slot at a static
        geom removes that geometry from the ray-cast and from every shadow ray without rebuilding
        anything; the tree keeps its shape and one static surface is listed twice, which changes no hit.
        """
        try:
            c.bvh_ids[slots] = c.bvh_substitute
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            return c.bgra[:n][..., [2, 1, 0]], c.seg[:n].clone()
        finally:
            # ALWAYS, even on a failed launch: this table is the context's own, and a context left with
            # its movers pointed at the table renders every later frame without the robot.
            c.bvh_ids.copy_(c.bvh_ids0)
            torch.cuda.synchronize()

    def _movers_off(self, c: _Ctx, n: int, plates, g) -> torch.Tensor:
        """``(n,res,res,3)`` MJWarp's own render of the same scene WITHOUT the movers, per row.

        This is what tells the composite which static pixels a mover shadows, and it is the whole
        answer to the one thing a baked plate structurally cannot hold: hiding a geom to bake the plate
        hides its shadow too, and Meta-World has a casting light. Measured on ``reach-v3`` before and
        after, 8 states at 480 px: all-pixel exactness 0.8017 -> 0.8826, mean 2.396 -> 0.920/255,
        within-8 0.898 -> 0.978, and interior 0.99923 -> 0.99978. The shadow was 8.6% of the frame at
        a mean of 18.9/255 -- a visibly missing shadow, not a quantisation residual.

        It costs ONE extra graph replay, and only on the first render of a given episode+appearance at
        a given resolution: the movers-off frame is a function of the model and the lights and NOT of
        qpos, so it is cached on the plate. A rollout of 500 states through one episode pays it once.
        """
        if c.bvh_substitute < 0 or not len(self._b.casting):
            return c.bgra[:n][..., [2, 1, 0]]  # nothing casts: no pixel can be shadowed
        # Only the groups THIS chunk holds: a plate the chunk does not touch has no movers-off frame
        # and must not be stacked into the gather.
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
        """``(1, len(mats), 3)`` texture-mean multiplier for a knob's materials. Built once."""
        cache = self.__dict__.setdefault("_scale_cache", {})
        hit = cache.get(name)
        if hit is None:
            hit = cache[name] = torch.as_tensor(
                self._b.mat_tex_mean[np.asarray(mats, np.int64)], dtype=torch.float32, device=self.device
            ).reshape(1, -1, 3)
        return hit

    def _idx(self, key: str, values) -> torch.Tensor:
        """A device index tensor, built once per env. ``as_tensor`` per render is a host sync each."""
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
        """Write the chunk's state, episode and appearance into the context's per-world arrays."""
        b = self._b
        wp = c.wp
        n = hi - lo
        wp.to_torch(c.wd.qpos)[:n] = s[:, : b.nq]

        # The episode. body_pos/body_quat reach every body's ``xpos`` (a kinematics branch is an
        # ancestor chain, so static bodies are walked too); the geoms of a WELDED body are the ones
        # ``_geom_local_to_global`` returns early for, and those come from the plate's own CPU forward.
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

        # The appearance.
        gr = wp.to_torch(c.wm.geom_rgba)
        mr = wp.to_torch(c.wm.mat_rgba)
        for name, k in b.knobs.items():
            col = sty[f"{name}.color"][lo:hi]
            gr[:n, self._idx(f"{name}.geoms", k["geoms"]), :3] = col.unsqueeze(1)
            if len(k["mats"]):
                # ... times the texture mean, for the same reason and with the same effect as in
                # ``_bake``: a knob whose material MJWarp cannot texture must write the MODULATED
                # colour here, or its default would stop reproducing the official appearance on the
                # movers while still reproducing it on the plate.
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

    # ---- states -----------------------------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draws from a pool the OFFICIAL env produced: 64 seeded episodes, 4 states from each.

        Several states per episode rather than one, for two reasons that pull the same way: the plate
        is keyed on the episode, so a pool of one-state episodes would bake a plate per state and
        measure nothing about the cache; and ``_check_plate_is_constant`` needs two states that share
        an episode to ask its question at all.
        """
        rng = rng or self.rng
        pool = _state_pool(self.TASK)
        return pool[rng.integers(0, len(pool), int(batch))].copy()

    # ---- reference bridge -------------------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 480):
        """A FRESH official env with a viewer at ``size``. Never closed — see ``_raster``."""
        return _make_task_env(cls.TASK, size=int(size))

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for the state, with the model pinned, forwarded, and the state asserted.

        Three things this does that a naive loop does not, each of which is a measured trap:

        * it restores the bundle's PRISTINE snapshot of the 14 model arrays before writing the state,
          so the elements a reset varies WITHOUT reaching a pixel are the same constant here as in the
          plate. Without it the reference frame is a function of the reset history rather than of the
          state, which is what ``test_state_is_sufficient`` exists to catch.
        * it calls ``mj_forward`` AFTER writing the model, because ``_set_obj_xyz`` is what forwards
          inside ``reset_model`` and 28 of 50 tasks write ``body_pos``/``site_pos`` after their last
          call to it — so a frame taken straight out of ``reset`` is up to 169/255 away from the frame
          the state implies, and no stored state can reproduce it.
        * it asserts the round trip. If the reference did not reach the requested state, every parity
          number computed from it is comparing two different scenes.
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
    """Put ``(qpos, model slice)`` into a model/data pair, pinning everything the state does not name."""
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
    """``(POOL_EPISODES * POOL_PER_EPISODE, state_dim)`` from the official env's own reset distribution.

    Reset alone would be the wrong distribution twice over. The arm is placed by ``_reset_hand`` at the
    same pose in every episode, so a parity number computed from reset states says nothing about the
    seven joints that are most of the picture; and an object that has never been touched sits at the
    identity quaternion in every state, which makes ``check_state_dims_matter`` substitute a value it
    already has and report a live dimension as dead. So the pool rolls 0-25 random actions out of each
    reset — and yaws every free-joint object by a random angle, which is a state the env reaches the
    moment the gripper turns it and is what an offline dataset holds. The draw is pinned through
    ``_seeded_reset`` (route 3), so the pool is identical in a fresh process.
    """
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
