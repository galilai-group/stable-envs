"""Every optimisation that claims to be invisible, held to that claim: fast path == slow path, exactly.

Three files used to make this argument independently — ``test_fused_cast.py`` for the Triton ray cast,
``test_mover_window.py`` for the screen-space windows, ``test_raster_culling.py`` for the 2-D tile
binning — and the third one's docstring said "modelled on ``test_mover_window.py``", which is the usual
sign that three files are one file. They are one file now, because the claim is identical in all three
and only the switch differs:

    somewhere in this renderer there are two implementations of one function. One is the definition and
    one is the one that runs. The fast one is admissible ONLY because it is bit-identical to the slow
    one on the states the envs actually reach, so run both over the registry and diff the uint8 frames
    with a threshold of zero.

What each case is really guarding is different, and that is why each carries its own note below rather
than being folded into a single sentence: the ray cast is guarding Triton's *floating-point flags*, the
windows are guarding a *geometric* argument about what a pass can reach, and the culling is guarding a
claim about *bounding boxes* that is false for one primitive. A regression in any of them produces a
picture that is almost right, which is the kind no eye and no mean-absolute-difference catches.

WHAT MERGING DID NOT DO. No assertion moved. Every env list, batch size, output size, seed and skip
condition below is the one its original file used — the ray cast still sweeps 84 px and 256 px at batch
8, the windows still use batch 16, the culling still runs a second pass at ``supersample=2`` on batch 4
with its own seed, and the two properties that are NOT of this shape (an unknown geom code must be
refused rather than silently treated as a plane; a zero-area polygon must never be culled) are still
asserted, at the bottom, unchanged. What merging removed is three copies of "build an env, draw twice,
subtract" and three different opinions about what a missing dependency means — see ``_shared.py`` for
the second one, which is the reason this suite went red for four tests today over a Box2D nothing here
uses, and green over a DM Control family that was not there at all.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import assert_frames_identical, family_params, make_or_skip, resolve_family, sampled_states

import stable_envs as se
from stable_envs.render import raster2d as R2
from stable_envs.render import raycast3d as R3

#: Env-module names that legitimately explain an empty family. Used by ``resolve_family`` to tell
#: "this interpreter cannot import dm_control" (a skip) from "the sweep selected nothing" (a failure).
RAYCAST_MODULES = ("dmcontrol", "mujoco_locomotion", "ogbench", "maze2d")
MOVER_MODULES = ("dmcontrol", "mujoco_locomotion")


@contextlib.contextmanager
def _forced_backend(backend: str | None):
    """Pin ``raycast3d``'s dispatch for the duration, and put it back whatever happens.

    The pre-condition matters as much as the restore: ``FORCE_BACKEND`` is a module global, so a test
    that left it set would make every later comparison in the process compare a backend against
    itself and pass for the wrong reason.
    """
    assert R3.FORCE_BACKEND is None, (
        "raycast3d.FORCE_BACKEND was already pinned when this test started, so an earlier test leaked "
        "it — every backend comparison after that point compares one backend against itself"
    )
    try:
        R3.FORCE_BACKEND = backend
        yield
    finally:
        R3.FORCE_BACKEND = None


@contextlib.contextmanager
def _unbounded(cls):
    """Every window declines and every restricted ray set becomes the whole frame, for the duration.

    Emptying ``MOVER_TILES`` alone is not enough any more, and that is worth spelling out: an env whose
    movers span most of the image (finger, reacher) DECLINES every window on its own, so with only that
    switch both legs of the comparison would take the same code path and the test would pass without
    testing anything.
    """
    saved = {k: getattr(cls, k) for k in ("MOVER_TILES", "SHADOW_COMPACTION", "COVER_COMPACTION") if hasattr(cls, k)}
    cls.MOVER_TILES = ()
    for k in ("SHADOW_COMPACTION", "COVER_COMPACTION"):
        if k in saved:
            setattr(cls, k, False)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(cls, k, v)


def _scene(env, batch: int, seed: int):
    """The env's ``Scene2D`` for a batch of its own states, or ``None`` if it does not build one."""
    states = sampled_states(env, batch, seed)
    try:
        return env.scene(torch.as_tensor(states, device=env.device, dtype=torch.float32))
    except (NotImplementedError, TypeError, AttributeError):
        return None


def _raster_size(env):
    return (env.H, env.W) if hasattr(env, "H") and hasattr(env, "W") else env.size


def _classic_enabled() -> bool:
    try:
        from stable_envs.render import gfxdraw_classic as FC
    except ImportError:
        return False
    return bool(FC.HAVE_TRITON)


# ------------------------------------------------------------------ the cases


@dataclass(frozen=True)
class FastPath:
    """One optimisation, and everything needed to run it against the thing it replaces."""

    id: str
    #: What a difference would MEAN. Goes into the failure message, because "these differ by 3/255" is
    #: not actionable and "one of the windows is clipping a shadow the full frame draws" is.
    why: str
    #: Registered env names this optimisation is reachable from, and the modules whose absence excuses
    #: an empty list.
    names: tuple
    modules: tuple
    num_envs: int
    seed: int
    #: Output sizes to build the env at; ``(None,)`` means "the env's own size".
    sizes: tuple
    #: ``(env) -> payload``, computed ONCE and handed to both legs. This is not a convenience: the
    #: claim is that two renderings of ONE input agree, and re-deriving the input per leg would make a
    #: nondeterministic ``sample_states`` or ``scene()`` show up here as a culling difference, which is
    #: the wrong file to go looking in. Returning ``None`` means the env cannot exercise this case.
    prepare: Callable
    #: ``(env, payload, fast: bool) -> uint8 frames``. Called twice per case, on the same payload.
    frames: Callable
    #: ``(env) -> str | None``. A reason this env cannot exercise the optimisation, or None.
    inapplicable: Callable = field(default=lambda env: None)
    #: Extra gate on the whole case, evaluated once at collection.
    enabled: Callable = field(default=lambda: True)
    disabled_reason: str = ""


def _states_prepare(seed: int):
    return lambda env: sampled_states(env, env.B, seed)


def _raycast_frames(env, states, fast: bool):
    with _forced_backend("triton" if fast else "torch"):
        return env._render(states).clone()


def _mover_frames(env, states, fast: bool):
    if fast:
        return env._render(states).clone()
    with _unbounded(type(env)):
        return env._render(states).clone()


@contextlib.contextmanager
def _composite_compact_merge(on: bool):
    """Pin ``raycast3d.COMPOSITE_COMPACT_MERGE`` for the duration, and put it back whatever happens."""
    assert R3.COMPOSITE_COMPACT_MERGE is True, (
        "raycast3d.COMPOSITE_COMPACT_MERGE was already off when this test started, so an earlier "
        "test leaked it and every comparison after that point has been the window path against "
        "itself"
    )
    try:
        R3.COMPOSITE_COMPACT_MERGE = on
        yield
    finally:
        R3.COMPOSITE_COMPACT_MERGE = True


def _compact_merge_frames(env, states, fast: bool):
    with _composite_compact_merge(fast):
        return env._render(states).clone()


@contextlib.contextmanager
def _composite_compaction(on: bool):
    """Pin ``raycast3d.COVER_COMPACTION`` for the duration, and put it back whatever happens."""
    assert R3.COVER_COMPACTION is True, (
        "raycast3d.COVER_COMPACTION was already off when this test started, so an earlier test leaked "
        "it and every comparison after that point has been the slow path against itself"
    )
    try:
        R3.COVER_COMPACTION = on
        yield
    finally:
        R3.COVER_COMPACTION = True


def _composite_frames(env, states, fast: bool):
    """RAYCAST-2b's claim, exercised on the path where the switch is still live.

    ``COMPOSITE_COMPACT_MERGE`` is pinned OFF here on purpose. RAYCAST-6's composite compacts per
    ROW rather than over the union, so it never reads ``COVER_COMPACTION`` at all -- and with the
    default left on, this case would flip a flag the running code ignores and compare the compact
    path with itself. A green light for that reason is worse than no case: the union compaction is
    still what ``_composite_window`` does, ``_composite_window`` is still what
    ``COMPOSITE_COMPACT_MERGE=False`` selects, and that is where the claim has to keep holding.
    """
    with _composite_compact_merge(False), _composite_compaction(fast):
        return env._render(states).clone()


CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")


def _classic_frames(env, states, fast: bool):
    """``render`` with the fused per-pixel kernel on, then off. One state list, two spellings."""
    was = env.use_fused_render
    try:
        env.use_fused_render = bool(fast)
        return env._render(states).clone()
    finally:
        env.use_fused_render = was


def _classic_prepare(seed: int, extreme: bool):
    """The frozen corpus, or states chosen to put every primitive in its awkward case.

    ``extreme`` is not "more random". Each row below is a case the per-primitive path and the fused
    one could disagree on and nothing else would notice: a cart driven off the surface exercises
    SDL_gfx's line clipping, which restarts the Wu accumulator from a clipped endpoint; an angle
    many turns from zero exercises the truncation of a large float to Sint16; a MountainCar position
    outside the track puts the car's box entirely off-screen, where the fill's span clamps do the
    work; and ``|torque| > 2`` drives Pendulum's sprite index to the end of its baked table.
    """

    def prepare(env):
        import numpy as np

        n = env.B
        if not extreme:
            import pathlib

            npz = (
                pathlib.Path(__file__).resolve().parents[2]
                / "tests/fixtures/regression/join/step4/states"
                / f"{type(env).__name__}.npz"
            )
            if npz.is_file():
                pool = np.load(npz)["states"]
                return np.asarray(pool[np.arange(n) % len(pool)], np.float32)
            return sampled_states(env, n, seed)
        rng = np.random.default_rng(seed)
        wide = {
            "CartPole": lambda: np.stack([rng.uniform(-8, 8, n), rng.uniform(-15, 15, n)], 1),
            "Pendulum": lambda: np.stack([rng.uniform(-40, 40, n), rng.uniform(-6, 6, n)], 1),
            "Acrobot": lambda: rng.uniform(-40, 40, (n, 2)),
            "MountainCar": lambda: rng.uniform(-5, 4, (n, 1)),
            "MountainCarContinuous": lambda: rng.uniform(-5, 4, (n, 1)),
        }
        gen = wide.get(type(env).__name__)
        return None if gen is None else np.asarray(gen(), np.float32)

    return prepare


@contextlib.contextmanager
def _primitive_bounds(on: bool):
    """Pin ``raycast3d.PRIMITIVE_BOUNDS`` for the duration, and put back whatever it was.

    Both legs are pinned explicitly, so the comparison is real whatever the module default is -- and
    the default MOVES: RAYCAST-4 shipped it on, RAYCAST-5 turned it off while the maze regression was
    open. Restoring a hard ``True`` here (as this did) would have quietly turned the bound back on for
    every later test in the process the moment the default changed.
    """
    was = R3.PRIMITIVE_BOUNDS
    try:
        R3.PRIMITIVE_BOUNDS = on
        yield
    finally:
        R3.PRIMITIVE_BOUNDS = was


def _bounds_frames(env, states, fast: bool):
    """Both legs through the TORCH cast: the bound is the CPU path and the fused kernel has no bound."""
    with _forced_backend("torch"), _primitive_bounds(fast):
        return env._render(states).clone()


def _scene_prepare(seed: int):
    return lambda env: _scene(env, env.B, seed)


def _cull_frames_factory(supersample):
    """``cull=True`` against ``cull=False`` on ONE ``Scene2D``.

    Goes through ``render_triton`` rather than ``env.render`` on purpose: culling is a property of the
    rasterizer and the envs that reach it do so through a ``Scene2D``, so driving it directly is what
    lets the sweep cover every registered scene rather than only the ones whose ``render`` happens to
    route through the default backend.
    """

    def frames(env, scene, fast: bool):
        ss = env.ss if supersample is None else supersample
        return R2.render_triton(scene, _raster_size(env), supersample=ss, cull=fast).clone()

    return frames


CASES: tuple[FastPath, ...] = (
    FastPath(
        id="raycast3d-fused-vs-torch",
        why=(
            "``raycast3d`` keeps two implementations of one function on purpose — "
            "``cast_nearest_torch`` is the readable definition and the Triton kernel is the one that "
            "runs — and every env in the PARITY tier is allowed the fast one only because the two "
            "agree to the last bit. Triton's DEFAULT fp32 lowering does NOT agree (FMA contraction, "
            "``div.full``, ``sqrt.approx``: 0.7-2.1 M of 4.2 M random values differ per shape), so "
            "what this really guards is that ``enable_fp_fusion=False`` / ``div_rn`` / ``sqrt_rn`` "
            "are all still in raycast3d._tri_isect and _launch. Drop any one and this fails. The "
            "comparison is on the WHOLE uint8 frame rather than on an isolated cast because the "
            "cast's output feeds a shadow test and a nearest-hit composite, and a one-ulp difference "
            "only matters once it reaches a pixel."
        ),
        names=tuple(
            n
            for n in sorted(se.registry.REGISTRY)
            if n.startswith(("DMC", "MuJoCo", "AntMaze", "PointMaze", "HumanoidMaze", "Maze2d"))
        ),
        modules=RAYCAST_MODULES,
        num_envs=8,
        seed=0,
        sizes=(84, 256),
        prepare=_states_prepare(0),
        frames=_raycast_frames,
        enabled=lambda: R3.HAVE_TRITON,
        disabled_reason="raycast3d has no Triton build in this interpreter, so there is no fused path",
    ),
    FastPath(
        id="mover-window-vs-full-frame",
        why=(
            "The 3-D renderers bound four passes rather than running them over the frame: the primary "
            "mover cast and the shadow cast to a window around the movers (``_mover_idx``), the "
            "reflection to a window around the movers' MIRROR IMAGES (``_mirror_idx``), and the "
            "shadow-from-static test to the samples where the lit and unlit plates actually differ "
            "(``_shadow_candidates``). Each is claimed to be conservative — outside it the pass "
            "provably cannot change a pixel — and each claim is a different argument. A difference "
            "means one of them is clipping geometry, a shadow or a reflection the full frame draws. "
            "Deliberately NOT through the official renderer: the reference path needs ``dm_control``, "
            "whose EGL teardown aborts the interpreter on this cluster, and the property is a "
            "property of our own two code paths. Parity against MuJoCo is test_conformance.py's job."
        ),
        names=tuple(n for n in sorted(se.registry.REGISTRY) if n.startswith(("DMC", "MuJoCo"))),
        modules=MOVER_MODULES,
        num_envs=16,
        seed=0,
        sizes=(84, 256),
        prepare=_states_prepare(0),
        frames=_mover_frames,
        inapplicable=lambda env: None if hasattr(env, "_mover_idx") else "does not use the windowed mover path",
    ),
    FastPath(
        id="raycast3d-primitive-bounds",
        why=(
            "``raycast3d``'s torch cast tests a primitive only against the rays whose line passes "
            "within that primitive's own bounding sphere. The claim is that a ray outside the "
            "INFLATED bound misses within the intersector's own rounding budget (k = 16 ulp) -- not "
            "in exact arithmetic, because _capsule in float32 at 100 m reports hits outside the "
            "primitive's true bound -- so intersect returns INF there and loses the ``t < best`` "
            "test, and skipping it cannot change t, the normal or the winning index. A difference "
            "means a bound is too SMALL somewhere, and the pixels it drops are on a silhouette "
            "where nothing else looks. Both legs are forced through the torch backend because the bound "
            "is the CPU path: the fused Triton kernel keeps every primitive in registers and has no "
            "bound to switch off, which is also why the GPU digests cannot move. The unit-ray "
            "invariant the bound rests on is checked at runtime, so a bundle that violated it would "
            "take the unbounded path and this case would pass vacuously -- "
            "tests/unit/test_raycast3d_primitive_bounds.py is what stops that being silent."
        ),
        names=tuple(
            n
            for n in sorted(se.registry.REGISTRY)
            if n.startswith(("DMC", "MuJoCo", "AntMaze", "PointMaze", "HumanoidMaze"))
        ),
        modules=RAYCAST_MODULES,
        num_envs=8,
        seed=0,
        # THE ENV'S OWN SIZE, not 84. This case first ran at 84 px only and passed on every env while
        # the CPU replay of the same change moved a pixel on HumanoidmazeGiantNavigate at its
        # certified 224 px. The bound's failure mode is a long lever arm -- a far camera and a small
        # primitive -- and building a giant maze at 84 px shortens exactly the thing under test.
        sizes=(None,),
        prepare=_states_prepare(0),
        frames=_bounds_frames,
    ),
    FastPath(
        id="composite-cover-compaction",
        why=(
            "``raycast3d.composite`` shades only the rays that landed on a mover, as the UNION over "
            "the batch, instead of every ray of the screen window. The window is a rectangle around "
            "the movers' projected bound and the mover inside it is a sphere or a small articulated "
            "body, so at 224 px / ss=2 that was ~45 elementwise passes and four material gathers "
            "running over eight times more samples than can be visible, with the result thrown away "
            "by a ``where``. The claim is that restricting the SET cannot change a VALUE: every "
            "ray's colour is a function of that ray alone, and at a position no row covers the "
            "composite takes the plate whatever the shader would have said. A difference means that "
            "argument is wrong somewhere -- a covered ray outside the union, or a shaded value that "
            "is not +0.0 where it is dropped -- and it would show up as a handful of pixels on the "
            "agent's silhouette, which is exactly what no mean-absolute-difference catches. The "
            "union is over ray POSITIONS, not pixels, because the window moves per environment."
        ),
        names=tuple(n for n in sorted(se.registry.REGISTRY) if n.startswith(("AntMaze", "PointMaze", "HumanoidMaze"))),
        modules=("ogbench", "maze2d"),
        num_envs=8,
        seed=0,
        sizes=(None,),
        prepare=_states_prepare(0),
        frames=_composite_frames,
        inapplicable=lambda env: (
            None
            if type(env).__module__.rsplit(".", 1)[-1] in ("ogbench", "maze2d")
            else "does not render through raycast3d.composite"
        ),
    ),
    FastPath(
        id="composite-compact-merge",
        why=(
            "``raycast3d.composite`` sizes its window from the BOUND instead of from the ``tile`` "
            "ceiling the caller passes, shades the covered (row, sample) pairs instead of their "
            "union over the batch, and gathers/blends/scatters only the output pixels some row "
            "covers. Three claims, all about values rather than tolerances: a ray outside the "
            "projected span cannot reach a mover, so it gets ``t = INF`` and the composite writes "
            "back the plate byte the broadcast already wrote; ``shade`` is elementwise in every "
            "argument, so shading a gathered subset returns what the superset held there; and at "
            "an output pixel with ``covf == 0`` the blend is ``patch * 1.0 + 0.0`` on a float that "
            "came out of a uint8, which ``round`` returns and ``clamp`` cannot move. A difference "
            "means one of those is wrong -- most likely the span bound is not conservative, and "
            "the symptom would be a clipped edge of the agent where the old 32-pixel window had "
            "slack and the fitted one does not. ``tests/unit/test_raycast3d_composite_compact.py`` "
            "is the differential fuzz; this case is the same claim on the environments, at their "
            "own sizes, because the span's lever arm is a far camera and a small primitive and "
            "building a giant maze at 84 px shortens exactly the thing under test."
        ),
        names=tuple(n for n in sorted(se.registry.REGISTRY) if n.startswith(("AntMaze", "PointMaze", "HumanoidMaze"))),
        modules=("ogbench", "maze2d"),
        num_envs=8,
        seed=1,
        sizes=(None,),
        prepare=_states_prepare(1),
        frames=_compact_merge_frames,
        inapplicable=lambda env: (
            None
            if type(env).__module__.rsplit(".", 1)[-1] in ("ogbench", "maze2d")
            else "does not render through raycast3d.composite"
        ),
    ),
    FastPath(
        id="raster2d-tile-culling",
        why=(
            "``raster2d`` skips a primitive whose world-space bounding box misses the pixel tile a "
            "program owns. That is a claim about geometry — a box is a BOUND, so nothing outside it "
            "can be painted — and a difference means a primitive paints outside its own box; see "
            "``_bbox_kernel`` on collinear polygons. Rather than reason about which envs can produce "
            "such a polygon, both paths are run over EVERY registered scene. What this can NOT "
            "separate, and does not claim to: the two paths are two COMPILATIONS of one kernel and "
            "ptxas may contract a multiply-add in one and not the other, which flips a pixel whose "
            "centre falls within ~3e-5 world units of the edge deciding ``sdf <= 0``. Measured at 600 "
            "px, batch 1024 over 368 M pixels that is 3 pixels on intersection-v0 and 1 on "
            "racetrack-v0, and the UNCULLED path disagrees with itself at another block size on "
            "exactly those pixels. The batches here are small enough that the coincidence does not "
            "arise."
        ),
        names=tuple(sorted(se.registry.REGISTRY)),
        modules=(),
        num_envs=8,
        seed=0,
        sizes=(None,),
        prepare=_scene_prepare(0),
        frames=_cull_frames_factory(None),
    ),
    FastPath(
        id="gfxdraw-classic-fused-vs-per-primitive",
        why=(
            "``render/gfxdraw.py``'s per-primitive path is the executable specification of six "
            "measured SDL_gfx rules, and ``render/gfxdraw_classic.py`` is a rewrite of the same "
            "rules into one closed form per pixel. The rewrite is admissible only while the two "
            "agree to the bit, and the places they can stop agreeing are specific: the order two "
            "edges of a polygon write a shared vertex (a scatter reads every destination before "
            "writing any, a register walk cannot), the two passes of the aacircle fringe, the "
            "fringe-before-fill order inside one primitive, SDL's blit rule for Pendulum's sprite "
            "against SDL_gfx's pixel writer, and the 16.16 scanline fill's span clamps. Every one "
            "of those is worth a handful of pixels, which is why the bar is zero."
        ),
        names=CLASSIC,
        modules=(),
        num_envs=8,
        seed=0,
        sizes=(None,),
        prepare=_classic_prepare(0, extreme=False),
        frames=_classic_frames,
        inapplicable=lambda env: (
            None if getattr(env, "fused_capable", False) else "no fused kernel describes this frame"
        ),
        enabled=lambda: _classic_enabled(),
        disabled_reason="triton is unavailable, so there is no fused path to compare against",
    ),
    FastPath(
        id="gfxdraw-classic-fused-vs-per-primitive-extreme",
        why=(
            "The same claim on states the corpus does not reach. The fused path's windows and its "
            "clipped edge table are BOUNDS; a corpus chosen for coverage of the reachable set "
            "cannot test a bound. These states put the cart off the surface, the angles many turns "
            "from zero, the car outside the track and the torque past the end of its sprite table."
        ),
        names=CLASSIC,
        modules=(),
        num_envs=8,
        seed=11,
        sizes=(None,),
        prepare=_classic_prepare(11, extreme=True),
        frames=_classic_frames,
        inapplicable=lambda env: (
            None if getattr(env, "fused_capable", False) else "no fused kernel describes this frame"
        ),
        enabled=lambda: _classic_enabled(),
        disabled_reason="triton is unavailable, so there is no fused path to compare against",
    ),
    FastPath(
        id="raster2d-tile-culling-ss2",
        why=(
            "The same claim at ``supersample=2``, where a program owns four times as many sample "
            "points as pixels. The tile box is grown to whole pixels for exactly this reason, and "
            "getting that wrong shows up ONLY at ``ss > 1`` — so the default supersample cannot be "
            "the only setting under test. The tile's world box must cover every subsample the program "
            "evaluates, not just the pixel centres."
        ),
        names=tuple(sorted(se.registry.REGISTRY)),
        modules=(),
        num_envs=4,
        seed=1,
        sizes=(None,),
        prepare=_scene_prepare(1),
        frames=_cull_frames_factory(2),
        inapplicable=lambda env: (
            None
            if (_raster_size(env) if isinstance(_raster_size(env), int) else max(_raster_size(env))) <= 512
            else "supersampling a frame this large is not worth the test time"
        ),
    ),
)

CASES_BY_ID = {c.id: c for c in CASES}


def _matrix():
    """``(case id, env name, size)`` for every case, with an EMPTY marker where a family is empty."""
    out = []
    for c in CASES:
        for name in family_params(c.names, family=c.id, modules=c.modules):
            for size in c.sizes:
                sid = "native" if size is None else f"{size}px"
                out.append(pytest.param(c.id, name.values[0], size, id=f"{c.id}-{name.id}-{sid}"))
    return out


@pytest.mark.parametrize("case_id,name,size", _matrix())
def test_the_fast_path_is_bit_identical_to_the_path_it_replaces(case_id, name, size):
    """One optimisation, one env, one output size: draw it both ways and subtract.

    The threshold is zero and it is not negotiable. These are not two renderers that ought to agree;
    they are two spellings of one, and the fast spelling is only admissible because the two are the
    same arithmetic. A tolerance here would be a licence for the exact regression the case exists to
    catch, since every one of them fails by a handful of pixels rather than by a visible amount.
    """
    case = CASES_BY_ID[case_id]
    resolve_family(name, family=case.id, modules=case.modules, subject="the fast/slow agreement check")
    if not case.enabled():
        pytest.skip(f"{case.id}: {case.disabled_reason}")

    kw = {"num_envs": case.num_envs, "device": "cuda"}
    if size is not None:
        kw["size"] = size
    env = make_or_skip(name, **kw)
    reason = case.inapplicable(env)
    if reason:
        pytest.skip(f"{name}: {reason}")

    payload = case.prepare(env)
    if payload is None:
        pytest.skip(f"{name} does not build the input {case.id} switches on (no Scene2D)")

    # ONE payload, two renderings of it. See FastPath.prepare on why this is not a convenience.
    fast = case.frames(env, payload, True)
    slow = case.frames(env, payload, False)
    at = "" if size is None else f" @{size}px"
    assert_frames_identical(fast, slow, what=f"{name}{at} [{case.id}]", why=case.why)


# ------------------------------------------------------------------ the two that are not of that shape


def test_an_unknown_geom_code_is_refused_by_both_cast_backends():
    """The kernel's last branch is the PLANE, so an unrecognised code must not silently BE one.

    Asserted on BOTH backends, because a fallthrough is exactly the kind of difference the frame-level
    comparison above cannot see: the envs in the sweep never pass an unsupported code, so a torch path
    that raises and a Triton path that quietly rasterises a plane would agree on every frame this file
    draws and diverge on the first env that adds an ellipsoid.
    """
    if not R3.HAVE_TRITON:
        pytest.skip("raycast3d has no Triton build in this interpreter, so there is one path only")
    dev = "cuda"
    o = torch.zeros(1, 8, 3, device=dev)
    d = torch.nn.functional.normalize(torch.randn(1, 8, 3, device=dev), dim=-1)
    sz = torch.ones(1, 1, 3, device=dev)
    pos = torch.zeros(1, 1, 3, device=dev)
    R = torch.eye(3, device=dev).view(1, 1, 3, 3)
    for backend in ("torch", "triton"):
        with pytest.raises(ValueError, match="no intersector"):
            R3.cast_nearest([8], sz, pos, R, o, d, backend=backend)  # 8 = mjGEOM_SDF


def test_the_ellipsoid_twins_are_bit_identical_on_both_casts():
    """``_ellipsoid`` (torch) and the ``kind == 4`` branch of ``_tri_isect`` are two spellings of one
    function, admissible together only because they agree to the bit -- the argument the module makes
    for every other kind. The ellipsoid kind exists for DM Control's quadruped torso; this is the
    synthetic-scene check (rotated, anisotropic, some degenerate, some with the origin inside), and
    the env-level one is the ``raycast3d-fused-vs-torch`` case above once a cell carries an ellipsoid
    through ``composite``."""
    if not R3.HAVE_TRITON:
        pytest.skip("raycast3d has no Triton build in this interpreter, so there is one path only")
    dev = "cuda"
    g = torch.Generator().manual_seed(7)
    B, K, N = 3, 6, 4096
    kinds = [R3.ELLIPSOID, R3.SPHERE, R3.ELLIPSOID, R3.CAPSULE, R3.ELLIPSOID, R3.BOX]
    sizes = torch.rand(B, K, 3, generator=g) * 0.4 + 0.02
    sizes[1, 0] = 0.0  # a degenerate one
    pos = (torch.rand(B, K, 3, generator=g) - 0.5) * 1.5
    q = torch.randn(B, K, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).view(B, K, 3, 3)
    o = (torch.rand(B, N, 3, generator=g) - 0.5) * 0.2 + torch.tensor([0.0, -2.0, 0.4])
    o[2] = pos[2, 0].view(1, 3) + (torch.rand(N, 3, generator=g) - 0.5) * 0.01  # inside one
    d = torch.nn.functional.normalize((torch.rand(B, N, 3, generator=g) - 0.5) + torch.tensor([0.0, 1.0, 0.0]), dim=-1)
    args = [t.contiguous().to(dev) for t in (sizes, pos, R, o, d)]
    ft, fn, fi = R3.cast_nearest(kinds, *args, backend="triton")
    st, sn, si = R3.cast_nearest(kinds, *args, backend="torch")
    assert torch.equal(ft, st) and torch.equal(fi, si), "ellipsoid: fused and torch casts disagree"
    hit = st < R3.INF
    assert torch.equal(fn[hit], sn[hit]), "ellipsoid: fused and torch normals disagree on hits"
    lim = torch.full((B, N), 3.0, device=dev)
    fb = R3.any_blocking(kinds, *args, lim, backend="triton")
    sb = R3.any_blocking(kinds, *args, lim, backend="torch")
    assert torch.equal(fb, sb), "ellipsoid: fused and torch any_blocking disagree"


def test_a_flat_polygon_is_not_culled():
    """A polygon with no interior paints an unbounded set, and must therefore be left alone.

    This is the one case where the bounding box is not a bound, and it is asserted directly rather than
    hoped for: collinear vertices make every edge normal parallel, so ``max`` over the edge planes
    ``<= 0`` is the whole LINE through them, which reaches pixels arbitrarily far from the vertex hull.
    No registered env is known to produce one, which is why the registry sweep above cannot stand in
    for this.

    The line is placed at y = 32.5 rather than at a whole number, and that is the test working rather
    than a detail. A pixel samples at its centre, so a line on an integer world coordinate is hit by no
    sample at all and the polygon paints nothing — which would make this pass for the wrong reason. At
    64 world units across 64 pixels the centres ARE the half-integers.
    """
    from stable_envs.render.scene import KIND_POLY, Scene2D

    dev = "cuda"
    z = lambda *a: torch.tensor(a, dtype=torch.float32, device=dev).view(1, 1, -1)  # noqa: E731
    sc = Scene2D(
        kind=torch.full((1, 1), KIND_POLY, dtype=torch.int32, device=dev),
        geom=z(10.0, 32.5, 20.0, 32.5, 20.0, 32.5, 10.0, 32.5),
        fill=z(255.0, 0.0, 0.0),
        line=z(255.0, 0.0, 0.0),
        line_w=torch.zeros((1, 1), device=dev),
        background=torch.zeros((1, 3), device=dev),
        world=(0.0, 0.0, 64.0, 64.0),
    )
    full = R2.render_triton(sc, 64, cull=False).clone()
    culled = R2.render_triton(sc, 64, cull=True)
    assert int(full[..., 0].sum()) > 0, "the flat polygon painted nothing, so this proves nothing"
    assert torch.equal(full, culled), (
        "a collinear polygon was culled by its bounding box, but its halfplane test is unbounded — "
        "_bbox_kernel must mark a zero-area polygon uncullable"
    )


def test_every_declared_fast_path_is_actually_reachable():
    """Each case above must name at least one thing this build can switch off. No dead cases.

    A case whose switch was renamed still collects, still runs, and still passes — it just compares the
    fast path against itself. That is the failure mode this whole file is written against, one level
    up, so it is asserted rather than assumed: the ray-cast case needs ``FORCE_BACKEND``, the window
    case needs at least one env carrying ``MOVER_TILES``, and both culling cases need
    ``render_triton`` to take a ``cull`` argument.
    """
    import inspect

    assert hasattr(R3, "FORCE_BACKEND"), (
        "raycast3d.FORCE_BACKEND is gone, so the fused-vs-torch case cannot pin either backend and has "
        "been comparing the default against itself"
    )
    # NOT "is True": the default is a release decision that moves, and pinning it here would make
    # this assertion fail for a reason that has nothing to do with reachability. What the case needs
    # is that the switch EXISTS and that turning it on actually reaches the bound -- i.e. that at
    # least one kind is bounded. A `_BOUNDED_KINDS` emptied by a future measurement would make the
    # case compare the unbounded path against itself, and that is the failure worth catching.
    assert isinstance(getattr(R3, "PRIMITIVE_BOUNDS", None), bool), (
        "raycast3d.PRIMITIVE_BOUNDS is gone, so the per-primitive-bound case cannot switch the bound "
        "off and has been comparing the bounded path against itself"
    )
    assert getattr(R3, "_BOUNDED_KINDS", frozenset()), (
        "raycast3d._BOUNDED_KINDS is empty, so PRIMITIVE_BOUNDS=True bounds NOTHING and the "
        "per-primitive-bound case compares the unbounded path against itself"
    )
    assert getattr(R3, "COVER_COMPACTION", None) is True, (
        "raycast3d.COVER_COMPACTION is gone or already off, so the composite case cannot switch the "
        "compaction off and has been comparing the compacted path against itself"
    )
    # COVER_COMPACTION now governs ONLY ``_composite_window``: RAYCAST-6's composite compacts per
    # row and never reads it. So the cover-compaction case must run with COMPOSITE_COMPACT_MERGE
    # pinned off, and a future edit that drops that pin would make it green for the wrong reason.
    assert "_composite_compact_merge(False)" in inspect.getsource(_composite_frames), (
        "the composite-cover-compaction case no longer pins COMPOSITE_COMPACT_MERGE off, so it "
        "flips a switch the running composite ignores and compares the compact path with itself"
    )
    assert "COVER_COMPACTION" in inspect.getsource(R3._composite_window), (
        "raycast3d._composite_window no longer branches on COVER_COMPACTION, so pinning that switch "
        "changes nothing on either path and the case is vacuous"
    )
    assert getattr(R3, "COMPOSITE_COMPACT_MERGE", None) is True, (
        "raycast3d.COMPOSITE_COMPACT_MERGE is gone or already off, so the compact-merge case cannot "
        "switch the compaction off and has been comparing the compacted path against itself"
    )
    assert callable(getattr(R3, "_composite_window", None)), (
        "raycast3d._composite_window is gone, so COMPOSITE_COMPACT_MERGE=False has no verbatim body "
        "to fall back to and the compact-merge case compares the new path against itself"
    )
    assert "cull" in inspect.signature(R2.render_triton).parameters, (
        "raster2d.render_triton no longer takes cull=, so both culling cases are drawing the same picture twice"
    )
    if any(n.startswith(("DMC", "MuJoCo")) for n in se.registry.REGISTRY):
        switchable = [
            n
            for n in se.registry.REGISTRY
            if any(
                hasattr(se.registry.REGISTRY[n], k) for k in ("MOVER_TILES", "SHADOW_COMPACTION", "COVER_COMPACTION")
            )
        ]
        assert switchable, (
            "no registered class carries MOVER_TILES / SHADOW_COMPACTION / COVER_COMPACTION, so "
            "_unbounded() switches nothing off and the window case compares the windowed path against "
            "itself for every env in it"
        )
