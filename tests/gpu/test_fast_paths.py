"""Fast-path renderers must be bit-identical to their reference paths on sampled registry states.

Each case pairs an optimization with the implementation it replaces, renders one input both ways,
and compares the uint8 frames with a threshold of zero.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import assert_frames_identical, family_params, make_or_skip, resolve_family, sampled_states

import stable_envs as se
from stable_envs.render import raster2d as R2
from stable_envs.render import raycast3d as R3
from stable_envs.testing._sampling import has_render_rows
from tests.helpers import raycast_reference as reference

#: Env modules whose import failure explains an empty family (see ``resolve_family``).
RAYCAST_MODULES = ("dmcontrol", "ogbench")
MOVER_MODULES = ("dmcontrol",)


@contextlib.contextmanager
def _forced_backend(backend: str | None):
    with (
        patch.object(R3, "cast_nearest", partial(R3.cast_nearest, backend=backend or "auto")),
        patch.object(R3, "any_blocking", partial(R3.any_blocking, backend=backend or "auto")),
    ):
        yield


@contextlib.contextmanager
def _unbounded(cls):
    """Disable mover windows and ray-set compaction on ``cls`` for the duration.

    Compaction is switched off too: envs whose movers span most of the image already decline every
    window, so clearing ``MOVER_TILES`` alone would leave both legs on the same path.
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
    if not has_render_rows(env):
        return None  # a full State renders without a Scene2D
    states = sampled_states(env, batch, seed)
    try:
        return env.scene(torch.as_tensor(states, device=env.device, dtype=torch.float32))
    except (NotImplementedError, TypeError, AttributeError):
        return None


def _raster_size(env):
    return (env.H, env.W) if hasattr(env, "H") and hasattr(env, "W") else env.size


def _classic_enabled() -> bool:
    try:
        from stable_envs.render.gfxdraw import classic as FC
    except ImportError:
        return False
    return bool(FC.HAVE_TRITON)


@dataclass(frozen=True)
class FastPath:
    """One optimization and what is needed to render it against the path it replaces."""

    id: str
    #: What a difference means; included in the failure message.
    why: str
    #: Registered env names this optimization is reachable from, and the modules whose import failure
    #: explains an empty list.
    names: tuple
    modules: tuple
    num_envs: int
    seed: int
    #: Output sizes to build the env at; ``(None,)`` means "the env's own size".
    sizes: tuple
    #: ``(env) -> payload``, computed once and passed to both legs so input nondeterminism cannot
    #: appear as a fast/slow difference. ``None`` means the env cannot exercise this case.
    prepare: Callable
    #: ``(env, payload, fast: bool) -> uint8 frames``. Called twice per case, on the same payload.
    frames: Callable
    #: ``(env) -> str | None``. A reason this env cannot exercise the optimization, or None.
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


def _compact_merge_frames(env, states, fast: bool):
    if fast:
        return env._render(states).clone()
    with patch.object(R3, "composite", reference.composite_window):
        return env._render(states).clone()


def _composite_frames(env, states, fast: bool):
    with patch.object(R3, "composite", partial(reference.composite_window, cover_compaction=fast)):
        return env._render(states).clone()


CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")


def _classic_frames(env, states, fast: bool):
    """Render with the fused per-pixel kernel on (``fast``) or off."""
    was = env.use_fused_render
    try:
        env.use_fused_render = bool(fast)
        return env._render(states).clone()
    finally:
        env.use_fused_render = was


def _classic_prepare(seed: int, extreme: bool):
    """The frozen state corpus, or (``extreme``) states that put each primitive in an edge case.

    Extreme states cover: a cart off the surface (SDL_gfx line clipping restarts the Wu accumulator
    from a clipped endpoint), angles many turns from zero (float to Sint16 truncation), a MountainCar
    position outside the track (fill span clamps), and ``|torque| > 2`` (end of Pendulum's sprite table).
    """

    def prepare(env):
        import numpy as np

        n = env.B
        if not extreme:
            import pathlib

            npz = (
                pathlib.Path(__file__).resolve().parents[2]
                / "tests/fixtures/states/fidelity"
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
    with (
        patch.object(R3, "cast_nearest_torch", partial(R3.cast_nearest_torch, bounds=on)),
        patch.object(R3, "any_blocking_torch", partial(R3.any_blocking_torch, bounds=on)),
    ):
        yield


def _bounds_frames(env, states, fast: bool):
    """Both legs use the torch cast; only it has primitive bounds to switch."""
    with _forced_backend("torch"), _primitive_bounds(fast):
        return env._render(states).clone()


def _trackcom_prepare(seed: int):
    """The env's states and a per-lane variation that recolors the floor, from one generator."""

    def prepare(env):
        import numpy as np

        style = env.sample_variation(env.B, rng=np.random.default_rng(seed + 1))
        return sampled_states(env, env.B, seed), torch.as_tensor(style, device=env.device)

    return prepare


def _trackcom_reflection(env):
    from stable_envs.envs.dmcontrol.trackcom import _launch_worlds

    return env._context(env.size, _launch_worlds(env.size, env.B)).reflection


def _trackcom_floor_frames(env, payload, fast: bool):
    """The floor pass gathering corner shades from its per-world table, then shading every corner in place."""
    states, style = payload
    flag = _trackcom_reflection(env)._scene.tabulated
    was = int(flag.numpy()[0])
    try:
        flag.fill_(int(fast and was))
        return torch.cat([env._render(states).clone(), env._render(states, style=style).clone()])
    finally:
        flag.fill_(was)


def _trackcom_inapplicable(env):
    if not hasattr(env, "_floor_slots") or env.device.type != "cuda":
        return "draws no TrackCom floor reflection pass"
    reflection = _trackcom_reflection(env)
    if reflection is None:
        return "draws no TrackCom floor reflection pass"
    if not reflection._scene.tabulated.numpy()[0]:
        return "its floor pass shades every corner in place"
    return None


def _scene_prepare(seed: int):
    return lambda env: _scene(env, env.B, seed)


def _cull_frames_factory(supersample):
    """``cull=True`` against ``cull=False`` on one ``Scene2D``.

    Calls ``render_triton`` directly so every env that builds a ``Scene2D`` is covered, whatever
    backend its ``render`` uses.
    """

    def frames(env, scene, fast: bool):
        ss = env.ss if supersample is None else supersample
        return R2.render_triton(scene, _raster_size(env), supersample=ss, cull=fast).clone()

    return frames


CASES: tuple[FastPath, ...] = (
    FastPath(
        id="raycast3d-fused-vs-torch",
        why=(
            "The Triton ray cast must match ``cast_nearest_torch`` bit for bit. Triton's default fp32 "
            "lowering does not (FMA contraction, ``div.full``, ``sqrt.approx``), so check that "
            "``enable_fp_fusion=False``, ``div_rn`` and ``sqrt_rn`` are still in raycast3d._tri_isect "
            "and _launch."
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
            "A restricted pass clips geometry, a shadow or a reflection the full frame draws. The "
            "bounded passes are the mover and shadow casts (``_mover_idx``), the reflection around "
            "the movers' mirror images (``_mirror_idx``) and the shadow-from-static test "
            "(``_shadow_candidates``)."
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
            "A primitive bounding sphere in the torch cast is too small: a ray outside the inflated "
            "bound (16 ulp budget) still hits the primitive, so skipping it drops silhouette pixels."
        ),
        names=tuple(
            n
            for n in sorted(se.registry.REGISTRY)
            if n.startswith(("DMC", "MuJoCo", "AntMaze", "PointMaze", "HumanoidMaze"))
        ),
        modules=RAYCAST_MODULES,
        num_envs=8,
        seed=0,
        # The env's own size: the bound fails on a long lever arm (far camera, small primitive), which
        # a giant maze built at 84 px shortens.
        sizes=(None,),
        prepare=_states_prepare(0),
        frames=_bounds_frames,
    ),
    FastPath(
        id="composite-cover-compaction",
        why=(
            "``raycast3d.composite`` shading only the batch union of mover-covered ray positions changed "
            "a value: a covered ray lies outside the union, or a dropped shaded value is not +0.0."
        ),
        names=tuple(n for n in sorted(se.registry.REGISTRY) if n.startswith(("AntMaze", "PointMaze", "HumanoidMaze"))),
        modules=("ogbench",),
        num_envs=8,
        seed=0,
        sizes=(None,),
        prepare=_states_prepare(0),
        frames=_composite_frames,
        inapplicable=lambda env: (
            None
            if type(env).__module__.rsplit(".", 1)[-1] in ("ogbench",)
            else "does not render through raycast3d.composite"
        ),
    ),
    FastPath(
        id="composite-compact-merge",
        why=(
            "The compact ``raycast3d.composite`` (bound-sized window, covered (row, sample) pairs, "
            "covered output pixels only) differs from the fixed-window reference. The most likely "
            "cause is a span bound that is not conservative, clipping an edge of the agent."
        ),
        names=tuple(n for n in sorted(se.registry.REGISTRY) if n.startswith(("AntMaze", "PointMaze", "HumanoidMaze"))),
        modules=("ogbench",),
        num_envs=8,
        seed=1,
        sizes=(None,),
        prepare=_states_prepare(1),
        frames=_compact_merge_frames,
        inapplicable=lambda env: (
            None
            if type(env).__module__.rsplit(".", 1)[-1] in ("ogbench",)
            else "does not render through raycast3d.composite"
        ),
    ),
    FastPath(
        id="raster2d-tile-culling",
        why=(
            "A primitive paints outside its own world-space bounding box, so ``raster2d`` tile culling "
            "drops it; see ``_bbox_kernel`` on collinear polygons. (The two paths are separate "
            "compilations, so a pixel within ~3e-5 world units of an edge can flip at large batch "
            "sizes; the batches here are too small for that.)"
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
            "The fused ``render/gfxdraw/classic.py`` kernel differs from the per-primitive gfxdraw path. "
            "Likely spots: shared-vertex write order between polygon edges, the two aacircle fringe "
            "passes, fringe-before-fill order, Pendulum's sprite blit, and the 16.16 fill span clamps."
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
            "The fused classic kernel differs from the per-primitive path on out-of-range states (cart "
            "off the surface, large angles, car off the track, torque past the sprite table), which "
            "exercise the fused path's window and clipped-edge bounds."
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
            "``raster2d`` tile culling at ``supersample=2`` drops a primitive: the tile's world box must "
            "cover every subsample the program evaluates, not just the pixel centres."
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
    FastPath(
        id="trackcom-floor-table-vs-in-place",
        why=(
            "The TrackCom floor corner-shade table differs from in-place shading. The table kernel must "
            "repeat the pixel kernel's corner loop so both round alike, and the in-place shading must "
            "stay behind a runtime flag because compiling it out reorders the pixel's float math."
        ),
        names=tuple(n for n in sorted(se.registry.REGISTRY) if n.startswith("DMC")),
        modules=MOVER_MODULES,
        num_envs=48,
        seed=0,
        sizes=(None,),
        prepare=_trackcom_prepare(0),
        frames=_trackcom_floor_frames,
        inapplicable=_trackcom_inapplicable,
    ),
)

CASES_BY_ID = {c.id: c for c in CASES}


def _matrix():
    """``(case id, env name, size)`` for every case, with an ``EMPTY`` marker for an empty family."""
    out = []
    for c in CASES:
        for name in family_params(c.names, family=c.id):
            for size in c.sizes:
                sid = "native" if size is None else f"{size}px"
                out.append(pytest.param(c.id, name.values[0], size, id=f"{c.id}-{name.id}-{sid}"))
    return out


@pytest.mark.parametrize("case_id,name,size", _matrix())
def test_the_fast_path_is_bit_identical_to_the_path_it_replaces(case_id, name, size):
    """The fast path renders the same bytes as the path it replaces, for one env and output size."""
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

    fast = case.frames(env, payload, True)
    slow = case.frames(env, payload, False)
    at = "" if size is None else f" @{size}px"
    assert_frames_identical(fast, slow, what=f"{name}{at} [{case.id}]", why=case.why)


def test_an_unknown_geom_code_is_refused_by_both_cast_backends():
    """Both cast backends raise on an unsupported geom code instead of treating it as a plane.

    No registered env passes an unsupported code, so the frame comparisons cannot catch a fallthrough.
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
    """The torch ``_ellipsoid`` and the Triton ellipsoid branch agree bit for bit on a synthetic scene.

    The scene has rotated, anisotropic and degenerate ellipsoids and rays starting inside one.
    """
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
    """A zero-area polygon is never culled by tile culling.

    Collinear vertices make the halfplane test cover the whole line through them, beyond the vertex
    bounding box. The line sits at y = 32.5 because pixel centers are at half-integers here; on an
    integer coordinate it would paint nothing and the test would pass vacuously.
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
        "a collinear polygon was culled by its bounding box; _bbox_kernel must mark a zero-area polygon uncullable"
    )


def test_every_declared_fast_path_is_actually_reachable():
    """The switches each fast-path case toggles exist, so no case compares a path against itself."""
    import inspect

    assert R3._BOUNDED_KINDS
    for fn in (R3.cast_nearest, R3.any_blocking):
        assert "backend" in inspect.signature(fn).parameters
    for fn in (R3.cast_nearest_torch, R3.any_blocking_torch):
        assert "bounds" in inspect.signature(fn).parameters
    assert R3.composite is not reference.composite_window
    assert "cover_compaction" in inspect.signature(reference.composite_window).parameters
    assert "cull" in inspect.signature(R2.render_triton).parameters, (
        "raster2d.render_triton does not take cull=, so both culling cases draw the same picture twice"
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
            "_unbounded() switches nothing off and the window case compares the windowed path against itself"
        )


def test_raycast_overrides_route_to_distinct_paths(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(1)

    monkeypatch.setattr(R3, "cast_nearest", record)
    monkeypatch.setattr(R3, "any_blocking", record)
    for backend in ("triton", "torch"):
        with _forced_backend(backend):
            R3.cast_nearest()
            R3.any_blocking()
        assert calls[-2:] == [{"backend": backend}] * 2
    assert R3.cast_nearest is record and R3.any_blocking is record
    monkeypatch.setattr(R3, "cast_nearest_torch", record)
    monkeypatch.setattr(R3, "any_blocking_torch", record)
    for on in (True, False):
        with _primitive_bounds(on):
            R3.cast_nearest_torch()
            R3.any_blocking_torch()
        assert calls[-2:] == [{"bounds": on}] * 2
    assert R3.cast_nearest_torch is record and R3.any_blocking_torch is record
    production = lambda **kwargs: record(path="compact", **kwargs)
    oracle = lambda **kwargs: record(path="window", **kwargs)
    monkeypatch.setattr(R3, "composite", production)
    monkeypatch.setattr(reference, "composite_window", oracle)
    env = SimpleNamespace(_render=lambda states: R3.composite())
    _compact_merge_frames(env, None, True)
    _compact_merge_frames(env, None, False)
    _composite_frames(env, None, True)
    _composite_frames(env, None, False)
    assert calls[-4:] == [
        {"path": "compact"},
        {"path": "window"},
        {"path": "window", "cover_compaction": True},
        {"path": "window", "cover_compaction": False},
    ]
    assert R3.composite is production
