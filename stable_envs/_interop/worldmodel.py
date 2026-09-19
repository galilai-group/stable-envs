"""Make a stable-envs renderer usable by ``stable_worldmodel`` without touching the caller's code.

The mismatch this module exists to close is structural, not cosmetic. ``stable_worldmodel`` drives
environments through ``swm.World``, which builds a plain Python list of single ``gym.Env`` objects
(``world.env_pool.EnvPool.__init__`` is literally ``[fn() for fn in env_fns]``) and then loops over
them one at a time, calling ``env.step(action_i)`` and ``env.render()`` per environment per step. A
stable-envs env is the opposite shape: no ``reset``, a ``step`` that takes ``(B, action_dim)``, and a
``render`` that is a pure function of a ``(B, state_dim)`` stack. Neither is wrong; they are answers
to different questions, and something has to translate.

Three adapters are provided, and they are genuinely different trades rather than three spellings of
one thing:

``StableEnvGym`` — **drop-in, one env at a time.**
    A real ``gym.Env`` over a batch-1 renderer, registered under ``stable_envs/<Name>-v0``. An
    existing swm workflow changes exactly one string:

        world = swm.World("stable_envs/TwoRoom-v0", num_envs=8, max_episode_steps=100)
        world.set_policy(swm.policy.RandomPolicy())
        world.reset(seed=0); ...

    Everything downstream — the wrapper stack, ``info['pixels']``, the policy, the solver — is
    untouched, because the facade satisfies the same duck-typed contract the official envs do. It
    buys correctness and portability and no speed: N envs is still N render calls.

``BatchedEnvPool`` — **the same World, one launch per step.**
    A replacement for ``swm.world.env_pool.EnvPool`` backed by ONE ``batch=N`` renderer, so a step
    of the whole world is one ``step`` launch and one ``render`` launch. This is where the speed
    actually is, and it is one line at the call site::

        world.envs = stable_envs.interop.BatchedEnvPool("TwoRoom", num_envs=64)

    The cost is that it replaces the wrapper stack too, so it has to produce the same ``info`` keys
    itself. It does — see ``INFO_KEYS`` — and the example asserts key-for-key equality against a
    stock World rather than claiming it.

``swap_renderer`` — **the official env, our pixels.**
    Keeps the official env for dynamics, reward and termination, and replaces only ``render()``.
    This is the arm that can be pixel-compared, because both arms then have identical trajectories
    by construction and the ONLY difference is the renderer. It is also the answer for the envs we
    render but do not simulate.

Appearance
----------

``stable_worldmodel`` and ``stable_envs`` both call the appearance knobs a "variation space" and both
use the same class names, so most leaves map by identity. The ones that do not are recorded in
``VARIATION_MAPS`` with the reason, in both directions: a leaf swm has and we do not (because it is
dynamics, or because the official env declares it and never reads it), and a leaf we have and swm
does not. ``style_from_variation`` turns a swm variation config into one of our flat style rows and
``variation_from_style`` inverts it, so an existing swm config drives our renderer unchanged and a
stored style row can be handed back to swm.

The one substantive unit difference is PushT's scales. swm declares ``agent.scale`` /``block.scale``
/``goal.scale`` as ABSOLUTE sizes in ``Box(20, 60)`` with default 40; stable-envs declares them as
MULTIPLIERS on the reference geometry with default 1.0, because that is what makes "our defaults are
the official appearance" an identity a test can assert. The conversion is therefore ``ours =
theirs / 40``, and it is in the table rather than in a caller's head.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .. import spaces as se_spaces
from ..dynamics import has_dynamics as _has_dynamics
from ..registry import REGISTRY, make

#: Keys ``BatchedEnvPool`` puts in the stacked info dict, and where each comes from in a stock World.
#: Recorded here because the pool bypasses ``MegaWrapper`` and therefore owes the same contract:
#:
#:   pixels, render_time     AddPixelsWrapper
#:   observation             the observation itself, filed under this name because it is a bare Box
#:   state, proprio,         the raw env's own info
#:   distance_to_target,
#:   goal_state, env_name
#:   reward, terminated,     EverythingToInfoWrapper
#:   truncated, action,
#:   step_idx, id
#:   goal                    an ADDITION: the goal frame, which the official TwoRoom keeps private
#:   variation.<path>        EverythingToInfoWrapper, one per watched variation key
#:
#: ``_needs_flush`` is NOT here: ``World._run_iter`` writes it into the dict after the pool returns,
#: and ``WorldModelPolicy.get_action`` pops it.
INFO_KEYS = (
    "pixels",
    "render_time",
    "observation",
    "state",
    "proprio",
    "distance_to_target",
    "reward",
    "terminated",
    "truncated",
    "action",
    "step_idx",
    "id",
    "goal",
    "goal_state",
    "env_name",
)


# ====================================================================== variation mapping


@dataclass(frozen=True)
class VariationMap:
    """How one stable-envs env's appearance relates to its ``stable_worldmodel`` counterpart.

    Written down rather than inferred by name matching. Name matching gets the easy 80% right and
    then silently mis-maps the interesting cases — PushT's absolute-vs-multiplier scales would map
    cleanly by name and be wrong by a factor of 40 — and it cannot express "swm has this leaf and we
    deliberately do not", which is the half of the mapping a user actually needs to read.
    """

    #: the gymnasium id of the official swm env this env reproduces
    swm_id: str
    #: leaves with the same dotted path and the same units in both libraries
    identical: tuple[str, ...] = ()
    #: ``our_path -> swm_path`` for leaves that mean the same thing under different names
    renamed: Mapping[str, str] = field(default_factory=dict)
    #: ``our_path -> (swm_path, factor)`` with ``ours = theirs * factor``
    scaled: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    #: ``our_path -> (swm_path, factor, offset)`` with ``ours = theirs * factor + offset``. The
    #: general form of ``scaled``, kept separate rather than widening it because ``scaled``'s
    #: two-tuple is what the four pre-Step-8 maps are written in and a silent arity change is the one
    #: edit a reader cannot see. It exists for the leaves whose two libraries agree on WHAT the knob
    #: means and disagree on how it is spelled: an inverted binary switch is ``(-1.0, 1.0)``, a unit
    #: shift is ``(1.0, k)``. The inverse is ``theirs = (ours - offset) / factor``, so a ``factor`` of
    #: zero is not a mapping and is refused.
    affine: Mapping[str, tuple[str, float, float]] = field(default_factory=dict)
    #: ``our_path -> swm_path`` for COLOURS we hold as 0-255 channels and swm holds in [0, 1].
    #: Separate from ``scaled`` because a factor of 255 alone is the wrong conversion: ``to_u8``
    #: truncates and swm's own ``_color_swaps`` writes ``uint8(v * 255 + 0.5)``, so 0.3 is 77 to swm
    #: and 76 through a bare multiply. The rounding is part of the mapping, not of the caller.
    rgb255: Mapping[str, str] = field(default_factory=dict)
    #: ``our_path -> (swm_path, table)`` for DISCRETE leaves whose alternatives are indexed
    #: differently. ``table[swm_index] = our_index``; an swm index with no entry is not reachable and
    #: must be named in ``unsupported``.
    enum: Mapping[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict)
    #: ``swm_path -> why we do not carry it``. Every swm leaf must appear in exactly one of the
    #: mapping fields or in one of the four unmapped buckets, and ``check_mapping`` asserts that.
    #: ``swm_only`` is the original undifferentiated bucket and is kept for the maps that predate the
    #: Step-8 taxonomy; new maps should use one of the three below instead.
    swm_only: Mapping[str, str] = field(default_factory=dict)
    #: ``swm_path -> reason``: the leaf has a real effect upstream (pixels, or dynamics on a cell that
    #: simulates) and THIS cell cannot express it. The reason must name the mechanism.
    unsupported: Mapping[str, str] = field(default_factory=dict)
    #: ``swm_path -> measurement``: the leaf reaches no pixel and no state IN SWM ITSELF. The reason
    #: must carry the measurement that proves it, not an opinion.
    dead_upstream: Mapping[str, str] = field(default_factory=dict)
    #: ``swm_path -> reason``: the leaf is not a question this cell answers at all — a pure dynamics
    #: parameter on a cell that ships a renderer and no simulator.
    not_applicable: Mapping[str, str] = field(default_factory=dict)
    #: ``swm_path -> caveat`` for leaves that ARE mapped but not over their whole range or not with
    #: identical semantics. A mapped leaf with a documented gap is still mapped; hiding the gap in
    #: prose beside the map is how a range gap becomes a surprise.
    notes: Mapping[str, str] = field(default_factory=dict)
    #: ``our_path -> why swm has no counterpart``
    ours_only: Mapping[str, str] = field(default_factory=dict)
    #: ``swm_path -> what is NOT reproduced`` for a leaf that IS mapped (it must appear in one of the
    #: mapping fields) but whose upstream effect this cell reproduces only in part -- Fetch's
    #: ``background.color`` writes the floor material AND the skybox texture, and the map carries the
    #: floor half. Unlike ``notes`` this is a classification, not a caveat: ``classification`` reports
    #: such a leaf as ``partial`` so a generated completeness table cannot count it as full support,
    #: and the reason must carry the measured residual. Added by the coordinator for Workstream B under
    #: Sami's ruling that a feasible, required leaf is not declared unsupported for want of a renderer
    #: capability; the capability decision is recorded separately.
    partial: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for ours, (theirs, factor, _) in self.affine.items():
            if not float(factor):
                raise ValueError(
                    f"{ours} -> {theirs}: an affine mapping with factor 0 is not "
                    f"invertible, so the leaf cannot round-trip; declare it unsupported"
                )
        stray = set(self.partial) - self.mapped_swm_paths()
        if stray:
            raise ValueError(
                f"partial names {sorted(stray)}, which no mapping field reaches; a leaf "
                f"that is not mapped at all belongs in unsupported, not partial"
            )

    def pairs(self) -> dict[str, tuple[str, float]]:
        """``our_path -> (swm_path, factor)`` over every leaf mapped by a pure multiply.

        ``affine`` is NOT folded in here: this returns two-tuples and a caller that unpacked three
        would be a silent breakage of the four pre-Step-8 maps. See ``affine_pairs``.
        """
        out = {p: (p, 1.0) for p in self.identical}
        out.update({a: (b, 1.0) for a, b in self.renamed.items()})
        out.update(dict(self.scaled))
        return out

    def affine_pairs(self) -> dict[str, tuple[str, float, float]]:
        """``our_path -> (swm_path, factor, offset)`` over EVERY numerically mapped leaf.

        The multiply-only leaves appear here with ``offset = 0``, so a caller that wants one uniform
        rule has one; ``pairs`` stays the two-tuple view the older code is written against.
        """
        out = {o: (t, f, 0.0) for o, (t, f) in self.pairs().items()}
        out.update({o: (t, float(f), float(k)) for o, (t, f, k) in self.affine.items()})
        return out

    def unmapped(self) -> dict[str, str]:
        """``swm_path -> reason`` over all four unmapped buckets, which are disjoint by construction."""
        out = dict(self.swm_only)
        for bucket in (self.unsupported, self.dead_upstream, self.not_applicable):
            out.update(bucket)
        return out

    def mapped_swm_paths(self) -> set[str]:
        """Every swm leaf this map reaches, by whichever mechanism."""
        return (
            {t for t, _ in self.pairs().values()}
            | {t for t, _, _ in self.affine.values()}
            | set(self.rgb255.values())
            | {t for t, _ in self.enum.values()}
        )

    def classification(self) -> dict[str, str]:
        """``swm_path -> one of mapped / partial / unsupported / dead_upstream / not_applicable / swm_only``.

        The Step-8 taxonomy, per leaf, from the map itself rather than from prose beside it.
        """
        out = {p: "mapped" for p in self.mapped_swm_paths()}
        out.update({p: "partial" for p in self.partial})
        for label, bucket in (
            ("swm_only", self.swm_only),
            ("unsupported", self.unsupported),
            ("dead_upstream", self.dead_upstream),
            ("not_applicable", self.not_applicable),
        ):
            for p in bucket:
                out[p] = label
        return out


_TWOROOM_STRUCTURE = (
    "background.color",
    "wall.color",
    "wall.thickness",
    "wall.axis",
    "door.color",
    "door.number",
    "door.position",
    "door.size",
)
_TWOROOM_SWM_ONLY = {
    "agent.position": "this is our STATE, not our style — see state_names",
    "agent.speed": "dynamics only; carried as env.speed because it changes no pixel",
    "task.min_steps": "a task definition, not an appearance; changes no pixel",
    "wall.border_color": "declared by TwoRoomEnv and read by nothing (measured: 0/255 diff)",
    "rendering.render_target": "selects TwoRoom vs TwoRoomTarget, which are separate classes",
    "target.color": "drawn only when render_target is on; see TwoRoomTarget",
    "target.radius": "drawn only when render_target is on; see TwoRoomTarget",
    "target.position": "TwoRoomTarget carries it in the STATE, because the frame reads it",
}
_PIECEWISE_SWM_ONLY = {
    "agent.position": "this is our STATE, not our style",
    "agent.speed": "dynamics only; carried as env.speed",
    "task.min_steps": "a task definition; changes no pixel",
    "target.color": "drawn only when render_target is on, which this port does not reproduce",
    "target.radius": "drawn only when render_target is on",
    "target.position": "drawn only when render_target is on",
    "rendering.render_target": "not reproduced; see the swm_nav module docstring",
    "rendering.render_zones": "selects Piecewise vs PiecewiseZones, which are separate classes",
    "rendering.render_bias_field": "rasterises a matplotlib quiver; not reproduced, see swm_nav",
    **{f"zones.bias_{i}": "dynamics only; carried as env.zone_bias" for i in range(4)},
}

#: our class name -> how its appearance relates to the official swm env's
VARIATION_MAPS: dict[str, VariationMap] = {
    "TwoRoom": VariationMap(
        swm_id="swm/TwoRoom-v1",
        identical=("agent.color", "agent.radius") + _TWOROOM_STRUCTURE,
        swm_only=_TWOROOM_SWM_ONLY,
    ),
    "TwoRoomTarget": VariationMap(
        swm_id="swm/TwoRoom-v1",
        identical=("agent.color", "agent.radius", "target.color", "target.radius") + _TWOROOM_STRUCTURE,
        swm_only={k: v for k, v in _TWOROOM_SWM_ONLY.items() if k not in ("target.color", "target.radius")},
    ),
    "Piecewise": VariationMap(
        swm_id="swm/Piecewise-v0",
        identical=("agent.color", "agent.radius", "background.color", "border.color"),
        swm_only={
            **_PIECEWISE_SWM_ONLY,
            **{f"zones.color_{i}": "drawn only with render_zones; see PiecewiseZones" for i in range(4)},
        },
    ),
    "PiecewiseZones": VariationMap(
        swm_id="swm/Piecewise-v0",
        identical=("agent.color", "agent.radius", "background.color", "border.color")
        + tuple(f"zones.color_{i}" for i in range(4)),
        swm_only=_PIECEWISE_SWM_ONLY,
    ),
    "PushT": VariationMap(
        swm_id="swm/PushT-v1",
        identical=(
            "agent.color",
            "block.color",
            "goal.color",
            "goal.position",
            "goal.angle",
            "background.color",
            "rendering.render_goal",
        ),
        # ours is a MULTIPLIER on the reference geometry (default 1.0); swm's is an ABSOLUTE size in
        # Box(20, 60) with default 40. Mapping them by name alone is wrong by a factor of 40.
        scaled={
            "agent.scale": ("agent.scale", 1.0 / 40.0),
            "block.scale": ("block.scale", 1.0 / 40.0),
            "goal.scale": ("goal.scale", 1.0 / 40.0),
        },
        swm_only={
            "agent.shape": "swm's PushT can swap the agent between 8 pymunk shapes; ours is "
            "gym_pusht's disc, which is shape index 0 and the default",
            "block.shape": "same: 8 shapes, of which ours is the T (index 2, the default)",
            "agent.angle": "the disc agent is rotationally symmetric, so it changes no pixel",
            "block.angle": "swm's RESET angle for the block; ours is state dimension block_angle",
            "agent.start_position": "a reset distribution, not an appearance; ours is the state",
            "block.start_position": "a reset distribution, not an appearance; ours is the state",
            "agent.velocity": "dynamics; gym_pusht's observation does not carry it either",
        },
        ours_only={
            "agent.inner_color": "pymunk's debug draw paints a lighter inner disc; swm derives it "
            "from agent.color with light_color() and does not expose it",
            "block.outline_color": "same: derived from block.color in swm, separate here",
            "walls.color": "swm hard-codes LightGray on the four wall segments",
            "walls.visible": "swm always draws the walls",
        },
    ),
}


#: Step 8's maps live one module per family under ``compat/variation_maps`` and install themselves
#: here; see that package's docstring for why they are not more entries in the literal above.
from .variation_maps import install as _install_variation_maps

_install_variation_maps(VARIATION_MAPS)  # noqa: E402,E501


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested ``{'agent': {'color': v}}`` or already-dotted ``{'agent.color': v}`` -> dotted."""
    out: dict[str, Any] = {}
    for k, v in values.items():
        p = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, p + "."))
        else:
            out[p] = v
    return out


def variation_map(env) -> VariationMap | None:
    """The mapping for an env or env name, or ``None`` if this env has no swm counterpart."""
    name = env if isinstance(env, str) else type(env).__name__
    return VARIATION_MAPS.get(name)


def style_from_variation(env, variation: Mapping[str, Any], base=None, strict: bool = False) -> np.ndarray:
    """A swm variation config -> one of our flat ``(style_dim,)`` style rows.

    ``variation`` is whatever ``swm``'s ``variation_space.value`` gives (nested) or what goes into
    ``options['variation_values']`` (dotted); both are accepted. Leaves swm has and we do not are
    ignored, which is the whole reason ``VariationMap.swm_only`` exists — with ``strict=True`` an
    unrecognised leaf raises instead, which is what a test wants and what a user pasting a config
    does not.
    """
    m = variation_map(env)
    if m is None:
        raise KeyError(
            f"{type(env).__name__} has no recorded stable_worldmodel counterpart; add a "
            f"VariationMap to stable_envs.interop.VARIATION_MAPS"
        )
    flat = _flatten(dict(variation))
    row = np.array(env.default_style() if base is None else base, np.float32).reshape(-1).copy()
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    for ours, (theirs, factor) in m.pairs().items():
        if theirs not in flat or ours not in layout:
            continue
        o, w = layout[ours]
        row[o : o + w] = np.asarray(flat[theirs], np.float64).reshape(-1)[:w] * factor
    for ours, (theirs, factor, offset) in m.affine.items():
        if theirs not in flat or ours not in layout:
            continue
        o, w = layout[ours]
        row[o : o + w] = np.asarray(flat[theirs], np.float64).reshape(-1)[:w] * factor + offset
    for ours, theirs in m.rgb255.items():
        if theirs not in flat or ours not in layout:
            continue
        o, w = layout[ours]
        # swm's own quantisation, transcribed from ``GymControlWrapper.render``:
        # ``(clip(v, 0, 1) * 255 + 0.5).astype(uint8)``. Written here so a colour that round-trips
        # through this mapping lands on the same uint8 swm would have written.
        v = np.clip(np.asarray(flat[theirs], np.float64).reshape(-1)[:w], 0.0, 1.0)
        row[o : o + w] = np.floor(v * 255.0 + 0.5)
    for ours, (theirs, table) in m.enum.items():
        if theirs not in flat or ours not in layout:
            continue
        idx = int(np.asarray(flat[theirs]).reshape(-1)[0])
        if not 0 <= idx < len(table):
            raise KeyError(
                f"{type(env).__name__}: swm {theirs} = {idx} has no entry in the map's "
                f"alternative table {tuple(table)}; an unreachable alternative belongs in "
                f"`unsupported`, not in `enum`"
            )
        o, _ = layout[ours]
        row[o] = float(table[idx])
    if strict:
        known = set(m.unmapped()) | m.mapped_swm_paths()
        unknown = sorted(set(flat) - known)
        if unknown:
            raise KeyError(
                f"variation leaves with no entry in the {type(env).__name__} map: "
                f"{unknown}. Add them to `identical`/`renamed`/`scaled` if we render "
                f"them, or to `swm_only` with the reason if we deliberately do not."
            )
    return row


def variation_from_style(env, style) -> dict[str, Any]:
    """Inverse of ``style_from_variation``: a stored style row -> a swm-shaped dotted config."""
    m = variation_map(env)
    if m is None:
        raise KeyError(f"{type(env).__name__} has no recorded stable_worldmodel counterpart")
    row = np.asarray(style, np.float32).reshape(-1)
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    out: dict[str, Any] = {}
    for ours, (theirs, factor) in m.pairs().items():
        if ours not in layout:
            continue
        o, w = layout[ours]
        v = row[o : o + w] / factor
        out[theirs] = float(v[0]) if w == 1 else v.astype(np.float64)
    for ours, (theirs, factor, offset) in m.affine.items():
        if ours not in layout:
            continue
        o, w = layout[ours]
        v = (row[o : o + w].astype(np.float64) - offset) / factor
        out[theirs] = float(v[0]) if w == 1 else v
    for ours, theirs in m.rgb255.items():
        if ours not in layout:
            continue
        o, w = layout[ours]
        out[theirs] = row[o : o + w].astype(np.float64) / 255.0
    for ours, (theirs, table) in m.enum.items():
        if ours not in layout:
            continue
        o, _ = layout[ours]
        mine = int(round(float(row[o])))
        # FIRST swm index that selects our alternative. A table may send two swm indices to one of
        # ours (it does not here); the inverse then names the lower, and the round trip is still the
        # same appearance, which is what the inverse is for.
        back = [k for k, v in enumerate(table) if int(v) == mine]
        if back:
            out[theirs] = int(back[0])
    return out


def style_from_swm_env(env, swm_env, base=None) -> np.ndarray:
    """Read a LIVE swm env's variation space and produce our style row for it."""
    return style_from_variation(env, swm_variation_space(swm_env).value, base=base)


def swm_variation_space(swm_env):
    """The variation space of a live swm env, wherever in its wrapper stack it lives.

    ``unwrapped.variation_space`` is right for the envs whose variation space is on the env itself
    (swm_nav, PushT, the DM Control wrappers, which are ``gym.Env`` and return ``self`` from
    ``unwrapped``) and WRONG for the five classic-control ids, where ``GymControlWrapper`` wraps a
    stock ``gymnasium`` env and ``unwrapped`` reaches past the wrapper to a ``CartPoleEnv`` that has
    no variation space at all. Walking the stack is right for both.
    """
    node = swm_env
    while node is not None:
        space = getattr(node, "variation_space", None)
        if space is not None:
            return space
        node = getattr(node, "env", None)
    raise AttributeError(f"{type(swm_env).__name__} and nothing it wraps has a variation_space")


def check_mapping(env) -> dict:
    """Every leaf of both variation spaces, classified. Used by the example to print a table.

    Returns ``covered`` (swm leaves we render), ``declared_unmapped`` (swm leaves with a recorded
    reason), ``undocumented`` (swm leaves in NEITHER — a gap, and the thing this function exists to
    find), and ``ours_only``.
    """
    import gymnasium
    import stable_worldmodel  # noqa: F401

    m = variation_map(env)
    if m is None:
        return {"error": f"no VariationMap for {type(env).__name__}"}
    ref = gymnasium.make(m.swm_id)
    swm_leaves = set(swm_variation_space(ref).names())
    ref.close()
    mapped = m.mapped_swm_paths()
    unmapped = m.unmapped()
    ours = set(env.variation_space.names())
    ours_mapped = set(m.affine_pairs()) | set(m.rgb255) | set(m.enum)
    cls = m.classification()
    return {
        "swm_id": m.swm_id,
        "covered": sorted(swm_leaves & mapped),
        "declared_unmapped": sorted(swm_leaves & set(unmapped)),
        "undocumented": sorted(swm_leaves - mapped - set(unmapped)),
        "mapped_but_absent_from_swm": sorted(mapped - swm_leaves),
        "ours_only": sorted(ours - ours_mapped),
        "ours_only_documented": sorted(set(m.ours_only)),
        # the Step-8 four-way taxonomy, one label per swm leaf actually offered by this env
        "classification": {p: cls.get(p, "undocumented") for p in sorted(swm_leaves)},
        "reasons": {p: unmapped[p] for p in sorted(swm_leaves & set(unmapped))},
        "notes": {p: m.notes[p] for p in sorted(swm_leaves & set(m.notes))},
    }


# ====================================================================== single-env facade


class StableEnvGym(gym.Env):
    """A gymnasium-shaped single environment backed by a stable-envs batched renderer.

    A real ``gym.Env`` subclass rather than a duck type, because ``gymnasium.make`` wraps whatever
    the entry point returns in ``PassiveEnvChecker``/``OrderEnforcing``/``TimeLimit`` and those
    wrappers reach for ``metadata``, ``render_mode``, ``spec`` and ``np_random``. The contract
    ``stable_worldmodel`` itself needs is smaller and worth listing in one place, because it is what
    a change on either side has to be checked against:

        reset(seed=, options=) -> (obs, info)      step(action) -> 5-tuple      render() -> HWC uint8
        close()                                    action_space / observation_space (real gym spaces)
        unwrapped                                  np_random (with .integers)
        variation_space                            (optional, but every swm env has one)

    **What this adapter carries and what it does not.** It carries the renderer, the appearance, the
    state and — where the underlying env ships one — the dynamics. It does NOT carry a reward
    function, because a reward is a task and stable-envs does not define tasks: it defines what a
    frame is. ``reward`` is the negative distance to the episode's goal state and ``terminated`` is
    that distance falling under ``success_radius``, which defaults to 0 so nothing ever terminates
    unless the caller asks for it. A workflow that needs the official reward should keep the official
    env and swap only its renderer — that is what ``swap_renderer`` is for.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        se_name: str,
        size: int | None = None,
        device: str = "cuda",
        render_mode: str = "rgb_array",
        success_radius: float = 0.0,
        default_variations: tuple[str, ...] = (),
        init_value: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.se_name = se_name
        self.env = make(se_name, num_envs=1, size=size, device=device, **kwargs)
        self.render_mode = render_mode
        self.success_radius = float(success_radius)
        self.default_variations = tuple(default_variations)
        self.variation_space = self.env.make_variation_space()
        if init_value:
            self.variation_space.set_init_value(init_value)
        self.env_name = se_name

        d = self.env.state_dim
        big = np.float32(np.finfo(np.float32).max)
        # A BARE Box rather than a Dict, so that ``EverythingToInfoWrapper`` files it under
        # ``observation`` exactly as it does for the official envs. A Dict observation is exploded
        # into its own keys instead, and the resulting info dict is then missing ``observation`` —
        # measured against a stock World, which is how this was found.
        self.observation_space = gym.spaces.Box(low=-big, high=big, shape=(d,), dtype=np.float32)
        adim = int(getattr(self.env, "action_dim", 1))
        lo = np.broadcast_to(np.float32(getattr(self.env, "action_low", -1.0)), (adim,))
        hi = np.broadcast_to(np.float32(getattr(self.env, "action_high", 1.0)), (adim,))
        self.action_space = gym.spaces.Box(low=lo.copy(), high=hi.copy(), shape=(adim,), dtype=np.float32)
        self._state = np.zeros(d, np.float32)
        self._goal_state = np.zeros(d, np.float32)
        self._goal = None

    # -------------------------------------------------------------- gym surface

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        se_spaces.reset_variation_space(self.variation_space, seed, options or {}, self.default_variations)
        style = self.variation_space.pack()
        options = options or {}
        self._state = np.asarray(
            options.get("state", self.env.sample_states(1, rng=self.np_random)[0]), np.float32
        ).reshape(-1)
        self._goal_state = np.asarray(
            options.get("goal_state", self.env.sample_states(1, rng=self.np_random)[0]), np.float32
        ).reshape(-1)
        self.env.set_state(self._state[None], style=style[None])
        # The goal frame is this same renderer at a different state — which is exactly how the
        # official envs produce theirs (TwoRoomEnv renders the agent AT the target position), so a
        # corpus of states regenerates the goals with no extra machinery.
        self._goal = self.env.render(self._goal_state[None], style=style[None])[0].cpu().numpy()
        self.env.set_state(self._state[None], style=style[None])
        return self._obs(), self._info()

    def step(self, action):
        if not _has_dynamics(self.env):
            raise NotImplementedError(
                f"{self.se_name} ships a renderer and no batched GPU simulator, so this facade "
                f"cannot advance it. Use stable_envs.interop.swap_renderer to keep the "
                f"official env's dynamics and replace only its pixels, or drive it from states you "
                f"already have."
            )
        a = np.asarray(action, np.float32).reshape(1, -1)
        # ``step`` is stateful and its return type is the env's business (Step 9 made it a StepResult
        # TensorDict for migrated cells, a packed tensor for the rest); ``get_state()`` is the packed
        # row on either path and is what this facade has always reported.
        self.env.step(a)
        self._state = self.env.get_state()[0].detach().cpu().numpy().copy()
        dist = float(np.linalg.norm(self._state - self._goal_state))
        terminated = bool(self.success_radius > 0.0 and dist < self.success_radius)
        return self._obs(), -dist, terminated, False, self._info()

    def render(self) -> np.ndarray:
        return self.env.render()[0].detach().cpu().numpy()

    def close(self) -> None:
        pass

    # -------------------------------------------------------------- state access

    def _obs(self) -> np.ndarray:
        return self._state.astype(np.float32).copy()

    def _info(self) -> dict:
        """The key set the official swm envs put in ``info``, plus ``goal``.

        ``state``, ``proprio``, ``goal_state``, ``env_name`` and ``distance_to_target`` are what
        ``TwoRoomEnv._get_info`` and its relatives return, so a policy or a dataset writer that
        reads them keeps working. ``goal`` is an ADDITION: TwoRoomEnv renders a goal frame and keeps
        it privately, PushT publishes one, and a goal-conditioned workflow wants it — so it is always
        present here, and the example reports the key set as a superset rather than as identical.
        """
        s = self._state.astype(np.float32)
        return {
            "env_name": self.env_name,
            "goal": self._goal,
            "state": s.copy(),
            "proprio": s.copy(),
            "goal_state": self._goal_state.astype(np.float32).copy(),
            "distance_to_target": float(np.linalg.norm(s - self._goal_state)),
        }

    def _set_state(self, state) -> None:
        """``swm`` drives envs from a dataset through ``evaluate(callables=...)``, which calls this."""
        self._state = np.asarray(state, np.float32).reshape(-1)
        self.env.set_state(self._state[None])

    def _set_goal_state(self, goal_state) -> None:
        self._goal_state = np.asarray(goal_state, np.float32).reshape(-1)
        self._goal = self.env.render(self._goal_state[None])[0].cpu().numpy()
        self.env.set_state(self._state[None])

    def get_state(self) -> np.ndarray:
        return self._state.copy()


#: ``gymnasium`` id -> our class name, filled in by ``register_gym_ids``
GYM_IDS: dict[str, str] = {}


def register_gym_ids(names=None, prefix: str = "stable_envs", **kwargs) -> dict[str, str]:
    """Register ``<prefix>/<Name>-v0`` for every stable-envs env, so ``gym.make`` reaches them.

    This is what makes the adapter drop-in rather than "drop-in after you write a factory": every
    entry point ``stable_worldmodel`` has — ``World(env_name=...)``, a hydra config, a CLI flag —
    takes a gymnasium id and nothing else. Re-registering an existing id is a no-op rather than an
    error, so calling this twice in one process (a notebook, a test module) is safe.
    """
    from gymnasium.envs import registration

    out = {}
    for name in sorted(names or REGISTRY):
        env_id = f"{prefix}/{name}-v0"
        out[env_id] = name
        GYM_IDS[env_id] = name
        if env_id in registration.registry:
            continue
        registration.register(id=env_id, entry_point=_entry_point, kwargs={"se_name": name, **kwargs})
    return out


def _entry_point(**kwargs) -> StableEnvGym:
    """Module-level so the id is picklable; ``gymnasium`` stores the entry point by reference."""
    return StableEnvGym(**kwargs)


# ====================================================================== batched pool


class _PoolSlot:
    """One environment's worth of a ``BatchedEnvPool``: its variation space and its row index.

    ``swm`` reaches individual envs in two places — ``EnvPool.variation_space`` reads
    ``envs[0].unwrapped``, and ``World.evaluate(callables=...)`` calls methods on ``envs[i]``
    directly — so the pool cannot be a bare tensor. A slot is the smallest object that satisfies
    both while owning no renderer of its own.
    """

    def __init__(self, pool: BatchedEnvPool, index: int):
        self.pool = pool
        self.index = index
        self.variation_space = pool.renderer.make_variation_space()
        self.np_random = np.random.default_rng(index)
        self.env_name = pool.se_name

    @property
    def unwrapped(self) -> _PoolSlot:
        return self

    @property
    def action_space(self):
        return self.pool.single_action_space

    @property
    def observation_space(self):
        return self.pool.single_observation_space

    def get_state(self) -> np.ndarray:
        return self.pool.states[self.index].copy()

    def _set_state(self, state) -> None:
        self.pool.states[self.index] = np.asarray(state, np.float32).reshape(-1)
        self.pool._push_states()

    def _set_goal_state(self, goal_state) -> None:
        self.pool.goal_states[self.index] = np.asarray(goal_state, np.float32).reshape(-1)
        self.pool._render_goals(np.arange(self.pool.num_envs) == self.index)

    def render(self) -> np.ndarray:
        return self.pool.frames[self.index]

    def close(self) -> None:
        pass


class BatchedEnvPool:
    """``swm.world.env_pool.EnvPool``, backed by ONE batched renderer instead of N single envs.

    Assign it over a World's pool and the workflow above it does not change::

        world = swm.World("stable_envs/TwoRoom-v0", num_envs=64)
        world.envs = BatchedEnvPool("TwoRoom", num_envs=64)      # one line
        world.set_policy(policy); world.evaluate(episodes=64)

    What changes is where the time goes: a stock World renders 64 frames with 64 calls into a CPU
    renderer, and this renders them with one kernel launch. The step is batched too wherever the
    stable-envs env ships one.

    Two behaviours are inherited from ``EnvPool`` deliberately because ``World`` depends on them and
    a "simplification" of either is a silent behaviour change:

      * ``reset``/``step`` take a ``mask`` and leave masked environments completely alone, including
        their slot in the stacked info dict. The batched ``step`` advances every row, so the masked
        rows are restored from a snapshot afterwards — one extra copy rather than a scatter, because
        a partial launch would cost more than the copy at every batch size that matters. The restore
        goes through ``set_state``, which on some envs also resets dynamical state the compact state
        does not carry (PushT zeroes the agent's velocity, deliberately and exactly as ``reset``
        does), so a masked step is not free of side effects on those envs. It is only taken in the
        ``wait`` reset mode, where the frozen environments are finished anyway.
      * the stacked info dict is allocated on the first full reset and written in place afterwards,
        with a leading ``(N, 1, ...)`` shape. Anything that appears only at step time is dropped by
        ``World``, so every key is produced on both paths.
    """

    def __init__(
        self,
        se_name: str,
        num_envs: int,
        size: int | None = None,
        device: str = "cuda",
        success_radius: float = 0.0,
        default_variations: tuple[str, ...] = (),
        watch_variations: tuple[str, ...] = (),
        image_shape=None,
        **kwargs,
    ):
        from gymnasium.vector.utils import batch_space

        self.se_name = se_name
        self.renderer = make(se_name, num_envs=int(num_envs), size=size, device=device, **kwargs)
        self.success_radius = float(success_radius)
        self.default_variations = tuple(default_variations)
        self.watch_variations = tuple(watch_variations)
        if image_shape is not None and tuple(image_shape)[:2] != (self.renderer.size, self.renderer.size):
            raise ValueError(
                f"image_shape={tuple(image_shape)} is not the renderer's native "
                f"{self.renderer.size}x{self.renderer.size}. A stock World resizes with PIL on the "
                f"CPU per environment per step, which is neither batched nor bit-reproducible on "
                f"another Pillow; this pool renders at the native size on purpose. Construct the "
                f"renderer at the size you want (size=) or resize the returned frames yourself."
            )

        self.envs = [_PoolSlot(self, i) for i in range(int(num_envs))]
        self.single_action_space, self.single_observation_space = _spaces_for(self.renderer)
        self._action_space = batch_space(self.single_action_space, self.num_envs)
        self._observation_space = batch_space(self.single_observation_space, self.num_envs)

        d = self.renderer.state_dim
        self.states = np.zeros((self.num_envs, d), np.float32)
        self.goal_states = np.zeros((self.num_envs, d), np.float32)
        self.style = np.tile(self.renderer.default_style(), (self.num_envs, 1))
        self.seeds = np.zeros(self.num_envs, dtype=np.int64)
        self.frames = None
        self.goals = None
        self._step_idx = np.zeros(self.num_envs, np.int64)
        self._last_action = np.zeros((self.num_envs, self.single_action_space.shape[0]), np.float32)
        self._reward = np.zeros(self.num_envs, np.float32)
        self._terminated = np.zeros(self.num_envs, bool)
        self._render_time = np.zeros(self.num_envs, np.float32)
        self._stacked = None

    # -------------------------------------------------------------- EnvPool surface

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def variation_space(self):
        return self.envs[0].variation_space

    @property
    def single_variation_space(self):
        return self.variation_space

    def close(self) -> None:
        pass

    def reset(self, seed=None, options=None, mask=None):
        seeds = _broadcast(seed, self.num_envs, increment=True)
        opts = _broadcast(options, self.num_envs)
        sel = np.ones(self.num_envs, bool) if mask is None else np.asarray(mask, bool)
        for i in range(self.num_envs):
            if not sel[i]:
                continue
            slot = self.envs[i]
            if seeds[i] is not None:
                slot.np_random = np.random.default_rng(seeds[i])
                self.seeds[i] = seeds[i]
            o = opts[i] or {}
            se_spaces.reset_variation_space(slot.variation_space, seeds[i], o, self.default_variations)
            self.style[i] = slot.variation_space.pack()
            self.states[i] = np.asarray(
                o.get("state", self.renderer.sample_states(1, rng=slot.np_random)[0]), np.float32
            ).reshape(-1)
            self.goal_states[i] = np.asarray(
                o.get("goal_state", self.renderer.sample_states(1, rng=slot.np_random)[0]), np.float32
            ).reshape(-1)
            self._step_idx[i] = 0
            self._reward[i] = 0.0
            self._terminated[i] = False
            self._last_action[i] = 0.0
        self._push_states()
        self._render_goals(sel)
        self._render_frames()
        return None, self._infos()

    def step(self, actions, mask=None):
        a = np.asarray(actions, np.float32).reshape(self.num_envs, -1)
        sel = np.ones(self.num_envs, bool) if mask is None else np.asarray(mask, bool)
        rewards = np.zeros(self.num_envs, np.float32)
        terminateds = np.zeros(self.num_envs, bool)
        truncateds = np.zeros(self.num_envs, bool)

        if not _has_dynamics(self.renderer):
            raise NotImplementedError(
                f"{self.se_name} ships a renderer and no batched GPU simulator, so a pool of them "
                f"cannot be stepped. Keep the official envs and swap only their renderer "
                f"(swap_renderer), or render states you already have."
            )
        prev = self.renderer.get_state().clone()
        act = np.where(sel[:, None], a, 0.0).astype(np.float32)
        self.renderer.step(torch.as_tensor(act, device=self.renderer.device))
        new = self.renderer.get_state()
        if not sel.all():
            keep = torch.as_tensor(~sel, device=self.renderer.device)
            new = torch.where(keep.unsqueeze(1), prev, new)
            self.renderer.set_state(new)
        self.states = self.renderer.get_state().detach().cpu().numpy().copy()

        dist = np.linalg.norm(self.states - self.goal_states, axis=1)
        rewards[sel] = (-dist)[sel]
        if self.success_radius > 0.0:
            terminateds[sel] = (dist < self.success_radius)[sel]
        self._reward[sel] = rewards[sel]
        self._terminated[sel] = terminateds[sel]
        self._last_action[sel] = a[sel]
        self._step_idx[sel] += 1
        self._render_frames()
        return None, rewards, terminateds, truncateds, self._infos()

    # -------------------------------------------------------------- internals

    def _push_states(self) -> None:
        self.renderer.set_state(self.states, style=self.style)

    def _render_goals(self, sel) -> None:
        g = self.renderer.render(self.goal_states, style=self.style).detach().cpu().numpy()
        self.goals = g if self.goals is None else np.where(np.asarray(sel, bool).reshape(-1, 1, 1, 1), g, self.goals)
        self._push_states()

    def _render_frames(self) -> None:
        t0 = time.perf_counter()
        self.frames = self.renderer.render(self.states, style=self.style).detach().cpu().numpy()
        self._render_time[:] = (time.perf_counter() - t0) / max(1, self.num_envs)

    def _infos(self) -> dict:
        """The stacked info dict, in ``MegaWrapper``'s key set and ``EnvPool``'s ``(N,1,...)`` shape.

        Built fresh each call rather than written into a preallocated buffer. That is the one place
        this pool deliberately differs from ``EnvPool``, and it is the safer side of the difference:
        ``EnvPool`` returns the SAME dict object every step, so ``World``'s ``infos['_needs_flush']
        = done`` survives into steps where nothing was flushed, and only ``WorldModelPolicy``'s pop —
        which acts on a COPY made by ``_prepare_info`` — hides it. A fresh dict cannot carry a stale
        flag, and ``World`` reassigns ``self.infos`` from the return value on every call, so nothing
        upstream depends on the identity.

        Masked environments need no special handling: every array here is only written for the rows
        the caller selected, so an unselected row still holds what it held before.
        """
        rows = {
            "pixels": self.frames,
            "render_time": self._render_time,
            "observation": self.states,
            "state": self.states,
            "proprio": self.states,
            "distance_to_target": np.linalg.norm(self.states - self.goal_states, axis=1).astype(np.float32),
            "reward": self._reward,
            "terminated": self._terminated,
            "truncated": np.zeros(self.num_envs, bool),
            "action": self._last_action,
            "step_idx": self._step_idx,
            "id": np.arange(self.num_envs),
            "goal": self.goals,
            "goal_state": self.goal_states,
            "env_name": [self.se_name] * self.num_envs,
        }
        for path in self.watch_variations:
            node = self.envs[0].variation_space
            for part in path.split("."):
                node = node[part]
            rows[f"variation.{path}"] = np.stack(
                [
                    np.asarray(_leaf_value(self.envs[i].variation_space, path), np.float32).reshape(-1)
                    for i in range(self.num_envs)
                ]
            )
        stacked = {}
        for k, v in rows.items():
            stacked[k] = [[x] for x in v] if isinstance(v, list) else np.asarray(v)[:, None, ...]
        self._stacked = stacked
        return stacked


def _leaf_value(space, path):
    node = space
    for part in path.split("."):
        node = node[part]
    return node.value


def _spaces_for(renderer):
    d = renderer.state_dim
    big = np.float32(np.finfo(np.float32).max)
    obs = gym.spaces.Box(low=-big, high=big, shape=(d,), dtype=np.float32)
    adim = int(getattr(renderer, "action_dim", 1))
    lo = np.broadcast_to(np.float32(getattr(renderer, "action_low", -1.0)), (adim,)).copy()
    hi = np.broadcast_to(np.float32(getattr(renderer, "action_high", 1.0)), (adim,)).copy()
    return gym.spaces.Box(low=lo, high=hi, shape=(adim,), dtype=np.float32), obs


def _broadcast(arg, n: int, increment: bool = False) -> list:
    """``EnvPool._broadcast_arg``, reproduced because World depends on the seed-incrementing rule."""
    if arg is None:
        return [None] * n
    if isinstance(arg, list):
        return arg
    if isinstance(arg, np.ndarray):
        return list(arg)
    if increment and isinstance(arg, (int, np.integer)):
        return [int(arg) + i for i in range(n)]
    return [arg] * n


def attach_batched_pool(world, **kwargs) -> BatchedEnvPool:
    """Replace a World's env pool with a batched one built from the same env, in place.

    Reads the stable-envs env name off the World's existing pool, so the call site is a single line
    with no repeated configuration. The old pool is closed first, which for the ``StableEnvGym``
    facades is a no-op and for anything holding a renderer is not.
    """
    slot = world.envs.envs[0].unwrapped
    name = getattr(slot, "se_name", None)
    if name is None:
        raise TypeError(
            f"{type(slot).__name__} is not a stable-envs facade, so there is no batched renderer to "
            f"attach. Build the World on a stable_envs/<Name>-v0 id (register_gym_ids), or pass the "
            f"name explicitly to BatchedEnvPool."
        )
    kwargs.setdefault("size", slot.env.size)
    kwargs.setdefault("device", str(slot.env.device))
    kwargs.setdefault("success_radius", slot.success_radius)
    kwargs.setdefault("default_variations", slot.default_variations)
    world.envs.close()
    world.envs = BatchedEnvPool(name, num_envs=world.num_envs, **kwargs)
    return world.envs


# ====================================================================== renderer swap


def swap_renderer(
    gym_env,
    se_name: str,
    read_state: Callable[[Any], np.ndarray],
    style=None,
    size: int | None = None,
    device: str = "cuda",
):
    """Keep an OFFICIAL swm env's dynamics and replace only its ``render()``.

    The strongest evidence available for a renderer, and the reason it is worth a separate adapter:
    with the official env still deciding every transition, the two arms visit exactly the same
    states, so any difference downstream is the pixels and nothing else. It is also the only adapter
    that works for the envs we render and do not simulate.

    ``read_state`` maps the unwrapped official env to our compact state — four lines that have to be
    written once per env family and that are the entire coupling between the two libraries.
    """
    env = make(se_name, num_envs=1, size=size, device=device)
    if style is not None:
        env.set_style(np.asarray(style, np.float32).reshape(1, -1))
    u = getattr(gym_env, "unwrapped", gym_env)
    original = u.render

    def render(*a, **kw):
        return env.render(np.asarray(read_state(u), np.float32).reshape(1, -1))[0].cpu().numpy()

    u.render = render
    u._stable_envs_original_render = original
    u._stable_envs_renderer = env
    return gym_env


def pusht_state(u) -> np.ndarray:
    """``swm/PushT-v1`` (and ``gym_pusht/PushT-v0``) -> our 5-float PushT state.

    Both keep the agent and the block as live pymunk bodies, and both report the block's ORIGIN
    rather than its centre of gravity, which is the quantity our renderer draws from.
    """
    return np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)
