"""SWM variation correspondence: evidence, not a shipped stable-envs API.

These tables record how stable-worldmodel variation leaves relate to
stable-envs Variation leaves. They are imported only by tests and
regression scripts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

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
    #: two-tuple is what the four earlier maps are written in and a silent arity change is the one
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
    #: four-way taxonomy; new maps should use one of the three below instead.
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
    #: and the reason must carry the measured residual. A feasible, required leaf is not declared
    #: unsupported for want of a renderer capability.
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
        would be a silent breakage of the four earlier maps. See ``affine_pairs``.
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

        The four-way taxonomy, per leaf, from the map itself rather than from prose beside it.
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

    def replacement_class(self) -> dict[str, str]:
        """``swm_path -> NATIVE | PARTIAL | DEAD_UPSTREAM | UNSUPPORTED_MEANINGFUL``.

        Overlay on :meth:`classification` for environment-replacement readiness. The six stored
        buckets stay as they are so existing tests keep passing: a mass/speed/reset leaf that was
        ``not_applicable`` because it moved no pixel is still ``not_applicable`` there, and
        ``UNSUPPORTED_MEANINGFUL`` here. Do not treat "changes no pixel" as a reason to drop a
        real dynamics/reset/task capability.
        """
        cls = self.classification()
        sibling_native = self._sibling_native_paths()
        has_sibling = self._n_maps_for_swm_id() >= 2
        out: dict[str, str] = {}
        for path, bucket in cls.items():
            if bucket == "partial":
                out[path] = "PARTIAL"
            elif bucket == "mapped":
                out[path] = "NATIVE"
            elif bucket == "dead_upstream":
                out[path] = "DEAD_UPSTREAM"
            elif bucket == "unsupported":
                out[path] = "UNSUPPORTED_MEANINGFUL"
            else:
                out[path] = self._replacement_for_unmapped(path, sibling_native, has_sibling)
        return out

    def _n_maps_for_swm_id(self) -> int:
        return sum(1 for m in VARIATION_MAPS.values() if m.swm_id == self.swm_id)

    def _sibling_native_paths(self) -> set[str]:
        paths: set[str] = set()
        for other in VARIATION_MAPS.values():
            if other.swm_id != self.swm_id:
                continue
            paths |= other.mapped_swm_paths() - set(other.partial)
        return paths

    def _unmapped_reason(self, path: str) -> str:
        for bucket in (self.swm_only, self.unsupported, self.dead_upstream, self.not_applicable):
            if path in bucket:
                return str(bucket[path])
        return ""

    def _replacement_for_unmapped(self, path: str, sibling_native: set[str], has_sibling: bool) -> str:
        if path in sibling_native or (path in _CLASS_SPLIT_FLAGS and has_sibling):
            return "NATIVE"
        reason = self._unmapped_reason(path).lower()
        if _is_meaningful_config(path, reason):
            return "UNSUPPORTED_MEANINGFUL"
        if _looks_dead(reason):
            return "DEAD_UPSTREAM"
        return "UNSUPPORTED_MEANINGFUL"


#: Class-split rendering flags: the sibling class is the native surface (TwoRoom vs TwoRoomTarget).
_CLASS_SPLIT_FLAGS = frozenset({"rendering.render_target", "rendering.render_zones"})

_MEANINGFUL_PATH_NEEDLES = (
    "speed",
    "mass",
    "gravity",
    "density",
    "velocity",
    "start_position",
    "start_height",
    "start_horizontal",
    "start_tilt",
    "start_yaw",
    "min_steps",
    "physics.",
    "agent.shape",
    "block.shape",
)

_MEANINGFUL_REASON_NEEDLES = (
    "dynamics",
    "reset distribution",
    "reset pose",
    "reset placement",
    "reset orientation",
    "reset altitude",
    "a reset",
    "task definition",
    "task semantics",
    "our state",
    "not our style",
    "zone_bias",
    "pymunk shapes",
    "show_goal",
)

_DEAD_REASON_NEEDLES = (
    "dead upstream",
    "dead at the appearance",
    "conditionally dead",
    "never read",
    "read by nothing",
    "nothing ever reads",
    "rotationally symmetric",
    "reaches no model array",
    "no-op",
)


def _is_meaningful_config(path: str, reason: str) -> bool:
    p = path.lower()
    if any(n in p for n in _MEANINGFUL_PATH_NEEDLES):
        return True
    return any(n in reason for n in _MEANINGFUL_REASON_NEEDLES)


def _looks_dead(reason: str) -> bool:
    return any(n in reason for n in _DEAD_REASON_NEEDLES)


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
    "wall.border_color": "declared by TwoRoomEnv and read by nothing (measured: 0/255 diff)",
    "rendering.render_target": "selects TwoRoom vs TwoRoomTarget, which are separate classes",
    "target.color": "drawn only when render_target is on; see TwoRoomTarget",
    "target.radius": "drawn only when render_target is on; see TwoRoomTarget",
    "target.position": "TwoRoomTarget carries it in the STATE, because the frame reads it",
}
#: Read by nothing in SWM: TwoRoom's ``_constrain_target_by_min_steps`` is commented out as the
#: target leaf's ``constrain_fn`` and Piecewise has no reader at all.
_MIN_STEPS_DEAD = {
    "task.min_steps": "dead upstream, measured by variation_navigation.min_steps_probe: min_steps 15 "
    "and 114 against the default 25, 5 seeds x (reset + 8 steps) on TwoRoom and Piecewise, give "
    "identical State and 0 of 4,515,840 px moved per env",
}
_PIECEWISE_SWM_ONLY = {
    "agent.position": "this is our STATE, not our style",
    "target.color": "drawn only when render_target is on; see Piecewise",
    "target.radius": "drawn only when render_target is on; see Piecewise",
    "target.position": "Piecewise carries it in the STATE",
    "rendering.render_target": "drawn by Piecewise, which carries the target in its State",
    "rendering.render_zones": "selects Piecewise vs PiecewiseZones, which are separate classes",
}
_PIECEWISE_GATE = (
    "gated by tests/integration/test_variation_navigation.py against swm: reset State, rollout State and frames"
)

#: our class name -> how its appearance relates to the official swm env's
VARIATION_MAPS: dict[str, VariationMap] = {
    # ``agent.position`` and ``target.position`` are per-lane reset distributions (``Env.reset_leaves``)
    # that land in State, not style columns; tests/integration/test_variation_navigation.py gates
    # them against swm.
    "TwoRoom": VariationMap(
        swm_id="swm/TwoRoom-v1",
        identical=("agent.color", "agent.radius", "agent.speed", "agent.position", "target.position")
        + _TWOROOM_STRUCTURE,
        swm_only={k: v for k, v in _TWOROOM_SWM_ONLY.items() if k not in ("agent.position", "target.position")},
        dead_upstream=_MIN_STEPS_DEAD,
    ),
    "TwoRoomTarget": VariationMap(
        swm_id="swm/TwoRoom-v1",
        identical=("agent.color", "agent.radius", "agent.speed", "target.color", "target.radius") + _TWOROOM_STRUCTURE,
        swm_only={k: v for k, v in _TWOROOM_SWM_ONLY.items() if k not in ("target.color", "target.radius")},
        dead_upstream=_MIN_STEPS_DEAD,
    ),
    "Piecewise": VariationMap(
        swm_id="swm/Piecewise-v0",
        identical=(
            "agent.color",
            "agent.radius",
            "agent.speed",
            "agent.position",
            "background.color",
            "border.color",
            "target.color",
            "target.radius",
            "target.position",
            "rendering.render_target",
            "rendering.render_bias_field",
        )
        + tuple(f"zones.bias_{i}" for i in range(4)),
        swm_only={
            "rendering.render_zones": _PIECEWISE_SWM_ONLY["rendering.render_zones"],
            **{f"zones.color_{i}": "drawn only with render_zones; see PiecewiseZones" for i in range(4)},
        },
        dead_upstream=_MIN_STEPS_DEAD,
        notes={
            "agent.position": "a per-lane reset leaf in State; " + _PIECEWISE_GATE,
            "target.position": "a per-lane reset leaf in State; " + _PIECEWISE_GATE,
            "rendering.render_target": "the target is drawn from the full State's hidden target; "
            "compact rows alone cannot draw it. " + _PIECEWISE_GATE,
            "rendering.render_bias_field": "the arrows are rasterised by the same Matplotlib Agg quiver "
            "as swm, so their pixels are exact only at the same Matplotlib version. " + _PIECEWISE_GATE,
        },
    ),
    "PiecewiseZones": VariationMap(
        swm_id="swm/Piecewise-v0",
        identical=(
            "agent.color",
            "agent.radius",
            "agent.speed",
            "background.color",
            "border.color",
            "rendering.render_bias_field",
        )
        + tuple(f"zones.color_{i}" for i in range(4))
        + tuple(f"zones.bias_{i}" for i in range(4)),
        swm_only=_PIECEWISE_SWM_ONLY,
        dead_upstream=_MIN_STEPS_DEAD,
    ),
    # The public PushT owns swm's variation space leaf for leaf (same paths, units, ranges and
    # defaults) and runs the same reset and physics, so every leaf is identical. Start positions,
    # block.angle and agent.velocity are reset values that land in State; agent.shape and
    # block.shape rebuild the pymunk bodies. Gated per lane against swm, explicit and sampled, by
    # tests/integration/test_variation_pusht.py.
    "PushT": VariationMap(
        swm_id="swm/PushT-v1",
        identical=(
            "agent.color",
            "agent.scale",
            "agent.shape",
            "agent.angle",
            "agent.start_position",
            "agent.velocity",
            "block.color",
            "block.scale",
            "block.shape",
            "block.angle",
            "block.start_position",
            "goal.color",
            "goal.scale",
            "goal.position",
            "goal.angle",
            "background.color",
            "rendering.render_goal",
        ),
        notes={
            "agent.angle": "turns only a non-circular agent.shape; the default disc is symmetric",
        },
    ),
}


#: Certified maps live one module per family under ``variation_maps`` and install themselves
#: here; see that package's docstring for why they are not more entries in the literal above.
from .variation_maps import install as _install_variation_maps

_install_variation_maps(VARIATION_MAPS)  # noqa: E402,E501

# swm's PushTDiscrete subclasses PushT and keeps its variation space; test_variation_pusht gates both classes.
VARIATION_MAPS["PushTDiscrete"] = replace(VARIATION_MAPS["PushT"], swm_id="swm/PushT-Discrete-v1")


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
    """The mapping for an env or env name, or ``None`` if this env has no swm counterpart.

    Maps are keyed by the frozen support-matrix cell spelling. Live class names are translated
    at lookup rather than registered twice.
    """
    from .crosswalk import canonical_from_frozen, frozen_from_canonical

    name = env if isinstance(env, str) else type(env).__name__
    return (
        VARIATION_MAPS.get(name)
        or VARIATION_MAPS.get(canonical_from_frozen(name))
        or VARIATION_MAPS.get(frozen_from_canonical(name))
    )


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
            f"VariationMap to tests.helpers.reference.VARIATION_MAPS"
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
        # the four-way taxonomy, one label per swm leaf actually offered by this env
        "classification": {p: cls.get(p, "undocumented") for p in sorted(swm_leaves)},
        "reasons": {p: unmapped[p] for p in sorted(swm_leaves & set(unmapped))},
        "notes": {p: m.notes[p] for p in sorted(swm_leaves & set(m.notes))},
    }
