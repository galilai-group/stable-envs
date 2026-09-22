"""A per-episode ``*_quat`` slot must reach the frame, and it must reach it PER ROW.

``_episode_slots`` used to refuse every slot ending ``_size``, ``_quat`` or ``_dir`` with one reason:
"the mover primitives carry one size and one orientation for the whole batch". Half of that is true.
A per-episode SIZE really is shared -- ``_prims`` reads ``geom_size`` once at bake time and
``screen_span`` derives its bound from ``sc.prims.size`` -- and so is a light direction, captured once
per bake by ``_Lights``. ORIENTATION is not: ``_fk`` writes the drawn slot into the model, runs
``mj_kinematics``, and reads ``d.geom_xmat`` into a ``(B, P, 3, 3)`` tensor that ``R3.cast_nearest``,
``R3.any_blocking`` and ``_reflect`` all consume per row.

``manipulator-bring_ball`` is the domain that made the difference matter: its ``initialize_episode``
writes ``body_quat['target_ball']`` every episode, so the over-broad guard refused the whole domain.

What is asserted here is the claim the guard's replacement rests on, not the fact that manipulator
happens to load:

  * the guard still refuses ``_size`` and ``_dir``, and no longer refuses ``_quat``;
  * ``_fk`` returns an ``xmat`` that DIFFERS ACROSS ROWS when only the quaternion slots differ --
    i.e. the orientation really is per environment and not the first row's broadcast;
  * the OFFICIAL renderer moves pixels for those slots, so the state dimension is not decoration.

A change that made ``xmat`` batch-shared would render every environment at one draw's orientation and
would pass every existing test; this is the line that fails instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

try:
    import stable_envs as se
    import stable_envs.envs.dmcontrol as D
except ImportError as exc:  # dm_control / mujoco absent: legitimately skippable
    pytest.skip(
        f"stable_envs.envs.dmcontrol needs dm_control ({exc}); this file has nothing to assert without it",
        allow_module_level=True,
    )
except RuntimeError as exc:
    # Same narrow guard as test_dmcontrol_tendon.py: a machine with no usable offscreen GL raises
    # RuntimeError out of mujoco's gl_context at import, and only that is skippable. A dmcontrol.py
    # broken by an edit must reach the runner rather than silently skipping these assertions.
    if "MUJOCO_GL" not in str(exc) and "GL" not in str(exc):
        raise
    pytest.skip(f"no usable offscreen GL context here ({exc})", allow_module_level=True)

CELL = "DMCManipulatorBringBall"


def test_the_guard_refuses_size_and_direction_and_not_orientation():
    """The guard's SHAPE, read off the module rather than off its docstring.

    Asserted as a set membership so that widening it further -- to allow a per-episode ``geom_size``,
    which really would render every environment at one draw's size -- cannot happen silently.
    """
    import inspect

    src = inspect.getsource(D._episode_slots)
    assert '("_size", "_dir")' in src, (
        "the _episode_slots guard is no longer ('_size', '_dir'). A per-episode SIZE is shared by the "
        "batch at bake time (_prims / screen_span) and a light DIRECTION is captured once per bake "
        "(_Lights), so both must stay refused; ORIENTATION goes through mj_kinematics into the "
        f"per-row xmat and must stay allowed. Current source:\n{src}"
    )
    assert '"_quat"' not in src.split("bad = ")[1].split("\n")[0], (
        "the guard refuses per-episode orientation again; see this file's docstring for why that is "
        "not the same class of problem as a per-episode size"
    )


@pytest.fixture(scope="module")
def cell():
    if CELL not in se.registry.REGISTRY:
        pytest.skip(f"{CELL} is not registered in this interpreter")
    return se.registry.REGISTRY[CELL]


def test_manipulator_widens_the_state_with_the_quaternion_slots(cell):
    """The census finds the four floats ``Bring.initialize_episode`` writes, and names them."""
    slots = cell._slots()
    arrays = sorted({a for a, _ in slots})
    assert arrays == ["body_pos", "body_quat"], f"{CELL}'s episode draw is no longer target_ball's pose: {slots}"
    names = cell._slot_names(cell._model())
    assert set(names) == {"target_ball_pos_x", "target_ball_pos_z", "target_ball_quat_qw", "target_ball_quat_qy"}, (
        names
    )
    env = cell(num_envs=1, device="cpu", seed=0)
    assert env.state_dim == env.nq + 4 == 15


def test_fk_orientation_is_per_row(cell):
    """``_fk``'s ``xmat`` differs ACROSS ROWS when only the quaternion slots differ.

    This is the property the guard's removal rests on. The two rows share every ``qpos`` entry and
    every positional slot, so a batch-shared orientation would make the two ``xmat`` stacks equal.
    """
    env = cell(num_envs=2, device="cpu", seed=0)
    names = list(env.state_names)
    iw, iy = names.index("target_ball_quat_qw"), names.index("target_ball_quat_qy")
    st = env.sample_states(1, np.random.default_rng(7))
    two = np.repeat(st, 2, axis=0)
    # row 1 gets the same target position and a rotation 0.7 rad further round y
    ang = 2.0 * np.arctan2(two[1, iy], two[1, iw]) + 0.7
    two[1, iw], two[1, iy] = np.cos(ang / 2), np.sin(ang / 2)

    sc = env._scene()
    _, xmat, _, _ = env._fk(torch.as_tensor(two, dtype=torch.float32), sc)
    assert xmat.shape[0] == 2 and xmat.dim() == 4, xmat.shape
    per_prim = (xmat[0] - xmat[1]).abs().amax(dim=(1, 2))
    assert bool((per_prim > 1e-6).any()), (
        "_fk returned the same orientation for two rows whose target_ball quaternion differs by "
        "0.7 rad: the mover orientation is batch-shared, so every environment would be drawn at one "
        "row's orientation. xmat max row-to-row difference per primitive: "
        f"{per_prim.tolist()}"
    )


def test_the_official_renderer_moves_for_the_quaternion_slots(cell):
    """The widened dimensions are real in the REFERENCE, whatever this renderer does with them.

    Deliberately asserted about ``physics.render`` and not about us: ``target_ball`` is an untextured
    SPHERE, so MuJoCo's tessellated per-vertex shading moves a few hundred pixels under rotation while
    an analytic per-pixel sphere is exactly rotation-invariant. That gap is declared in
    ``STATE_DIMS_NOTE`` and recorded as a ``check_state_dims_matter`` failure; what must not be true is
    that the state carries a dimension the reference itself ignores.
    """
    from stable_envs.testing import conformance as C

    env = cell(num_envs=1, device="cpu", seed=0)
    names = list(env.state_names)
    iw, iy = names.index("target_ball_quat_qw"), names.index("target_ball_quat_qy")
    st = env.sample_states(2, np.random.default_rng(11))
    # Through the POOL, never ``cls.reference_env`` into a local: the returned object owns an EGL
    # context and CPython frees it when the local goes out of scope, which corrupts every other live
    # context in the process. ``conformance.reference_env`` keeps it for the life of the process and
    # ``resource_guard`` fails this test if it is dropped -- see that function's docstring.
    try:
        ref = C.reference_env(cell, env.size)
        base = cell.reference_render(ref, st)
    except RuntimeError as exc:
        if "OpenGL" not in str(exc) and "MUJOCO_GL" not in str(exc):
            raise
        pytest.skip(f"dm_control official renderer has no usable GL backend here ({exc})")
    rot = st.copy()
    ang = 2.0 * np.arctan2(rot[:, iy], rot[:, iw]) + 0.6
    rot[:, iw], rot[:, iy] = np.cos(ang / 2), np.sin(ang / 2)
    moved = cell.reference_render(ref, rot)
    d = np.abs(base.astype(np.int16) - moved.astype(np.int16)).max(-1)
    assert int((d > 0).sum()) > 0, (
        "rotating target_ball changed 0 pixels in the OFFICIAL renderer, so the two quaternion slots "
        "would be dead in the reference as well as here and should not be in the state at all"
    )


def test_a_body_slot_becomes_one_visibility_leaf_over_its_geoms(cell):
    """``_hideable`` resolves a drawn BODY to the geoms hanging off it, named by the body.

    ``reacher`` and ``finger-turn`` draw the element that is drawn; manipulator draws a body, which
    has no rgba of its own. Without this the cell declares no ``visibility.*`` leaf at all and
    ``stable_worldmodel``'s ``render_target = 0`` is unreachable.
    """
    import mujoco

    hide = cell._hideable()
    assert len(hide) == 1, hide
    kind, name, ids = hide[0]
    assert (kind, name) == ("geom", "target_ball"), hide
    m = cell._model()
    assert len(ids) >= 1
    for gid in ids:
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) == "target_ball"

    env = cell(num_envs=2, device="cpu", seed=0)
    assert "visibility.target_ball" in env.variation_space.names()
    st = env.sample_states(2, np.random.default_rng(3))
    shown = env._render(torch.as_tensor(st)).cpu().numpy()
    hidden = env._render(torch.as_tensor(st), variation={"visibility.target_ball": [0]}).cpu().numpy()
    assert not np.array_equal(shown, hidden), (
        "hiding visibility.target_ball changed no pixel, so the body slot did not reach the geoms"
    )


def test_the_shape_leaf_default_is_the_compiled_model(cell):
    """``target_ball.shape``'s default is the index that matches the compiled model, and the other
    alternatives -- ``stable_worldmodel``'s 0.015 box and 0.015 sphere -- actually change the frame."""
    env = cell(num_envs=2, device="cpu", seed=0)
    leaf = env.variation_space["target_ball"]["shape"]
    assert leaf.n == 3, leaf.n
    default = int(np.asarray(leaf.value).reshape(-1)[0])
    st = env.sample_states(2, np.random.default_rng(13))
    base = env._render(torch.as_tensor(st)).cpu().numpy()
    same = env._render(torch.as_tensor(st), variation={"target_ball.shape": [default]}).cpu().numpy()
    assert np.array_equal(base, same), "target_ball.shape's default is not the default appearance"
    for k in range(leaf.n):
        if k == default:
            continue
        other = env._render(torch.as_tensor(st), variation={"target_ball.shape": [k]}).cpu().numpy()
        assert not np.array_equal(base, other), (
            f"target_ball.shape = {k} changed no pixel; a declared alternative that cannot be seen "
            f"fails check_variations_do_something and advertises support that is not there"
        )


# --------------------------------------------------------------------------- synthetic bodies
#
# ``manipulator`` is the only registered cell with a body slot and its ``target_ball`` body carries
# exactly ONE drawn geom, so the N>1 path and the both-kinds raise below are reachable from no env in
# the registry. Review asked for them to be covered rather than left to the next domain, and they are
# covered against a hand-built model instead of a real one: the question is about ``_hideable``'s
# resolution rule, not about any suite task.

_SYNTH_XML = """
<mujoco model="synthetic body slot">
  <asset><material name="mk" rgba=".6 .3 .3 1"/></asset>
  <worldbody>
    <light name="l" directional="true" pos="0 0 1" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 1"/>
    <body name="marker" pos="0 0 .3">
      <geom name="marker_a" type="sphere" size=".02" material="mk"/>
      <geom name="marker_b" type="box" size=".01 .01 .01" pos=".05 0 0" material="mk"/>
      <geom name="marker_hidden" type="sphere" size=".02" pos="-.05 0 0" rgba="1 0 0 0"/>
      <geom name="marker_grouped" type="sphere" size=".02" pos="0 .05 0" group="3" material="mk"/>
      {extra_site}
    </body>
  </worldbody>
</mujoco>
"""


def _synth_cls(extra_site=""):
    """A ``DMControlSuite`` subclass over a hand-built model whose episode draws ``body_pos[marker]``.

    Only ``_slots``/``_model``/``_pristine`` are overridden -- ``_hideable`` is the shipped one, which
    is the whole point.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_string(_SYNTH_XML.format(extra_site=extra_site))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "marker")
    pristine = dict(
        mat_rgba=np.array(m.mat_rgba, copy=True),
        geom_rgba=np.array(m.geom_rgba, copy=True),
        site_rgba=np.array(m.site_rgba, copy=True),
        geom_group=np.array(m.geom_group, copy=True),
        site_group=np.array(m.site_group, copy=True),
    )

    class Synth(D.DMControlSuite):
        _M, _PR = m, pristine

        @classmethod
        def _slots(cls):
            return (("body_pos", 3 * bid), ("body_pos", 3 * bid + 2))

        @classmethod
        def _model(cls):
            return cls._M

        @classmethod
        def _pristine(cls):
            return cls._PR

    return Synth


def test_a_body_slot_with_several_drawn_geoms_resolves_to_all_of_them():
    """N > 1, and the two filters that must still apply inside the body.

    ``marker`` carries four geoms: two drawn, one at alpha 0 and one in group 3. The leaf must cover
    the two drawn ones and neither of the others -- a leaf that included a hidden or undrawn geom
    would advertise a knob that cannot change a pixel, which is what
    ``check_variations_do_something`` fails on.
    """
    import mujoco

    cls = _synth_cls()
    hide = cls._hideable()
    assert len(hide) == 1, hide
    kind, name, ids = hide[0]
    assert (kind, name) == ("geom", "marker"), hide
    got = sorted(mujoco.mj_id2name(cls._M, mujoco.mjtObj.mjOBJ_GEOM, i) for i in ids)
    assert got == ["marker_a", "marker_b"], (
        f"the body leaf resolved to {got}; marker_hidden is alpha 0 and marker_grouped is in group 3, "
        f"and neither is drawn at the default MjvOption"
    )


def test_a_body_carrying_both_drawn_geoms_and_drawn_sites_raises():
    """One leaf cannot name two element kinds, and hiding half a marker silently is worse than failing.

    ``_apply_variation`` zeroes ``<kind>_rgba[eid, 3]`` for one ``kind``, so a body with both would
    have its geoms hidden and its site left drawn.
    """
    cls = _synth_cls(extra_site='<site name="marker_s" type="sphere" size=".02" group="0"/>')
    with pytest.raises(RuntimeError, match="drawn geoms .* and drawn sites"):
        cls._hideable()


# --------------------------------------------------------- render-invariant exclusion (Sami, approved)
#
# The three dimensions ball_y / target_ball_quat_qw / target_ball_quat_qy orient an UNTEXTURED SPHERE,
# which an analytic per-pixel intersector cannot draw a difference for. They are excluded from
# check_state_dims_matter by a VALIDATED contract: the env declares candidates, the check verifies
# against the compiled model that perturbing one moves only sphere geom_xmat, and refuses otherwise.
#
# These tests hold both halves. If the exclusion ever became a name-based pass, or the renderer ever
# started ignoring rotation generally, one of them fails.

_INVARIANT = ("ball_y", "target_ball_quat_qw", "target_ball_quat_qy")


def _rotate(states, names, dim, amount=0.6):
    p = states.copy()
    if names[dim].startswith("target_ball_quat"):
        iw, iy = names.index("target_ball_quat_qw"), names.index("target_ball_quat_qy")
        a = 2.0 * np.arctan2(p[:, iy], p[:, iw]) + amount
        p[:, iw], p[:, iy] = np.cos(a / 2), np.sin(a / 2)
    else:
        p[:, dim] = p[:, dim] + amount
    return p


def test_the_declared_dims_are_excluded_and_reported_never_silently_passed(cell):
    """``check_state_dims_matter`` excludes the three, REPORTS them with a reason, and finds no dead
    dimension. An excluded dim must never simply vanish from the output."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, device="cpu", seed=0)
    r = C.check_state_dims_matter(env, n=8)
    assert r["dead_dims"] == [], r
    assert set(r.get("excluded_dims", {})) == set(_INVARIANT), r
    assert not r.get("refused_exclusions"), r
    for name, why in r["excluded_dims"].items():
        assert "sphere orientation" in why, (name, why)


def test_excluded_dims_leave_pixels_byte_unchanged_at_both_sizes(cell):
    """Regression A: the claim is about PIXELS, so it is asserted on pixels and not only on the model."""
    for size in (224, 84):
        env = cell(num_envs=16, size=size, device="cpu", seed=0)
        names = list(env.state_names)
        st = env.sample_states(16, np.random.default_rng(55))
        base = env._render(torch.as_tensor(st)).cpu().numpy()
        for nm in _INVARIANT:
            moved = env._render(torch.as_tensor(_rotate(st, names, names.index(nm)))).cpu().numpy()
            assert np.array_equal(base, moved), (
                f"{nm} changed pixels at {size} px, so it is NOT render-invariant and must not be "
                f"excluded from check_state_dims_matter"
            )


def test_the_target_quat_dims_are_live_again_when_the_target_is_a_box(cell):
    """Regression B, and the one that makes the exclusion a statement about GEOMETRY.

    Under ``target_ball.shape = 0`` the target is a BOX. The two quaternion dims then move pixels, and
    the validated exclusion must REFUSE them -- if it did not, the mechanism would be excusing a
    renderer that ignores rotation rather than one drawing a rotation-invariant solid.

    ``ball_y`` stays excluded under the box, correctly: it orients the ``ball`` prop, which is a sphere
    in both styles and is not what ``target_ball.shape`` switches.
    """
    from stable_envs.testing import conformance as C

    box = {"target_ball.shape": [0]}
    env = cell(num_envs=16, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(16, np.random.default_rng(55))
    base = env._render(torch.as_tensor(st), variation=box).cpu().numpy()
    for nm in ("target_ball_quat_qw", "target_ball_quat_qy"):
        d = names.index(nm)
        moved = env._render(torch.as_tensor(_rotate(st, names, d)), variation=box).cpu().numpy()
        assert not np.array_equal(base, moved), f"{nm} is dead even with a BOX target"
        ok, why = C._validate_render_invariant(env, st, d, variation=box)
        assert not ok, f"{nm} was excluded under a BOX target: {why}"
        assert "NON-sphere" in why, why
    d = names.index("ball_y")
    ok, _ = C._validate_render_invariant(env, st, d, variation=box)
    assert ok, "ball_y orients the spherical ball prop and stays invariant whatever the target is"


def test_a_dimension_that_moves_position_is_refused_for_exclusion(cell):
    """The negative case the mechanism exists to catch: a dim the renderer merely failed to draw.

    Every remaining dimension moves a POSITION (arm joints, the ball's x/z, the target's x/z), which
    the frozen-array comparison forbids. None of them may be excludable, whatever an env declares.
    """
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(8, np.random.default_rng(12))
    for nm in ("arm_root", "ball_x", "target_ball_pos_x"):
        ok, why = C._validate_render_invariant(env, st, names.index(nm))
        assert not ok, f"{nm} was granted a render-invariance exclusion: {why}"


def test_an_env_that_declares_nothing_is_unaffected(cell):
    """The mechanism is opt-in: with no declaration, a dead dim is dead exactly as before."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, device="cpu", seed=0)
    env.RENDER_INVARIANT_DIMS = ()
    try:
        r = C.check_state_dims_matter(env, n=8)
        assert set(r["dead_dims"]) == set(_INVARIANT), r
        assert "excluded_dims" not in r, r
    finally:
        del env.RENDER_INVARIANT_DIMS  # back to the class attribute


# ------------------------------------------------- the two false positives review demonstrated (D1/D2)
#
# Both are about the rule trusting the ENV for something it must establish itself. Each is reproduced
# here as a WORKING false positive against the real cell -- the signature is wrapped, not faked -- so
# that removing either guard fails a test rather than silently widening the escape.


class _Wrapped:
    """The real env with its ``render_scene_signature`` filtered. Everything else is delegated."""

    def __init__(self, env, fn):
        self._env, self._fn = env, fn

    def __getattr__(self, k):
        return getattr(self._env, k)

    def render_scene_signature(self, states, variation=None):
        kw = {} if variation is None else {"variation": variation}
        return self._fn(self._env.render_scene_signature(states, **kw))


def test_a_missing_frozen_field_is_refused_and_named_not_skipped(cell):
    """D1. ``thumbtip`` rotates only the two fingertip SPHERES -- and TRANSLATES them.

    With ``geom_xpos`` dropped from the signature the old rule GRANTED it, because a frozen field the
    env does not emit was skipped. ``thumbtip`` moves 60 px at max 206/255, so that exclusion would
    have hidden a dimension the renderer draws perfectly well. A missing key must be a refusal that
    NAMES the key.
    """
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(8, np.random.default_rng(4))
    d = names.index("thumbtip")

    # it is a real, visible dimension
    base = env._render(torch.as_tensor(st)).cpu().numpy()
    pert = st.copy()
    pert[:, d] = pert[:, d] + 0.2
    moved = env._render(torch.as_tensor(pert)).cpu().numpy()
    assert not np.array_equal(base, moved), "thumbtip draws nothing; this test has lost its subject"

    blind = _Wrapped(env, lambda sig: {k: v for k, v in sig.items() if k != "geom_xpos"})
    ok, why = C._validate_render_invariant(blind, st, d)
    assert not ok, f"a signature missing geom_xpos granted thumbtip an exclusion: {why}"
    assert "geom_xpos" in why, why
    # and the unwrapped env refuses it too, on the positional change itself
    ok2, why2 = C._validate_render_invariant(env, st, d)
    assert not ok2 and "geom_xpos" in why2, why2


def test_a_lying_sphere_flag_cannot_grant_an_exclusion(cell):
    """D2. ``geom_is_sphere`` is an assertion by the party being judged.

    Set to all-True with a BOX target it granted the two quaternion dims, which move 201 px there.
    Sphere-ness is now derived from ``geom_type`` inside the rule, and a disagreeing flag is refused
    rather than ignored. ``mjGEOM_ELLIPSOID`` is why this matters beyond a hostile test: it is the
    sphere-LIKE kind that is not rotation-invariant, and it is in the registry now.
    """
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(8, np.random.default_rng(4))
    box = {"target_ball.shape": [0]}

    def all_spheres(sig):
        sig = dict(sig)
        sig["geom_is_sphere"] = np.ones(np.asarray(sig["geom_type"]).reshape(-1).shape, bool)
        return sig

    liar = _Wrapped(env, all_spheres)
    for nm in ("target_ball_quat_qw", "target_ball_quat_qy"):
        d = names.index(nm)
        ok, why = C._validate_render_invariant(liar, st, d, variation=box)
        assert not ok, f"a lying geom_is_sphere granted {nm} with a BOX target: {why}"
        assert "geom_is_sphere" in why or "NON-sphere" in why, why


def test_the_rule_derives_sphereness_even_when_the_env_omits_the_flag(cell):
    """The flag is now optional evidence, not the basis of the decision."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(8, np.random.default_rng(4))
    noflag = _Wrapped(env, lambda sig: {k: v for k, v in sig.items() if k != "geom_is_sphere"})
    ok, why = C._validate_render_invariant(noflag, st, names.index("ball_y"))
    assert ok, f"the rule could not derive sphere-ness from geom_type alone: {why}"
