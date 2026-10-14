"""Per-episode orientation slots reach the frame per row; render-invariant dimensions are validated, not trusted.

An episode may draw a body's orientation (``manipulator-bring_ball`` writes ``body_quat['target_ball']``).
Orientation goes through ``mj_kinematics`` into a per-row ``xmat``, so unlike a per-episode size or light
direction (shared by the whole batch at bake time) it is a legitimate state dimension.
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
    # Only a missing GL context skips; any other import failure must fail the file.
    if "GL" not in str(exc):
        raise
    pytest.skip(f"no usable offscreen GL context here ({exc})", allow_module_level=True)

CELL = "DMCManipulatorBringBall"


@pytest.fixture(scope="module")
def cell():
    if CELL not in se.registry.REGISTRY:
        pytest.skip(f"{CELL} is not registered in this interpreter")
    return se.registry.REGISTRY[CELL]


def test_manipulator_widens_the_state_with_the_quaternion_slots(cell):
    """The census finds the four floats ``Bring.initialize_episode`` writes."""
    slots = cell._slots()
    arrays = sorted({a for a, _ in slots})
    assert arrays == ["body_pos", "body_quat"], f"unexpected episode draw: {slots}"
    names = cell._slot_names(cell._model())
    assert set(names) == {"target_ball_pos_x", "target_ball_pos_z", "target_ball_quat_qw", "target_ball_quat_qy"}, (
        names
    )
    env = cell(num_envs=1, device="cpu", seed=0)
    assert env.state_dim == env.nq + 4 == 15


def test_fk_orientation_is_per_row(cell):
    """``_fk``'s ``xmat`` differs across rows when only the quaternion slots differ."""
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
        f"orientation is batch-shared; per-primitive row difference {per_prim.tolist()}"
    )


def test_the_official_renderer_moves_for_the_quaternion_slots(cell):
    """The quaternion dimensions move pixels in dm_control's own renderer.

    ``target_ball`` is an untextured sphere: MuJoCo's per-vertex shading moves a few hundred pixels under
    rotation while an analytic per-pixel sphere is rotation-invariant (declared in ``STATE_DIMS_NOTE``).
    The state must not carry a dimension the reference itself ignores.
    """
    from stable_envs.testing import conformance as C

    env = cell(num_envs=1, device="cpu", seed=0)
    names = list(env.state_names)
    iw, iy = names.index("target_ball_quat_qw"), names.index("target_ball_quat_qy")
    st = env.sample_states(2, np.random.default_rng(11))
    # ``conformance.reference_env`` keeps the EGL-owning reference alive for the process; a local would free it.
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
    assert int((d > 0).sum()) > 0, "rotating target_ball changed no pixel in dm_control's renderer"


def test_a_body_slot_becomes_one_visibility_leaf_over_its_geoms(cell):
    """``_hideable`` resolves a drawn body to the geoms hanging off it, named by the body."""
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
    assert not np.array_equal(shown, hidden), "hiding visibility.target_ball changed no pixel"


def test_the_shape_leaf_default_is_the_compiled_model(cell):
    """``target_ball.shape``'s default is the compiled model's shape; every other alternative changes the frame."""
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
        assert not np.array_equal(base, other), f"target_ball.shape = {k} changed no pixel"


# --------------------------------------------------------------------------- synthetic bodies
#
# ``manipulator`` is the only registered cell with a body slot and its ``target_ball`` carries one drawn
# geom, so the multi-geom path and the both-kinds refusal are exercised on a hand-built model.

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
    """A ``DMControlSuite`` over a hand-built model whose episode draws ``body_pos[marker]``; ``_hideable`` is the shipped one."""
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


def _drawing_env(array):
    """An env whose every reset moves ``array`` of a hand-built model; returns (env, model)."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(_SYNTH_XML.format(extra_site=""))
    rng = np.random.default_rng(0)

    class Env:
        def reset(self):
            view = getattr(m, array)
            view[0] += rng.uniform(0.1, 0.2, view[0].shape)

    return Env(), m


def test_episode_slots_refuse_size_and_light_direction_and_allow_orientation():
    env, m = _drawing_env("body_quat")
    assert {a for a, _ in D._episode_slots(env, m)} == {"body_quat"}
    for array in ("geom_size", "light_dir"):
        env, m = _drawing_env(array)
        with pytest.raises(RuntimeError, match="size or a light's direction"):
            D._episode_slots(env, m)


def test_a_body_slot_with_several_drawn_geoms_resolves_to_all_of_them():
    """``marker`` carries four geoms: two drawn, one at alpha 0 and one in group 3; the leaf covers only the drawn two."""
    import mujoco

    cls = _synth_cls()
    hide = cls._hideable()
    assert len(hide) == 1, hide
    kind, name, ids = hide[0]
    assert (kind, name) == ("geom", "marker"), hide
    got = sorted(mujoco.mj_id2name(cls._M, mujoco.mjtObj.mjOBJ_GEOM, i) for i in ids)
    assert got == ["marker_a", "marker_b"], got


def test_a_body_carrying_both_drawn_geoms_and_drawn_sites_raises():
    """A body with both drawn geoms and drawn sites raises: one leaf cannot name two element kinds."""
    cls = _synth_cls(extra_site='<site name="marker_s" type="sphere" size=".02" group="0"/>')
    with pytest.raises(RuntimeError, match="drawn geoms .* and drawn sites"):
        cls._hideable()


# --------------------------------------------------------- render-invariant exclusion
#
# ball_y / target_ball_quat_qw / target_ball_quat_qy orient an untextured sphere, which an analytic
# per-pixel intersector draws identically. ``check_state_dims_matter`` excludes them only after
# verifying against the compiled model that perturbing one moves nothing but sphere ``geom_xmat``.

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


@pytest.mark.nightly
def test_the_declared_dims_are_excluded_and_reported_never_silently_passed(cell):
    """The three are excluded and reported with a reason; no dimension is dead."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, device="cpu", seed=0)
    r = C.check_state_dims_matter(env, n=8)
    assert r["dead_dims"] == [], r
    assert set(r.get("excluded_dims", {})) == set(_INVARIANT), r
    assert not r.get("refused_exclusions"), r
    for name, why in r["excluded_dims"].items():
        assert "sphere orientation" in why, (name, why)


@pytest.mark.nightly
def test_excluded_dims_leave_pixels_byte_unchanged_at_both_sizes(cell):
    """Perturbing an excluded dimension leaves the frame byte-identical at 224 and 84 px."""
    for size in (224, 84):
        env = cell(num_envs=16, size=size, device="cpu", seed=0)
        names = list(env.state_names)
        st = env.sample_states(16, np.random.default_rng(55))
        base = env._render(torch.as_tensor(st)).cpu().numpy()
        for nm in _INVARIANT:
            moved = env._render(torch.as_tensor(_rotate(st, names, names.index(nm)))).cpu().numpy()
            assert np.array_equal(base, moved), f"{nm} changed pixels at {size} px"


@pytest.mark.nightly
def test_the_target_quat_dims_are_live_again_when_the_target_is_a_box(cell):
    """With a box target the two quaternion dims move pixels and the exclusion must refuse them.

    ``ball_y`` stays excluded: it orients the ``ball`` prop, a sphere in both styles.
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
    """Dimensions that move a position (arm joints, ball x/z, target x/z) are never excludable."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(8, np.random.default_rng(12))
    for nm in ("arm_root", "ball_x", "target_ball_pos_x"):
        ok, why = C._validate_render_invariant(env, st, names.index(nm))
        assert not ok, f"{nm} was granted a render-invariance exclusion: {why}"


@pytest.mark.nightly
def test_an_env_that_declares_nothing_is_unaffected(cell):
    """Exclusion is opt-in: with no declaration, the three dims are reported dead."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, device="cpu", seed=0)
    env.RENDER_INVARIANT_DIMS = ()
    try:
        r = C.check_state_dims_matter(env, n=8)
        assert set(r["dead_dims"]) == set(_INVARIANT), r
        assert "excluded_dims" not in r, r
    finally:
        del env.RENDER_INVARIANT_DIMS


# ------------------------------------------------- the rule must not trust the env
#
# Each test wraps the real cell's scene signature so the env supplies misleading evidence.


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
    """``thumbtip`` rotates the two fingertip spheres and translates them; a signature without ``geom_xpos`` must refuse and name the key."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(8, np.random.default_rng(4))
    d = names.index("thumbtip")

    # thumbtip is a visible dimension.
    base = env._render(torch.as_tensor(st)).cpu().numpy()
    pert = st.copy()
    pert[:, d] = pert[:, d] + 0.2
    moved = env._render(torch.as_tensor(pert)).cpu().numpy()
    assert not np.array_equal(base, moved), "perturbing thumbtip changed no pixel"

    blind = _Wrapped(env, lambda sig: {k: v for k, v in sig.items() if k != "geom_xpos"})
    ok, why = C._validate_render_invariant(blind, st, d)
    assert not ok, f"a signature missing geom_xpos granted thumbtip an exclusion: {why}"
    assert "geom_xpos" in why, why
    # The unwrapped env refuses it too, on the positional change itself.
    ok2, why2 = C._validate_render_invariant(env, st, d)
    assert not ok2 and "geom_xpos" in why2, why2


def test_a_lying_sphere_flag_cannot_grant_an_exclusion(cell):
    """``geom_is_sphere`` is the env's own claim; sphere-ness is derived from ``geom_type`` and a disagreeing flag is refused."""
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
    """The flag is optional evidence, not the basis of the decision."""
    from stable_envs.testing import conformance as C

    env = cell(num_envs=8, size=224, device="cpu", seed=0)
    names = list(env.state_names)
    st = env.sample_states(8, np.random.default_rng(4))
    noflag = _Wrapped(env, lambda sig: {k: v for k, v in sig.items() if k != "geom_is_sphere"})
    ok, why = C._validate_render_invariant(noflag, st, names.index("ball_y"))
    assert ok, f"the rule could not derive sphere-ness from geom_type alone: {why}"
