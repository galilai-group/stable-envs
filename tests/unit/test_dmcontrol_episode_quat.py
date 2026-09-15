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
except ImportError as exc:                       # dm_control / mujoco absent: legitimately skippable
    pytest.skip(f"stable_envs.envs.dmcontrol needs dm_control ({exc}); this file has nothing to "
                f"assert without it", allow_module_level=True)
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
        f"per-row xmat and must stay allowed. Current source:\n{src}")
    assert '"_quat"' not in src.split("bad = ")[1].split("\n")[0], (
        "the guard refuses per-episode orientation again; see this file's docstring for why that is "
        "not the same class of problem as a per-episode size")


@pytest.fixture(scope="module")
def cell():
    if CELL not in se.REGISTRY:
        pytest.skip(f"{CELL} is not registered in this interpreter")
    return se.REGISTRY[CELL]


def test_manipulator_widens_the_state_with_the_quaternion_slots(cell):
    """The census finds the four floats ``Bring.initialize_episode`` writes, and names them."""
    slots = cell._slots()
    arrays = sorted({a for a, _ in slots})
    assert arrays == ["body_pos", "body_quat"], (
        f"{CELL}'s episode draw is no longer target_ball's pose: {slots}")
    names = cell._slot_names(cell._model())
    assert set(names) == {"target_ball_pos_x", "target_ball_pos_z",
                          "target_ball_quat_qw", "target_ball_quat_qy"}, names
    env = cell(batch=1, device="cpu", seed=0)
    assert env.state_dim == env.nq + 4 == 15


def test_fk_orientation_is_per_row(cell):
    """``_fk``'s ``xmat`` differs ACROSS ROWS when only the quaternion slots differ.

    This is the property the guard's removal rests on. The two rows share every ``qpos`` entry and
    every positional slot, so a batch-shared orientation would make the two ``xmat`` stacks equal.
    """
    env = cell(batch=2, device="cpu", seed=0)
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
        f"{per_prim.tolist()}")


def test_the_official_renderer_moves_for_the_quaternion_slots(cell):
    """The widened dimensions are real in the REFERENCE, whatever this renderer does with them.

    Deliberately asserted about ``physics.render`` and not about us: ``target_ball`` is an untextured
    SPHERE, so MuJoCo's tessellated per-vertex shading moves a few hundred pixels under rotation while
    an analytic per-pixel sphere is exactly rotation-invariant. That gap is declared in
    ``STATE_DIMS_NOTE`` and recorded as a ``check_state_dims_matter`` failure; what must not be true is
    that the state carries a dimension the reference itself ignores.
    """
    from stable_envs.testing import conformance as C
    env = cell(batch=1, device="cpu", seed=0)
    names = list(env.state_names)
    iw, iy = names.index("target_ball_quat_qw"), names.index("target_ball_quat_qy")
    st = env.sample_states(2, np.random.default_rng(11))
    # Through the POOL, never ``cls.reference_env`` into a local: the returned object owns an EGL
    # context and CPython frees it when the local goes out of scope, which corrupts every other live
    # context in the process. ``conformance.reference_env`` keeps it for the life of the process and
    # ``resource_guard`` fails this test if it is dropped -- see that function's docstring.
    ref = C.reference_env(cell, env.size)
    base = cell.reference_render(ref, st)
    rot = st.copy()
    ang = 2.0 * np.arctan2(rot[:, iy], rot[:, iw]) + 0.6
    rot[:, iw], rot[:, iy] = np.cos(ang / 2), np.sin(ang / 2)
    moved = cell.reference_render(ref, rot)
    d = np.abs(base.astype(np.int16) - moved.astype(np.int16)).max(-1)
    assert int((d > 0).sum()) > 0, (
        "rotating target_ball changed 0 pixels in the OFFICIAL renderer, so the two quaternion slots "
        "would be dead in the reference as well as here and should not be in the state at all")


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

    env = cell(batch=2, device="cpu", seed=0)
    assert "visibility.target_ball" in env.variation_space.names()
    st = env.sample_states(2, np.random.default_rng(3))
    shown = env.render(torch.as_tensor(st)).cpu().numpy()
    hidden = env.render(torch.as_tensor(st),
                        variation={"visibility.target_ball": [0]}).cpu().numpy()
    assert not np.array_equal(shown, hidden), (
        "hiding visibility.target_ball changed no pixel, so the body slot did not reach the geoms")


def test_the_shape_leaf_default_is_the_compiled_model(cell):
    """``target_ball.shape``'s default is the index that matches the compiled model, and the other
    alternatives -- ``stable_worldmodel``'s 0.015 box and 0.015 sphere -- actually change the frame."""
    env = cell(batch=2, device="cpu", seed=0)
    leaf = env.variation_space["target_ball"]["shape"]
    assert leaf.n == 3, leaf.n
    default = int(np.asarray(leaf.value).reshape(-1)[0])
    st = env.sample_states(2, np.random.default_rng(13))
    base = env.render(torch.as_tensor(st)).cpu().numpy()
    same = env.render(torch.as_tensor(st),
                      variation={"target_ball.shape": [default]}).cpu().numpy()
    assert np.array_equal(base, same), "target_ball.shape's default is not the default appearance"
    for k in range(leaf.n):
        if k == default:
            continue
        other = env.render(torch.as_tensor(st), variation={"target_ball.shape": [k]}).cpu().numpy()
        assert not np.array_equal(base, other), (
            f"target_ball.shape = {k} changed no pixel; a declared alternative that cannot be seen "
            f"fails check_variations_do_something and advertises support that is not there")
