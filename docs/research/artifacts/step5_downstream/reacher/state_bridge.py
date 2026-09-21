"""Read a ``stable_worldmodel`` DM Control env's live state in the layout our renderer expects.

The two libraries agree on what a DM Control frame depends on and disagree on how to say it.
``stable_envs`` states a DM Control scene as ``qpos`` plus whatever the task's ``initialize_episode``
DREW into the model that episode -- for reacher, the two ``geom_pos`` entries that place the target
-- and discovers that second half by diffing episodes rather than transcribing it
(``stable_envs.envs.dmcontrol._episode_slots``). ``stable_worldmodel`` keeps ``qpos`` and ``qvel``
in its ``info`` dict and leaves the drawn half in the model where the episode put it.

So the bridge is: take ``qpos`` from the official env's physics, and read the drawn half out of the
official env's own MODEL. Both halves come from the env under test, which is the point -- nothing is
inferred, re-simulated or assumed, so the frame our renderer draws is a frame of the state the
official env is actually in at that instant.

The one thing that cannot be read by index is the slot itself. ``stable_worldmodel`` compiles its
reacher through ``mjcf.from_xml_string`` after editing materials and geom types, and a recompiled
model is free to order its geoms differently from ``suite.load``'s. Indices are therefore resolved
to NAMES on our model once, and looked up by name on theirs -- an ordering difference then produces
a KeyError instead of a target drawn at the arm's coordinates.
"""

from __future__ import annotations

import numpy as np

#: MuJoCo array prefix -> the object type its rows are indexed by. Only the prefixes
#: ``_episode_slots`` can return are listed; a new one must be added deliberately.
_OBJ = {"geom": "mjOBJ_GEOM", "site": "mjOBJ_SITE", "body": "mjOBJ_BODY", "light": "mjOBJ_LIGHT"}


def slot_names(se_env) -> tuple[tuple[str, str, int], ...]:
    """Our flat ``(array, index)`` slots, resolved to ``(array, element name, component)``."""
    import mujoco

    m = se_env._model()
    out = []
    for arr, k in se_env._slots():
        a = np.asarray(getattr(m, arr))
        ncomp = a.shape[1] if a.ndim == 2 else 1
        i, comp = divmod(int(k), ncomp)
        kind = arr.split("_")[0]
        name = mujoco.mj_id2name(m, getattr(mujoco.mjtObj, _OBJ[kind]), i)
        if name is None:
            raise KeyError(f"{arr}[{i}] has no name in our model; the slot cannot be matched by name")
        out.append((arr, name, comp))
    return tuple(out)


def read_state(se_env, swm_env, slots=None, target_override=None) -> np.ndarray:
    """``(state_dim,)`` for our renderer, read off a live ``stable_worldmodel`` env.

    ``target_override`` moves the elements named in it before they are read. That was Step 5's way
    of reproducing an element the official env has made INVISIBLE, at a time when ``stable_envs``'
    DM Control appearance space carried material colour, light and camera fovy and no visibility, so
    a geom the reference declines to draw could only be matched by putting it where the camera cannot
    see it. Step 6 gave the env a ``visibility.target`` style leaf and
    ``swap_renderer.Renderer._variation`` uses it, so the state read here keeps the position the
    episode actually drew. The override is kept, unused by the default path, because
    ``tests/fixtures/regression/join/step5/reacher/`` was measured under it and has to stay reproducible.
    """
    import mujoco

    slots = slot_names(se_env) if slots is None else slots
    p = swm_env.env.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    qpos = np.asarray(p.data.qpos, np.float64).reshape(-1)
    tail = []
    for arr, name, comp in slots:
        if target_override is not None and name in target_override:
            tail.append(float(np.asarray(target_override[name], np.float64).reshape(-1)[comp]))
            continue
        kind = arr.split("_")[0]
        i = mujoco.mj_name2id(m, getattr(mujoco.mjtObj, _OBJ[kind]), name)
        if i < 0:
            raise KeyError(f"the official env's model has no {kind} named {name!r}")
        tail.append(float(np.asarray(getattr(m, arr))[i].reshape(-1)[comp]))
    st = np.concatenate([qpos, np.asarray(tail, np.float64)])
    if st.shape[0] != se_env.state_dim:
        raise ValueError(f"built a {st.shape[0]}-vector for a state_dim of {se_env.state_dim}")
    return st.astype(np.float32)
