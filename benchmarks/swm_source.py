"""The stable-worldmodel (swm) source of every benchmark: one swm gym env per instance.

Each stable-envs class maps to its swm id through ``tests/helpers/reference/crosswalk.py``. DM Control and Fetch
also go through swm's own wrappers. ``SwmSource`` gives the calls that the benchmarks use: ``reset``, ``step``,
``render``, ``set_state`` (write one stable-envs State row into the swm env) and ``get_state`` (read it back).
"""

from __future__ import annotations

import importlib.metadata as md
import json

import numpy as np

#: Constructor keyword arguments of swm ids, as a function of the frame size.
SOURCE_KWARGS = {
    "swm/PushT-v1": lambda size: {"resolution": size},
    "swm/PushT-Discrete-v1": lambda size: {"resolution": size},
    "swm/PFRocketLanding-v0": lambda size: {"render_resolution": (size, size)},
    "swm/OGBCube-v0": lambda size: {"width": size, "height": size},
    "swm/OGBScene-v0": lambda size: {"width": size, "height": size},
    "swm/OGBMaze-v0": lambda size: {"width": size, "height": size},
}
#: Families whose swm env draws with MuJoCo, so it can draw on the GPU with EGL.
MUJOCO_FAMILIES = ("dmcontrol", "fetch", "ogbench")


def source_id(name: str) -> str:
    """The swm id that class ``name`` is compared with."""
    import stable_envs as snv
    from tests.helpers.reference.crosswalk import swm_ids_for_class

    if snv.info(name)["family"] == "craftax":
        return f"swm/{name}-v1"
    ids = swm_ids_for_class(name)
    if not ids:
        raise LookupError(f"{name} has no stable-worldmodel id")
    return ids[0]


def draws_on_gpu(name: str) -> bool:
    """Whether the swm env of ``name`` can draw on the GPU (MuJoCo with EGL)."""
    import stable_envs as snv

    return snv.info(name)["family"] in MUJOCO_FAMILIES


def versions() -> str:
    """The swm commit and the versions of the source libraries, as a JSON string."""
    out = {}
    try:
        url = json.loads(md.distribution("stable-worldmodel").read_text("direct_url.json") or "{}")
        out["stable-worldmodel"] = url.get("vcs_info", {}).get("commit_id") or md.version("stable-worldmodel")
    except md.PackageNotFoundError:
        pass
    if "stable-worldmodel" in out and len(out["stable-worldmodel"]) < 12:
        try:
            import subprocess
            from pathlib import Path

            import stable_worldmodel

            root = Path(stable_worldmodel.__file__).resolve().parents[1]
            sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
            out["stable-worldmodel"] = sha.stdout.strip() or out["stable-worldmodel"]
        except (ImportError, OSError):
            pass
    for lib in ("gymnasium", "mujoco", "dm-control", "ogbench", "gymnasium-robotics", "pymunk", "craftax", "PyFlyt"):
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    return json.dumps(out, sort_keys=True)


#: Short names of the State fields that the mapping uses, and the stable-envs field names they can have.
FIELDS = {
    "agent": ("current.body.agent.position", "agent.position"),
    "block": ("current.body.block.position",),
    "angle": ("current.body.block.angle",),
    "velocity": ("current.body.agent.velocity",),
    "block_velocity": ("current.body.block.velocity",),
    "block_angular_velocity": ("current.body.block.angular_velocity",),
    "qpos": ("qpos", "physics.qpos"),
    "qvel": ("qvel", "physics.qvel"),
}


def rows(state) -> list[dict]:
    """One ``{short name: float64 vector}`` dict per instance of a batched stable-envs State (see ``FIELDS``).

    Call it in a process that built the env of the State: a folded State needs its schema to read fields.
    """
    from stable_envs import get_field

    n = int(state.batch_size[0])
    fields = {}
    for short, names in FIELDS.items():
        for name in names:
            try:
                value = get_field(state, name)
            except KeyError:
                continue
            fields[short] = np.asarray(value.detach().cpu().numpy(), np.float64).reshape(n, -1)
            break
    return [{k: v[i] for k, v in fields.items()} for i in range(n)]


def _leaf(row: dict, short: str):
    return row.get(short)


def _mujoco(u):
    """The MuJoCo ``(model, data)`` of an unwrapped swm env, or None."""
    for get in (lambda: u.env.physics, lambda: u.physics):
        try:
            p = get()
            return p.model.ptr if hasattr(p.model, "ptr") else p.model, p.data.ptr if hasattr(
                p.data, "ptr"
            ) else p.data
        except AttributeError:
            pass
    for m, d in (("model", "data"), ("_model", "_data")):
        if hasattr(u, m) and hasattr(u, d):
            return getattr(u, m), getattr(u, d)
    return None


class SwmSource:
    """One swm env of class ``name`` at frame size ``size``; actions are converted to its action space."""

    def __init__(self, name: str, size: int = 224):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401  registers the swm/ ids

        self.name, self.id = name, source_id(name)
        kwargs = SOURCE_KWARGS[self.id](size) if self.id in SOURCE_KWARGS else {}
        self.env = gym.make(self.id, render_mode="rgb_array", **kwargs)
        self.kwargs = kwargs
        space = self.env.action_space
        self.discrete = hasattr(space, "n") and not hasattr(space, "nvec")
        self.shape, self.dtype = getattr(space, "shape", ()), getattr(space, "dtype", None)

    def reset(self, seed=None):
        self.env.reset(seed=seed)

    def step(self, action) -> bool:
        """Take one action; return whether the episode ended."""
        a = np.asarray(action)
        a = int(a.reshape(-1)[0]) if self.discrete else a.reshape(self.shape).astype(self.dtype)
        out = self.env.step(a)
        return bool(out[2] or out[3])

    def render(self) -> np.ndarray:
        return np.asarray(self.env.render())

    def close(self):
        self.env.close()

    def set_state(self, state: dict, velocity: bool = True) -> None:
        """Write one instance of a stable-envs State (a dict from :func:`rows`) into the swm env.

        swm's PushT setter takes one physics step after it writes the poses. ``velocity=False`` writes a zero agent
        velocity, so that step does not move the agent before a frame is drawn.

        Raises:
            NotImplementedError: If there is no mapping for this class.
        """
        u = self.env.unwrapped
        agent, block = _leaf(state, "agent"), _leaf(state, "block")
        if block is not None and hasattr(u, "_set_state"):
            vel = _leaf(state, "velocity") if velocity else None
            angle = _leaf(state, "angle")
            u._set_state(np.concatenate([agent, block, angle[:1], vel if vel is not None else np.zeros(2)]))
            return
        if agent is not None and hasattr(u, "_set_state"):
            u._set_state(agent[:2])
            return
        qpos, qvel = _leaf(state, "qpos"), _leaf(state, "qvel")
        mj = _mujoco(u)
        if qpos is None or mj is None:
            raise NotImplementedError(f"no swm state mapping for {self.name}")
        model, data = mj
        qvel = np.zeros(model.nv) if qvel is None else qvel
        if hasattr(u, "set_state"):
            u.set_state(qpos[: model.nq], qvel[: model.nv])
            return
        import mujoco

        data.qpos[:], data.qvel[:] = qpos[: model.nq], qvel[: model.nv]
        mujoco.mj_forward(model, data)

    def read(self) -> dict:
        """The fields that swm exposes, as ``{short name: float64 vector}`` (see ``FIELDS``)."""
        u = self.env.unwrapped
        if hasattr(u, "block") and hasattr(u, "agent"):
            return {
                "agent": np.array(u.agent.position, np.float64),
                "velocity": np.array(u.agent.velocity, np.float64),
                "block": np.array(u.block.position, np.float64),
                "angle": np.array([u.block.angle], np.float64),
                "block_velocity": np.array(u.block.velocity, np.float64),
                "block_angular_velocity": np.array([u.block.angular_velocity], np.float64),
            }
        if hasattr(u, "agent_position"):
            return {"agent": np.asarray(u.agent_position, np.float64).reshape(-1)[:2]}
        mj = _mujoco(u)
        if mj is None:
            raise NotImplementedError(f"no swm state mapping for {self.name}")
        model, data = mj
        return {
            "qpos": np.array(data.qpos[: model.nq], np.float64),
            "qvel": np.array(data.qvel[: model.nv], np.float64),
        }

    def get_state(self, like: dict) -> np.ndarray:
        """Read the swm env's state as the vector that :func:`state_vector` makes of the row ``like``."""
        u = self.env.unwrapped
        if _leaf(like, "block") is not None:
            return np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)
        if _leaf(like, "agent") is not None:
            return np.asarray(u.agent_position, np.float64).reshape(-1)[:2]
        mj = _mujoco(u)
        if mj is None:
            raise NotImplementedError(f"no swm state mapping for {self.name}")
        model, data = mj
        qvel = np.asarray(data.qvel[: model.nv]) if _leaf(like, "qvel") is not None else np.zeros(0)
        return np.concatenate([np.asarray(data.qpos[: model.nq]), qvel])


def state_vector(state: dict) -> np.ndarray:
    """The vector of one State instance (a dict from :func:`rows`) that :meth:`SwmSource.get_state` reads."""
    block = _leaf(state, "block")
    if block is not None:
        return np.concatenate([_leaf(state, "agent"), block, _leaf(state, "angle")[:1]])
    agent = _leaf(state, "agent")
    if agent is not None:
        return agent[:2]
    qpos, qvel = _leaf(state, "qpos"), _leaf(state, "qvel")
    if qpos is None:
        raise NotImplementedError("no qpos leaf in the State")
    return np.concatenate([qpos, qvel if qvel is not None else np.zeros(0)])
