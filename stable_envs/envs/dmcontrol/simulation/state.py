import numpy as np
import torch
from tensordict import TensorDict

_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}

#: State fields that hold episode-drawn model entries: ``key -> (named model field, index)``.
MODEL_FIELDS = {
    ("task", "target_pos"): ("geom_pos", "target"),
    ("model", "target_site_xz"): ("site_pos", ("target", ["x", "z"])),
    ("model", "target_body_xz"): ("body_pos", ("target_ball", ["x", "z"])),
    ("model", "target_body_qwqy"): ("body_quat", ("target_ball", ["qw", "qy"])),
    ("model", "target_geom_xy"): ("geom_pos", ("target", ["x", "y"])),
    ("model", "target_light_xy"): ("light_pos", ("target_light", ["x", "y"])),
}
_COLUMNS = {"x": 0, "y": 1, "z": 2, "qw": 0, "qx": 1, "qy": 2, "qz": 3}
_RNG = (("rng", "keys"), ("rng", "position"), ("rng", "has_gauss"), ("rng", "cached_gaussian"))


def core_fields(nq, nv, na=0):
    """Fields that resume a dm_control lane: physics, lifecycle, reward and task RNG."""
    fields = {
        "qpos": (nq, torch.float64),
        "qvel": (nv, torch.float64),
        "time": (1, torch.float64),
        "warmstart": (nv, torch.float64),
        ("lifecycle", "step_count"): (1, torch.int64),
        ("lifecycle", "reset_next_step"): (1, torch.bool),
        ("task", "cumulative_reward"): (1, torch.float64),
        ("rng", "keys"): (624, torch.int64),
        ("rng", "position"): (1, torch.int64),
        ("rng", "has_gauss"): (1, torch.bool),
        ("rng", "cached_gaussian"): (1, torch.float64),
    }
    if na:
        fields["activation"] = (na, torch.float64)
    return fields


def model_entries(model, fields):
    """Resolve the episode-drawn model entries of ``fields`` to ``(key, model field, row, columns)``."""
    import mujoco

    entries = []
    for key in fields:
        if key not in MODEL_FIELDS:
            continue
        field, index = MODEL_FIELDS[key]
        name, columns = index if isinstance(index, tuple) else (index, None)
        kind = field.split("_")[0]
        row = mujoco.mj_name2id(model, getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"), name)
        if row < 0:
            raise KeyError(f"{key}: no {kind} named {name!r}")
        width = getattr(model, field).shape[1]
        entries.append((key, field, row, list(range(width)) if columns is None else [_COLUMNS[c] for c in columns]))
    return entries


class RNGColumns:
    """The task RNG State of every lane; a lane is read again only after code that can draw from it ran."""

    def __init__(self, num_envs):
        self.columns = {
            ("rng", "keys"): np.zeros((num_envs, 624), np.int64),
            ("rng", "position"): np.zeros((num_envs, 1), np.int64),
            ("rng", "has_gauss"): np.zeros((num_envs, 1), np.bool_),
            ("rng", "cached_gaussian"): np.zeros((num_envs, 1), np.float64),
        }
        self.stale = np.ones(num_envs, bool)

    def read(self, envs):
        for lane in np.flatnonzero(self.stale):
            algorithm, *values = envs[lane].task.random.get_state()
            if algorithm != "MT19937":
                raise ValueError(f"Unsupported DMC RNG algorithm: {algorithm}")
            for key, value in zip(_RNG, values, strict=True):
                self.columns[key][lane] = value
        self.stale[:] = False
        return {key: value.copy() for key, value in self.columns.items()}


def export_state(lanes, fields, device, extra=None):
    """Export fresh State columns of a ``DMCLanes`` batch with one device transfer per field.

    CPU tensors retain fresh host arrays; later simulation and exports cannot overwrite the snapshot.
    """
    views, bases, batch = lanes._views, lanes._bases, lanes.num_envs
    values = {
        "qpos": np.stack(views.data("qpos")),
        "qvel": np.stack(views.data("qvel")),
        "time": np.array([d.time for d in views.datas]),
        "warmstart": np.stack(views.data("qacc_warmstart")),
        ("lifecycle", "step_count"): np.array([b._step_count for b in bases]),
        ("lifecycle", "reset_next_step"): np.array([b._reset_next_step for b in bases]),
        ("task", "cumulative_reward"): lanes._cumulative.copy(),
        **lanes._rng.read(lanes.envs),
        **(extra or {}),
    }
    if "activation" in fields:
        values["activation"] = np.stack(views.data("act"))
    for key, field, row, columns in lanes._model_entries:
        values[key] = np.stack([view[row] for view in views.model(field)])[:, columns]
    return TensorDict(
        {
            key: torch.from_numpy(np.asarray(values[key], _DTYPES[dtype]).reshape(batch, width)).to(device)
            for key, (width, dtype) in fields.items()
        },
        [batch],
        device=device,
    )


def restore_lane(env, row, fields, cumulative, lane):
    """Write one exported State row back into a lane, model entries before forward kinematics."""
    physics, base = env.physics, env._env
    for key in fields:
        if key in MODEL_FIELDS:
            field, index = MODEL_FIELDS[key]
            getattr(physics.named.model, field)[index] = row[key].numpy()
    physics.data.qpos[:] = row["qpos"].numpy()
    physics.data.qvel[:] = row["qvel"].numpy()
    physics.data.time = row["time"].item()
    if "activation" in fields:
        physics.data.act[:] = row["activation"].numpy()
    physics.forward()
    physics.data.qacc_warmstart[:] = row["warmstart"].numpy()
    base._step_count = row["lifecycle", "step_count"].item()
    base._reset_next_step = row["lifecycle", "reset_next_step"].item()
    env.task.random.set_state(
        (
            "MT19937",
            row["rng", "keys"].numpy().astype(np.uint32),
            row["rng", "position"].item(),
            int(row["rng", "has_gauss"].item()),
            row["rng", "cached_gaussian"].item(),
        )
    )
    cumulative[lane] = row["task", "cumulative_reward"].item()
