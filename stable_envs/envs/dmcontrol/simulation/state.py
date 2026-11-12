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


def new_columns(batch, fields):
    """Empty host State columns for ``batch`` lanes."""
    return {key: np.empty((batch, width), dtype=_DTYPES[dtype]) for key, (width, dtype) in fields.items()}


def model_columns(columns):
    """The episode-drawn model entries held by ``columns``: ``(key, named model field, index)``."""
    return [(key, *MODEL_FIELDS[key]) for key in columns if key in MODEL_FIELDS]


def write_row(columns, i, env, cumulative, model_fields):
    """Read live lane ``env`` into row ``i`` of ``columns``; lanes in different rows may be written concurrently."""
    physics, base = env.physics, env._env
    data = physics.data
    columns["qpos"][i] = data.qpos
    columns["qvel"][i] = data.qvel
    columns["time"][i] = data.time
    columns["warmstart"][i] = data.qacc_warmstart
    columns["lifecycle", "step_count"][i] = base._step_count
    columns["lifecycle", "reset_next_step"][i] = base._reset_next_step
    columns["task", "cumulative_reward"][i] = cumulative
    write_task_row(columns, i, env, model_fields)
    if "activation" in columns:
        columns["activation"][i] = data.act


def write_task_row(columns, i, env, model_fields):
    """Read the task RNG and episode-drawn model entries of lane ``env`` into row ``i`` of ``columns``."""
    algorithm, keys, position, has_gauss, cached = env.task.random.get_state()
    if algorithm != "MT19937":
        raise ValueError(f"Unsupported DMC RNG algorithm: {algorithm}")
    columns["rng", "keys"][i] = keys
    columns["rng", "position"][i] = position
    columns["rng", "has_gauss"][i] = has_gauss
    columns["rng", "cached_gaussian"][i] = cached
    for key, field, index in model_fields:
        columns[key][i] = getattr(env.physics.named.model, field)[index]


def columns_to_state(columns, device, extra=None):
    """Own the filled columns as a State; ``extra`` columns come from elsewhere."""
    batch = len(next(iter(columns.values())))
    for key, value in (extra or {}).items():
        columns[key][:] = value
    # On CPU, from_numpy retains these fresh arrays. No owner/buffer is reused
    # by a later step, reset or export; CUDA conversion also owns its storage.
    return TensorDict(
        {key: torch.from_numpy(value).to(device) for key, value in columns.items()}, [batch], device=device
    )


def export_columns(envs, cumulative, fields):
    """Read ``fields`` of live lane owners into fresh host arrays, ``{key: (lanes, width)}``."""
    columns = new_columns(len(envs), fields)
    model_fields = model_columns(columns)
    for i, env in enumerate(envs):
        write_row(columns, i, env, cumulative[i], model_fields)
    return columns


def export_state(envs, cumulative, fields, device, extra=None):
    """Export fresh State columns with one device transfer per field.

    Read live lane owners into owned host arrays. CPU tensors retain those arrays;
    later simulation and exports cannot overwrite the snapshot.
    """
    return columns_to_state(export_columns(envs, cumulative, fields), device, extra)


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
