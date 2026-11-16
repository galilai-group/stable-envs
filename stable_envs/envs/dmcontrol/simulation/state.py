import numpy as np
import torch
from tensordict import TensorDict

from ...._core.state_schema import Field, StateSchema
from ...._core.tensordicts import build_tensordict

_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}
_GROUPS = {torch.float64: "float64", torch.int64: "int64", torch.bool: "bool"}

#: State fields that hold episode-drawn model entries: ``key -> (named model field, index)``.
MODEL_FIELDS = {
    ("task", "target_pos"): ("geom_pos", "target"),
    ("model", "target_site_xz"): ("site_pos", ("target", ["x", "z"])),
    ("model", "target_body_xz"): ("body_pos", ("target_ball", ["x", "z"])),
    ("model", "target_body_qwqy"): ("body_quat", ("target_ball", ["qw", "qy"])),
    ("model", "target_geom_xy"): ("geom_pos", ("target", ["x", "y"])),
    ("model", "target_light_xy"): ("light_pos", ("target_light", ["x", "y"])),
}
_SCHEMAS: dict[str, StateSchema] = {}


def field_name(key):
    """The schema name of a State field key: its parts joined by ``/``."""
    return key if isinstance(key, str) else "/".join(key)


def state_section(key):
    """The top-level State key of simulator ``key``, which holds its groups."""
    return "dmc_" + key.replace("-", "_")


def state_schema(key, fields):
    """The State of simulator ``key``: ``fields`` in one float64, one int64 and one bool group.

    The groups sit under ``state_section(key)``, so States of two simulators never share a group key.
    """
    schema = _SCHEMAS.get(key)
    if schema is None:
        section = state_section(key)
        schema = _SCHEMAS[key] = StateSchema(
            f"DMC {key}",
            [
                Field(field_name(name), (width,), dtype, f"{section}/{_GROUPS[dtype]}")
                for name, (width, dtype) in fields.items()
            ],
        )
    return schema


def new_groups(schema, batch):
    """Empty host group arrays of ``schema`` for ``batch`` environments, ``{group key: (batch, size)}``."""
    return {key: np.empty((batch, group.size), _DTYPES[group.dtype]) for key, group in schema.groups.items()}


def group_views(schema, groups, fields):
    """``{field key: (environments, width) view}`` of ``groups`` for each key of ``fields``."""
    views = schema.views(groups, [field_name(key) for key in fields], numpy=True)
    return {key: views[field_name(key)] for key in fields}


def groups_to_state(groups, device):
    """Own the filled group arrays as a State."""
    tree: dict = {}
    for key, value in groups.items():
        section, name = key.split("/")
        # On CPU, from_numpy retains these fresh arrays. No owner/buffer is reused
        # by a later step, reset or export; CUDA conversion also owns its storage.
        tree.setdefault(section, {})[name] = torch.from_numpy(value).to(device)
    return build_tensordict(tree, [len(next(iter(groups.values())))], device)


def field_arrays(schema, state, fields):
    """``{field key: (environments, width) NumPy view}`` of a host State."""
    return group_views(schema, schema.buffers(state), fields)


def core_fields(nq, nv, na=0):
    """Fields that resume a dm_control environment: physics, lifecycle, reward and task RNG."""
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


class Columns(dict):
    """``{field key: (environments, width) view}`` of fresh host group arrays; ``groups`` holds the arrays."""

    def __init__(self, schema, batch, fields):
        self.groups = new_groups(schema, batch)
        super().__init__(group_views(schema, self.groups, fields))


def model_columns(columns):
    """The episode-drawn model entries held by ``columns``: ``(key, named model field, index)``."""
    return [(key, *MODEL_FIELDS[key]) for key in columns if key in MODEL_FIELDS]


def write_row(columns, i, env, cumulative, model_fields):
    """Read live environment ``env`` into row ``i`` of ``columns``; environments in different rows may be written concurrently."""
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
    """Read the task RNG and episode-drawn model entries of environment ``env`` into row ``i`` of ``columns``."""
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
    for key, value in (extra or {}).items():
        columns[key][:] = value
    return groups_to_state(columns.groups, device)


def export_state(envs, cumulative, schema, fields, device, extra=None):
    """Export a fresh State of live environment owners with one device transfer per group.

    Read live environment owners into owned host arrays. CPU tensors retain those arrays;
    later simulation and exports cannot overwrite the snapshot.
    """
    columns = Columns(schema, len(envs), fields)
    model_fields = model_columns(columns)
    for i, env in enumerate(envs):
        write_row(columns, i, env, cumulative[i], model_fields)
    return columns_to_state(columns, device, extra)


def restore_instance(env, values, i, cumulative):
    """Write row ``i`` of host field arrays ``values`` back into an environment, model entries before forward kinematics."""
    physics, base = env.physics, env._env
    for key, (field, index) in MODEL_FIELDS.items():
        if key in values:
            getattr(physics.named.model, field)[index] = values[key][i]
    physics.data.qpos[:] = values["qpos"][i]
    physics.data.qvel[:] = values["qvel"][i]
    physics.data.time = values["time"][i].item()
    if "activation" in values:
        physics.data.act[:] = values["activation"][i]
    physics.forward()
    physics.data.qacc_warmstart[:] = values["warmstart"][i]
    base._step_count = values["lifecycle", "step_count"][i].item()
    base._reset_next_step = values["lifecycle", "reset_next_step"][i].item()
    env.task.random.set_state(
        (
            "MT19937",
            values["rng", "keys"][i].astype(np.uint32),
            values["rng", "position"][i].item(),
            int(values["rng", "has_gauss"][i].item()),
            values["rng", "cached_gaussian"][i].item(),
        )
    )
    cumulative[i] = values["task", "cumulative_reward"][i].item()


def validate_state(schema, state, num_envs, key):
    """Check that ``state`` is a full State of simulator ``key`` with ``num_envs`` rows and finite values."""
    if not isinstance(state, TensorDict) or state.batch_size != torch.Size([num_envs]):
        raise TypeError(f"{key} requires full TensorDict State with batch [num_envs]")
    schema.validate(state, finite=True)
