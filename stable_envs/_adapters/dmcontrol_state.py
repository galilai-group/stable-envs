"""Owned, column-wise export of the accepted DMC resume fields.

Each call allocates fresh host columns, fills them from the lane owners, then
transfers each complete column once. No per-lane TensorDicts or CUDA transfers.
The field declarations and simulator restore paths remain authoritative.
"""

import numpy as np
import torch
from tensordict import TensorDict

_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}


def export_state(envs, cumulative, fields, device):
    batch = len(envs)
    columns = {key: np.empty((batch, width), dtype=_DTYPES[dtype]) for key, (width, dtype) in fields.items()}
    for i, env in enumerate(envs):
        physics, base = env.physics, env._env
        data = physics.data
        algorithm, keys, position, has_gauss, cached = env.task.random.get_state()
        if algorithm != "MT19937":
            raise ValueError(f"Unsupported DMC RNG algorithm: {algorithm}")
        columns["qpos"][i] = data.qpos
        columns["qvel"][i] = data.qvel
        columns["time"][i] = data.time
        columns["warmstart"][i] = data.qacc_warmstart
        columns["lifecycle", "step_count"][i] = base._step_count
        columns["lifecycle", "reset_next_step"][i] = base._reset_next_step
        columns["task", "cumulative_reward"][i] = cumulative[i]
        columns["rng", "keys"][i] = keys
        columns["rng", "position"][i] = position
        columns["rng", "has_gauss"][i] = has_gauss
        columns["rng", "cached_gaussian"][i] = cached
        if ("task", "target_pos") in columns:
            columns["task", "target_pos"][i] = physics.named.model.geom_pos["target"]
        if ("model", "target_site_xz") in columns:
            columns["model", "target_site_xz"][i] = physics.named.model.site_pos["target", ["x", "z"]]
        if ("model", "target_body_xz") in columns:
            columns["model", "target_body_xz"][i] = physics.named.model.body_pos["target_ball", ["x", "z"]]
            columns["model", "target_body_qwqy"][i] = physics.named.model.body_quat["target_ball", ["qw", "qy"]]
        if "activation" in columns:
            columns["activation"][i] = data.act
    # On CPU, from_numpy retains these fresh arrays. No owner/buffer is reused
    # by a later step, reset or export; CUDA conversion also owns its storage.
    return TensorDict({key: torch.from_numpy(value).to(device) for key, value in columns.items()}, [batch], device=device)
