"""Batches of PushT replay owners on CPU."""

from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.space_nodes import nodes
from ...._execution.process import BROADCAST, PER_LANE
from ...._execution.protocol import SimulatorAdapter
from .replay import HISTORY_KEYS, PushTReplay, validate_seed


def _leaves(space):
    return {p.strip("/").replace("/", "."): n for p, n in nodes(space) if not hasattr(n, "spaces")}


def lane_options(space, options, num_envs, lane):
    """Select lane ``lane`` of each ``variation_values`` entry shaped ``(num_envs, *leaf.shape)``."""
    values = (options or {}).get("variation_values")
    if not isinstance(values, dict):
        return options
    selected = {}
    for key, value in values.items():
        node = space
        for part in str(key).split("."):
            node = getattr(node, "spaces", {}).get(part)
        array = np.asarray(value) if node is not None and not hasattr(node, "spaces") else None
        if array is not None and array.shape == (num_envs, *node.shape):
            value = array[lane].copy() if node.shape else array[lane]
        selected[key] = value
    return {**options, "variation_values": selected}


def resolve_variation(space, overrides, num_envs, flat_keys=frozenset()):
    """Return each lane's checked ``{leaf: value}`` overrides; a value shaped ``(num_envs, *leaf)`` is per lane.

    ``flat_keys`` names values that come from flat Variation rows and must convert exactly to the leaf dtype.
    """
    leaves = _leaves(space)
    lanes = []
    for lane in range(num_envs):
        values = {}
        for key, raw in overrides.items():
            if key not in leaves:
                raise ValueError(f"Unknown PushT Variation: {key}")
            node = leaves[key]
            value = np.asarray(raw.detach().cpu() if torch.is_tensor(raw) else raw)
            if value.shape == (num_envs, *node.shape):
                value = value[lane]
            if value.shape != node.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid PushT Variation shape/value: {key}")
            # Keep Python scalar versus NumPy scalar/array representation;
            # source reset arithmetic and checkpoint equality depend on it.
            converted = copy.deepcopy(raw) if type(raw) in (int, float, bool) else value.copy()
            if key in flat_keys:
                dtype = np.dtype(node.dtype)
                converted = value.astype(dtype)
                if dtype.kind in "iub" and not np.array_equal(converted, value):
                    raise ValueError(f"Nonintegral PushT Variation: {key}")
                if hasattr(node, "n"):
                    converted = int(converted)
            if not node.contains(converted):
                raise ValueError(f"Invalid PushT Variation range/dtype: {key}")
            values[key] = converted
        lanes.append(values)
    return lanes


def state_columns(state, template, num_envs):
    """Check State keys, dtypes and batch against one lane's ``template`` row; return host columns."""
    if not isinstance(state, TensorDict) or state.batch_size != torch.Size([num_envs]):
        raise ValueError("PushT full State requires batch [num_envs]")
    if set(state.keys(True, True)) != {tuple(key.split("/")) for key in template}:
        raise ValueError("PushT full State keys differ")
    # Transfer each whole field once, then take owned lane copies on the host.
    fields = {}
    for key, sample in template.items():
        value = state[tuple(key.split("/"))]
        if value.dtype != torch.from_numpy(sample).dtype or value.shape[0] != num_envs:
            raise ValueError(f"Invalid PushT State dtype/batch: {key}")
        fields[key] = value.detach().cpu().numpy()
    return fields


class PushTReplayAdapter(SimulatorAdapter):
    """Batch CPU PushT owners with CPU/CUDA State storage.

    Stage every lane before installing a reset or restored State. Export pads
    histories to the longest lane with zero, invalid entries; padding adds no
    episode time limit.
    """

    process_methods = {"install_variation": PER_LANE, "variation_rows": BROADCAST}
    action_dim = 2

    def __init__(self, batch, device="cpu", seed=0, *, discrete=False, config=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("PushT batch must be a positive integer")
        validate_seed(seed)
        self.num_envs, self.device = batch, torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("PushT supports CPU physics with CPU/CUDA State storage")
        self.envs = []
        self._prepared = None
        try:
            for lane in range(batch):
                self.envs.append(
                    PushTReplay(discrete=discrete, config=config, seed=None if seed is None else int(seed) + lane)
                )
        except BaseException:
            self.close()
            raise

    @property
    def template(self):
        """A lane whose configuration, Variation space and State fields describe the batch."""
        return self.envs[0]

    def get_state(self):
        rows = [env.get_state() for env in self.envs]
        capacity = max(int(row["history/lengths"]) for row in rows)
        result = {}
        for key in rows[0]:
            if key in HISTORY_KEYS:
                sample = rows[0][key]
                values = np.zeros((self.num_envs, capacity, *sample.shape[1:]), sample.dtype)
                for lane, row in enumerate(rows):
                    values[lane, : len(row[key])] = row[key]
            else:
                values = np.stack([row[key] for row in rows])
            result[tuple(key.split("/"))] = torch.as_tensor(values, device=self.device)
        return TensorDict(result, [self.num_envs], device=self.device)

    def validate(self, state):
        fields = state_columns(state, self.envs[0].get_state(), self.num_envs)
        rows = [
            {key: np.asarray(value[lane]).copy() for key, value in fields.items()} for lane in range(self.num_envs)
        ]
        for env, row in zip(self.envs, rows, strict=True):
            env.validate(row)
        return rows

    def _swap(self, staged):
        previous, self.envs = self.envs, staged
        for env in previous:
            env.close()

    def prepare_restore(self, state):
        """Replay every lane of ``state`` into new owners; the live lanes do not change."""
        if self._prepared is not None:
            raise RuntimeError("A PushT restore is already prepared")
        rows = self.validate(state)
        staged = []
        try:
            for env, row in zip(self.envs, rows, strict=True):
                staged.append(env.reconstruct(row))
        except BaseException:
            for env in staged:
                env.close()
            raise
        self._prepared = staged

    def commit_restore(self):
        if self._prepared is None:
            raise RuntimeError("No PushT restore is prepared")
        staged, self._prepared = self._prepared, None
        self._swap(staged)

    def discard_restore(self):
        staged, self._prepared = self._prepared, None
        for env in staged or ():
            env.close()

    def set_state(self, state):
        self.prepare_restore(state)
        self.commit_restore()

    def reset(self, seed=None, options=None):
        """Reset every lane; ``options`` may also be a list with one options mapping per lane."""
        validate_seed(seed)
        space = self.envs[0].env.variation_space
        lanes = (
            options
            if isinstance(options, list)
            else [lane_options(space, options, self.num_envs, lane) for lane in range(self.num_envs)]
        )
        staged = []
        try:
            for lane, env in enumerate(self.envs):
                staged.append(env.reset_candidate(None if seed is None else int(seed) + lane, lanes[lane]))
        except BaseException:
            for env in staged:
                env.close()
            raise
        self._swap(staged)
        return self.get_state()

    def install_variation(self, lanes, immediate=frozenset()):
        """Set each lane's checked ``resolve_variation`` values as reset values.

        Leaves in ``immediate`` also change the running episode.
        """
        staged = []
        for replay, values in zip(self.envs, lanes, strict=True):
            space = copy.deepcopy(replay.env.variation_space)
            leaves = _leaves(space)
            for key, converted in values.items():
                node = leaves[key]
                node.set_init_value(converted, reset_value=False)
                if key in immediate:
                    node.set_value(copy.deepcopy(converted))
            staged.append(space)
        for replay, space in zip(self.envs, staged, strict=True):
            replay.env.variation_space = space
            replay.env.goal_pose = np.concatenate([space["goal"]["position"].value, [space["goal"]["angle"].value]])

    def variation_rows(self, attribute, keys):
        """Return each lane's Variation leaves ``keys`` as one flat row of ``attribute`` values."""
        rows = []
        for replay in self.envs:
            leaves = _leaves(replay.env.variation_space)
            rows.append(np.concatenate([np.asarray(getattr(leaves[k], attribute)).reshape(-1) for k in keys]))
        return np.stack(rows)

    def check_step(self, actions, active):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 2):
            raise ValueError("PushT actions require shape [num_envs, 2]")
        for action in actions:
            self.template.validate_action(action)
        if torch.is_tensor(active):
            active = active.detach().cpu().numpy()
        if active is not None and (
            not isinstance(active, np.ndarray) or active.dtype != np.bool_ or active.shape != (self.num_envs,)
        ):
            raise ValueError("PushT active mask requires bool [num_envs]")
        return actions, active

    def probe_actions(self):
        return np.zeros((1, 2), np.int64 if self.template.discrete else np.float64)

    def step(self, actions, *, active=None, capture_state=True):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 2):
            raise ValueError("PushT actions require shape [num_envs, 2]")
        for env, action in zip(self.envs, actions, strict=True):
            env.validate_action(action)
        if active is None:
            active = np.ones(self.num_envs, np.bool_)
        elif torch.is_tensor(active):
            active = active.detach().cpu().numpy()
        if not isinstance(active, np.ndarray) or active.dtype != np.bool_ or active.shape != (self.num_envs,):
            raise ValueError("PushT active mask requires bool [num_envs]")
        outputs = []
        for env, action, enabled in zip(self.envs, actions, active, strict=True):
            if enabled:
                out = env.step(action)
                outputs.append((out[1], out[2], out[3]))
            else:
                success, distance = env.env.eval_state(env.env.goal_state, env.env._get_obs())
                outputs.append((-distance, success, False))
        return TensorDict(
            dict(
                **(dict(state=self.get_state()) if capture_state else {}),
                reward=torch.tensor([v[0] for v in outputs], dtype=torch.float64, device=self.device),
                success=torch.tensor([v[1] for v in outputs], dtype=torch.bool, device=self.device),
                terminated=torch.tensor([v[1] for v in outputs], dtype=torch.bool, device=self.device),
                truncated=torch.tensor([v[2] for v in outputs], dtype=torch.bool, device=self.device),
            ),
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        self.discard_restore()
        for env in self.envs:
            env.close()
        self.envs = []
