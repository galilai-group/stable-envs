"""Public PushT environments built on the source task.

``PushT`` and ``PushTDiscrete`` expose complete, replayable State and render on
CPU with the source Pygame renderer. They take the ``PushT`` registry name from
the compact renderer in ``envs.pusht.legacy``.
"""

import copy

import numpy as np
import torch
from tensordict import TensorDict

from ...env import Env
from ...registry import REGISTRY, register
from ...spaces import Box, Dict, Discrete
from ...specs import CompositeSpec, Spec
from . import legacy


class PushT(Env):
    """Source-aligned continuous PushT with complete, replayable numeric State.

    Actions are relative controls in [-1, 1], not absolute pixel targets.
    Physics and the exact Pygame renderer run on CPU. ``device='cuda'``
    selects tensor/output storage, not GPU simulation or GPU rasterization.
    Restoring a checkpoint replays its history; ordinary stepping does not.
    """

    family = domain = "pusht"
    name = "push-t"
    reference_id = "swm/PushT-v1"
    default_size = 224
    state_dim = 0
    execution_mode = "DELEGATED_CPU"
    cpu_render_support = "SUPPORTED_PYGAME"
    action_dim = 2
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    discrete = False

    def __init__(
        self,
        num_envs=1,
        size=None,
        supersample=1,
        device=None,
        backend="auto",
        seed=0,
        *,
        render_action=False,
        with_target=True,
    ):
        from .projection import PushTProjection
        from .simulation.replay_batch import PushTReplayAdapter

        if backend not in ("auto", "pygame"):
            raise ValueError("Source-aligned PushT supports backend='pygame' only")
        if supersample != 1:
            raise ValueError("The source PushT renderer has no supersample setting")
        device = Env.resolve_device(device)
        resolution = self.default_size if size is None else size
        if render_action and resolution < 96:
            raise ValueError("Source PushT action markers require size >= 96")
        self._simulator = PushTReplayAdapter(
            num_envs,
            device,
            seed,
            discrete=self.discrete,
            config=dict(resolution=resolution, render_action=render_action, with_target=with_target),
        )
        self._projection = PushTProjection()
        try:
            super().__init__(num_envs, resolution, 1, device, "pygame", 0 if seed is None else seed)
            row = self._simulator.envs[0].get_state()
            self._fixed_fields = {k: (v.shape, v.dtype) for k, v in row.items() if not k.startswith("history/")}
            self._render_fields = [
                (tuple(k.split("/")), shape, torch.from_numpy(np.empty((), dtype)).dtype)
                for k, (shape, dtype) in self._fixed_fields.items()
                if k.startswith(("recipe/", "anchor/", "current/", "owned/variation/")) and k != "current/goal_image"
            ]
        except BaseException:
            self._simulator.close()
            self._projection.close()
            raise

    def make_variation_space(self):
        def convert(node):
            if hasattr(node, "spaces"):
                return Dict({k: convert(v) for k, v in node.spaces.items()})
            if hasattr(node, "n"):
                return Discrete(int(node.n), start=int(node.start), init_value=copy.deepcopy(node.init_value))
            return Box(node.low, node.high, dtype=node.dtype, init_value=copy.deepcopy(node.init_value))

        return convert(self._simulator.envs[0].env.variation_space)

    @property
    def state_spec(self):
        # History has a growing axis; the spec describes the current checkpoint only.
        fields = {}
        for key, value in self.get_state().items(True, True):
            branch = fields
            for part in key[:-1]:
                branch = branch.setdefault(part, {})
            branch[key[-1]] = Spec(tuple(value.shape[1:]), dtype=value.dtype, device=self.device)
        return CompositeSpec(fields, device=self.device)

    @property
    def action_spec(self):
        return Spec(
            (2,),
            dtype=self.action_dtype,
            low=self.action_low,
            high=self.action_high,
            discrete_n=16 if self.discrete else None,
            device=self.device,
        )

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        return self._simulator.reset(seed, options)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step(self, actions, *, active=None):
        return self._simulator.step(actions, active=active)

    @staticmethod
    def _stack_states(values):
        from .simulation.history import stack_states

        return stack_states(values)

    def _variation_rows(self, attribute):
        from .simulation.numeric import nodes

        rows = []
        for lane in self._simulator.envs:
            leaves = {
                p.strip("/").replace("/", "."): n
                for p, n in nodes(lane.env.variation_space)
                if not hasattr(n, "spaces")
            }
            rows.append(
                np.concatenate(
                    [
                        np.asarray(getattr(leaves[k], attribute)).reshape(-1)
                        for k, _, _ in self.variation_space.flat_layout()
                    ]
                )
            )
        return torch.as_tensor(np.stack(rows), device=self.device)

    def get_variation(self):
        self._check_open("get_variation")
        return self._variation_rows("value")

    def sample_variation(self, batch=None, paths=None, rng=None):
        """Sample valid integer colors and source discrete ranges (including start=1)."""
        self._check_open("sample_variation")
        batch = self.B if batch is None else batch
        if type(batch) is not int or batch < 1:
            raise ValueError("PushT Variation sample batch must be positive")
        if paths is not None and set(paths) - set(self.variation_names):
            raise KeyError("Unknown PushT Variation path")
        rng = self.rng if rng is None else rng
        result = np.tile(self.default_variation(), (batch, 1))
        for key, offset, width in self.variation_space.flat_layout():
            if paths is not None and key not in paths:
                continue
            node = self.variation_space._walk(key)
            if hasattr(node, "n"):
                values = rng.integers(node.start, node.start + node.n, (batch, width))
            else:
                low, high = np.asarray(node.low).reshape(-1), np.asarray(node.high).reshape(-1)
                values = rng.uniform(low, high.astype(np.float64) + (node.dtype.kind in "iub"), (batch, width))
                if node.dtype.kind in "iub":
                    values = np.floor(values)
            result[:, offset : offset + width] = values
        return result

    #: Leaves only the frame reads; ``set_variation`` applies them to the running episode as well.
    #: Body shapes and scales need a rebuilt pymunk space, and start poses and velocity exist only
    #: at reset, so those wait for the next reset.
    immediate_variation_leaves = frozenset(
        {
            "agent.color",
            "block.color",
            "goal.color",
            "goal.scale",
            "goal.position",
            "goal.angle",
            "background.color",
            "rendering.render_goal",
        }
    )

    def set_variation(self, variation=None, **overrides):
        """Configure the values every later reset restores.

        Appearance leaves (``immediate_variation_leaves``) also apply to the running episode;
        the others take effect at the next reset.
        """
        from .simulation.numeric import nodes

        self._check_open("set_variation")
        if isinstance(variation, dict):
            overrides = {**variation, **overrides}
            variation = None
        flat_keys = set()
        if variation is not None:
            rows = np.asarray(variation.detach().cpu() if torch.is_tensor(variation) else variation)
            if rows.shape == (self.variation_dim,):
                rows = np.broadcast_to(rows, (self.B, self.variation_dim))
            if rows.shape != (self.B, self.variation_dim):
                raise ValueError("Invalid PushT Variation batch")
            for key, off, width in self.variation_space.flat_layout():
                overrides[key] = rows[:, off : off + width].reshape(self.B, *self.variation_space._walk(key).shape)
                flat_keys.add(key)
        staged = []
        for lane, replay in enumerate(self._simulator.envs):
            space = copy.deepcopy(replay.env.variation_space)
            leaves = {p.strip("/").replace("/", "."): n for p, n in nodes(space) if not hasattr(n, "spaces")}
            for key, raw in overrides.items():
                if key not in leaves:
                    raise ValueError(f"Unknown PushT Variation: {key}")
                node = leaves[key]
                value = np.asarray(raw.detach().cpu() if torch.is_tensor(raw) else raw)
                if value.shape == (self.B, *node.shape):
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
                node.set_init_value(converted, reset_value=False)
                if key in self.immediate_variation_leaves:
                    node.set_value(copy.deepcopy(converted))
            staged.append(space)
        for replay, space in zip(self._simulator.envs, staged, strict=True):
            replay.env.variation_space = space
            replay.env.goal_pose = np.concatenate([space["goal"]["position"].value, [space["goal"]["angle"].value]])
        return self._variation_rows("init_value")

    def to_render_state(self, state=None):
        """Return an owned, history-free copy of the State fields the CPU renderer reads."""
        state = self.get_state() if state is None else state
        self._render_fields_of(state)
        return state.select(*(key for key, _, _ in self._render_fields)).clone()

    def _render_fields_of(self, state):
        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        if not isinstance(state, TensorDict) or len(state.batch_size) != 1 or not state.batch_size[0]:
            raise ValueError("PushT render requires a nonempty State batch")
        flat = dict(state.items(include_nested=True, leaves_only=True))
        fields = {}
        for key, shape, dtype in self._render_fields:
            value = flat.get(key)
            if value is None:
                raise KeyError(key)
            if value.shape != (*state.batch_size, *shape) or value.dtype != dtype:
                raise ValueError(f"Invalid PushT render field: {key}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite PushT render field: {key}")
            fields["/".join(key)] = value
        return fields

    def render(self, state=None, *, size=None, out=None):
        self._check_open("render")
        size = self.size if size is None else size
        if type(size) is not int or size < 1:
            raise ValueError("PushT render size must be a positive integer")
        if self._simulator.envs[0].config["render_action"] and size < 96:
            raise ValueError("Source PushT action markers require size >= 96")
        fields = {k: v.cpu().numpy() for k, v in self._render_fields_of(state).items()}
        rows = [{k: v[i] for k, v in fields.items()} for i in range(len(fields["recipe/config"]))]
        frames = torch.as_tensor(self._projection.render(rows, size), device=self.device)
        if out is not None:
            if (
                out.shape != frames.shape
                or out.dtype != frames.dtype
                or out.device != frames.device
                or not out.is_contiguous()
            ):
                raise ValueError("Invalid PushT render output buffer")
            out.copy_(frames)
            return out
        return frames

    def _close(self):
        self._projection.close()
        self._simulator.close()


class PushTDiscrete(PushT):
    """Source-aligned PushT with two discrete polar-action indices in [0, 16).

    Complete State restores by replay. Physics and rendering run on CPU;
    CUDA selects output storage only.
    """

    name = "push-t-discrete"
    reference_id = "swm/PushT-Discrete-v1"
    discrete = True
    action_dtype = torch.int64
    action_low, action_high = 0, 15


if "PushT" in REGISTRY:
    assert REGISTRY["PushT"] is legacy.PushT
    del REGISTRY["PushT"]
register(PushT)
register(PushTDiscrete)
