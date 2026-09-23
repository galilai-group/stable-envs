"""Public parameterized Fetch semantics; renderer dependencies stay optional."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from .._adapters.fetch_pickandplace import FetchAdapter
from ..env import Env
from ..spaces import Box, Dict


class FetchSemantics:
    execution_mode = "DELEGATED_CPU"
    action_dim = 4
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    cpu_render_support = "CPU_BACKEND_MISSING"
    _reset_keys = {
        "agent.start_position",
        "block.start_position",
        "block.angle",
        "block.mass",
        "goal.start_position",
        "object.color",
    }

    def __init__(self, num_envs=1, size=None, supersample=None, device="cpu", backend="auto", seed=0):
        device = Env.resolve_device(device)
        self._simulator = FetchAdapter(
            num_envs, device, seed, self.SCENE, "dense" if "Dense" in self.REF_ID else "sparse"
        )
        self._has_object = self._simulator.has_object
        self._nq = self._simulator.fields["qpos"][0]
        self._reset_keys = set(type(self)._reset_keys)
        if not self._has_object:
            self._reset_keys -= {"block.mass", "block.angle", "block.start_position", "object.color"}
        self._reset_keys.add("table.color")
        self._configured_mass = self._simulator._mass.copy()
        self.state_dim = self._nq + 3  # Private frozen renderer width only.
        from .fetch import GOAL_NAMES, OBJECT_JOINT, ROBOT_JOINTS

        self.state_names = (*ROBOT_JOINTS, *(OBJECT_JOINT if self._has_object else ()), *GOAL_NAMES)
        Env.__init__(
            self,
            num_envs=num_envs,
            size=size,
            supersample=1 if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )

    def make_variation_space(self):
        # Derive precisely the existing renderer defaults without compiling its
        # surgical model or allocating a GL/MJWarp context in the CPU profile.
        from .fetch import BLUE, _effective_rgba

        m = self._simulator.envs[0].unwrapped.model
        floor = _effective_rgba(m, m.geom("floor0").id)[:3]
        table = next(
            _effective_rgba(m, g)[:3]
            for g in range(m.ngeom)
            if m.geom_matid[g] >= 0 and m.material(int(m.geom_matid[g])).name == "table_mat"
        )
        arm = next(
            _effective_rgba(m, g)[:3]
            for g in range(m.ngeom)
            if np.allclose(_effective_rgba(m, g)[:3], BLUE, atol=2e-3)
        )
        colors = dict(floor=floor, table=table, arm=arm, goal=m.site_rgba[m.site("target0").id, :3])
        if self._has_object:
            colors["object"] = _effective_rgba(m, m.geom("object0").id)[:3]
        result = {
            k: Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=v.tolist())}) for k, v in colors.items()
        }
        result["light"] = Dict({"diffuse": Box(low=0.0, high=2.0, shape=(1,), init_value=[1.0])})
        result["camera"] = Dict(
            {
                k: Box(low=lo, high=hi, shape=(1,), init_value=[v])
                for k, lo, hi, v in [
                    ("distance_scale", 0.6, 1.6, 1.0),
                    ("azimuth_delta", -25.0, 25.0, 0.0),
                    ("elevation_delta", -20.0, 20.0, 0.0),
                ]
            }
        )
        if self._has_object:
            result["block"] = Dict(
                {"mass": Box(low=0.01, high=50.0, shape=(1,), init_value=self._simulator._mass[:1].tolist())}
            )
        return Dict(result)

    def _layout(self):
        return tuple(
            (
                key if isinstance(key, tuple) else (key,),
                width,
                0 if key == "qpos" else self._nq if key == "goal" else None,
            )
            for key, (width, _) in self._simulator.fields.items()
        )

    def _layout_dtypes(self):
        return {key if isinstance(key, tuple) else (key,): dtype for key, (_, dtype) in self._simulator.fields.items()}

    def _pack_render_state(self, state):
        return torch.cat([state["qpos"], state["goal"]], dim=-1).to(device=self.device, dtype=torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, self.state_dim)
        return TensorDict(
            {"qpos": rows[:, : self._nq].clone(), "goal": rows[:, self._nq :].clone()}, [len(rows)], device=self.device
        )

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        options = options or {}
        if set(options) - {"variation", "variation_values"}:
            raise ValueError("Unsupported Fetch reset option")
        sampled = options.get("variation", [])
        explicit = options.get("variation_values", {})
        if (set(sampled) | set(explicit)) - self._reset_keys:
            raise ValueError("Uncertified Fetch reset Variation")
        # Variation reset uses its own Gymnasium space tree and reseeds even
        # for seed=None, as SWM does. It never consumes simulator PCG64.
        values = []
        for lane in range(self.B):
            v = {}
            if sampled:
                spaces = _historical_sampling_tree(self._has_object)
                spaces.seed(None if seed is None else int(seed) + lane)
                for key in sampled:
                    node = spaces
                    for part in key.split("."):
                        node = node[part]
                    v[key] = node.sample()
            v.update(explicit)
            values.append(v)
        for key in ("object.color", "table.color"):
            if any(key in v for v in values):
                Env.set_variation(self, {key: np.stack([v[key] for v in values])})
        self._simulator.configure_object_mass(self._configured_mass)
        state = self._simulator.reset(seed, values)
        if self._has_object:
            off = next(off for name, off, _ in self.variation_space.flat_layout() if name == "block.mass")
            self._style.values[:, off] = torch.as_tensor(
                self._simulator._mass, device=self.device, dtype=torch.float32
            )
        return state

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def step(self, actions):
        self._check_open("step")
        return self._simulator.step(actions)

    def set_variation(self, variation=None, **overrides):
        result = super().set_variation(variation, **overrides)
        if not self._has_object:
            return result
        layout = {name: (off, width) for name, off, width in self.variation_space.flat_layout()}
        off, _ = layout["block.mass"]
        self._configured_mass = self.get_style()[:, off].cpu().numpy().astype(np.float64).copy()
        self._simulator.configure_object_mass(self._configured_mass)
        return result

    def _render(self, *args, **kwargs):
        if self.device.type != "cuda":
            raise NotImplementedError("CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp")
        if not hasattr(self, "_b"):
            from .fetch import _build_bundle, _require_gpu_renderer

            _require_gpu_renderer()
            self._b = _build_bundle(self.SCENE)
            self._check_plate_is_constant()
        return super()._render(*args, **kwargs)

    def _close(self):
        self._simulator.close()
        super()._close()


def _historical_sampling_tree(has_object=True):
    """Upstream Gymnasium seeding tree; only bounded leaves are exposed here."""
    import gymnasium as gym

    def box(lo, hi, shape):
        return gym.spaces.Box(lo, hi, shape=shape, dtype=np.float64)

    tree = gym.spaces.Dict(
        {
            "table": gym.spaces.Dict({"color": box(0.0, 1.0, (3,))}),
            "object": gym.spaces.Dict({"color": box(0.0, 1.0, (3,))}),
            "background": gym.spaces.Dict({"color": box(0.0, 1.0, (3,))}),
            "light": gym.spaces.Dict({"intensity": box(0.0, 1.0, (1,))}),
            "camera": gym.spaces.Dict({"angle_delta": box(-10.0, 10.0, (1, 2))}),
            "agent": gym.spaces.Dict(
                {"start_position": gym.spaces.Box(np.array([1.25, 0.6]), np.array([1.45, 0.9]), dtype=np.float64)}
            ),
            "goal": gym.spaces.Dict(
                {
                    "start_position": gym.spaces.Box(
                        np.array([1.15, 0.6, 0.4247]), np.array([1.45, 0.9, 0.4247]), dtype=np.float64
                    )
                }
            ),
            "rendering": gym.spaces.Dict({"transparent_arm": gym.spaces.Discrete(2)}),
            "block": gym.spaces.Dict(
                {
                    "start_position": gym.spaces.Box(np.array([1.15, 0.6]), np.array([1.45, 0.9]), dtype=np.float64),
                    "angle": box(-np.pi, np.pi, (1,)),
                    "mass": box(0.01, 50.0, (1,)),
                }
            ),
        }
    )

    if not has_object:
        del tree.spaces["block"]
    return tree
