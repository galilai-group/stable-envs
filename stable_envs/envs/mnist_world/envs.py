"""MNIST World, the 2-D environment of Flow Equivariant World Models (Lillemark et al., ICML 2026).

A toroidal ``world_size`` x ``world_size`` black canvas holds ``num_digits`` MNIST training digits, each
moving by a constant integer velocity every step and wrapping around the edges. The agent sees a
``window_size`` x ``window_size`` view centred on its camera, and its action is the integer ``(dx, dy)``
offset that moves the camera. Overlapping digits add, clipped at 255.

The paper's generator is not published; the rules here are measured on the published dataset
(``flowm123/mnist-world``, see ``scripts/testing/freeze_mnist_world_corpus.py``):

- A digit's position is the ``(x, y)`` of its top-left pixel. Positions start uniform over the world.
  Velocities are uniform in ``{-1, 0, 1}`` per axis (6,141 draws over four splits; the paper's
  Table 8 says -2 to 2, the data never holds 2). Frame ``t`` shows the digits at ``start + t * velocity``.
- The camera starts at ``(0, 0)`` and frame ``t`` shows it at the sum of the first ``t`` actions, modulo the
  world. The window's top-left pixel is ``camera - window_size // 2``. A recorded episode matches these
  rules on every step: positions and camera exactly, frames to 0.27-0.53 / 255 mean (H.264 noise).
- Digits are unmodified MNIST training images; overlaps add and clip (on pixels where two digits are
  bright, a sum fits the frames to 1.0 / 255 mean and a maximum to 23.3).
- The dataset's actions are uniform in ``[-max_self_motion, max_self_motion]`` per axis after a scripted
  first 50 steps. That is the recording policy, not the environment; ``step`` takes any action in range.

Presets follow the paper's Table 8: ``dynamic_po`` (default; world 50, 5 digits, self-motion up to 10),
``static_po`` (digits do not move), ``dynamic_fo`` (world 32 = window, 3 digits) and ``dynamic_fo_no_sm``
(``dynamic_fo`` without self-motion: the only action is ``(0, 0)``). The files the dataset publishes as
``dynamic_fo`` are a world of 80 with 8 digits; ``preset="dynamic_fo", world_size=80, num_digits=8`` makes it.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.tensordicts import build_tensordict
from ...env import Env
from ...registry import register
from ...spaces import Dict
from . import data

PRESETS = {
    "dynamic_po": {"world_size": 50, "num_digits": 5, "max_self_motion": 10, "max_speed": 1},
    "static_po": {"world_size": 50, "num_digits": 5, "max_self_motion": 10, "max_speed": 0},
    "dynamic_fo": {"world_size": 32, "num_digits": 3, "max_self_motion": 10, "max_speed": 1},
    "dynamic_fo_no_sm": {"world_size": 32, "num_digits": 3, "max_self_motion": 0, "max_speed": 1},
}
#: Options every preset shares: the 32 x 32 view and the dataset's 200-frame episodes.
COMMON = {"window_size": 32, "max_episode_steps": 199}


def resolve_config(preset: str, options: dict) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"unknown MNIST World preset {preset!r}; the presets are {sorted(PRESETS)}")
    config = {**COMMON, **PRESETS[preset]}
    unknown = sorted(set(options) - set(config))
    if unknown:
        raise TypeError(f"unknown MNIST World option(s) {unknown}; the options are {sorted(config)}")
    config.update(options)
    for key in ("world_size", "num_digits", "window_size"):
        if int(config[key]) < 1:
            raise ValueError(f"MNIST World {key} must be positive, got {config[key]}")
    if config["world_size"] < data.DIGIT:
        raise ValueError(f"MNIST World world_size must hold a {data.DIGIT}-pixel digit, got {config['world_size']}")
    if config["max_self_motion"] < 0 or config["max_speed"] < 0:
        raise ValueError("MNIST World max_self_motion and max_speed must not be negative")
    if config["max_episode_steps"] is not None and config["max_episode_steps"] < 1:
        raise ValueError("MNIST World max_episode_steps must be positive or None")
    return {k: (v if v is None else int(v)) for k, v in config.items()}


@register
class MNISTWorld(Env):
    """MNIST World lanes in batched torch, on CPU or CUDA.

    State per lane, all int64: ``digits.image`` (the MNIST training index of each digit),
    ``digits.label``, ``digits.position`` and ``digits.velocity`` (``x, y`` per digit), ``camera``
    (``x, y``), ``step`` (steps since reset), and ``rng.seed`` / ``rng.episode``, from which the next
    unseeded reset draws. A State restores only into an env of the same configuration.

    Reward is 0 and episodes never terminate; ``truncated`` is set once ``step`` reaches
    ``max_episode_steps`` (199 by default: the dataset's 200 frames). Frames are the grey view repeated
    over three channels.
    """

    family = "mnist_world"
    domain = "mnist_world"
    name = "mnist-world"
    reference_id = "flowm123/mnist-world"
    fidelity_tier = "parity"
    action_dim = 2
    action_dtype = torch.int64
    default_size = 32

    def __init__(
        self,
        num_envs=1,
        size=None,
        supersample=1,
        device=None,
        backend="auto",
        seed=0,
        preset="dynamic_po",
        mnist_dir=None,
        **options,
    ):
        """Build ``num_envs`` lanes of ``preset`` updated with ``options``.

        ``options`` are ``world_size``, ``num_digits``, ``max_self_motion``, ``max_speed``,
        ``window_size`` and ``max_episode_steps`` (None for no limit). ``mnist_dir`` holds the MNIST
        training archives; by default they download once into ``$XDG_CACHE_HOME/stable-envs/mnist``.
        """
        self.config = resolve_config(preset, options)
        self.preset = preset
        window = self.config["window_size"]
        if size not in (None, window) or supersample != 1:
            raise ValueError(f"MNISTWorld draws its {window} x {window} view; size and supersample are fixed")
        if backend not in ("auto", "torch"):
            raise ValueError(f"MNISTWorld draws with torch, not {backend!r}")
        self.frame_shape = (window, window)
        self.action_low = -self.config["max_self_motion"]
        self.action_high = self.config["max_self_motion"]
        self._mnist_dir = str(mnist_dir or data.default_dir())
        super().__init__(num_envs, window, supersample, device, backend, seed)
        self._images, self._labels = data.device_table(self._mnist_dir, self.device)
        k = self.config["num_digits"]
        world = self.config["world_size"]
        speed = self.config["max_speed"]
        self.fields = {
            ("digits", "image"): (k, 0, data.NUM_IMAGES - 1),
            ("digits", "label"): (k, 0, 9),
            ("digits", "position"): (2 * k, 0, world - 1),
            ("digits", "velocity"): (2 * k, -speed, speed),
            ("camera",): (2, 0, world - 1),
            ("step",): (1, 0, None),
            ("rng", "seed"): (1, None, None),
            ("rng", "episode"): (1, 0, None),
        }
        self.state_bounds = {key: (lo, hi) for key, (_, lo, hi) in self.fields.items() if lo is not None}
        self._state: dict[tuple[str, ...], torch.Tensor] = {}
        self.reset(seed=seed)

    def make_variation_space(self):
        return Dict({})

    def _layout(self):
        return tuple((key, width, None) for key, (width, _, _) in self.fields.items())

    def _layout_dtypes(self):
        return {key: torch.int64 for key in self.fields}

    # State

    def _snapshot(self) -> TensorDict:
        tree: dict = {}
        for key, value in self._state.items():
            node = tree
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = value.clone()
        return build_tensordict(tree, [self.B], self.device)

    def _checked(self, state, what: str, batch: int | None = None) -> dict[tuple[str, ...], torch.Tensor]:
        """The fields of ``state`` on this env's device, after checking every one of them."""
        self._check_open(what)
        name = type(self).__name__
        if not isinstance(state, TensorDict) or len(state.batch_size) != 1:
            raise TypeError(f"{name}.{what}: expected a State TensorDict with one batch dimension")
        n = int(state.batch_size[0])
        if batch is not None and n != batch:
            raise ValueError(f"{name}.{what}: expected State batch [{batch}], got [{n}]")
        keys = {(k,) if isinstance(k, str) else tuple(k) for k in state.keys(True, True)}
        if keys != set(self.fields):
            raise ValueError(
                f"{name}.{what}: State keys differ from {sorted(self.fields)}: {sorted(keys ^ set(self.fields))}"
            )
        out = {}
        for key, (width, lo, hi) in self.fields.items():
            value = state[key]
            label = ".".join(key)
            if value.dtype != torch.int64:
                raise TypeError(f"{name}.{what}: State field {label} must be int64, got {value.dtype}")
            if tuple(value.shape) != (n, width):
                raise ValueError(
                    f"{name}.{what}: State field {label} has shape {tuple(value.shape)}, not {(n, width)}"
                )
            value = value.to(self.device)
            if (lo is not None and bool((value < lo).any())) or (hi is not None and bool((value > hi).any())):
                raise ValueError(f"{name}.{what}: State field {label} lies outside [{lo}, {hi}]")
            out[key] = value
        if not torch.equal(self._labels[out["digits", "image"]], out["digits", "label"]):
            raise ValueError(f"{name}.{what}: digits.label is not the label of digits.image")
        return out

    def get_state(self):
        """An independent snapshot of every lane."""
        self._check_open("get_state")
        return self._snapshot()

    def set_state(self, state):
        """Restore every lane from a State with ``num_envs`` lanes; the State is copied."""
        fields = self._checked(state, "set_state", self.B)
        self._state = {key: value.clone() for key, value in fields.items()}

    def reset(self, seed=None):
        """Reset every lane and return the State.

        Lane ``i`` of ``reset(seed)`` draws from ``numpy.random.default_rng([seed + i, 0])``. Each later
        unseeded reset of the lane takes the next episode number in place of 0.
        """
        self._check_open("reset")
        if seed is not None:
            seeds = np.arange(self.B, dtype=np.int64) + int(seed)
            episodes = np.zeros(self.B, np.int64)
        else:
            seeds = self._state["rng", "seed"][:, 0].cpu().numpy()
            episodes = self._state["rng", "episode"][:, 0].cpu().numpy() + 1
        k, world, speed = self.config["num_digits"], self.config["world_size"], self.config["max_speed"]
        images = np.empty((self.B, k), np.int64)
        positions = np.empty((self.B, 2 * k), np.int64)
        velocities = np.empty((self.B, 2 * k), np.int64)
        for lane in range(self.B):
            rng = np.random.default_rng([int(seeds[lane]) & (2**63 - 1), int(episodes[lane])])
            images[lane] = rng.integers(0, data.NUM_IMAGES, k)
            positions[lane] = rng.integers(0, world, 2 * k)
            velocities[lane] = rng.integers(-speed, speed + 1, 2 * k)
        image = torch.as_tensor(images, device=self.device)
        self._state = {
            ("digits", "image"): image,
            ("digits", "label"): self._labels[image],
            ("digits", "position"): torch.as_tensor(positions, device=self.device),
            ("digits", "velocity"): torch.as_tensor(velocities, device=self.device),
            ("camera",): torch.zeros(self.B, 2, dtype=torch.int64, device=self.device),
            ("step",): torch.zeros(self.B, 1, dtype=torch.int64, device=self.device),
            ("rng", "seed"): torch.as_tensor(seeds, device=self.device).reshape(self.B, 1),
            ("rng", "episode"): torch.as_tensor(episodes, device=self.device).reshape(self.B, 1),
        }
        return self._snapshot()

    # Dynamics

    def _actions(self, actions) -> torch.Tensor:
        a = actions if torch.is_tensor(actions) else torch.as_tensor(np.asarray(actions))
        bound = self.config["max_self_motion"]
        if a.is_floating_point() or a.dtype == torch.bool or tuple(a.shape) != (self.B, 2):
            raise ValueError(f"MNISTWorld: actions must be ({self.B}, 2) integers in [-{bound}, {bound}]")
        a = a.to(device=self.device, dtype=torch.int64)
        if bool((a.abs() > bound).any()):
            raise ValueError(f"MNISTWorld: actions must be ({self.B}, 2) integers in [-{bound}, {bound}]")
        return a

    def _step(self, actions, *, active=None):
        self._check_open("step")
        a = self._actions(actions)
        world = self.config["world_size"]
        s = self._state
        position = (s["digits", "position"] + s["digits", "velocity"]) % world
        camera = (s["camera",] + a) % world
        step = s["step",] + 1
        if active is not None:
            keep = active.to(self.device).reshape(self.B, 1)
            position = torch.where(keep, position, s["digits", "position"])
            camera = torch.where(keep, camera, s["camera",])
            step = torch.where(keep, step, s["step",])
        s["digits", "position"], s["camera",], s["step",] = position, camera, step
        limit = self.config["max_episode_steps"]
        truncated = torch.zeros(self.B, dtype=torch.bool, device=self.device) if limit is None else step[:, 0] >= limit
        result = {
            "state": self._snapshot(),
            "reward": torch.zeros(self.B, dtype=torch.float32, device=self.device),
            "terminated": torch.zeros(self.B, dtype=torch.bool, device=self.device),
            "truncated": truncated,
        }
        return build_tensordict(result, [self.B], self.device)

    # Rendering

    def _draw(self, image, position, camera) -> torch.Tensor:
        """``(N, V, V, 3)`` uint8 views of ``N`` lanes from their digit indices, positions and cameras."""
        world, window = self.config["world_size"], self.config["window_size"]
        n, k = image.shape
        offsets = torch.arange(window, device=self.device) - window // 2
        rows = (camera[:, 1:2] + offsets) % world  # N, V
        cols = (camera[:, 0:1] + offsets) % world
        xy = position.reshape(n, k, 2)
        u = (rows[:, None, :] - xy[..., 1:2]) % world  # N, K, V
        v = (cols[:, None, :] - xy[..., 0:1]) % world
        inside = (u < data.DIGIT)[..., :, None] & (v < data.DIGIT)[..., None, :]  # N, K, V, V
        pixels = self._images[
            image[:, :, None, None],
            u.clamp(max=data.DIGIT - 1)[..., :, None],
            v.clamp(max=data.DIGIT - 1)[..., None, :],
        ]
        grey = torch.where(inside, pixels.to(torch.int32), 0).sum(1).clamp(max=255).to(torch.uint8)
        return grey[..., None].expand(n, window, window, 3).contiguous()

    def render(self, state=None):
        """Draw ``(N, V, V, 3)`` uint8 frames from a State of any batch, or from the live lanes."""
        if state is None:
            return self._render_live()
        fields = self._checked(state, "render")
        return self._draw(fields["digits", "image"], fields["digits", "position"], fields["camera",])

    def _render_live(self):
        self._check_open("render")
        s = self._state
        return self._draw(s["digits", "image"], s["digits", "position"], s["camera",])

    def _close(self):
        self._state = {}
