"""What the batched Roadster backends share: episode resets, State export, validation and restore.

A backend keeps the task bookkeeping as ``(num_envs, width)`` tensors in the State layout on its compute
device and the MuJoCo state in its physics batch. Resets draw each environment's episode with the source's
random draws on the host, then write every environment to the physics batch in one call.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.physics import state_key
from ...._execution.protocol import SimulatorAdapter
from ..source.sim import FUEL_CAPACITY
from .adapter import (
    _NUMPY_DTYPES,
    FX_ENTRY,
    FX_HISTORY,
    MAX_ROUTE,
    MAX_SEED,
    MAX_TARGETS,
    PHYSICS_LAYOUT,
    PHYSICS_LEAVES,
    RAIN_MAX,
    RENDER_FIELDS,
    SEED_BITS,
    SEED_MASK,
    STATUSES,
    TASKS,
    WIND_MAX,
    _from_limbs,
    _limbs,
    host_actions,
    integration_layout,
    physics_rows,
    state_fields,
)
from .world import STATUS, Episode, Index, check_task, episode_base, physics_model

#: MjData names of the integration-state components whose State name differs.
DATA_NAMES = {"warmstart": "qacc_warmstart"}
_RNG_KEYS = (
    ("weather", "rng_state"),
    ("weather", "rng_increment"),
    ("weather", "rng_has_uint32"),
    ("weather", "rng_uinteger"),
)


def rng_columns(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, int]:
    """A PCG64 generator's state as the four State columns."""
    st = rng.bit_generator.state
    if st["bit_generator"] != "PCG64":
        raise RuntimeError(f"Expected PCG64, got {st['bit_generator']!r}")
    return _limbs(st["state"]["state"]), _limbs(st["state"]["inc"]), int(st["has_uint32"]), int(st["uinteger"])


class RoadsterBatch(SimulatorAdapter):
    """``num_envs`` environments of one Roadster task stepped together; see the backends.

    Wind and rain are installed with ``configure_physics`` and apply at the next ``reset``, as the source sets
    the weather at reset. Environment ``i`` resets with ``seed + i``, as the source env does with that seed.
    """

    backend = ""

    def __init__(self, task: str, num_envs: int, device="cpu"):
        check_task(task)
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.task = task
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Roadster stores State on CPU or CUDA")
        self._closed = True
        self.model = physics_model()
        self.index = Index(self.model)
        self.layout = integration_layout(self.model)
        unknown = set(self.layout) - {"time", "qpos", "qvel", "warmstart", "ctrl", "qfrc_applied", "xfrc_applied"}
        unknown -= {"mocap_pos", "mocap_quat"}
        if unknown:
            raise RuntimeError(f"Unexpected MuJoCo state components {sorted(unknown)}")
        self.fields = state_fields(self.model)
        self.physics_layout = PHYSICS_LAYOUT
        self._base = episode_base(task)
        self._physics = np.zeros((num_envs, 2))  # installed for the next reset: wind speed, rain
        self._episode = np.zeros((num_envs, 2))  # the values the current episode was reset with
        self._rngs: list[np.random.Generator] = []
        self._build()
        self._closed = False
        self.reset(0)  # every environment in a valid episode, so a step before the first reset is defined

    # ------------------------------------------------------------------ backend hooks
    def _build(self) -> None:
        """Create the physics batch, the compute tensors (``_alloc``) and their views."""
        raise NotImplementedError

    def _sim(self, name: str) -> torch.Tensor:
        """The live ``(num_envs, ...)`` physics tensor of the integration-state component ``name``."""
        raise NotImplementedError

    def _write_physics(self, values: dict[str, torch.Tensor], qpos_pre: torch.Tensor, has_pre: torch.Tensor):
        """Write the integration state of every environment, then the derived quantities at ``qpos_pre``."""
        raise NotImplementedError

    def _after_reset(self) -> None:
        """Set what the source computes from the reset pose: the autopilot heading and the first cost."""
        raise NotImplementedError

    def _apply_grip(self) -> None:
        """Write each environment's tyre grip into its model."""
        raise NotImplementedError

    # ------------------------------------------------------------------ storage
    def _alloc(self, device: torch.device) -> None:
        n = self.num_envs
        self._compute = device
        self._s = {
            key: torch.zeros((n, width), dtype=dtype, device=device)
            for key, (width, dtype) in self.fields.items()
            if not (key[0] == "sim" and key[1] in self.layout) and key[0] != "physics" and key != ("fx", "history")
        }
        self._live: TensorDict | None = None
        self._ring = torch.zeros((n, FX_HISTORY, FX_ENTRY), dtype=torch.float64, device=device)
        self._head = torch.zeros(n, dtype=torch.int64, device=device)
        self._leaves = torch.zeros((n, 2), dtype=torch.float64, device=device)

    def _col(self, group: str, name: str) -> torch.Tensor:
        """The ``(num_envs,)`` view of a one-wide State column, or ``(num_envs, width)`` otherwise."""
        t = self._s[(group, name)]
        return t[:, 0] if t.shape[1] == 1 else t

    # ------------------------------------------------------------------ lifecycle
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def configure_physics(self, values) -> None:
        """Install per-environment ``(wind speed, rain)`` float64 rows for the next reset."""
        self._check_open()
        self._physics = physics_rows(values, self.num_envs)

    def reset(self, seed: int | None = None) -> TensorDict:
        """Reset every environment with ``seed + i`` (the source seeds the task draw and the gusts with it)."""
        self._check_open()
        n = self.num_envs
        if seed is not None and not 0 <= int(seed) <= MAX_SEED - n + 1:
            raise ValueError(f"Roadster seeds lie in [0, {MAX_SEED}]; environment i resets with seed + i")
        eps = [
            Episode(self.task, None if seed is None else int(seed) + i, float(w), float(r), self._base)
            for i, (w, r) in enumerate(self._physics)
        ]
        host = {
            ("task", "status"): [STATUS[e.status] for e in eps],
            ("task", "airborne_once"): [e.airborne_once for e in eps],
            ("task", "target_idx"): [e.target_idx for e in eps],
            ("task", "region_y"): [e.region_y for e in eps],
            ("task", "allow_ground_landing"): [e.allow_ground_landing for e in eps],
            ("task", "fuel"): [FUEL_CAPACITY] * n,
            ("task", "ground_hold"): [3] * n,
            ("task", "id"): [TASKS.index(self.task)] * n,
            ("goal", "position"): [e.goal[0] for e in eps],
            ("goal", "yaw"): [e.goal[1] for e in eps],
            ("goal", "velocity"): [e.goal[2] for e in eps],
            ("route", "count"): [e.route_count for e in eps],
            ("route", "positions"): [e.route.reshape(-1) for e in eps],
            ("route", "velocities"): [e.route_vel.reshape(-1) for e in eps],
            ("route", "up"): [e.route_up for e in eps],
            ("route", "touch"): [e.route_touch for e in eps],
            ("jam", "present"): [e.jam is not None for e in eps],
            ("jam", "extent"): [
                [0.0] * 3 if e.jam is None else [e.jam[k] for k in ("x_start", "x_end", "top")] for e in eps
            ],
            ("weather", "seed_hi"): [e.seed >> SEED_BITS for e in eps],
            ("weather", "seed_lo"): [e.seed & SEED_MASK for e in eps],
            ("weather", "rain"): [e.rain for e in eps],
            ("weather", "wind_mean"): [e.wind_mean for e in eps],
            ("weather", "gust_std"): [e.gust_std for e in eps],
            ("weather", "grip"): [e.grip for e in eps],
        }
        self._rngs = [e.wind_rng for e in eps]
        rngs = [rng_columns(rng) for rng in self._rngs]
        for k, key in enumerate(_RNG_KEYS):
            host[key] = [r[k] for r in rngs]
        for t in self._s.values():
            t.zero_()
        for key, value in host.items():
            dtype = _NUMPY_DTYPES[self.fields[key][1]]
            self._s[key].copy_(torch.from_numpy(np.asarray(value, dtype).reshape(n, -1)))
        self._ring.zero_()
        self._head.zero_()
        qpos = torch.from_numpy(np.stack([e.qpos for e in eps]))
        values = {
            "qpos": qpos,
            "qvel": torch.from_numpy(np.stack([e.qvel for e in eps])),
            "mocap_pos": torch.from_numpy(np.stack([e.mocap_pos for e in eps])),
            "mocap_quat": torch.from_numpy(np.stack([e.mocap_quat for e in eps])),
        }
        self._apply_grip()
        self._write_physics(values, qpos, torch.zeros(n, dtype=torch.bool))
        self._after_reset()
        self._episode = self._physics.copy()
        self._leaves.copy_(torch.from_numpy(self._episode))
        return self.get_state()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_physics()

    def _close_physics(self) -> None:
        pass

    # ------------------------------------------------------------------ State
    def _live_state(self) -> TensorDict:
        """A TensorDict over the live storage of every State field, refreshed; not a snapshot."""
        n = self.num_envs
        if self._live is None:
            self._fx_out = torch.zeros((n, FX_HISTORY * FX_ENTRY), dtype=torch.float64, device=self._compute)
            self._fx_order = torch.arange(FX_HISTORY, device=self._compute)
            self._casts = []
            physics = [state_key(leaf) for leaf in PHYSICS_LEAVES]
            leaves = {}
            for key, (_, dtype) in self.fields.items():
                if key[0] == "sim" and key[1] in self.layout:
                    view = self._sim(key[1]).reshape(n, -1)
                    if view.dtype != dtype:
                        buffer = torch.zeros(view.shape, dtype=dtype, device=self._compute)
                        self._casts.append((view, buffer))
                        view = buffer
                elif key == ("fx", "history"):
                    view = self._fx_out
                elif key in physics:
                    j = physics.index(key)
                    view = self._leaves[:, j : j + 1]
                else:
                    view = self._s[key]
                leaves[key] = view
            self._live = TensorDict(leaves, [n], device=self._compute)
        order = (self._head[:, None] + self._fx_order) % FX_HISTORY
        torch.gather(
            self._ring, 1, order[:, :, None].expand(-1, -1, FX_ENTRY), out=self._fx_out.view(n, FX_HISTORY, FX_ENTRY)
        )
        for view, buffer in self._casts:
            buffer.copy_(view)
        return self._live

    def get_state(self) -> TensorDict:
        self._check_open()
        live = self._live_state()
        return live.clone() if self._compute == self.device else live.to(self.device)

    def render_fields(self) -> dict:
        """Host arrays of the State fields a render row is packed from; no snapshot."""
        self._check_open()
        live = self._live_state()
        return {key: live[key].cpu().numpy().copy() for key in RENDER_FIELDS}

    def validate(self, state, *, batch: int | None = None) -> None:
        """Require the complete, typed resume schema with finite values and consistent task fields.

        Compact render rows and partial TensorDicts fail. Nothing is changed.
        """
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
        if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
            raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
        have, want = set(state.keys(True, True)), set(self.fields)
        if have != want:
            missing, extra = sorted(want - have, key=str), sorted(have - want, key=str)
            raise ValueError(
                "Full State keys do not match the Roadster resume schema"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        n = int(state.batch_size[0])
        for key, (width, dtype) in self.fields.items():
            value = state[key]
            if value.shape != (n, width):
                raise ValueError(f"State field {key!r} has invalid shape {tuple(value.shape)}")
            if value.dtype != dtype:
                raise ValueError(f"State field {key!r} has dtype {value.dtype}; expected {dtype}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"State field {key!r} contains non-finite values")
        host = state.cpu()

        def within(key, lo, hi):
            v = host[key]
            if bool((v < lo).any() or (v > hi).any()):
                raise ValueError(f"State field {key!r} must lie in [{lo}, {hi}]")

        task = TASKS.index(self.task)
        within(("task", "id"), task, task)
        within(("task", "status"), 0, len(STATUSES) - 1)
        within(("task", "target_idx"), 0, MAX_TARGETS)
        within(("task", "ground_hold"), 0, 3)
        within(("task", "fuel"), 0.0, np.inf)
        within(("task", "sim_steps"), 0, np.iinfo(np.int64).max)
        within(("lifecycle", "elapsed_steps"), 0, np.iinfo(np.int64).max)
        within(("route", "count"), 0, MAX_ROUTE)
        count = host["route", "count"].reshape(-1)
        idx = host["task", "route_idx"].reshape(-1)
        if bool((idx < 0).any() or (idx > torch.clamp(count - 1, min=0)).any()):
            raise ValueError("State field ('task', 'route_idx') must index the route")
        within(("weather", "rain"), 0.0, RAIN_MAX)
        within(("weather", "seed_hi"), 0, SEED_MASK)
        within(("weather", "seed_lo"), 0, SEED_MASK)
        within(("weather", "grip"), 0.0, 1.0)
        within(("weather", "gust_std"), 0.0, np.inf)
        within(("weather", "rng_state"), 0, 0xFFFF_FFFF)
        within(("weather", "rng_increment"), 0, 0xFFFF_FFFF)
        within(("weather", "rng_has_uint32"), 0, 1)
        within(("weather", "rng_uinteger"), 0, 0xFFFF_FFFF)
        within(state_key("weather.wind"), 0.0, WIND_MAX)
        within(state_key("weather.rain"), 0.0, RAIN_MAX)
        within(("fx", "count"), 0, FX_HISTORY)
        hist = host["fx", "history"].reshape(n, FX_HISTORY, FX_ENTRY)
        counts = host["fx", "count"].reshape(-1)
        steps = hist[:, :, 0]
        valid = torch.arange(FX_HISTORY)[None, :] < counts[:, None]
        ordered = (steps[:, 1:] > steps[:, :-1]) | ~valid[:, 1:]
        if bool(((steps < 0) & valid).any()) or not bool(ordered.all()):
            raise ValueError("State field ('fx', 'history') must list steps in increasing order")

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        self.validate(state, batch=self.num_envs)
        n = self.num_envs
        src = state.to(self._compute)
        for key, t in self._s.items():
            t.copy_(src[key])
        self._ring.copy_(src["fx", "history"].reshape(n, FX_HISTORY, FX_ENTRY))
        self._head.zero_()
        host = state.cpu()
        self._rngs = []
        for i in range(n):
            rng = np.random.default_rng()
            rng.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _from_limbs(host["weather", "rng_state"][i].numpy()),
                    "inc": _from_limbs(host["weather", "rng_increment"][i].numpy()),
                },
                "has_uint32": int(host["weather", "rng_has_uint32"][i].item()),
                "uinteger": int(host["weather", "rng_uinteger"][i].item()),
            }
            self._rngs.append(rng)
        values = {name: src["sim", name] for name in self.layout}
        self._apply_grip()
        self._write_physics(values, src["sim", "qpos_pre"], src["sim", "has_qpos_pre"][:, 0])
        # The restored episode's physics leaves become the installed ones; the next reset returns to the
        # Variation defaults.
        for j, leaf in enumerate(PHYSICS_LEAVES):
            self._physics[:, j] = host[state_key(leaf)].numpy().reshape(-1)
        self._episode = self._physics.copy()
        self._leaves.copy_(torch.from_numpy(self._episode))

    # ------------------------------------------------------------------ stepping
    def check_step(self, actions, active):
        """Return host ``(actions, active)``, or raise before any environment moves."""
        actions = host_actions(actions, self.num_envs)
        if active is not None:
            active = np.asarray(active.detach().cpu() if torch.is_tensor(active) else active, dtype=bool)
            if active.shape != (self.num_envs,):
                raise ValueError(f"Roadster active mask requires shape ({self.num_envs},), got {active.shape}")
        return actions, active

    def probe_actions(self) -> np.ndarray:
        return np.zeros((1, 4))

    def _gust_rows(self, ids: np.ndarray, count: int) -> tuple[np.ndarray, dict]:
        """``(num_envs, count, 3)`` standard normal gust draws of the windy environments ``ids``.

        Returns the draws and each drawing environment's generator state before them, to rewind an
        environment that stops early (``_rewind_gusts``).
        """
        noise = np.zeros((self.num_envs, count, 3))
        saved = {}
        for i in ids:
            rng = self._rngs[i]
            saved[int(i)] = rng.bit_generator.state
            noise[i] = rng.standard_normal((count, 3))
        return noise, saved

    def _rewind_gusts(self, saved: dict, substeps: np.ndarray, count: int) -> None:
        """Leave each drawing environment's generator after the draws its ``substeps`` of ``count`` used.

        Updates the generator columns of State.
        """
        if not saved:
            return
        ids = np.fromiter(saved, np.int64)
        cols = []
        for i in ids:
            rng, used = self._rngs[i], int(substeps[i])
            if used < count:
                rng.bit_generator.state = saved[int(i)]
                if used:
                    rng.standard_normal((used, 3))
            cols.append(rng_columns(rng))
        index = torch.from_numpy(ids).to(self._compute)
        for k, key in enumerate(_RNG_KEYS):
            value = np.asarray([c[k] for c in cols], np.int64).reshape(len(ids), -1)
            self._s[key][index] = torch.from_numpy(value).to(self._compute)
