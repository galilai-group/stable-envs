"""Online checks consume source-witnessed traces; family adapters build them.

Source pixels must come from forward-running upstream lanes, never a candidate
render or a source restored using candidate serialization.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict

from stable_envs.testing.evidence import EvidenceRun
from stable_envs.testing.metrics import normalized_mse
from tests.helpers.reference.semantics import compare_state


class PixelEnv(Protocol):
    device: torch.device

    def set_state(self, state: TensorDict) -> object: ...
    def get_state(self) -> TensorDict: ...
    def step(self, actions: NDArray, *, states: slice | None, frames: slice | None) -> TensorDict: ...


@dataclass
class PixelTrace:
    initial: TensorDict
    states: list[TensorDict]
    tasks: dict[str | tuple[str, ...], torch.Tensor]  # [B,T,...], including the one-action case.
    frames: NDArray[np.uint8]  # [B,T,H,W,3], including held terminal lanes.
    valid: torch.Tensor
    actions: NDArray  # [B,T,A]; the family chooses the action dtype.
    nan_tasks: frozenset[str | tuple[str, ...]] = frozenset()


def check_online_pixels(
    env: PixelEnv,
    trace: PixelTrace,
    *,
    states: slice | None,
    frames: slice | None,
    mse_max: float,
    failure_dir: str | Path,
    compare_states: Callable[[TensorDict, TensorDict], object] = compare_state,
) -> NDArray[np.float64]:
    """Check one capture selection against an independently witnessed rollout."""
    batch, horizon = trace.actions.shape[:2]
    assert horizon > 0 and len(trace.states) == horizon
    assert trace.valid.shape == (batch, horizon) and trace.valid.dtype == torch.bool
    assert trace.frames.shape[:2] == (batch, horizon) and trace.frames.dtype == np.uint8
    assert trace.nan_tasks <= trace.tasks.keys(), "NaN policy names an absent task output"
    if not np.isfinite(mse_max) or mse_max < 0:
        raise ValueError("A finite nonnegative source-pixel threshold is required")
    result = None
    try:
        env.set_state(trace.initial.clone())
        compare_states(env.get_state(), trace.initial)
        actions = trace.actions[:, 0] if horizon == 1 else trace.actions
        result = env.step(actions.copy(), states=states, frames=frames)
        assert all(value.device.type == env.device.type for value in result.values(True, True))
        compare_states(result["state"], trace.states[-1])
        compare_states(env.get_state(), trace.states[-1])
        torch.testing.assert_close(result["valid"].cpu(), trace.valid, atol=0, rtol=0)
        for key, expected in trace.tasks.items():
            torch.testing.assert_close(
                result[key].cpu(),
                expected[:, 0] if horizon == 1 else expected,
                atol=0,
                rtol=0,
                equal_nan=key in trace.nan_tasks,
            )
        for name, selection in (("states", states), ("frames", frames)):
            mask = "state_valid" if name == "states" else "frame_valid"
            if selection is None:
                assert name not in result and mask not in result
                continue
            indices = list(range(horizon))[selection]
            torch.testing.assert_close(result[mask].cpu(), trace.valid[:, indices], atol=0, rtol=0)
            assert tuple(result[name].shape[:2]) == (batch, len(indices))
            if name == "states":
                for j, t in enumerate(indices):
                    compare_states(result[name][:, j], trace.states[t])
        indices = [] if frames is None else list(range(horizon))[frames]
        errors = np.empty((batch, 0), dtype=np.float64)
        if frames is not None:
            errors = normalized_mse(result["frames"].cpu().numpy(), trace.frames[:, indices])
            assert np.all(errors <= mse_max), f"Source pixel MSE {errors.tolist()} exceeds {mse_max}"
        held = result.clone()
        env.step(trace.actions[:, 0].copy(), states=None, frames=None)
        compare_state(result, held)
        return errors
    except Exception as error:
        run = EvidenceRun(failure_dir)
        arrays = dict(actions=trace.actions, source_frames=trace.frames, valid=trace.valid.numpy())
        for label, state in [("initial", trace.initial), *[(f"state/{i}", s) for i, s in enumerate(trace.states)]]:
            arrays.update(
                {
                    label + "/" + (k if isinstance(k, str) else "/".join(k)): v.cpu().numpy()
                    for k, v in state.items(True, True)
                }
            )
        if result is not None and "frames" in result:
            arrays["ours"] = result["frames"].cpu().numpy()
        run.arrays("failure.npz", **arrays)
        run.json(
            "failure.json",
            dict(
                error=str(error),
                states=repr(states),
                frames=repr(frames),
                mse_max=mse_max,
                nan_tasks=sorted(map(repr, trace.nan_tasks)),
            ),
        )
        run.seal()
        raise
