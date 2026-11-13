"""Interfaces between an environment and its batch of simulators."""

from abc import abstractmethod
from typing import Protocol, cast, runtime_checkable

import numpy as np
from tensordict import TensorDict
from torch import Tensor


class SimulatorAdapter(Protocol):
    """Internal interface for an owned batch of simulators.

    Concrete adapters inherit this interface. Env handles action sequences and
    rendering; adapters advance one transition per lane. Family-specific options
    and asynchronous execution are additional capabilities. Restore has two
    phases so that a process batch restores every worker or none.
    """

    num_envs: int
    _staged_restore: TensorDict | None = None
    #: Family methods that a process adapter runs on every worker: name -> "per_lane" or "broadcast".
    process_methods: dict[str, str] = {}
    #: Attributes that a process adapter copies from a one-lane prototype.
    process_attributes: tuple[str, ...] = ()

    @abstractmethod
    def reset(self, seed: int | None = None) -> TensorDict:
        """Reset the batch and return an owned full-state snapshot."""
        ...

    @abstractmethod
    def get_state(self) -> TensorDict:
        """Return an independent snapshot, including hidden simulator and RNG state."""
        ...

    @abstractmethod
    def set_state(self, state: TensorDict) -> None:
        """Restore a complete snapshot for all lanes."""
        ...

    def validate(self, state: TensorDict) -> object:
        """Raise if ``state`` cannot restore every lane; change no simulator."""
        raise NotImplementedError(f"{type(self).__name__} has no State validation")

    def prepare_restore(self, state: TensorDict) -> None:
        """Validate ``state`` and hold it for ``commit_restore``; change no simulator."""
        self.validate(state)
        self._staged_restore = state

    def commit_restore(self) -> None:
        """Restore the state that ``prepare_restore`` accepted."""
        state = self._staged_restore
        if state is None:
            raise RuntimeError("No restore is prepared")
        self._staged_restore = None
        self.set_state(state)

    def discard_restore(self) -> None:
        """Drop a prepared restore."""
        self._staged_restore = None

    def check_step(self, actions, active):
        """Return host ``(actions, active)`` for ``step``, or raise without changing a lane.

        A process adapter calls this method with itself as ``self`` before it sends a step to
        any worker, so it may read only ``num_envs`` and ``process_attributes``.
        """
        return actions, active

    def probe_actions(self) -> np.ndarray:
        """Return valid actions for one lane; a process adapter steps them once to size its buffers."""
        return np.zeros((1, getattr(self, "action_dim")))

    @abstractmethod
    def step(self, actions: Tensor | np.ndarray) -> TensorDict:
        """Advance one action per lane and return state and the defined task outputs."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release owned simulator resources and workers."""
        ...


@runtime_checkable
class StepPipeline(Protocol):
    """One in-flight transition; receive owned outputs before reusing worker storage."""

    def submit_step(
        self, actions: Tensor | np.ndarray, *, active: Tensor | np.ndarray | None = None, capture_state: bool = True
    ) -> None: ...

    def receive_step(self) -> TensorDict: ...

    def abort_step(self) -> None:
        """Release workers after an interrupted request."""
        ...


_PIPELINE_TYPES: dict[type, bool] = {}
_PIPELINE_METHODS = ("submit_step", "receive_step", "abort_step")


def step_pipeline(adapter: SimulatorAdapter) -> StepPipeline | None:
    """Return ``adapter`` when it supports submit/receive stepping, else None."""
    cls = type(adapter)
    pipelined = _PIPELINE_TYPES.get(cls)
    if pipelined is None:
        pipelined = isinstance(adapter, StepPipeline)
        # The answer holds for every instance when the class alone decides it.
        if all(hasattr(cls, name) for name in _PIPELINE_METHODS) or not hasattr(cls, "__getattr__"):
            _PIPELINE_TYPES[cls] = pipelined
    return cast(StepPipeline, adapter) if pipelined else None
