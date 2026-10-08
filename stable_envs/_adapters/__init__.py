from typing import Protocol, runtime_checkable

import numpy as np
from tensordict import TensorDict
from torch import Tensor


class SimulatorAdapter(Protocol):
    """Internal interface for an owned batch of simulators.

    Implementations provide these methods without inheriting this protocol.
    Env handles action sequences and rendering; adapters advance one transition
    per lane. Family-specific options and asynchronous execution are additional
    capabilities, not requirements of this interface.
    """

    num_envs: int

    def reset(self, seed: int | None = None) -> TensorDict:
        """Reset the batch and return an owned full-State snapshot."""
        ...

    def get_state(self) -> TensorDict:
        """Return an independent snapshot, including hidden simulator and RNG state."""
        ...

    def set_state(self, state: TensorDict) -> None:
        """Restore a complete snapshot for all lanes."""
        ...

    def step(self, actions: Tensor | np.ndarray) -> TensorDict:
        """Advance one action per lane and return State and the defined task outputs."""
        ...

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


def step_pipeline(adapter: SimulatorAdapter) -> StepPipeline | None:
    return adapter if isinstance(adapter, StepPipeline) else None
