from typing import Protocol

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

    def step(self, actions: Tensor) -> TensorDict:
        """Advance one action per lane and return State and the defined task outputs."""
        ...

    def close(self) -> None:
        """Release owned simulator resources and workers."""
        ...
