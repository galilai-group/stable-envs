from __future__ import annotations

from ._delegated import DelegatedSemantics
from .simulation.continuation import _make_env


class DMCContinuationSemantics(DelegatedSemantics):
    """Cheetah, Hopper and Walker on CPU dm_control lanes."""

    @classmethod
    def _load_env(cls, seed: int):
        # Registered Cheetah uses a different material from the suite default.
        return _make_env(cls.DOMAIN, seed)
