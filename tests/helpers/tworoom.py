"""TwoRoom parity oracle using retained Torch dynamics and shared reset/configuration."""

from stable_envs.envs.navigation.envs import TwoRoom, _HiddenTargetSemantics


class TorchTwoRoom(TwoRoom):
    """Reference kernel for parity checks; never registered as a public environment."""

    execution_mode = "NATIVE_TORCH"
    __init__ = _HiddenTargetSemantics.__init__
    _step = _HiddenTargetSemantics._step
