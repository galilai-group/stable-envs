import numpy as np


def validate_action(action: np.ndarray) -> None:
    """Validate finite float32/float64 commands before touching the engine.

    PyFlyt can apply forces before rejecting booster or gimbal bounds. This check
    must precede stepping so rejected calls cannot alter replay history.
    """
    if not isinstance(action, np.ndarray) or action.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Rocket action must be a float32 or float64 numpy array")
    if action.shape != (7,):
        raise ValueError("Rocket action must have shape (7,)")
    if not np.isfinite(action).all():
        raise ValueError("Rocket action must contain only finite values")
    if np.any(action[:3] < -1) or np.any(action[:3] > 1):
        raise AssertionError("Rocket fin commands must be within [-1, 1]")
    if not 0 <= action[3] <= 1:
        raise AssertionError("Rocket ignition must be within [0, 1]")
    if not 0 <= action[4] <= 1:
        raise AssertionError("Rocket throttle must be within [0, 1]")
    if np.any(action[5:] < -1) or np.any(action[5:] > 1):
        raise AssertionError("Rocket gimbal commands must be within [-1, 1]")


def validate_actions(actions: np.ndarray) -> None:
    """Validate a ``[batch, 7]`` array as ``validate_action`` validates each row, and raise its first error."""
    if actions.dtype in (np.dtype("float32"), np.dtype("float64")) and actions.ndim == 2 and actions.shape[1] == 7:
        low = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0])
        if np.isfinite(actions).all() and ((actions >= low) & (actions <= 1)).all():
            return
    for action in actions:
        validate_action(action)
