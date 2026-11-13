"""Detect whether an environment implements dynamics, and the error raised when it does not."""


def has_dynamics(env):
    """Return whether ``env`` overrides the base ``_step``."""
    base = getattr(env, "_se_dynamics_base", None)
    if base is None:
        return hasattr(env, "step")
    return type(env)._step is not base._step


def no_dynamics(env, what):
    """Raise NotImplementedError for ``what`` on a render-only environment."""
    raise NotImplementedError(
        f"{type(env).__name__}.{what}: full simulator semantics are not implemented for this environment; "
        "it renders supplied states only. Compact renderer rows cannot substitute for State."
    )
