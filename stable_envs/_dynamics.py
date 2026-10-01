"""Private simulator capability checks."""


def has_dynamics(env):
    base = getattr(env, "_se_dynamics_base", None)
    if base is None:
        return hasattr(env, "step")
    return type(env)._step is not base._step


def no_dynamics(env, what):
    raise NotImplementedError(
        f"{type(env).__name__}.{what}: full simulator semantics are not implemented for this cell. "
        "See docs/research/semantic_recovery_matrix.json. Compact renderer rows cannot substitute for State."
    )
