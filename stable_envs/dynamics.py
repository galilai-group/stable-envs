"""The pure batched dynamics protocol: ``transition(state, action)`` and ``unroll(state, actions)``.

Two functions, and the whole content of the module is what "pure" means here:

  * ``transition`` reads the state it is GIVEN and writes nothing. It does not touch ``env.state``, no
    hidden buffer, no cached velocity. That is why PushT's agent velocity had to become part of the
    public state (contract §3): as long as one dynamical variable lived in an attribute, a function
    that looked pure was not, and two calls with the same ``state`` could disagree.
  * ``unroll`` applies known, OPEN-LOOP actions. It is not a policy rollout and never will be —
    policies, planners, MPC and dataset collection belong to ``stable_worldmodel`` (plan §2). The
    distinction is load-bearing: ``unroll`` can be a pure function of ``(state, actions)`` precisely
    because nothing in the loop consults an agent.

MOST ENVIRONMENTS IN THIS REPOSITORY HAVE NO DYNAMICS, and that is a design statement rather than a
gap: rendering and simulation are separable, and an env earns a simulator only where its dynamics can
be ported AND checked against the reference. Those cells raise from ``step``/``transition`` with the
exact reason and keep ``reset``/``render`` working, which is what ``no_dynamics`` below writes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict


@runtime_checkable
class Dynamics(Protocol):
    """What a cell with a batched simulator implements. Structural: no base class to inherit."""

    def transition(self, state: TensorDict, action: Any) -> TensorDict:
        """``(state, action) -> next_state``, pure in ``state``. Never touches the env's buffers."""
        ...

    def unroll(self, state: TensorDict, actions: Any) -> TensorDict:
        """``(state, (T, num_envs, ...) actions) -> (T, num_envs) states``, open loop."""
        ...


def has_dynamics(env) -> bool:
    """Does this env ship a batched simulator?

    Asked of the CLASS rather than by ``hasattr(env, "step")``, because every ``Env`` now defines
    ``step`` — it has to, in order to raise the precise refusal below instead of an ``AttributeError``
    from inside somebody's loop. Duck-typed stand-ins (the test suite has several) keep the old answer.
    """
    base = getattr(env, "_se_dynamics_base", None)
    if base is None:
        return hasattr(env, "step")
    cls = type(env)
    return any(getattr(cls, m, None) is not getattr(base, m, None) for m in ("step", "transition"))


def no_dynamics(env, what: str):
    """Raise the one refusal a renderer-only cell owes its caller: what is missing and what to do."""
    name = type(env).__name__
    raise NotImplementedError(
        f"{name}.{what}: {name} ships a renderer and no batched simulator, so it has no dynamics — "
        f"PushT has dynamics; {name} does not. This is deliberate: rendering and simulation are "
        f"separable in this package and a cell earns a simulator only where its dynamics can be "
        f"ported AND checked against the reference. reset() and render() work as usual; to produce "
        f"trajectories, roll out the official CPU environment and render the states it gives you "
        f"(env.render(states) takes any batch)."
    )


def unroll_from_transition(env, state: TensorDict, actions) -> TensorDict:
    """The generic ``unroll``: apply ``actions[t]`` with ``transition``, stack the results.

    ``actions`` is ``(T, num_envs, ...)`` — TIME FIRST, matching the loop that consumes it — and the
    trajectory comes back stacked on a leading time axis of length ``T`` holding the state AFTER each
    action. The initial state is not included: the caller already holds it, and returning ``T + 1``
    states makes every index in a downstream dataset off by one against the actions beside it.

    Written once, here, so an env that implements ``transition`` gets ``unroll`` for free and the two
    cannot drift apart. An env with a faster fused path may override ``unroll``; a test asserts the two
    agree to the bit.
    """
    acts = actions if torch.is_tensor(actions) else torch.as_tensor(actions)
    n = int(getattr(env, "num_envs", 0) or 0)
    if acts.ndim < 2 or (n and int(acts.shape[1]) != n):
        raise ValueError(
            f"{type(env).__name__}.unroll: actions are (T, num_envs, ...) — time first — and this "
            f"input has shape {tuple(acts.shape)} for {n or 'num_envs'} environment(s). One action "
            f"for every environment at every one of the T steps; there is no policy here to fill "
            f"them in. A single step's actions are what `transition(state, action)` takes."
        )
    out = []
    s = state
    for t in range(int(acts.shape[0])):
        s = env.transition(s, acts[t])
        out.append(s)
    return torch.stack(out, dim=0)


__all__ = ["Dynamics", "has_dynamics", "no_dynamics", "unroll_from_transition"]
