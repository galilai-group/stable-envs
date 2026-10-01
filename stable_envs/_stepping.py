"""Shared public action-shape/capture contract, independent of execution strategy.

The serial sequence executor is the correctness reference for optimized executors.
Neither snapshot restoration nor implicit reset is used to advance a sequence.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict


def _actions(env, actions):
    # Keep action dtype and device: upstream scaling arithmetic can depend on dtype.
    a = actions if torch.is_tensor(actions) else np.asarray(actions)
    shape = tuple(env.action_spec.shape)
    b = env.num_envs
    # Preserve the existing discrete scalar spelling without making [B,T] ambiguous.
    if shape == (1,) and a.shape == (b,) and not env.action_spec.dtype.is_floating_point:
        a = a.reshape(b, 1)
    if tuple(a.shape) == (b, *shape):
        return a, False
    if a.ndim == len(shape) + 2 and a.shape[0] == b and tuple(a.shape[2:]) == shape and a.shape[1] > 0:
        return a, True
    raise ValueError(
        f"{type(env).__name__}.step: actions must be [B, {', '.join(map(str, shape))}] "
        f"or [B, T, {', '.join(map(str, shape))}] with B={b} and T>0; got {tuple(a.shape)}"
    )


def _done(result):
    done = torch.zeros(result.batch_size[0], dtype=torch.bool, device=result.device)
    for key in ("terminated", "truncated", "done"):
        if key in result.keys():
            done |= result[key].bool().reshape(len(done), -1).any(-1)
    return done


def _masked_outputs(result, active):
    """Invalid task-output slots are zero; held endpoint States are not padding."""
    out = result.exclude("state").clone()
    if "state" in result.keys():
        out["state"] = result["state"]
    for key, value in out.items(True, True):
        path = (key,) if isinstance(key, str) else key
        if path[0] != "state":
            value[~active] = 0
    return out


@torch.no_grad()
def step(env, actions, *, capture=None):
    env._check_open("step")
    if not env.has_dynamics:
        from ._dynamics import no_dynamics

        no_dynamics(env, "step")
    if capture not in (None, "states", "final_frame", "all_frames"):
        raise ValueError('capture must be None, "states", "final_frame", or "all_frames"')
    acts, sequence = _actions(env, actions)
    pixels = capture != "states"
    if pixels and not getattr(env, "produces_rgb", True):
        raise ValueError(f'{type(env).__name__} has no RGB output; use capture="states"')
    if not sequence:
        out = env._step(acts)
        if pixels:
            frame = env.render(out["state"])
            out["frame"] = frame
            if capture == "all_frames":
                out["frames"] = frame.unsqueeze(1)
        return out

    if torch.is_tensor(acts) and acts.device.type != "cpu" and getattr(env, "execution_mode", None) == "DELEGATED_CPU":
        # A supplied chunk needs one action transfer, not one synchronized transfer per timestep.
        acts = acts.detach().cpu().numpy()
    history, states, frames, validity = [], [], [], []
    advance = getattr(env, "_advance_without_snapshot", env._step) if capture == "final_frame" else env._step
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    result = None
    for t in range(acts.shape[1]):
        validity.append(active.clone())
        if bool(active.all()):
            result = advance(acts[:, t])
        elif bool(active.any()):
            # Family hook must actually skip these lanes, not step then restore them.
            result = advance(acts[:, t], active=active)
        out = _masked_outputs(result, active)
        history.append(out.exclude("state"))
        if capture in (None, "states"):
            states.append(out["state"])
        if capture == "all_frames":
            frames.append(env.render(out["state"]))
        active = active & ~_done(result)

    # A [B] result can contain [B,T] histories alongside its [B] final State.
    final_state = result["state"] if "state" in result.keys() else env.get_state()
    result = TensorDict({"state": final_state}, [env.num_envs], device=env.device)
    result.update(torch.stack(history, dim=1))
    result["valid"] = torch.stack(validity, dim=1)
    if states:
        result["states"] = torch.stack(states, dim=1)
    if frames:
        result["frames"] = torch.stack(frames, dim=1)
        result["frame"] = result["frames"][:, -1]
    elif pixels:
        result["frame"] = env.render(result["state"])
    return result
