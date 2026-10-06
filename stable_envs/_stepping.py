"""Dispatch public action batches and sequences with independent capture slices.

Finished lanes hold their endpoints without implicit reset. The process path
keeps at most one transition in flight and renders owned snapshots in the parent.
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
    # Accept discrete [B] actions without treating an ambiguous [B,T] array as a sequence.
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


def _masked_outputs(result, active, *, all_active):
    """Zero invalid task outputs while retaining held endpoint States."""
    out = result.exclude("state").clone()
    if "state" in result.keys():
        out["state"] = result["state"]
    if not all_active:
        inactive = (~active).nonzero(as_tuple=True)[0]
        for key, value in out.items(True, True):
            path = (key,) if isinstance(key, str) else key
            if path[0] != "state":
                value.index_fill_(0, inactive, 0)
    return out


def _selection(value, length, name):
    if value is None:
        return None
    if not isinstance(value, slice):
        raise TypeError(f"{name} must be a slice or None")
    # Python handles clipping, negative indices/strides, and rejects zero strides.
    return tuple(range(length)[value])


@torch.no_grad()
def step(env, actions, *, states=slice(None), frames=slice(-1, None)):
    env._check_open("step")
    if not env.has_dynamics:
        from ._dynamics import no_dynamics

        no_dynamics(env, "step")
    acts, sequence = _actions(env, actions)
    if not sequence:
        acts = acts.unsqueeze(1) if torch.is_tensor(acts) else np.expand_dims(acts, 1)
    length = acts.shape[1]
    state_indices = _selection(states, length, "states")
    frame_indices = _selection(frames, length, "frames")
    if frame_indices is not None and not getattr(env, "produces_rgb", True):
        raise ValueError(f"{type(env).__name__} has no RGB output; use frames=None")
    state_times, frame_times = set(state_indices or ()), set(frame_indices or ())
    if torch.is_tensor(acts) and acts.device.type != "cpu" and getattr(env, "execution_mode", None) == "DELEGATED_CPU":
        acts = acts.detach().cpu().numpy()
    history, saved_states, saved_frames, validity = [], {}, {}, []
    advance = getattr(env, "_advance_without_snapshot", env._step)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    all_active = True
    result, live_state, live_frame = None, None, None
    pipeline = getattr(env, "_step_pipeline", lambda: None)()
    pending = False

    def submit(t, mask, all_active):
        pipeline.submit_step(
            acts[:, t],
            # The initial mask is already known to be all true; avoid a GPU readback.
            active=None if all_active else mask,
            capture_state=t in state_times or t in frame_times or t == length - 1,
        )

    try:
        if pipeline is not None:
            submit(0, active, all_active)
            pending = True
        for t in range(length):
            validity.append(active.clone())
            if pipeline is not None:
                if pending:
                    result = pipeline.receive_step()
                    pending = False
                    live_state, live_frame = result.get("state", None), None
            elif all_active:
                result = advance(acts[:, t])
                live_state, live_frame = result.get("state", None), None
            elif bool(active.any()):
                # Skip finished lanes through the family hook, without stepping and restoring them.
                result = advance(acts[:, t], active=active)
                live_state, live_frame = result.get("state", None), None
            needs_state = t in state_times or t in frame_times or t == length - 1
            if needs_state and live_state is None:
                live_state = env.get_state()
            next_active = active & ~_done(result) if t + 1 < length else active
            # Reuse the routing decision for padding to avoid another readback or empty indexing of large images.
            next_all_active = bool(next_active.all()) if t + 1 < length else all_active
            if pipeline is not None and t + 1 < length and (next_all_active or bool(next_active.any())):
                # Received outputs own their storage, so workers can reuse buffers while the parent renders.
                submit(t + 1, next_active, next_all_active)
                pending = True
            history.append(_masked_outputs(result, active, all_active=all_active).exclude("state"))
            if t in state_times:
                saved_states[t] = live_state
            if t in frame_times:
                if live_frame is None:
                    live_frame = env.render(live_state)
                saved_frames[t] = live_frame
            active = next_active
            all_active = next_all_active
    except BaseException:
        if pending:
            pipeline.abort_step()
        raise

    out = TensorDict({"state": live_state}, [env.num_envs], device=env.device)
    out.update(torch.stack(history, dim=1) if sequence else history[0])
    out["valid"] = torch.stack(validity, dim=1)
    if state_indices is not None:
        out["states"] = (
            getattr(env, "_stack_states", lambda values: torch.stack(values, dim=1))(
                [saved_states[t] for t in state_indices]
            )
            if state_indices
            else live_state.unsqueeze(1)[:, :0].clone()
        )
        out["state_valid"] = out["valid"][:, list(state_indices)]
    if frame_indices is not None:
        out["frames"] = (
            torch.stack([saved_frames[t] for t in frame_indices], dim=1)
            if frame_indices
            else torch.empty((env.num_envs, 0, *env.frame_spec.shape), dtype=env.frame_spec.dtype, device=env.device)
        )
        out["frame_valid"] = out["valid"][:, list(frame_indices)]
    return out
