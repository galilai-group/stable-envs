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
    spec = env.action_spec
    shape = tuple(spec.shape)
    b = env.num_envs
    # Accept discrete [B] actions without treating an ambiguous [B,T] array as a sequence.
    if shape == (1,) and a.shape == (b,) and not spec.dtype.is_floating_point:
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


def _task_outputs(result, active, *, all_active):
    """Own the task outputs and zero them where a lane had already finished."""
    out = result.exclude("state").clone()
    if not all_active:
        inactive = (~active).nonzero(as_tuple=True)[0]
        for value in out.values(True, True):
            value.index_fill_(0, inactive, 0)
    return out


def _stack_states(values):
    """Stack owned State snapshots along a new time axis, leaf by leaf."""
    first = values[0]
    return first.apply(
        lambda *leaves: torch.stack(leaves, dim=1), *values[1:], batch_size=[*first.batch_size, len(values)]
    )


def _selection(value, length, name):
    if value is None:
        return None
    if not isinstance(value, slice):
        raise TypeError(f"{name} must be a slice or None")
    # Python handles clipping, negative indices/strides, and rejects zero strides.
    return tuple(range(length)[value])


def _columns(valid, indices):
    """Copy the selected time columns; forward strides avoid uploading an index tensor."""
    stride = indices[1] - indices[0] if len(indices) > 1 else 1
    if not indices or stride < 0:
        return valid[:, list(indices)]
    return valid[:, indices[0] : indices[-1] + 1 : stride].clone()


@torch.no_grad()
def step(env, actions, *, states=slice(None), frames=slice(-1, None)):
    env._check_open("step")
    if not env.has_dynamics:
        from .dynamics import no_dynamics

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
    # Draws the live lanes without rebuilding them from the State snapshot.
    render_live = getattr(env, "_render_live", None)
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
            # The all-true mask is never handed to the family, so it needs no copy.
            validity.append(active if all_active else active.clone())
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
            history.append(_task_outputs(result, active, all_active=all_active))
            if t in state_times:
                saved_states[t] = live_state
            if t in frame_times:
                if live_frame is None:
                    live_frame = (
                        render_live() if render_live is not None and pipeline is None else env.render(live_state)
                    )
                saved_frames[t] = live_frame
            active = next_active
            all_active = next_all_active
    except BaseException:
        if pending:
            pipeline.abort_step()
        raise

    valid = torch.stack(validity, dim=1)
    out = {"state": live_state}
    out.update((torch.stack(history, dim=1) if sequence else history[0]).items())
    out["valid"] = valid
    if state_indices is not None:
        out["states"] = (
            getattr(env, "_stack_states", _stack_states)([saved_states[t] for t in state_indices])
            if state_indices
            else live_state.unsqueeze(1)[:, :0].clone()
        )
        out["state_valid"] = _columns(valid, state_indices)
    if frame_indices is not None:
        out["frames"] = (
            torch.stack([saved_frames[t] for t in frame_indices], dim=1)
            if frame_indices
            else torch.empty((env.num_envs, 0, *env.frame_spec.shape), dtype=env.frame_spec.dtype, device=env.device)
        )
        out["frame_valid"] = _columns(valid, frame_indices)
    return TensorDict(out, [env.num_envs], device=env.device)
