"""Dispatch public action batches and sequences with independent capture slices.

Finished lanes hold their endpoints without implicit reset. The process path
keeps at most one transition in flight and renders owned snapshots in the parent.
"""

from __future__ import annotations

import contextlib
import sys
import threading

import numpy as np
import torch
from tensordict import TensorDict

from .tensordicts import build_tensordict, exclude_leaves, leaf_tree, new_node, pack_to_device, stack_leaves, to_device

_NUMPY_DTYPES = {torch.uint8: np.uint8, torch.float32: np.float32, torch.float64: np.float64, torch.bool: np.bool_}


def _actions(env, actions):
    # Keep action dtype and device: upstream scaling arithmetic can depend on dtype.
    a = actions if torch.is_tensor(actions) else np.asarray(actions)
    # The action spec is fixed once the env exists; build it once.
    layout = env.__dict__.get("_action_layout")
    if layout is None:
        spec = env.action_spec
        layout = env.__dict__["_action_layout"] = (tuple(spec.shape), spec.dtype.is_floating_point)
    shape, floating = layout
    b = env.num_envs
    # Accept discrete [B] actions without treating an ambiguous [B,T] array as a sequence.
    if shape == (1,) and a.shape == (b,) and not floating:
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


def _task_outputs(result, active, *, all_active, owned=False, shared=()):
    """Own the task outputs and zero them where a lane had already finished.

    ``shared`` leaves are tensors the family never writes; they pass through and are padded when stacked.
    """
    # Families may guarantee fresh output storage. Padding still needs a copy:
    # once every lane finishes, result can be reused for later sequence entries.
    copy = not owned or not all_active
    if shared:
        out = result.exclude("state", "frame", *shared)
        out = out.clone() if copy else out
    else:
        out = exclude_leaves(result, ("state", "frame"), clone=copy)
    if not all_active:
        # A mask fill reads nothing back from an accelerator.
        inactive = ~active
        for value in out.values(True, True):
            value.masked_fill_(inactive.view(-1, *([1] * (value.ndim - 1))), 0)
    for key in shared:
        out[key] = result[key]
    return out


def _same_view(a, b):
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()


def _refs(spares, key) -> int:
    return sys.getrefcount(spares[key][0])


def _padded_buffer(spares, key, like, shape):
    """Return a tensor for a padded copy of a shared output.

    On CPU the copy goes into memory that NumPy owns and that is kept for the next copy. A tensor
    made by ``torch.from_numpy`` holds one reference to the array until its last view is freed, so
    the reference count tells when no returned tensor uses the memory. Reused memory needs no new
    pages.
    """
    if like.device.type != "cpu" or like.dtype not in _NUMPY_DTYPES:
        return torch.empty(shape, dtype=like.dtype, device=like.device)
    if key in spares and spares[key][0].shape == tuple(shape) and _refs(spares, key) == spares[key][1]:
        return torch.from_numpy(spares[key][0])
    # Count the references of the idle array the same way the check above does.
    spares[key] = [np.empty(shape, dtype=_NUMPY_DTYPES[like.dtype]), 0]
    spares[key][1] = _refs(spares, key)
    return torch.from_numpy(spares[key][0])


def _stack_outputs(history, valid, shared=(), spares=None):
    """Stack per-step task outputs along a new time axis.

    A shared leaf that is one tensor at every step becomes a view of it along time. It is copied
    only when a lane finished, to zero the slots after the end.
    """
    if not shared:
        return stack_leaves(history)
    out = stack_leaves([entry.exclude(*shared) for entry in history])
    ended = (~valid).nonzero(as_tuple=True)
    padded = bool(ended[0].numel())
    for key in shared:
        leaves = [entry[key] for entry in history]
        first = leaves[0]
        if all(_same_view(leaf, first) for leaf in leaves):
            value = first.unsqueeze(1).expand(first.shape[0], len(leaves), *first.shape[1:])
            if padded:
                value = _padded_buffer({} if spares is None else spares, key, first, value.shape).copy_(value)
        else:
            value = torch.stack(leaves, dim=1)
        if padded:
            value[ended] = 0
        out[key] = value
    return out


def _stack_states(values):
    """Stack owned state snapshots along a new time axis, leaf by leaf."""
    return stack_leaves(values)


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


# torch's intra-op thread count is process-wide. Steps on several Python threads share one record: the count
# before the first override, and the counts still requested. The newest request applies; the first count
# returns when the last override ends.
_THREADS_LOCK = threading.Lock()
_THREAD_REQUESTS: list[int] = []
_THREADS_BEFORE = [0]


def _threads_outside_steps() -> int:
    """The intra-op thread count that applies when no step overrides it."""
    with _THREADS_LOCK:
        return _THREADS_BEFORE[0] if _THREAD_REQUESTS else torch.get_num_threads()


@contextlib.contextmanager
def _threads(count):
    with _THREADS_LOCK:
        if not _THREAD_REQUESTS:
            _THREADS_BEFORE[0] = torch.get_num_threads()
        _THREAD_REQUESTS.append(count)
        if torch.get_num_threads() != count:
            torch.set_num_threads(count)
    try:
        yield
    finally:
        with _THREADS_LOCK:
            _THREAD_REQUESTS.remove(count)
            target = _THREAD_REQUESTS[-1] if _THREAD_REQUESTS else _THREADS_BEFORE[0]
            if torch.get_num_threads() != target:
                torch.set_num_threads(target)


def _host_pipeline(env, pipeline) -> bool:
    """Whether the step keeps the outputs of ``pipeline`` and its bookkeeping on the host."""
    return pipeline is not None and getattr(pipeline, "host_outputs", False) and env.device.type != "cpu"


def _native_cpu_step(env) -> bool:
    """Whether a step advances on a CPU thread pool of its own (``mujoco.batch``, Numba), not on torch threads."""
    return getattr(env, "native_step_threads", False)


def step(env, actions, *, states=slice(None), frames=slice(-1, None), state=True):
    if not torch.is_grad_enabled():
        return _dispatch(env, actions, states, frames, state)
    # Switching the thread's grad mode costs less per step than entering the no_grad context manager.
    torch._C._set_grad_enabled(False)
    try:
        return _dispatch(env, actions, states, frames, state)
    finally:
        torch._C._set_grad_enabled(True)


def _dispatch(env, actions, states, frames, state):
    env._check_open("step")
    if type(state) is not bool:
        raise TypeError(f"state must be a bool, got {state!r}")
    pipeline = getattr(env, "_step_pipeline", lambda: None)()
    short = pipeline is None and frames is None
    if not _host_pipeline(env, pipeline) and not _native_cpu_step(env):
        out = _host_single_step(env, actions, states, state) if short else None
        if out is None:
            out = _step(env, actions, states, frames, pipeline, render_threads=None, final_state=state)
        return out
    # Bookkeeping runs in one thread: idle intra-op threads spin on the cores of the workers or of the
    # native pool, and a parallel region waits for every one of its threads.
    threads = _threads_outside_steps()
    with _threads(1):
        out = _host_single_step(env, actions, states, state) if short else None
        if out is None:
            out = _step(env, actions, states, frames, pipeline, render_threads=threads, final_state=state)
        return out


def _single_step(env, action, sequence, state_indices, final_state, advance):
    """One action without frames or a step pipeline: one transition, no masks to route and no history to stack."""
    host_step = getattr(env, "_host_step", None)
    if host_step is not None:
        out = host_step(action, lambda leaves: _moved_leaves(env, leaves, sequence, state_indices, final_state))
        if out is not None:
            return out
    needs_state = final_state or state_indices is not None
    result = (env._step if needs_state else advance)(action)
    live_state = result.get("state", None) if needs_state else None
    if needs_state and live_state is None:
        live_state = env.get_state()
    if getattr(env, "_step_outputs_owned", False) and type(result) is TensorDict and not sequence:
        # Owned outputs pass through as they are: their nodes need no rebuild.
        outputs = {k: v for k, v in result._tensordict.items() if k != "state" and k != "frame"}
    else:
        outputs = exclude_leaves(result, ("state", "frame"), clone=not getattr(env, "_step_outputs_owned", False))
        if sequence:
            # The outputs are a copy or owned: a time axis of views costs no copy and no device kernel.
            outputs = stack_leaves([outputs], view=True)
    flags = _true_flags(env, 1 + (state_indices is not None), env.device)
    out = {"state": live_state} if final_state else {}
    out.update(outputs.items())
    out["valid"] = flags[0]
    if state_indices is not None:
        stack = getattr(env, "_stack_states", _stack_states)
        out["states"] = stack([live_state]) if state_indices else live_state.unsqueeze(1)[:, :0].clone()
        out["state_valid"] = flags[1] if state_indices else flags[1][:, :0]
    return build_tensordict(out, [env.num_envs], env.device)


#: Bytes of all-true flags that one accelerator slab holds; a step takes a fresh slice of it.
_FLAG_SLAB_BYTES = 64 * 1024


def _true_flags(env, count, device):
    """``count`` all-true ``(num_envs, 1)`` bool flags that no other result shares.

    On an accelerator each call slices unused flags from a slab filled once, instead of a fill per step.
    """
    n = env.num_envs
    if device.type == "cpu":
        return torch.ones((count, n, 1), dtype=torch.bool).unbind(0)
    slab = env.__dict__.get("_true_flag_slab")
    if slab is None or slab[1] + count > len(slab[0]) or slab[2] != device:
        rows = max(count, _FLAG_SLAB_BYTES // max(n, 1))
        slab = env.__dict__["_true_flag_slab"] = [torch.ones((rows, n, 1), dtype=torch.bool, device=device), 0, device]
    start = slab[1]
    slab[1] = start + count
    return slab[0][start : start + count].unbind(0)


#: All-true ``(num_envs, 1)`` host flags per lane count; packing copies them, so the result never shares them.
_HOST_TRUE: dict = {}
#: ``copies`` of ``pack_to_device`` that give the ``states`` of one action from the State.
_STATE_COPIES = (("state",), ("states",), ("state_valid",))


def _moved_leaves(env, leaves, sequence, state_indices, final_state):
    """The step result of host NumPy ``leaves`` on the device, from one copy, and the moved State.

    ``leaves`` maps key paths to ``(num_envs, ...)`` arrays: the State under ``"state"`` and the task outputs. The
    State is packed even when the result leaves it out, so the family can read the moved State.
    """
    n = env.num_envs
    if env.device.type == "cpu":
        return _host_result(env, leaves, sequence, state_indices, final_state)
    ones = _HOST_TRUE.get(n)
    if ones is None:
        ones = _HOST_TRUE[n] = np.ones((n, 1), np.bool_)
    out, batches, state = [], [], []
    for key, value in leaves.items():
        if key[0] == "state":
            state.append((key, value))
        elif sequence:
            out.append((key, value.reshape(n, 1, *value.shape[1:])))
            if len(key) > 1:
                batches.append((key[:1], (n, 1)))
        else:
            out.append((key, value))
    out = state + out
    out.append((("valid",), ones))
    copies = None
    if state_indices is not None:
        width = len(state_indices)
        if width == 1:
            # The State of the one action: copies of the State leaves from the same transfer, in (B, 1, ...) shape.
            copies = _STATE_COPIES
        else:
            out += [(("states", *key[1:]), value.reshape(n, 1, *value.shape[1:])[:, :width]) for key, value in state]
        out.append((("state_valid",), ones[:, :width]))
        batches.append((("states",), (n, width)))
    moved = pack_to_device(out, (n,), env.device, tuple(dict.fromkeys(batches)), copies)
    source = moved._tensordict
    live = source["state"]
    if not final_state:
        moved = new_node({k: v for k, v in source.items() if k != "state"}, moved.batch_size, moved.device)
    return moved, live


def _host_single_step(env, actions, states, final_state):
    """``_step`` of one action without frames for a family with a host step (``_host_step``), or None.

    The same result as ``_step`` and ``_single_step``, with fewer calls: a step of one environment costs tens of
    microseconds, so the call path is a large part of it.
    """
    host_step = getattr(env, "_host_step", None)
    if host_step is None or getattr(env, "_shared_outputs", ()):
        return None
    acts, sequence = _actions(env, actions)
    if sequence:
        if acts.shape[1] != 1:
            return None
        acts = acts[:, 0]
    state_indices = _selection(states, 1, "states")
    if torch.is_tensor(acts) and getattr(env, "execution_mode", None) == "DELEGATED_CPU":
        acts = acts.detach().cpu().numpy()

    def move(leaves):
        return _moved_leaves(env, leaves, sequence, state_indices, final_state)

    # A family may leave the State out of ``leaves`` when the result doesn't hold it; on an accelerator the family
    # can read the moved State, so it always comes.
    move.needs_state = final_state or state_indices is not None or env.device.type != "cpu"
    return host_step(acts, move)


def _host_builder(keys, n, sequence, state_indices):
    """``build(leaves)``: the step result of the host ``leaves`` (``{key path: array}`` with ``keys``) on the CPU.

    Every leaf is a copy that the result owns. The leaves and nodes are those of ``pack_to_device`` in
    ``_moved_leaves``: the State, the outputs (with a time axis for a ``sequence``), ``valid``, then the
    ``states`` of ``state_indices`` and ``state_valid``. The builder is generated once per structure, so a call
    runs no loop over the leaves.
    """
    state = [key for key in keys if key[0] == "state"]
    outputs = [key for key in keys if key[0] != "state"]
    width = None if state_indices is None else len(state_indices)
    names: dict = {"F": torch.from_numpy, "N": new_node, "C": torch.device("cpu"), "T": torch.ones, "B": torch.bool}
    names["R"] = lambda value: value.reshape(n, 1, *value.shape[1:])
    leaves = []
    for key in state:
        leaves.append((key, f"F(V[K{len(names)}].copy())"))
        names[f"K{len(names)}"] = key
    for key in outputs:
        copy = f"V[K{len(names)}].copy()"
        leaves.append((key, f"F(R({copy}))" if sequence else f"F({copy})"))
        names[f"K{len(names)}"] = key
    leaves.append((("valid",), f"T(({n}, 1), dtype=B)"))
    batches = {(): (n,)}
    if sequence:
        batches.update({key[:1]: (n, 1) for key in outputs if len(key) > 1})
    if width is not None:
        for key in state:
            leaves.append((("states", *key[1:]), f"F(R(V[K{len(names)}].copy())[:, :{width}])"))
            names[f"K{len(names)}"] = key
        leaves.append((("state_valid",), f"T(({n}, 1), dtype=B)[:, :{width}]"))
        batches[("states",)] = (n, width)
    tree: dict = {}
    for key, expression in leaves:
        node = tree
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = expression

    def source(node, path, batch):
        batch = batches.get(path, batch)
        size = f"S{len(names)}"
        names[size] = torch.Size(batch)
        items = ", ".join(
            f"{name!r}: {source(c, (*path, name), batch) if type(c) is dict else c}" for name, c in node.items()
        )
        return f"N({{{items}}}, {size}, C)"

    return eval("lambda V: " + source(tree, (), (n,)), names)  # noqa: S307 - the source holds only key literals


def _host_result(env, leaves, sequence, state_indices, final_state):
    """``_moved_leaves`` on the CPU: each leaf a copy that the result owns, without packing."""
    cache = env.__dict__.setdefault("_host_builders", {})
    plan_key = (tuple(leaves), sequence, state_indices)
    build = cache.get(plan_key)
    if build is None:
        if len(cache) >= 16:
            cache.clear()
        build = cache[plan_key] = _host_builder(plan_key[0], env.num_envs, sequence, state_indices)
    result = build(leaves)
    source = result._tensordict
    live = source.get("state")
    if not final_state and live is not None:
        result = new_node({k: v for k, v in source.items() if k != "state"}, result.batch_size, result.device)
    return result, live


def _sequence_hook(env, acts, sequence, state_indices, frame_indices, *, final_state, final):
    """Step through the family's ``_step_sequence`` hook, or return None when the family has none or declines.

    The hook gets ``(B, T, *action_shape)`` actions, the selected step indices of states and frames (None for
    no selection), and ``final``: whether to build the final state, which the result returns unless
    ``final_state`` is False and an empty ``states`` selection copies. It returns None, or a dict of
    ``(B, T, ...)`` task outputs with ``valid``, and ``state``, ``states`` and ``frames`` when they are asked
    for. Finished lanes hold their state and their later task outputs are zero, as in the step loop.
    """
    hook = getattr(env, "_step_sequence", None)
    if hook is None:
        return None
    got = hook(acts if sequence else acts[:, None], state_indices or None, frame_indices or None, final)
    if got is None:
        return None
    valid = got.pop("valid")
    live_state, saved_states, saved_frames = got.pop("state", None), got.pop("states", None), got.pop("frames", None)
    out = {"state": live_state} if final_state else {}
    out.update(got.items() if sequence else ((key, value[:, 0]) for key, value in got.items()))
    out["valid"] = valid
    if state_indices is not None:
        out["states"] = saved_states if state_indices else live_state.unsqueeze(1)[:, :0].clone()
        out["state_valid"] = _columns(valid, state_indices)
    if frame_indices is not None:
        if frame_indices:
            out["frames"] = saved_frames
        else:
            out["frames"] = torch.empty(
                (env.num_envs, 0, *env.frame_spec.shape), dtype=env.frame_spec.dtype, device=env.device
            )
        out["frame_valid"] = _columns(valid, frame_indices)
    return build_tensordict(out, [env.num_envs], env.device)


def _step(env, actions, states, frames, pipeline, *, render_threads, final_state=True):
    if not env.has_dynamics:
        from .dynamics import no_dynamics

        no_dynamics(env, "step")
    # One action batch steps as it is; a sequence steps its time slices.
    acts, sequence = _actions(env, actions)
    length = acts.shape[1] if sequence else 1
    state_indices = _selection(states, length, "states")
    frame_indices = _selection(frames, length, "frames")
    if frame_indices is not None and not getattr(env, "produces_rgb", True):
        raise ValueError(f"{type(env).__name__} has no RGB output; use frames=None")
    state_times, frame_times = set(state_indices or ()), set(frame_indices or ())
    # The final State: returned unless ``state=False``, and the source of an empty ``states`` selection.
    final = final_state or state_indices == ()
    if pipeline is None:
        # A family that steps the whole sequence in one call takes it here; the step loop serves the rest.
        fused = _sequence_hook(env, acts, sequence, state_indices, frame_indices, final_state=final_state, final=final)
        if fused is not None:
            return fused
    if torch.is_tensor(acts) and acts.device.type != "cpu" and getattr(env, "execution_mode", None) == "DELEGATED_CPU":
        acts = acts.detach().cpu().numpy()
    advance = getattr(env, "_advance_without_snapshot", env._step)
    outputs_owned = getattr(env, "_step_outputs_owned", False)
    shared = tuple(getattr(env, "_shared_outputs", ()))
    if length == 1 and frame_indices is None and pipeline is None and not shared:
        return _single_step(env, acts[:, 0] if sequence else acts, sequence, state_indices, final_state, advance)
    history, saved_states, saved_frames, validity = [], {}, {}, []
    # Draws the live lanes without rebuilding them from the state snapshot.
    render_live = getattr(env, "_render_live", None)
    all_active = True
    result, live_state, live_frame = None, None, None
    # A process pipeline on an accelerator keeps the outputs and this bookkeeping on the host until
    # the end of the step; then the outputs move to the device. So does a family that steps host copies of an
    # accelerator's lanes (``_steps_on_host``): its transitions stay on the host and the result moves in one copy.
    host_steps = pipeline is None and env.device.type != "cpu" and getattr(env, "_steps_on_host", bool)()
    host = host_steps or _host_pipeline(env, pipeline)
    # A family that masks finished lanes on the device takes the mask as it is: no readback between actions.
    device_masks = pipeline is None and not host_steps and getattr(env, "_masks_on_device", bool)()
    work = torch.device("cpu") if host else env.device
    if host_steps:
        acts = acts.detach().cpu() if torch.is_tensor(acts) else acts
    # Host steps with several States: each State's host arrays go straight into its time slot of one buffer per leaf,
    # with no State export per action and no stack (``_host_state_leaves``).
    slots, rows = {}, None
    if (
        host_steps
        and state_indices
        and len(state_indices) > 1
        and getattr(env, "_host_state_leaves", None) is not None
        and getattr(env, "_stack_states", None) is None
    ):
        slots = {t: k for k, t in enumerate(state_indices)}
    # One action needs one all-true flag column per output; a single allocation serves them all.
    single = length == 1 and state_indices in (None, (0,)) and frame_indices in (None, (0,))
    flags = None
    if single:
        count = 1 + (state_indices is not None) + (frame_indices is not None)
        flags = _true_flags(env, count, work)
        active = flags[0].view(-1)
    else:
        active = torch.ones(env.num_envs, dtype=torch.bool, device=work)
    # Frames drawn from the live lanes, here or in pipeline workers, need no state snapshot.
    pipeline_frames = pipeline is not None and getattr(pipeline, "live_frames", False)
    draws_live = render_live is not None and (pipeline is None or pipeline_frames)
    # Families whose frames depend only on packed render rows keep the rows and draw every frame in one call.
    draw_rows = getattr(env, "_render_frame_rows", None) if pipeline is None and render_live is None else None
    pending = False

    def submit(t, mask, all_active):
        frame = t in frame_times
        pipeline.submit_step(
            acts[:, t] if sequence else acts,
            # The initial mask is already known to be all true; avoid a GPU readback.
            active=None if all_active else mask,
            capture_state=t in state_times or (final and t == length - 1) or (frame and not draws_live),
            **({"capture_frame": True} if frame and pipeline_frames else {}),
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
                    result = pipeline.receive_step(host=True) if host else pipeline.receive_step()
                    pending = False
                    live_state, live_frame = result.get("state", None), result.get("frame", None)
            elif host_steps:
                wanted = (t in state_times and t not in slots) or (final and t == length - 1)
                wanted = wanted or (t in frame_times and not draws_live)
                result = env._host_advance(
                    acts[:, t] if sequence else acts, active=None if all_active else active, capture_state=wanted
                )
                live_state, live_frame = result.get("state", None), None
                if t in slots:
                    leaves = env._host_state_leaves()
                    if rows is None:
                        n, width = env.num_envs, len(state_indices)
                        rows = {key: np.empty((n, width, *v.shape[1:]), v.dtype) for key, v in leaves.items()}
                    for key, value in leaves.items():
                        rows[key][:, slots[t]] = value
            elif all_active:
                # A single step that needs its State exports it with the outputs, in one call.
                captured = length == 1 and (final or 0 in state_times or (0 in frame_times and not draws_live))
                result = (env._step if captured else advance)(acts[:, t] if sequence else acts)
                live_state, live_frame = result.get("state", None), None
            elif device_masks or bool(active.any()):
                # Skip finished lanes through the family hook, without stepping and restoring them.
                result = advance(acts[:, t] if sequence else acts, active=active)
                live_state, live_frame = result.get("state", None), None
            needs_state = (t in state_times and t not in slots) or (final and t == length - 1)
            needs_state = needs_state or (t in frame_times and not draws_live)
            if needs_state and live_state is None:
                live_state = env.get_state().to(work)
            next_active = active & ~_done(result) if t + 1 < length else active
            # Reuse the routing decision for padding to avoid another readback or empty indexing of large images.
            if t + 1 < length:
                next_all_active = False if device_masks else bool(next_active.all())
            else:
                next_all_active = all_active
            if pipeline is not None and t + 1 < length and (next_all_active or bool(next_active.any())):
                # Received outputs own their storage, so workers can reuse buffers while the parent renders.
                submit(t + 1, next_active, next_all_active)
                pending = True
            history.append(_task_outputs(result, active, all_active=all_active, owned=outputs_owned, shared=shared))
            if t in state_times and t not in slots:
                saved_states[t] = live_state
            if t in frame_times and draw_rows is not None:
                saved_frames[t] = env._pack_render_state(live_state)
            elif t in frame_times:
                with _threads(render_threads or _threads_outside_steps()):
                    if live_frame is None and pipeline_frames:
                        # Workers copied the leaves on receipt; the next step is already running.
                        live_frame = pipeline.take_frame() if hasattr(pipeline, "take_frame") else None
                    if live_frame is None and draws_live and pipeline is None:
                        live_frame = render_live()
                    elif live_frame is None:
                        # Workers drew no frame after the last transition when every lane finished before it.
                        live_state = env.get_state().to(work) if live_state is None else live_state
                        live_frame = env.render(live_state.to(env.device))
                saved_frames[t] = live_frame
            active = next_active
            all_active = next_all_active
    except BaseException:
        if pending:
            pipeline.abort_step()
        raise
    finally:
        if host_steps:
            env._end_host_steps()

    valid = flags[0] if single else torch.stack(validity, dim=1)
    out = {"state": live_state} if final_state else {}
    spares = env.__dict__.setdefault("_padded_output_buffers", {}) if shared else None
    out.update((_stack_outputs(history, valid, shared, spares) if sequence else history[0]).items())
    out["valid"] = valid
    if state_indices is not None:
        if rows is not None:
            tree = leaf_tree({key[1:]: value for key, value in rows.items()})
            out["states"] = build_tensordict(tree, [env.num_envs, len(state_indices)], work)
        else:
            out["states"] = (
                getattr(env, "_stack_states", _stack_states)([saved_states[t] for t in state_indices])
                if state_indices
                else live_state.unsqueeze(1)[:, :0].clone()
            )
        out["state_valid"] = flags[1] if single else _columns(valid, state_indices)
    if frame_indices is not None:
        if not frame_indices:
            out["frames"] = torch.empty(
                (env.num_envs, 0, *env.frame_spec.shape), dtype=env.frame_spec.dtype, device=env.device
            )
        elif draw_rows is not None:
            with _threads(_threads_outside_steps()):
                out["frames"] = draw_rows(torch.stack([saved_frames[t] for t in frame_indices], dim=1))
        elif len(frame_indices) == 1 and getattr(env, "_frames_owned", False):
            # Each frame owns new storage, so one frame needs no copy.
            out["frames"] = saved_frames[frame_indices[0]].to(env.device).unsqueeze(1)
        else:
            out["frames"] = torch.stack([saved_frames[t].to(env.device) for t in frame_indices], dim=1)
        out["frame_valid"] = flags[-1] if single else _columns(valid, frame_indices)
    if not host:
        return build_tensordict(out, [env.num_envs], env.device)
    frames = out.get("frames")
    moved = to_device(
        build_tensordict({k: v for k, v in out.items() if k != "frames"}, [env.num_envs], work), env.device
    )
    return build_tensordict(
        {key: frames if key == "frames" else moved[key] for key in out}, [env.num_envs], env.device
    )
