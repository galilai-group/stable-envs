"""Fixed-seed state and style draws shared by conformance checks and the test runner."""

import hashlib

import numpy as np
import torch
from tensordict import TensorDict

STATE_SEED = 20260809
STYLE_SEED = 20260810

#: Upper bound (exclusive) on the random steps after each seeded reset in :func:`rollout_states`.
ROLLOUT_MAX_STEPS = 20


def has_render_rows(env) -> bool:
    """Whether ``env`` renders flat float rows; an env without them renders its full State.

    An object without ``render_state_dim``, such as a test stand-in, renders rows.
    """
    return int(getattr(env, "render_state_dim", 1)) > 0


def _rollout(env, n: int, rng: np.random.Generator, variation=None) -> TensorDict:
    B = int(env.B)
    steps = int(rng.integers(1, ROLLOUT_MAX_STEPS))
    chunks = []
    for lo in range(0, int(n), B):
        if variation is not None:
            rows = np.asarray(variation)[lo : lo + B]
            if len(rows) < B:
                rows = np.concatenate([rows, np.repeat(rows[-1:], B - len(rows), 0)])
            env.set_variation(rows)
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(steps):
            env.step(env.action_spec.sample(B, rng), frames=None)
        chunks.append(env.get_state())
    return torch.cat(chunks)[: int(n)]


def rollout_states(env, n: int, rng: np.random.Generator, variation=None) -> TensorDict:
    """Draw ``n`` full States with the public API: seeded ``reset``, random ``step``, ``get_state``.

    Each chunk of ``num_envs`` lanes takes a reset seed from ``rng``. Every chunk takes the same number
    of steps, so the chunks have the same history width and concatenate. ``env`` keeps its State.
    ``variation`` holds ``(n, variation_dim)`` rows installed for each chunk before its reset; those
    States come from a new env of the same class, so ``env`` also keeps its Variation.
    """
    if variation is not None:
        with type(env)(num_envs=env.B, size=env.size, device=str(env.device)) as other:
            return _rollout(other, n, rng, variation)
    saved = env.get_state()
    try:
        return _rollout(env, n, rng)
    finally:
        env.set_state(saved)


def draw_states(env, n: int, rng: np.random.Generator):
    """``n`` render rows from the env's sampler, or ``n`` full States when the env has no render rows."""
    if has_render_rows(env):
        return env.sample_states(int(n), rng=rng)
    return rollout_states(env, n, rng)


def render_any(env, states, **kw) -> torch.Tensor:
    """Render render rows with ``env._render`` or full States with ``env.render``.

    A full State carries its own Variation, so ``style`` and ``variation`` must be None for one.
    """
    if not isinstance(states, TensorDict):
        return env._render(states, **kw)
    style, variation = kw.pop("style", None), kw.pop("variation", None)
    if style is not None or variation:
        raise TypeError(f"{type(env).__name__}: a full State carries its Variation; render it without style")
    return env.render(states, **kw)


def state_arrays(states: TensorDict) -> dict[str, np.ndarray]:
    """``{"a/b/c": array}`` for every leaf of ``states``, for ``np.savez``."""
    return {"/".join(k): v.detach().cpu().numpy() for k, v in states.items(True, True)}


def state_from_arrays(arrays, device) -> TensorDict:
    """Rebuild the TensorDict that :func:`state_arrays` flattened."""
    tree: dict = {}
    for key, value in arrays.items():
        *parents, leaf = key.split("/")
        node = tree
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = torch.as_tensor(np.asarray(value), device=device)
    batch = len(next(iter(arrays.values())))
    return TensorDict(tree, batch_size=[batch], device=device)


def digest(x) -> str:
    """Hash shape, dtype and bytes of an array, a tensor, or every leaf of a TensorDict in key order."""
    h = hashlib.sha256()
    if isinstance(x, TensorDict):
        for key, value in sorted(state_arrays(x).items()):
            h.update(key.encode())
            h.update(digest(value).encode())
        return h.hexdigest()
    if torch.is_tensor(x):
        x = x.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(x)
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sample_inputs(env, n: int):
    """Draw the fixed-seed inputs shared by the parent test and worker.

    For an env without render rows the states are full States under the default Variation. The style
    rows are drawn the same way for every env.
    """
    rng = np.random.default_rng(STATE_SEED)
    if has_render_rows(env):
        states = np.asarray(env.sample_states(n, rng=rng), np.float32)
    else:
        states = rollout_states(env, n, rng)
    style = (
        np.asarray(env.sample_style(n, rng=np.random.default_rng(STYLE_SEED)), np.float32)
        if env.style_dim
        else np.zeros((n, 0), np.float32)
    )
    return states, style


def styled_inputs(env, n: int, style) -> TensorDict:
    """The full States of :func:`sample_inputs`, drawn under the per-lane Variation rows ``style``."""
    return rollout_states(env, n, np.random.default_rng(STATE_SEED), variation=style)
