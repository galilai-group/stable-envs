"""Checks for envs whose State has no flat render row, using only the public State API.

Each function measures the same property as the ``stable_envs.testing.conformance`` check of the same
name, with full States from :func:`stable_envs.testing._sampling.rollout_states` in place of render
rows. A full State carries its own Variation, so a probe that changes the Variation installs it with
``set_variation`` and draws new States with ``reset``. Such probes run on a new env of the same class,
so the caller's env keeps its Variation.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from tensordict import TensorDict

from stable_envs.testing import conformance as C
from stable_envs.testing._sampling import rollout_states

#: Reset seed and step count of the Variation probes.
PROBE_SEED, PROBE_STEPS = 3, 3


@contextlib.contextmanager
def fresh(env):
    """A new env of the same class, size, batch and device; closed on exit."""
    other = type(env)(num_envs=env.B, size=env.size, device=str(env.device))
    try:
        yield other
    finally:
        other.close()


def frames(env, states) -> np.ndarray:
    return env.render(states).detach().cpu().numpy()


def max_diff(a, b) -> int:
    return int(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max())


def state_diff(a: TensorDict, b: TensorDict) -> list[str]:
    """The leaves of ``a`` that are missing from ``b`` or differ from it."""
    out = []
    for key, value in a.items(True, True):
        other = b.get(key, None)
        if other is None or other.shape != value.shape or not torch.equal(other.to(value.device), value):
            out.append("/".join(key))
    return out


def frames_under(env, variation=None, seed: int = PROBE_SEED, steps: int = PROBE_STEPS) -> np.ndarray:
    """Frames after ``set_variation`` (the default first, then ``variation``), a seeded reset and steps.

    With ``variation=None`` the env's installed Variation is used as it is.
    """
    if variation is not None:
        env.set_variation(env.default_variation())
        env.set_variation(variation)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(env.action_spec.sample(env.B, rng), frames=None)
    return env.render().detach().cpu().numpy()


def _alternative(node, default):
    """A value of a variation leaf different from ``default``: the other Discrete value or the far bound."""
    if hasattr(node, "n"):
        start = int(getattr(node, "start", 0))
        return start if int(np.asarray(default)) != start else start + 1
    lo = np.broadcast_to(node.low, node.shape).astype(np.float64)
    hi = np.broadcast_to(node.high, node.shape).astype(np.float64)
    d = np.asarray(default, np.float64)
    return np.where(np.abs(d - lo) > np.abs(d - hi), lo, hi).astype(node.dtype)


def check_state_roundtrip(env) -> dict:
    """``get_state()`` after ``set_state(s)`` gives back ``s`` leaf for leaf."""
    s = rollout_states(env, env.B, np.random.default_rng(0))
    env.set_state(s)
    return {"mismatched_leaves": state_diff(s, env.get_state())}


def check_state_reaches_the_render(env) -> dict:
    """States from different reset seeds draw different frames, and steps change the frame."""
    with fresh(env) as other:
        start = frames_under(other, steps=0)
        later = frames_under(other)
    unique = len(np.unique(later.reshape(env.B, -1), axis=0))
    return {"duplicate_frames": env.B - unique, "steps_change_the_frame": not np.array_equal(start, later)}


def check_batch_invariance(env, n: int = 8) -> dict:
    s = rollout_states(env, env.B, np.random.default_rng(0))
    batched = frames(env, s)
    worst = 0
    for i in range(min(n, env.B)):
        worst = max(worst, max_diff(frames(env, s[i : i + 1])[0], batched[i]))
    return {"max_abs_diff": worst}


def check_determinism(env) -> dict:
    s = rollout_states(env, env.B, np.random.default_rng(0))
    return {"identical": bool(np.array_equal(frames(env, s), frames(env, s)))}


def check_style_is_state(env) -> dict:
    """A new env given ``set_state(s)`` draws the frame ``render(s)`` draws, for States under a sampled Variation."""
    with fresh(env) as other:
        s = rollout_states(env, env.B, np.random.default_rng(0), variation=env.sample_style(env.B))
        other.set_state(s)
        return {"max_abs_diff": max_diff(frames(env, s), other.render().detach().cpu().numpy())}


def check_default_style_is_official(env) -> dict:
    """An env that never had a Variation installed draws what one given ``default_variation()`` draws."""
    with fresh(env) as a, fresh(env) as b:
        return {"max_abs_diff": max_diff(frames_under(a), frames_under(b, b.default_variation()))}


def check_per_env_variation(env) -> dict:
    """With a leaf at ``low`` on the first half of the lanes and ``high`` on the rest, each lane draws its own value.

    Each half must equal the frames of a uniform ``low`` or ``high`` install on the same seeds.
    """
    if env.B < 2:
        return {"skipped": "batch too small"}
    h = env.B // 2
    collapsed = []
    with fresh(env) as other:
        for path in C._frame_variation_paths(env):
            node = env.variation_space._walk(path)
            if not hasattr(node, "low"):
                continue
            lo = np.broadcast_to(node.low, node.shape).astype(node.dtype)
            hi = np.broadcast_to(node.high, node.shape).astype(node.dtype)
            ramp = np.stack([lo] * h + [hi] * (env.B - h))
            try:
                f_lo = frames_under(other, {path: lo})
                f_hi = frames_under(other, {path: hi})
                f = frames_under(other, {path: ramp})
            except Exception:  # noqa: BLE001
                continue  # the env may not accept a per-lane form for this leaf
            if not (np.array_equal(f[:h], f_lo[:h]) and np.array_equal(f[h:], f_hi[h:])):
                collapsed.append(path)
    return {"collapsed_variations": collapsed}


def check_variations_do_something(env) -> dict:
    """Appearance leaves whose alternative value leaves the frames of the same seeds unchanged."""
    inert = []
    with fresh(env) as other:
        base = frames_under(other, other.default_variation())
        for path in C._frame_variation_paths(env):
            node = env.variation_space._walk(path)
            try:
                if np.array_equal(frames_under(other, {path: _alternative(node, node.value)}), base):
                    inert.append(path)
            except Exception as e:  # noqa: BLE001
                inert.append(f"{path} ({type(e).__name__}: {e})")
    return {"inert_variations": inert}


def check_colors_apply_exactly(env) -> dict:
    """RGB leaves whose requested color is on no pixel; the result has the conformance check's keys."""
    wrong, best = [], {}
    drawn = set(C._frame_variation_paths(env))
    with fresh(env) as other:
        for path, _, w in env.variation_space.flat_layout():
            node = env.variation_space._walk(path)
            if path not in drawn or w != 3 or not hasattr(node, "high") or int(np.max(node.high)) != 255:
                continue
            for target in ([17, 33, 200], [200, 17, 33]):
                f = frames_under(other, {path: np.asarray(target, node.dtype)})
                d = np.abs(f.astype(np.int16) - np.asarray(target, np.int16)).sum(-1)
                best[f"{path}->{target}"] = int(d.min())
                if not (d == 0).any():
                    wrong.append(f"{path}->{target}")
    return {
        "colors_not_applied": wrong,
        "best_abs_diff": best,
        "applicability": C.color_exactness_applicability(env, sorted(best)),
    }


def check_chunked_render_matches(env, states, chunks=(1, 7, 64, 4096)) -> dict:
    """``render`` over consecutive chunks of ``states`` gives the frames of one ``render`` call."""
    ref = frames(env, states)
    n, worst, where = len(states), 0, ""
    for c in chunks:
        got = np.concatenate([frames(env, states[lo : lo + int(c)]) for lo in range(0, n, int(c))])
        d = max_diff(got, ref)
        if d > worst:
            worst, where = d, f"chunk={c}"
    return {"max_abs_diff": worst, "worst_case": where, "chunks": tuple(chunks)}


def check_input_form_invariance(env, states) -> dict:
    """The same State on the CPU, on the env's device, cloned, or with strided leaves draws one frame."""
    ref = frames(env, states)

    def strided(td):
        out = td.clone()
        for key, value in td.items(True, True):
            pad = torch.zeros((value.shape[0], 2, *value.shape[1:]), dtype=value.dtype, device=value.device)
            pad[:, 0] = value
            out.set(key, pad[:, 0])
        return out

    forms = {"device": states, "cpu": states.cpu(), "clone": states.clone(), "strided": strided(states)}
    bad = {}
    for name, s in forms.items():
        try:
            d = max_diff(frames(env, s), ref)
        except Exception as e:  # noqa: BLE001
            bad[name] = f"{type(e).__name__}: {e}"
            continue
        if d:
            bad[name] = f"differs by {d}/255"
    return {"disagreements": bad, "forms": tuple(forms)}


def check_order_independence(env, states, seed: int = 0) -> dict:
    n = len(states)
    perm = np.random.default_rng(seed).permutation(n)
    base = frames(env, states)
    permuted = frames(env, states[torch.as_tensor(perm)])
    f = frames(env, torch.cat([states[:1], states[:1]]))
    return {"permutation_max_abs_diff": max_diff(permuted, base[perm]), "duplicate_max_abs_diff": max_diff(f[0], f[1])}


def check_render_is_pure(env, states) -> dict:
    """``render`` changes neither State nor Variation, and ``set_state(get_state())`` twice changes neither."""
    env.set_state(states)
    s0, k0 = env.get_state(), env.get_variation().detach().cpu().numpy().copy()
    f0 = env.render().detach().cpu().numpy()
    env.render(rollout_states(env, env.B, np.random.default_rng(1)))
    s1, k1 = env.get_state(), env.get_variation().detach().cpu().numpy().copy()
    env.set_state(env.get_state())
    env.set_state(env.get_state())
    s2 = env.get_state()
    f2 = env.render().detach().cpu().numpy()
    return {
        "state_changed_by_render": float(len(state_diff(s0, s1))),
        "style_changed_by_render": float(np.abs(k1 - k0).max()) if k0.size else 0.0,
        "state_changed_by_double_set": float(len(state_diff(s0, s2))),
        "frame_changed_by_double_set": max_diff(f2, f0),
    }


def check_set_state_copies(env) -> dict:
    """Zeroing the State handed to ``set_state`` leaves the env's State unchanged."""
    s = rollout_states(env, env.B, np.random.default_rng(0))
    env.set_state(s)
    stored = env.get_state()
    for _, value in s.items(True, True):
        value.zero_()
    return {"aliased_leaves": state_diff(stored, env.get_state())}
