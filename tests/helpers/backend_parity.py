"""Reset and short-rollout agreement between two simulation backends of one class.

Reset randomness comes from the family's host generators, so both backends must start from the same State for a
seed. The physics can differ in precision (float64 on CPU, float32 on GPU), so the rollout check has a tolerance.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch

import stable_envs as se


def _key(key) -> str:
    return "/".join(key) if isinstance(key, tuple) else key


def compare_backend_states(actual, expected, *, atol: float, exclude: Iterable[str] = (), what: str = "") -> None:
    """Assert equal keys, exact integer and bool leaves, and float leaves within ``atol``.

    ``exclude`` holds ``/``-joined key prefixes that one backend computes differently, such as solver caches.
    """
    skip = tuple(exclude)
    keep = lambda key: not _key(key).startswith(skip)  # noqa: E731
    a = {_key(k): v for k, v in actual.items(True, True) if keep(k)}
    e = {_key(k): v for k, v in expected.items(True, True) if keep(k)}
    assert set(a) == set(e), f"{what}: keys differ: {sorted(set(a) ^ set(e))}"
    for key in sorted(a):
        x, y = a[key].detach().cpu(), e[key].detach().cpu()
        assert x.shape == y.shape, f"{what} {key}: shape {tuple(x.shape)} != {tuple(y.shape)}"
        if x.is_floating_point() or y.is_floating_point():
            torch.testing.assert_close(
                x.double(), y.double(), atol=atol, rtol=0, equal_nan=True, msg=lambda m, k=key: f"{what} {k}: {m}"
            )
        else:
            assert torch.equal(x.to(y.dtype), y), f"{what} {key}: integer or bool values differ"


def check_backend_parity(
    name: str,
    backends: Sequence[str],
    *,
    num_envs: int = 4,
    device: str = "cuda",
    seeds: Sequence[int] = (0, 7),
    steps: int = 3,
    reset_atol: float = 1e-5,
    step_atol: float = 1e-3,
    exclude: Iterable[str] = (),
    **make_kwargs,
) -> None:
    """Reset ``name`` on each of ``backends`` with the same seeds, compare the States, then a few steps.

    Args:
        name: Registered class name.
        backends: Two or more ``simulation_backend`` values; the first is the reference.
        num_envs: Batch size of every environment.
        device: Device of every environment.
        seeds: Reset seeds.
        steps: Steps after each reset, with the same random actions on every backend.
        reset_atol: Tolerance of float leaves after a reset (float32 rounding of the reset values).
        step_atol: Tolerance of float leaves after each step (float64 against float32 physics).
        exclude: ``/``-joined State key prefixes left out of the comparison.
        **make_kwargs: Other ``se.make`` arguments.
    """
    exclude = tuple(exclude)
    envs = []
    try:
        for backend in backends:
            envs.append(se.make(name, num_envs=num_envs, device=device, simulation_backend=backend, **make_kwargs))
        reference, *others = envs
        for seed in seeds:
            want = reference.reset(seed=seed)
            for env in others:
                compare_backend_states(
                    env.reset(seed=seed), want, atol=reset_atol, exclude=exclude,
                    what=f"{name} reset seed={seed} {env.simulation_backend} vs {reference.simulation_backend}",
                )
            rng = np.random.default_rng(seed)
            for t in range(steps):
                actions = torch.as_tensor(reference.action_spec.sample(num_envs, rng=rng))
                want = reference.step(actions.to(reference.device), states=slice(0, 0), frames=None)["state"]
                for env in others:
                    got = env.step(actions.to(env.device), states=slice(0, 0), frames=None)["state"]
                    compare_backend_states(
                        got, want, atol=step_atol, exclude=exclude,
                        what=f"{name} seed={seed} step {t} {env.simulation_backend} vs {reference.simulation_backend}",
                    )
    finally:
        for env in envs:
            env.close()
