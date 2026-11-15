"""A State from one simulation backend either resumes on another backend of the class or is refused.

Every registered class that declares two or more ``simulation_backends`` is covered. A State saved on backend X,
loaded into an environment built on backend Y, must resume within the tolerances of the backend parity check, or
``set_state`` must refuse it with an error that names both backends. It must never load and then resume on
different physics.
"""

import itertools

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.backend_parity import compare_backend_states

#: ci/families.toml profile of each ``Env.family``.
PROFILES = {
    "classic_control": "classic",
    "dmcontrol": "dmc",
    "fetch": "fetch",
    "miniworld": "miniworld",
    "mshuman": "mshuman",
    "navigation": "navigation",
    "ogbench": "ogbench",
    "pusht": "pusht",
    "pyflyt": "rocket",
    "roadster": "roadster",
    "craftax": "craftax",
}
STEP_ATOL = 1e-3
#: A State from a float64 backend resumes in float32, so values carry a relative rounding error that contacts grow.
STEP_RTOL = 1e-2
#: Solver warm-start caches: each solver writes its own, and they don't change the next State beyond the tolerance.
SOLVER_CACHES = ("warmstart",)


def _cases():
    cases = []
    for name, cls in sorted(se.registry.REGISTRY.items()):
        backends = tuple(getattr(cls, "simulation_backends", ()) or ())
        if len(backends) < 2:
            continue
        profile = PROFILES.get(getattr(cls, "family", ""), "core")
        for device in ("cpu", "cuda"):
            marks = [pytest.mark.family(profile)]
            if device == "cuda":
                marks.append(pytest.mark.gpu)
            for saved, loaded in itertools.permutations(backends, 2):
                cases.append(pytest.param(name, device, saved, loaded, id=f"{name}-{device}-{saved}-{loaded}", marks=marks))
    return cases


def _make(name, device, backend):
    """The environment on ``backend``, or None when the backend does not run on this device or machine."""
    size = {} if se.registry.REGISTRY[name].native_size else {"size": 32}
    try:
        return se.make(name, num_envs=3, device=device, simulation_backend=backend, **size)
    except (ValueError, ImportError, RuntimeError):
        return None


@pytest.mark.parametrize("name,device,saved,loaded", _cases())
def test_state_resumes_on_another_backend_or_is_refused(name, device, saved, loaded):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = _make(name, device, saved)
    if source is None:
        pytest.skip(f"{name}: simulation_backend={saved!r} is not available on {device}")
    target = _make(name, device, loaded)
    if target is None:
        source.close()
        pytest.skip(f"{name}: simulation_backend={loaded!r} is not available on {device}")
    try:
        rng = np.random.default_rng(5)
        source.reset(seed=5)
        for _ in range(2):
            source.step(source.action_spec.sample(3, rng=rng).to(source.device), states=None, frames=None)
        state = source.get_state()
        try:
            target.set_state(state)
        except (ValueError, TypeError) as error:
            message = str(error)
            assert saved in message and loaded in message, f"refusal must name both backends: {message}"
            return
        for t in range(3):
            actions = source.action_spec.sample(3, rng=rng)
            want = source.step(actions.to(source.device), states=slice(0, 0), frames=None)["state"]
            got = target.step(actions.to(target.device), states=slice(0, 0), frames=None)["state"]
            compare_backend_states(
                got, want, atol=STEP_ATOL, rtol=STEP_RTOL, exclude=SOLVER_CACHES,
                what=f"{name} {saved} State on {loaded}, step {t}",
            )
    finally:
        source.close()
        target.close()
