from contextlib import closing

import numpy as np
import torch

from tests.helpers.reference.dmc_semantics import DMCReference
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import Registration
from tests.helpers.reference.semantics import compare_state


def dmc_trace(
    case: Registration, seed: int, dtype: str, horizon: int, device: str
) -> tuple[PixelTrace, dict[str, str | bool]]:
    """Witness forward source rollouts and verify the actual source GL device."""
    from OpenGL import GL

    with closing(DMCReference(case.domain, 2)) as source:
        source.reset(seed)
        initial = source.get_state()
        identities = []
        for wrapper in source.envs:
            wrapper.render(width=64, height=64)
            with wrapper.env.physics.contexts.gl.make_current() as context:
                vendor = context.call(GL.glGetString, GL.GL_VENDOR).decode()
                renderer = context.call(GL.glGetString, GL.GL_RENDERER).decode()
            software = any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "swrast"))
            assert software if device == "cpu" else "nvidia" in vendor.lower() and not software, (vendor, renderer)
            identities.append(dict(vendor=vendor, renderer=renderer, software=software))
        assert identities[0] == identities[1], "Source lanes use different GL backends"
        compare_state(source.get_state(), initial)
        for wrapper, prefix in zip(source.envs, (0, 7), strict=True):
            for _ in range(prefix):
                output = wrapper.step(np.zeros(case.width, dtype=dtype))
                assert not (output[2] or output[3]), "Source prefix unexpectedly terminated"
        initial = source.get_state()
        actions = np.random.default_rng(seed + 1000).uniform(-0.2, 0.2, (2, horizon, case.width)).astype(dtype)
        states, frames, outputs = [], [], []
        for t in range(horizon):
            result = source.step(actions[:, t])
            # These pinned default wrappers continue through internal DMC resets.
            assert not bool((result["terminated"] | result["truncated"]).any())
            states.append(result["state"].clone())
            outputs.append(result.exclude("state").clone())
            frames.append(np.stack([wrapper.render(width=64, height=64).copy() for wrapper in source.envs]))
            compare_state(source.get_state(), states[-1])
        return PixelTrace(
            initial,
            states,
            {key: torch.stack([row[key] for row in outputs], dim=1) for key in outputs[0].keys(True, True)},
            np.stack(frames, axis=1),
            torch.ones((2, horizon), dtype=torch.bool),
            actions,
            nan_tasks=frozenset({"success"})
            if case.domain in {"cheetah", "hopper", "walker", "finger", "manipulator", "quadruped", "humanoid"}
            else frozenset(),
        ), identities[0]
