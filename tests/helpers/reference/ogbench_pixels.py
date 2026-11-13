"""Source-only Maze States and independently witnessed Cube/Scene checkpoints."""

from contextlib import ExitStack, closing

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference.bridges import load_source
from tests.helpers.reference.isolated import isolated_call
from tests.helpers.reference.ogbench_manipulation_semantics import assert_oracle, load_oracle
from tests.helpers.reference.ogbench_maze_semantics import OGBMazeReference
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import Registration
from tests.helpers.reference.semantics import compare_state


def source_gl_identity(gl, family, cutoffs):
    """Drain the source's pending GL errors, then read the GL vendor and renderer.

    The pinned Cube/Scene XML sets GL_SPOT_CUTOFF=100, which Mesa rejects with
    GL_INVALID_VALUE in glLight; the renderer keeps GL's retained cutoff. That
    error is recorded and allowed; any other pending error raises.
    """
    prior = []
    for _ in range(16):
        error = int(gl.glGetError())
        if error == int(gl.GL_NO_ERROR):
            break
        prior.append(error)
    else:
        raise RuntimeError("Source GL error queue did not clear")
    illegal = [float(v) for v in cutoffs if not (0 <= v <= 90 or v == 180)]
    if prior and not (family in ("Cube", "Scene") and illegal == [100.0] and set(prior) == {int(gl.GL_INVALID_VALUE)}):
        raise RuntimeError(f"Unexpected source rendering GL errors: {prior}; light cutoffs={list(cutoffs)}")
    vendor = gl.glGetString(gl.GL_VENDOR).decode()
    name = gl.glGetString(gl.GL_RENDERER).decode()
    if int(gl.glGetError()) != int(gl.GL_NO_ERROR):
        raise RuntimeError("Source GL identity query generated an error")
    return dict(
        vendor=vendor,
        renderer=name,
        software=any(x in name.lower() for x in ("llvmpipe", "softpipe", "swrast")),
        source_render_gl_errors=prior,
        source_light_cutoffs=[float(v) for v in cutoffs],
    )


def _identity(renderer, device: str, family: str, cutoffs: np.ndarray) -> dict:
    from OpenGL import GL

    renderer._gl_context.make_current()
    result = source_gl_identity(GL, family, cutoffs)
    assert (
        result["software"] if device == "cpu" else "nvidia" in result["vendor"].lower() and not result["software"]
    ), result
    return result


def _close_renderer(owner, attribute: str) -> None:
    renderer = getattr(owner, attribute, None)
    if renderer is not None:
        renderer.close()
        setattr(owner, attribute, None)


def ogbench_trace(
    case: Registration, seed: int, dtype: str, horizon: int, device: str, source_root: str
) -> tuple[PixelTrace, list[dict]]:
    return isolated_call(__name__, "_ogbench_trace", (case, seed, dtype, horizon, device, source_root))


def _ogbench_trace(
    case: Registration, seed: int, dtype: str, horizon: int, device: str, source_root: str
) -> tuple[PixelTrace, list[dict]]:
    load_source(source_root, "ogbench")
    actions = np.random.default_rng(seed + 1000).uniform(-0.2, 0.2, (2, horizon, case.width)).astype(dtype)
    if case.domain == "Maze":
        return _maze_trace(seed, actions, device)
    return _manipulation_trace(case, seed, actions, device)


def _maze_trace(seed: int, actions: np.ndarray, device: str) -> tuple[PixelTrace, list[dict]]:
    with ExitStack() as stack:
        source = stack.enter_context(closing(OGBMazeReference(2, width=64, height=64)))
        for wrapper in source.envs:
            stack.callback(_close_renderer, wrapper.env, "custom_renderer")
        source.reset(seed)
        for lane, prefix in enumerate((0, 7)):
            for _ in range(prefix):
                output = source._in_legacy_rng(lane, source.envs[lane].step, np.zeros(2, actions.dtype))
                assert not (output[2] or output[3])
        initial = source.get_state()
        states, frames, tasks, identities = [], [], [], []
        for step in range(actions.shape[1]):
            result = source.step(actions[:, step].copy())
            assert not bool((result["terminated"] | result["truncated"]).any())
            state = source.get_state()
            images = []
            for lane, wrapper in enumerate(source.envs):
                image = source._in_legacy_rng(lane, wrapper.render).copy()
                assert image.shape == (64, 64, 3) and image.dtype == np.uint8
                identity = _identity(wrapper.env.custom_renderer, device, "Maze", wrapper.env.model.light_cutoff)
                if step == 0:
                    identities.append(identity)
                else:
                    assert identity == identities[lane]
                images.append(image)
            compare_state(source.get_state(), state)
            states.append(state)
            frames.append(np.stack(images))
            tasks.append(result.exclude("state").clone())
        return PixelTrace(
            initial,
            states,
            {key: torch.stack([row[key] for row in tasks], dim=1) for key in tasks[0].keys(True, True)},
            np.stack(frames, axis=1),
            torch.ones(actions.shape[:2], dtype=torch.bool),
            actions,
        ), identities


def _manipulation_trace(
    case: Registration, seed: int, actions: np.ndarray, device: str
) -> tuple[PixelTrace, list[dict]]:
    oracle = load_oracle()
    source_class = getattr(oracle, "Owned" + case.domain)
    with ExitStack() as stack:
        reference = stack.enter_context(se.make(case.name, num_envs=2, size=64, device="cpu"))
        reference.reset(seed=seed)
        sources = [
            stack.enter_context(closing(source_class(entropy_seed=lane, width=64, height=64))) for lane in range(2)
        ]
        for source in sources:
            stack.callback(_close_renderer, source, "_renderer")
        for lane, (source, prefix) in enumerate(zip(sources, (0, 7), strict=True)):
            source.reset(seed=seed + lane)
            raw = reference._simulator.envs[lane]
            for _ in range(prefix):
                action = np.zeros(5, actions.dtype)
                output = source.step(action.copy())
                assert not oracle.differences(output, raw.step(action.copy()))
                assert not (output[2] or output[3])
            assert_oracle(source, raw, oracle)
        initial = reference.get_state()
        states, frames, tasks, validity, identities = [], [], [], [], []
        active = np.ones(2, bool)
        for step in range(actions.shape[1]):
            validity.append(active.copy())
            row = np.zeros((2, 4), np.float64)
            images = []
            for lane, source in enumerate(sources):
                if active[lane]:
                    output = source.step(actions[lane, step].copy())
                    row[lane] = output[1], output[2], output[3], output[4]["success"]
                before = oracle.capture(source)
                image = source.render().copy()
                assert image.shape == (64, 64, 3) and image.dtype == np.uint8
                assert not oracle.differences(before, oracle.capture(source)), "Source render changed continuation"
                identity = _identity(source._renderer, device, case.domain, source.model.light_cutoff)
                if step == 0:
                    identities.append(identity)
                else:
                    assert identity == identities[lane]
                images.append(image)
            reference._simulator.step(actions[:, step], active=torch.from_numpy(active))
            for lane, source in enumerate(sources):
                assert_oracle(source, reference._simulator.envs[lane], oracle)
            states.append(reference.get_state())
            frames.append(np.stack(images))
            tasks.append(row)
            active &= ~(row[:, 1].astype(bool) | row[:, 2].astype(bool))
        for lane, source in enumerate(sources):
            action = -actions[lane, -1]
            raw = reference._simulator.envs[lane]
            assert not oracle.differences(source.step(action.copy()), raw.step(action.copy()))
            assert_oracle(source, raw, oracle)
        columns = torch.from_numpy(np.stack(tasks, axis=1))
        return PixelTrace(
            initial,
            states,
            {
                key: columns[:, :, i].to(torch.float64 if key == "reward" else torch.bool)
                for i, key in enumerate(("reward", "terminated", "truncated", "success"))
            },
            np.stack(frames, axis=1),
            torch.from_numpy(np.stack(validity, axis=1)),
            actions,
        ), identities
