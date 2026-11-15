"""Draw a small batch of a CUDA environment with its family's CPU renderer, and move the frames in one copy.

A GPU renderer pays a launch cost per call, so a small batch draws faster on the CPU. A class that has both
renderers declares ``render_rungs``, ``{device type: Rungs("cpu", "cuda", min_rows)}``, next to its renderer choice.
``on_host`` selects the rung with ``select_renderer``: the ``large_renderer_min_rows`` knob, else the tuning cache,
else the shipped ``min_rows``. ``draw_rows`` draws compact rows in a CPU copy of the environment, and ``upload``
moves CPU frames to the device in one pinned copy.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .. import tuning
from .._core.style import StyleTensor
from .backends import RENDER_KNOB, Measure, select_renderer

#: The rung that draws on the CPU.
CPU = "cpu"


def _measure(env) -> Measure:
    """A ``Measure`` that builds ``type(env)`` on its device and draws its reset State on one renderer."""
    cls, kwargs = type(env), dict(size=env.size, supersample=env.ss, device=env.device)

    def measure(renderer: str, rows: int):
        other = cls(num_envs=rows, **kwargs)
        state = other.reset(seed=0)
        value = rows + 1 if renderer == CPU else 1

        def draw():
            with tuning.override(**{RENDER_KNOB: value}):
                return other.render(state)

        return draw, other.close

    return measure


def on_host(env, path: str, rows: int, backend: str = "auto") -> bool:
    """Whether ``rows`` frames of ``env`` draw on the CPU renderer of its class.

    ``backend`` is the renderer that the call requests. Only ``"auto"`` switches; an explicit renderer draws every
    batch size.
    """
    rungs = getattr(env, "render_rungs", None)
    if not rungs or env.device.type == "cpu" or backend != "auto":
        return False
    measure = env.__dict__.get("_render_measure")
    if measure is None:
        measure = env.__dict__["_render_measure"] = _measure(env)
    return select_renderer(f"{path}/{type(env).__name__}", rows, env.device, rungs, measure=measure) == CPU


def pinned(shape: tuple[int, ...]) -> torch.Tensor:
    """An uninitialized pinned uint8 host buffer of ``shape`` for frames that ``upload`` moves."""
    return torch.empty(shape, dtype=torch.uint8, pin_memory=torch.cuda.is_available())


def upload(frames: torch.Tensor, device: torch.device, out: torch.Tensor | None = None) -> torch.Tensor:
    """Move CPU ``frames`` to ``device`` in one copy, into ``out`` when it is given.

    A pinned buffer moves asynchronously; the host allocator keeps it until the copy ends.
    """
    blocking = not frames.is_pinned()
    if out is not None:
        out.copy_(frames.view(out.shape), non_blocking=not blocking)
        return out
    return frames.to(device, non_blocking=not blocking)


def _twin(env):
    """A one-row CPU environment of the same class and frame settings, built on first use and kept on ``env``."""
    twin = env.__dict__.get("_host_twin")
    if twin is None or twin.closed:
        from .. import registry

        kwargs = getattr(env, "render_worker_kwargs", {})
        twin = registry.REGISTRY[type(env).__name__](
            num_envs=1, device="cpu", size=env.size, supersample=env.ss, **kwargs
        )
        env.__dict__["_host_twin"] = twin
    return twin


def close(env) -> None:
    """Close the CPU environment that ``draw_rows`` built for ``env``, if any."""
    twin = env.__dict__.pop("_host_twin", None)
    if twin is not None:
        twin.close()


def draw_rows(
    env, rows: torch.Tensor, style: StyleTensor, size: int, supersample: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Draw compact ``rows`` with the appearance ``style`` in a CPU copy of ``env`` and move the frames to its device.

    Rows and appearance leave the device in one copy when their dtypes match.
    """
    n = int(rows.shape[0])
    values = style.values
    if rows.device.type == "cpu":
        values = _host_values(env, values)
    elif rows.dtype == values.dtype and rows.device == values.device:
        both = torch.cat((rows, values.reshape(n, -1)), 1).cpu()
        rows, values = both[:, : rows.shape[1]], both[:, rows.shape[1] :]
    else:
        rows, values = rows.cpu(), values.cpu()
    host = StyleTensor(values, style._layout_list(), shared=style._shared)
    frames = pinned((n, size, size, 3))
    drawn = _twin(env)._render(rows, size=size, supersample=supersample, style=host, out=frames)
    if drawn is not frames and drawn.data_ptr() != frames.data_ptr():
        frames.copy_(drawn.reshape(frames.shape))
    return upload(frames, env.device, out)


def _host_values(env, values: torch.Tensor) -> torch.Tensor:
    """A host copy of appearance ``values``, kept on ``env`` until the values change in place or move.

    The cache holds the device tensor, so its memory is not reused by another tensor while the copy is kept.
    """
    if values.device.type == "cpu":
        return values
    hit = env.__dict__.get("_host_style")
    if (
        hit is not None
        and hit[0].data_ptr() == values.data_ptr()
        and hit[0].shape == values.shape
        and hit[0].stride() == values.stride()
        and hit[1] == values._version
    ):
        return hit[2]
    host = values.cpu()
    env.__dict__["_host_style"] = (values, values._version, host)
    return host


def draw_into(shape: tuple[int, ...], device: torch.device, draw: Callable[[torch.Tensor], torch.Tensor], out=None):
    """Run ``draw(buffer)`` on a pinned host buffer of ``shape`` and move the frames it returns to ``device``."""
    frames = pinned(shape)
    drawn = draw(frames)
    if drawn.data_ptr() != frames.data_ptr():
        frames.copy_(drawn.reshape(shape))
    return upload(frames, device, out)
