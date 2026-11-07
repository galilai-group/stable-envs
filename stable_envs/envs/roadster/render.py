"""The Roadster's world-model camera, drawn from compact render rows with MuJoCo's OpenGL renderer.

A render row is ``[time, qpos, qpos_pre, has_qpos_pre, ctrl, mocap_pos, mocap_quat, seed_hi, seed_lo, rain,
wind_mean, fx_count, fx_history]``; seed/rain/wind describe the weather look and the ``fx`` columns are the source's
exhaust history. The frame shows the live simulation: after a step MuJoCo's ``xpos``/``xmat`` lag ``qpos``
by one substep (``qpos_pre``), the jet flames and nozzle glow follow ``ctrl``, the blob shadow is cast from
``qpos``, the rain moves with the time and the exhaust smoke is regenerated from the history
(``source.fx.Smoke``). The flame flicker is seeded per frame so rendering is deterministic.

One render model, data and ``mujoco.Renderer`` per resolution live for the process and are shared by
every Roadster env; releasing a renderer can invalidate other live GL contexts. Large batches draw in
render worker processes, each with its own (``render_rows``).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..._core.mujoco_gl import mujoco_renderer
from .source import cinematic
from .source.fx import ENTRY_DIM, HISTORY
from .source.sim import RoadsterSim
from .source.weather import apply_render_flags

CAMERA = "wm"


def row_layout(model) -> dict[str, tuple[int, int]]:
    """``{column group: (offset, width)}`` of a render row for ``model``."""
    n = int(model.nmocap)
    groups = (
        ("time", 1),
        ("qpos", int(model.nq)),
        ("qpos_pre", int(model.nq)),
        ("has_qpos_pre", 1),
        ("ctrl", int(model.nu)),
        ("mocap_pos", 3 * n),
        ("mocap_quat", 4 * n),
        ("seed_hi", 1),  # the weather seed in two 24-bit halves, each exact in a float32 row
        ("seed_lo", 1),
        ("rain", 1),
        ("wind_mean", 3),
        ("fx_count", 1),
        ("fx_history", HISTORY * ENTRY_DIM),
    )
    out, offset = {}, 0
    for name, width in groups:
        out[name] = (offset, width)
        offset += width
    return out


def row_dim(model) -> int:
    layout = row_layout(model)
    off, width = layout["fx_history"]
    return off + width


def row_names(model) -> tuple[str, ...]:
    names: list[str] = []
    for group, (_, width) in row_layout(model).items():
        names.extend([group] if width == 1 else [f"{group}[{i}]" for i in range(width)])
    return tuple(names)


LOOKS = ("wm", "cinematic")


class _Renderer:
    """A render-only copy of the world at one resolution and look.

    ``wm``: the world-model camera (no MSAA, no shadows). ``cinematic``: the same camera and State on the
    dark-space scene (grid floor fading to black, no decorative scenery, a sun that follows the car with real
    shadows, 8x MSAA) with post-processing (bloom, filmic grade, vignette; no grain and no motion blur, so a
    State gives identical pixels). Both draw the same State; only appearance differs. ``postfx=False`` leaves
    the cinematic frame raw.
    """

    def __init__(self, size: int, look: str = "wm", postfx: bool = True):
        if look not in LOOKS:
            raise ValueError(f"look must be one of {LOOKS}, got {look!r}")
        self.cinematic = look == "cinematic"
        self.postfx = bool(postfx) and self.cinematic
        # The world the tasks simulate, driven only for drawing.
        self.sim = RoadsterSim(xml_patch=cinematic.patch_space if self.cinematic else None)
        self.model, self.data = self.sim.model, self.sim.data
        self.model.vis.quality.offsamples = 8 if self.cinematic else 0
        if self.cinematic and self.sim._shadow >= 0:  # real shadows replace the blob shadow
            self.model.geom_rgba[self.sim._shadow, 3] = 0.0
            self.sim._shadow = -1
        self.layout = row_layout(self.model)
        self.size = int(size)
        self._look = (0.0, 0, 0.0, 0.0)  # no rain, seed 0, no wind: the context is made for this look
        self.sim.weather.set(0, wind=(0.0, 0.0), rain=0.0)
        self.renderer = mujoco_renderer(self.model, height=self.size, width=self.size)

    def _set_look(self, rain: float, seed: int, wind: np.ndarray) -> None:
        """Configure the weather look as the source does at reset (same seed, rain and mean wind).

        The seed and wind only place and drift the rain drops. MuJoCo bakes the fog, which only the rain
        changes, into the render context when it is made, so the context is remade only when the rain does.
        """
        look = (float(rain), int(seed), float(wind[0]), float(wind[1]))
        if look == self._look:
            return
        self.sim.weather.set(int(seed), wind=(float(wind[0]), float(wind[1])), rain=float(rain))
        if look[0] != self._look[0]:
            self._remake_context()
        self._look = look

    def _remake_context(self) -> None:
        self.renderer._gl_context.make_current()
        old = self.renderer._mjr_context
        self.renderer._mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.renderer._mjr_context)
        self.renderer._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        old.free()

    def draw(self, rows: np.ndarray, out: np.ndarray) -> None:
        """Draw ``rows`` (n, row_dim) into ``out`` (n, size, size, 3) uint8.

        Rows are drawn grouped by rain, so the render context is remade at most once per rain strength.
        """
        d, lay = self.data, self.layout
        col = lambda name, row: row[lay[name][0] : lay[name][0] + lay[name][1]]
        look = lambda row: (
            float(col("rain", row)[0]),
            (int(round(col("seed_hi", row)[0])) << 24) | int(round(col("seed_lo", row)[0])),
            *map(float, col("wind_mean", row)[:2]),
        )
        order = sorted(range(len(rows)), key=lambda i: look(rows[i]))
        for i in order:
            row = rows[i]
            self._set_look(*look(row)[:2], col("wind_mean", row))
            d.time = float(col("time", row)[0])
            d.qpos[:] = col("qpos", row)
            d.ctrl[:] = col("ctrl", row)
            d.mocap_pos[:] = col("mocap_pos", row).reshape(-1, 3)
            d.mocap_quat[:] = col("mocap_quat", row).reshape(-1, 4)
            d.qvel[:] = 0.0
            self.sim.restore_derived(col("qpos_pre", row) if col("has_qpos_pre", row)[0] > 0.5 else None)
            self.sim.set_fx_state(int(round(col("fx_count", row)[0])), col("fx_history", row).reshape(-1, ENTRY_DIM))
            self.sim.draw_effects(np.random.default_rng(0))  # flame flicker: deterministic per frame
            self.renderer.update_scene(d, camera=CAMERA)
            apply_render_flags(self.renderer.scene, self.sim.weather, shadows=self.cinematic)
            if self.cinematic:  # the dark-space fade to black is fog
                self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
            frame = self.renderer.render()
            if self.postfx:
                frame = cinematic.postfx(frame)
            out[i] = frame


_RENDERERS: dict[tuple[int, str, bool], _Renderer] = {}


def renderer(size: int, look: str = "wm", postfx: bool = True) -> _Renderer:
    """The process-wide renderer for ``size``, ``look`` and ``postfx``; built on first use and never released."""
    key = (int(size), look, bool(postfx))
    r = _RENDERERS.get(key)
    if r is None:
        r = _RENDERERS[key] = _Renderer(int(size), look, postfx)
    return r


class Drawer:
    """Picklable factory of a render worker's drawer for one size and look (``_execution.render_pool``)."""

    def __init__(self, size: int, look: str):
        self.size, self.look = int(size), look

    def __call__(self):
        r = renderer(self.size, self.look)

        def draw(rows, frames):
            r.draw(np.asarray(rows, np.float64), frames)

        return draw


def render_rows(
    rows: torch.Tensor, size: int, device, out: torch.Tensor | None = None, look: str = "wm"
) -> torch.Tensor:
    """Render ``(n, row_dim)`` float32 rows to ``(n, size, size, 3)`` uint8 frames on ``device``.

    A batch large enough for the ``roadster-render`` tuning path draws in render worker processes, each
    with its own renderer; a frame depends only on its row, so the split does not change the frames.
    """
    from ..._execution import render_pool

    size = int(size)
    host = rows.detach().to("cpu", torch.float64).numpy()
    n = host.shape[0]
    workers = render_pool.pool(
        ("roadster", size, look),
        Drawer(size, look),
        n,
        "roadster-render",
        size,
        software_gl=True,
        name="Roadster render",
    )
    if workers is not None:
        with workers.lock:
            frames = workers.render(n, (size, size, 3), lambda part: host[part])
    else:
        frames = np.empty((n, size, size, 3), np.uint8)
        renderer(size, look).draw(host, frames)
    result = torch.from_numpy(frames)
    if out is not None:
        if tuple(out.shape) != tuple(result.shape) or out.dtype != torch.uint8:
            raise ValueError(
                f"out must be a uint8 buffer of shape {tuple(result.shape)}, got {tuple(out.shape)} {out.dtype}"
            )
        out.copy_(result.to(out.device))
        return out
    return result.to(device)
