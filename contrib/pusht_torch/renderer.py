"""Batched torch renderer for the public PushT classes."""

from __future__ import annotations

import numpy as np
import torch

from stable_envs.envs.pusht.projection import PushTProjection
from stable_envs.envs.pusht.simulation.numeric import get_value

from . import display_list
from .cvresize import LinearResize
from .pgdraw import SpanRaster, line_width

CANVAS = 512


class TorchRenderer:
    """Render ``swm/PushT-v1`` / ``swm/PushT-Discrete-v1`` State on ``device``.

    Frames equal ``env.render(state)`` wherever the installed OpenCV resizes through its generic
    path (Linux x86-64 wheels). Action markers (``render_action=True``) are drawn with OpenCV on CPU.
    """

    def __init__(self, device="cuda", chunk=256):
        self.device = torch.device(device)
        self.chunk = chunk
        self.raster = SpanRaster(CANVAS, CANVAS, self.device)
        self._templates = {}
        self._resize = {}
        self._scenes = PushTProjection()

    def render(self, env, state=None, size=None):
        size = env.size if size is None else size
        fields = {k: v.cpu().numpy() for k, v in env._render_fields_of(state).items()}
        rows = [{k: v[i] for k, v in fields.items()} for i in range(len(fields["recipe/config"]))]
        return self.render_rows(rows, size)

    def render_rows(self, rows, size):
        frames = [self._render(rows[i : i + self.chunk], size) for i in range(0, len(rows), self.chunk)]
        return torch.cat(frames) if len(frames) > 1 else frames[0]

    def close(self):
        self._scenes.close()

    def _template(self, row):
        key = tuple(
            row[f"anchor/variation/{body}/{leaf}/current/{part}"].tobytes()
            for body in ("agent", "block")
            for leaf in ("shape", "scale")
            for part in ("value", "type")
        ) + (row["recipe/config"].tobytes(),)
        if key not in self._templates:
            self._templates[key] = display_list.Template(self._scenes._scene(0, row)[1])
        return key

    def _render(self, rows, size):
        B = len(rows)
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(self._template(row), []).append(i)
        built = [
            (np.asarray(idx), display_list.build(self._templates[key], [rows[i] for i in idx]))
            for key, idx in groups.items()
        ]
        nops = max(len(ops) for _, ops in built)
        colors = np.zeros((B, nops, 3), np.uint8)
        circles, lines, polys, parts, aalines = [], [], {}, [], []
        for idx, ops in built:
            for k, (kind, rgb, geom, on) in enumerate(ops):
                colors[idx, k] = rgb
                meta = np.stack([idx, np.full(len(idx), k)], 1)
                if kind == "circle":
                    circles.append(np.concatenate([meta, geom[0], geom[1][:, None]], 1)[on])
                elif kind == "poly":
                    polys.setdefault(geom.shape[1], []).append(
                        np.concatenate([meta, geom[..., 0], geom[..., 1]], 1)[on]
                    )
                elif kind == "line":
                    arr = np.concatenate([meta, geom[0], geom[1], geom[2][:, None]], 1)[on]
                    inside = ((arr[:, 2:6] >= 0) & (arr[:, 2:6] < CANVAS)).all(1)
                    lines.append(arr[inside])
                    for f, o, *p in arr[~inside].tolist():
                        parts += self.raster.spans((f, o), line_width(*p, CANVAS, CANVAS))
                elif kind == "aaline":
                    for j in np.nonzero(on)[0]:
                        aalines.append(
                            (
                                int(idx[j]),
                                k,
                                tuple(map(float, geom[0][j])),
                                tuple(map(float, geom[1][j])),
                                tuple(int(v) for v in rgb[j]),
                            )
                        )
        if circles and len(c := np.concatenate(circles)):
            parts += self.raster.circles(c[:, :2], c[:, 2:])
        for n, items in polys.items():
            if len(a := np.concatenate(items)):
                parts += self.raster.polys(a[:, :2], a[:, 2 : 2 + n], a[:, 2 + n :])
        if lines and len(ln := np.concatenate(lines)):
            parts += self.raster.lines(ln[:, :2], ln[:, 2:])
        canvas = self.raster.blend_aalines(self.raster.paint(colors, parts), aalines)
        if size not in self._resize:
            self._resize[size] = LinearResize(CANVAS, size, self.device)
        frames = self._resize[size](canvas)
        return self._markers(frames, rows, size)

    @staticmethod
    def _markers(frames, rows, size):
        marked = [i for i, r in enumerate(rows) if r["recipe/config"][2] and r["current/latest_present"]]
        if not marked:
            return frames
        import cv2

        for i in marked:
            img = frames[i].cpu().numpy().copy()
            action = np.array(get_value(rows[i], "current/latest"))
            cv2.drawMarker(
                img,
                (action / 512 * 96).astype(np.int32),
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=int(8 / 96 * size),
                thickness=int(1 / 96 * size),
            )
            frames[i] = torch.as_tensor(img, device=frames.device)
        return frames
