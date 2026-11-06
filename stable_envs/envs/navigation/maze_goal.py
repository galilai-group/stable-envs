"""Matplotlib Agg images of the SimplePointMaze ``info.goal`` frame.

The source draws one figure per episode: background, walls, the goal disc, then the agent
disc, then the axes spines. Each distinct background is drawn once here. The agent disc is
then drawn over a copy of that background with ``draw_artist``. Agg draws the artists in
the same order onto the same pixels, so the bytes are the same as a full draw. Near the
frame edge the copy has no spines, and the spines are drawn again over the disc.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

# Pixels between the agent disc and the frame edge below which the spines can overlap the disc.
_EDGE_PX = 4
# Pixels copied around the disc bounds; antialiasing reaches at most one pixel past them.
_AA_PX = 3


def _imports():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, Rectangle
    except ImportError as exc:
        raise ImportError(
            "SimplePointMaze exact info.goal output requires matplotlib (validated with 3.10.9)"
        ) from exc
    return FigureCanvasAgg, Figure, Circle, Rectangle


def _figure(world, background, walls, wall_color, goal, goal_radius, goal_color):
    FigureCanvasAgg, Figure, Circle, Rectangle = _imports()
    fig = Figure(figsize=(5, 5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_xlim(0, world)
    ax.set_ylim(0, world)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(background)
    for position, shape in walls:
        ax.add_patch(Rectangle(position, shape[0], shape[1], facecolor=wall_color))
    ax.add_patch(Circle(goal, goal_radius, facecolor=goal_color, alpha=0.5))
    return fig, ax


def _rgb(fig) -> np.ndarray:
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]


class _Plate:
    """One drawn background and the figure that draws agent discs over it.

    ``region`` holds the whole background. ``bare`` holds it without the artists drawn after
    the patches (the spines), which are drawn again over a disc near the frame edge.
    """

    def __init__(self, world, background, walls, wall_color, goal, goal_radius, goal_color):
        self.fig, self.ax = _figure(world, background, walls, wall_color, goal, goal_radius, goal_color)
        self.fig.tight_layout(pad=0)
        canvas = self.fig.canvas
        canvas.draw()
        self.region = canvas.copy_from_bbox(self.fig.bbox)
        self.rgb = _rgb(self.fig).copy()
        patch_z = self.ax.patches[-1].get_zorder()
        children = [a for a in self.ax.get_children() if a is not self.ax.patch]
        self.above = [a for a in sorted(children, key=lambda a: a.get_zorder()) if a.get_zorder() > patch_z]
        shown = [a.get_visible() for a in self.above]
        for a in self.above:
            a.set_visible(False)
        canvas.draw()
        self.bare = canvas.copy_from_bbox(self.fig.bbox)
        for a, visible in zip(self.above, shown, strict=True):
            a.set_visible(visible)

    def _window(self, agent, radius):
        """Return the pixel rows and columns the disc can touch, or None near the spines."""
        h, w = self.rgb.shape[:2]
        (x0, y0), (x1, y1) = self.ax.transData.transform(
            [(agent[0] - radius, agent[1] - radius), (agent[0] + radius, agent[1] + radius)]
        )
        if min(x0, y0) < _EDGE_PX or x1 > w - _EDGE_PX or y1 > h - _EDGE_PX:
            return None
        rows = slice(max(0, int(np.floor(h - y1)) - _AA_PX), min(h, int(np.ceil(h - y0)) + _AA_PX))
        cols = slice(max(0, int(np.floor(x0)) - _AA_PX), min(w, int(np.ceil(x1)) + _AA_PX))
        return rows, cols

    def paint(self, agent, radius, color, out) -> None:
        """Write the frame with the agent disc into ``out``."""
        _, _, Circle, _ = _imports()
        window = self._window(agent, radius)
        canvas = self.fig.canvas
        canvas.restore_region(self.region if window is not None else self.bare)
        disc = Circle(agent, radius, facecolor=color)
        disc.set_figure(self.fig)
        disc.set_transform(self.ax.transData)
        disc.axes = self.ax
        disc.set_clip_path(self.ax.patch)
        self.ax.draw_artist(disc)
        if window is None:
            for artist in self.above:
                self.ax.draw_artist(artist)
            out[:] = _rgb(self.fig)
            return
        out[:] = self.rgb
        rows, cols = window
        out[rows, cols] = _rgb(self.fig)[rows, cols]


class GoalPainter:
    """Draw goal frames lane by lane and keep the most recent backgrounds."""

    def __init__(self, world: float, capacity: int = 8):
        self.world = float(world)
        self.capacity = int(capacity)
        self._plates: OrderedDict = OrderedDict()

    def _plate(self, key, args):
        plate = self._plates.get(key)
        if plate is None:
            plate = _Plate(self.world, *args)
            self._plates[key] = plate
            while len(self._plates) > self.capacity:
                self._plates.popitem(last=False)
        else:
            self._plates.move_to_end(key)
        return plate

    def paint(
        self,
        background: np.ndarray,
        number: np.ndarray,
        shapes: np.ndarray,
        positions: np.ndarray,
        wall_color: np.ndarray,
        goal: np.ndarray,
        goal_radius: np.ndarray,
        goal_color: np.ndarray,
        agent: np.ndarray,
        agent_radius: np.ndarray,
        agent_color: np.ndarray,
        out: np.ndarray,
    ) -> np.ndarray:
        """Write ``(B, H, W, 3)`` uint8 goal frames from per-lane NumPy rows into ``out``."""
        n = len(agent)
        for lane in range(n):
            count = int(number[lane])
            walls = list(zip(positions[lane, :count], shapes[lane, :count], strict=True))
            args = (
                background[lane],
                walls,
                wall_color[lane],
                goal[lane],
                float(goal_radius[lane]),
                goal_color[lane],
            )
            key = b"".join(
                np.ascontiguousarray(a).tobytes()
                for a in (
                    background[lane],
                    positions[lane, :count],
                    shapes[lane, :count],
                    wall_color[lane],
                    goal[lane],
                    goal_radius[lane],
                    goal_color[lane],
                )
            )
            self._plate(key, args).paint(agent[lane], float(agent_radius[lane]), agent_color[lane], out[lane])
        return out


class GoalDrawer:
    """Picklable factory of a render-worker drawer of goal frames."""

    def __init__(self, world: float):
        self.world = float(world)

    def __call__(self):
        painter = GoalPainter(self.world)

        def draw(rows, frames):
            painter.paint(**rows, out=frames)

        return draw


def goal_frames(painter: GoalPainter, rows: dict, size: int) -> np.ndarray:
    """Draw ``(B, size, size, 3)`` goal frames, in render worker processes when the batch is large."""
    from ..._execution import render_pool

    n = len(rows["agent"])
    shape = (size, size, 3)
    workers = render_pool.pool(
        ("SimplePointMaze.goal", painter.world),
        GoalDrawer(painter.world),
        n,
        "navigation",
        size,
        name="SimplePointMaze goal",
    )
    if workers is None:
        out = np.empty((n, *shape), np.uint8)
        painter.paint(**rows, out=out)
        return out
    with workers.lock:
        return workers.render(n, shape, lambda part: {key: value[part] for key, value in rows.items()})
