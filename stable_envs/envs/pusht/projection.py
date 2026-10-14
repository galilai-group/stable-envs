import numpy as np
from pymunk.space_debug_draw_options import SpaceDebugColor

from ..._core.space_nodes import nodes
from .simulation.numeric import get_value
from .simulation.task import PushT, PushTDiscrete


def variation_values(row, prefix):
    result = {}
    base = prefix + "/variation/"
    for key in row:
        if key.startswith(base) and key.endswith("/current/value"):
            parts = key[len(base) :].removesuffix("/current/value").split("/")
            branch = result
            for name in parts[:-1]:
                branch = branch.setdefault(name, {})
            branch[parts[-1]] = get_value(row, key.removesuffix("/value"))
    return result


class PushTProjection:
    """Render numeric State through a separate CPU Pygame scene per lane.

    Geometry changes rebuild the scene; other frames install poses, colours and
    captured contact segments. Rendering never steps the physics owner or replays
    action history. CUDA output placement does not make this a GPU renderer.
    """

    def __init__(self):
        self.scenes = []

    def _scene(self, lane, row):
        # Only body shapes and scales are baked into the pymunk scene; poses,
        # colours and the goal pose are installed on every render.
        key = tuple(
            row[f"anchor/variation/{body}/{leaf}/current/{part}"].tobytes()
            for body in ("agent", "block")
            for leaf in ("shape", "scale")
            for part in ("value", "type")
        )
        key += (row["recipe/config"].tobytes(),)
        while len(self.scenes) <= lane:
            self.scenes.append(None)
        cached = self.scenes[lane]
        if cached is None or cached[0] != key:
            discrete, resolution, marker, target = row["recipe/config"]
            scene = (PushTDiscrete if discrete else PushT)(
                resolution=int(resolution), render_action=bool(marker), with_target=bool(target)
            )
            try:
                scene.reset(
                    seed=0,
                    options=dict(
                        variation=[],
                        variation_values=variation_values(row, "anchor"),
                        state=row["anchor/start"].copy(),
                        goal_state=get_value(row, "anchor/goal"),
                    ),
                )
            except BaseException:
                scene.close()
                raise
            if cached is not None:
                cached[1].close()
            leaves = [("owned/variation" + path + "/current", node) for path, node in nodes(scene.variation_space)]
            leaves = [(field, node) for field, node in leaves if not hasattr(node, "spaces")]
            self.scenes[lane] = [key, scene, leaves, [None] * len(leaves)]
        return self.scenes[lane]

    def render(self, rows, size):
        frames = []
        for lane, row in enumerate(rows):
            cached = self._scene(lane, row)
            scene, leaves, installed = cached[1], cached[2], cached[3]
            for i, (field, node) in enumerate(leaves):
                current = (row[field + "/value"].tobytes(), row[field + "/type"].tobytes())
                if current != installed[i]:
                    node.set_value(get_value(row, field))
                    installed[i] = current
            for name in ("agent", "block"):
                body = getattr(scene, name)
                body.angle = float(row[f"current/body/{name}/angle"])
                body.position = row[f"current/body/{name}/position"].tolist()
                scene.space.reindex_shapes_for_body(body)
            goal = "owned/variation/goal/"
            scene.goal_pose = np.concatenate(
                [get_value(row, goal + "position/current"), [get_value(row, goal + "angle/current")]]
            )
            scene.latest_action = get_value(row, "current/latest") if row["current/latest_present"] else None
            count = int(row["current/render_segment_count"])
            segments = row["current/render_segments"]
            if not 0 <= count <= len(segments) or np.any(segments[count:]):
                raise ValueError("Invalid PushT render contact cache")
            original = scene.space.debug_draw

            def draw(options):
                flags = options.flags
                options.flags = flags & ~options.DRAW_COLLISION_POINTS
                try:
                    original(options)
                finally:
                    options.flags = flags
                for segment in segments[:count]:
                    options.draw_segment(segment[:2], segment[2:4], SpaceDebugColor(*segment[4:]))

            scene.render_size = size
            scene.space.debug_draw = draw
            try:
                frames.append(scene.render())
            finally:
                scene.space.debug_draw = original
        return np.stack(frames)

    def close(self):
        for cached in self.scenes:
            if cached is not None:
                cached[1].close()
        self.scenes.clear()
