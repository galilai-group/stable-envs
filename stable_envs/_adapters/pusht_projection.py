"""Source Pygame rendering from numeric State, without replaying its history.

Each lane retains one render-only scene. Geometry changes rebuild that scene;
ordinary frames install poses and numeric debug-draw contacts only. Physics
and rendering never share engines. This is a CPU renderer, including when
its output is explicitly copied to CUDA State storage.
"""

import numpy as np
from pymunk.space_debug_draw_options import SpaceDebugColor

from .pusht_numeric import get_value, nodes
from .pusht_task import PushT, PushTDiscrete


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
    def __init__(self):
        self.scenes = []

    def _scene(self, lane, row):
        # The reset recipe determines geometry and goal pose. Current colors
        # are installed below; RNG and action history never enter this key.
        keys = sorted(
            k
            for k in row
            if k.startswith("anchor/") and ("/current/" in k or k.startswith(("anchor/start", "anchor/goal/")))
        )
        key = tuple((k, row[k].dtype.str, row[k].shape, row[k].tobytes()) for k in keys)
        key += (("config", row["recipe/config"].tobytes()),)
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
            self.scenes[lane] = (key, scene)
        return self.scenes[lane][1]

    def render(self, rows, size):
        frames = []
        for lane, row in enumerate(rows):
            scene = self._scene(lane, row)
            for path, node in nodes(scene.variation_space):
                if not hasattr(node, "spaces"):
                    node.set_value(get_value(row, "owned/variation" + path + "/current"))
            for name in ("agent", "block"):
                body = getattr(scene, name)
                body.angle = float(row[f"current/body/{name}/angle"])
                body.position = row[f"current/body/{name}/position"].tolist()
                scene.space.reindex_shapes_for_body(body)
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
