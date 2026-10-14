import numpy as np
import pymunk


class ContactSegments(pymunk.SpaceDebugDrawOptions):
    def __init__(self):
        super().__init__()
        self.flags = self.DRAW_COLLISION_POINTS
        self.segments = []

    def draw_segment(self, a, b, color):
        self.segments.append([*a, *b, *color])


def capture_contacts(space):
    """Return padded contact segments and their count for immediate source rendering."""
    # A supported scene has at most ten shapes and Munk2D at most two contacts per shape pair.
    collector = ContactSegments()
    space.debug_draw(collector)
    count = len(collector.segments)
    if count > 90:
        raise RuntimeError("PushT visible contact cache exceeds the supported geometry bound")
    segments = np.zeros((90, 8), np.float64)
    segments[:count] = np.asarray(collector.segments, np.float64).reshape(-1, 8)
    return segments, np.array(count, np.int64)
