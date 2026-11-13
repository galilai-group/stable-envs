import numpy as np
import pymunk


class ContactSegments(pymunk.SpaceDebugDrawOptions):
    def __init__(self):
        super().__init__()
        self.flags = self.DRAW_COLLISION_POINTS
        self.segments = []

    def draw_segment(self, a, b, color):
        self.segments.append([*a, *b, *color])


def contact_segments(space, collector=None):
    """Return the visible contact segments of ``space``; ``collector`` is a reusable ``ContactSegments``."""
    # A supported scene has at most ten shapes and Munk2D at most two contacts per shape pair.
    collector = ContactSegments() if collector is None else collector
    collector.segments = []
    space.debug_draw(collector)
    if len(collector.segments) > 90:
        raise RuntimeError("PushT visible contact cache exceeds the supported geometry bound")
    return collector.segments


def capture_contacts(space, collector=None):
    """Return padded contact segments and their count for immediate source rendering."""
    found = contact_segments(space, collector)
    count = len(found)
    segments = np.zeros((90, 8), np.float64)
    segments[:count] = np.asarray(found, np.float64).reshape(-1, 8)
    return segments, np.array(count, np.int64)
