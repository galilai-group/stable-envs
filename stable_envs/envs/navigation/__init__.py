"""2-D point navigation tasks of stable-worldmodel: TwoRoom, Piecewise and SimplePointMaze.

State keeps the hidden reward target, so a restored state keeps the task.
"""

from .envs import Piecewise, PiecewiseZones, SimplePointMaze, TwoRoom, TwoRoomTarget

__all__ = ["Piecewise", "PiecewiseZones", "SimplePointMaze", "TwoRoom", "TwoRoomTarget"]
