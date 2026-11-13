"""Gymnasium-Robotics Fetch tasks: Reach, Push, Slide and PickAndPlace, each in three classes.

``Fetch<Task>`` has the sparse reward, ``Fetch<Task>Dense`` the dense one, and ``Fetch<Task>Dict`` the
sparse task under stable-worldmodel's name. The classes need the ``robotics`` extra.
"""

from .envs import *  # noqa: F403
from .envs import __all__  # noqa: F401
