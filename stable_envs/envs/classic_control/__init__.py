"""Gymnasium classic control: CartPole, Pendulum, Acrobot, MountainCar and MountainCarContinuous.

Frames match the source's ``pygame.gfxdraw`` drawing at its native size. CartPole and Acrobot need no extra;
MountainCar, MountainCarContinuous and Pendulum need pygame, which the ``classic`` extra installs.
"""

from .envs import Acrobot, CartPole, MountainCar, MountainCarContinuous, Pendulum

__all__ = ["Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum"]
