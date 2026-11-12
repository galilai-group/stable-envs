"""Gymnasium classic control: CartPole, Pendulum, Acrobot, MountainCar and MountainCarContinuous.

Frames match the source's ``pygame.gfxdraw`` drawing at its native size. The family needs no extra, but
MountainCar, MountainCarContinuous and Pendulum need pygame.
"""

from .envs import Acrobot, CartPole, MountainCar, MountainCarContinuous, Pendulum

__all__ = ["Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum"]
