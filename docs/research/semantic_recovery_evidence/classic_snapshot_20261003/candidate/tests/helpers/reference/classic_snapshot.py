"""Construct the frozen pre-snapshot adapter behind unchanged public renderers."""

from unittest.mock import patch

import stable_envs as se
from stable_envs._adapters import acrobot_process
from stable_envs.envs import _classic_semantics
from tests.helpers.reference.acrobot_before_snapshot import AcrobotProcessAdapter
from tests.helpers.reference.classic_before_snapshot import ClassicControlAdapter

NAMES = {"cartpole": "CartPole", "pendulum": "Pendulum", "acrobot": "Acrobot",
         "mountain_car": "MountainCar", "mountain_car_continuous": "MountainCarContinuous"}


def make(domain, batch, device, *, legacy=False):
    options = dict(num_envs=batch, device=device, seed=17)
    if not legacy:
        return se.make(NAMES[domain], **options)
    with (patch.object(_classic_semantics, "ClassicControlAdapter", ClassicControlAdapter),
          patch.object(acrobot_process, "AcrobotProcessAdapter", AcrobotProcessAdapter)):
        env = se.make(NAMES[domain], **options)
    env._advance_without_snapshot = env._step
    return env
