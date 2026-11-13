"""The Roadster simulator: a jet-powered car in MuJoCo with five planning tasks.

Modules:

  tasks       RoadsterPlanEnv: the goal-conditioned tasks with the 4-D autopilot action
  sim         RoadsterSim: raw drive / steer / nine-jet control, aerodynamics, course bookkeeping, effects
  control     Autopilot: the 50 Hz controller from the 4-D command to the raw action
  model       the procedural MJCF of the world (car, course, scenarios), with body_mesh, decor, scenarios,
              course and textures
  weather     wind and rain; fx: the exhaust smoke drawn from the exhaust history
  cinematic   the dark-space look and its post-processing
  expert      GoalExpert, the scripted expert
"""

from .expert import GoalExpert
from .tasks import HORIZON, TASKS, RoadsterPlanEnv

__all__ = ["HORIZON", "TASKS", "GoalExpert", "RoadsterPlanEnv"]
