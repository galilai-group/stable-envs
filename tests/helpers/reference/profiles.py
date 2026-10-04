"""Small explicit source profiles; each is a regression subset, not promotion.

Family-specific acceptance witnesses remain required in the recovery inventory.
No production environment imports this module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Registration:
    registration: str
    name: str
    family: str
    domain: str
    width: int
    dtypes: tuple[str, ...]


PROFILES = {
    "native": (
        Registration("swm/CartPoleControl-v1", "CartPole", "native", "cartpole", 1, ("int64",)),
        Registration("swm/PendulumControl-v1", "Pendulum", "native", "pendulum", 1, ("float32", "float64")),
    ),
    "fetch": (Registration("swm/FetchReach-v3", "FetchReach", "fetch", "Reach", 4, ("float32", "float64")),),
    "rocket": (
        Registration("swm/PFRocketLanding-v0", "PyFlytRocketLanding", "rocket", "rocket", 7, ("float32", "float64")),
    ),
}
PINS = {
    "native": {"numpy": "2.5.3", "gymnasium": "1.3.0"},
    "fetch": {"numpy": "2.5.3", "gymnasium": "1.3.0", "gymnasium-robotics": "1.4.2", "mujoco": "3.11.0"},
    "rocket": {"numpy": "2.5.3", "gymnasium": "1.3.0", "PyFlyt": "0.29.0", "pybullet": "3.2.7"},
}
# Every profile requires these functions for every declared registration, seed and
# action dtype. A -k/-m selection is useful development, but cannot pass a profile.
CONTRACTS = {
    "test_state.py::test_restore_and_ownership": "state",
    "test_simulation.py::test_independent_source_trajectory": "simulation",
    "test_task.py::test_independent_source_task_outputs": "task",
    "test_online.py::test_selected_states_follow_source": "online",
}
