"""Source profiles: the registrations, action dtypes and package pins of each contract profile."""

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
        Registration("swm/AcrobotControl-v1", "Acrobot", "native", "acrobot", 1, ("int64",)),
        Registration("swm/MountainCarControl-v0", "MountainCar", "native", "mountain_car", 1, ("int64",)),
        Registration(
            "swm/MountainCarContinuousControl-v0",
            "MountainCarContinuous",
            "native",
            "mountain_car_continuous",
            1,
            ("float32", "float64"),
        ),
    ),
    "fetch": tuple(
        Registration(f"swm/Fetch{task}{suffix}-v3", f"Fetch{task}{suffix}", "fetch", task, 4, ("float32", "float64"))
        for task in ("Reach", "Push", "Slide", "PickAndPlace")
        for suffix in ("", "Dense", "Dict")
    ),
    "dmc": tuple(
        Registration(f"swm/{source}DMControl-v0", name, "dmc", domain, width, ("float32", "float64"))
        for source, name, domain, width in (
            ("Acrobot", "DMCAcrobotBalance", "acrobot", 1),
            ("Pendulum", "DMCPendulumSwingup", "pendulum", 1),
            ("Cartpole", "DMCCartpoleSwingup", "cartpole", 1),
            ("Reacher", "DMCReacherHard", "reacher", 2),
            ("Cheetah", "DMCCheetahRun", "cheetah", 6),
            ("Hopper", "DMCHopperHop", "hopper", 4),
            ("Walker", "DMCWalkerWalk", "walker", 6),
            ("BallInCup", "DMCBallInCupCatch", "ball_in_cup", 2),
            ("Finger", "DMCFingerTurnHard", "finger", 2),
            ("Manipulator", "DMCManipulatorBringBall", "manipulator", 5),
            ("Quadruped", "DMCQuadrupedWalk", "quadruped", 12),
            ("Humanoid", "DMCHumanoidWalk", "humanoid", 21),
        )
    ),
    "navigation": tuple(
        Registration(registration, name, "navigation", domain, 2, ("float32", "float64"))
        for registration, name, domain in (
            ("swm/TwoRoom-v1", "TwoRoom", "two_room"),
            ("swm/Piecewise-v0", "Piecewise", "piecewise"),
            ("swm/SimplePointMaze-v0", "SimplePointMaze", "simple_point_maze"),
        )
    ),
    "ogbench": tuple(
        Registration(f"swm/OGB{domain}-v0", name, "ogbench", domain, width, ("float32", "float64"))
        for domain, name, width in (
            ("Maze", "PointMazeLargeNavigate", 2),
            ("Cube", "OGBCube", 5),
            ("Scene", "OGBScene", 5),
        )
    ),
    "pusht": (
        Registration("swm/PushT-v1", "PushT", "pusht", "continuous", 2, ("float32", "float64")),
        Registration("swm/PushT-Discrete-v1", "PushTDiscrete", "pusht", "discrete", 2, ("int64",)),
    ),
    "rocket": (
        Registration("swm/PFRocketLanding-v0", "PyFlytRocketLanding", "rocket", "rocket", 7, ("float32", "float64")),
    ),
}
PINS = {
    "object_permanence": {"numpy": "2.5.3", "mujoco": "3.12.0"},
    "dmc": {"numpy": "2.5.3", "gymnasium": "1.3.0", "dm-control": "1.0.45", "mujoco": "3.12.0"},
    "navigation": {"numpy": "2.5.3", "gymnasium": "1.3.0", "matplotlib": "3.10.9"},
    "ogbench": {
        "numpy": "2.2.6",
        "gymnasium": "1.2.3",
        "ogbench": "1.2.1",
        "mujoco": "3.12.0",
        "dm-control": "1.0.45",
    },
    "pusht": {"numpy": "2.2.6", "gymnasium": "1.2.3", "pymunk": "7.2.0", "pygame": "2.6.1"},
    "native": {"numpy": "2.5.3", "gymnasium": "1.3.0", "pygame": "2.6.1"},
    "fetch": {"numpy": "2.5.3", "gymnasium": "1.3.0", "gymnasium-robotics": "1.4.2", "mujoco": "3.11.0"},
    "rocket": {"numpy": "2.5.3", "gymnasium": "1.3.0", "PyFlyt": "0.29.0", "pybullet": "3.2.7"},
    "craftax": {"craftax": "1.6.1", "jax": "0.11.2", "jaxlib": "0.11.2"},
}
# Every profile requires these functions for every declared registration, seed and
# action dtype; a -k/-m selection cannot pass a profile.
CONTRACTS = {
    "test_contract_state.py::test_restore_and_ownership": "state",
    "test_simulation.py::test_independent_source_trajectory": "simulation",
    "test_task.py::test_independent_source_task_outputs": "task",
    "test_online.py::test_selected_states_follow_source": "online",
}
