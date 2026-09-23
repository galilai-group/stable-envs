"""Record pinned Fetch sources, symbols, task configuration and full State schemas."""

import hashlib
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import stable_worldmodel
from gymnasium_robotics.envs import fetch as upstream_fetch
from gymnasium_robotics.envs import robot_env
from gymnasium_robotics.envs.fetch import fetch_env
from gymnasium_robotics.utils import mujoco_utils
from stable_worldmodel.envs.gymnasium_robotics.fetch import FetchWrapper

import stable_envs.envs.fetch as fetch


def main():
    root = Path(stable_worldmodel.__file__).resolve().parents[1]
    pin = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert pin == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
    assert not subprocess.check_output(["git", "-C", str(root), "diff", "--name-only"], text=True)
    sources = {}

    def record(path):
        path = Path(path)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    record(root / "stable_worldmodel/envs/__init__.py")
    record(root / "stable_worldmodel/spaces.py")
    record(gymnasium_robotics.__file__)
    symbols = {}
    for obj in (
        FetchWrapper,
        robot_env.BaseRobotEnv,
        robot_env.MujocoRobotEnv,
        fetch_env.get_base_fetch_env,
        fetch_env.MujocoFetchEnv,
        mujoco_utils.ctrl_set_action,
        mujoco_utils.mocap_set_action,
        mujoco_utils.reset_mocap2body_xpos,
        gym.wrappers.TimeLimit,
    ):
        record(inspect.getfile(obj))
        symbols[f"{obj.__module__}.{obj.__name__}"] = dict(path=inspect.getfile(obj), source=inspect.getsource(obj))
    tasks = {}
    for task in ("PickAndPlace", "Reach", "Push", "Slide"):
        with getattr(fetch, f"Fetch{task}")() as env:
            u = env._simulator.envs[0].unwrapped
            record(inspect.getfile(type(u)))
            record(u.fullpath)
            cls = type(u)
            symbols[f"{cls.__module__}.{cls.__name__}.__init__"] = dict(
                path=inspect.getfile(cls), source=inspect.getsource(cls.__init__)
            )
            tasks[task] = dict(
                upstream=f"{cls.__module__}.{cls.__name__}",
                xml=u.fullpath,
                configuration={
                    key: getattr(u, key).tolist() if hasattr(getattr(u, key), "tolist") else getattr(u, key)
                    for key in (
                        "has_object",
                        "block_gripper",
                        "gripper_extra_height",
                        "target_in_the_air",
                        "target_offset",
                        "obj_range",
                        "target_range",
                        "distance_threshold",
                        "n_substeps",
                    )
                },
                state_schema={
                    ".".join(k) if isinstance(k, tuple) else k: dict(width=w, dtype=str(t))
                    for k, (w, t) in env._simulator.fields.items()
                },
                default_mass=env._simulator._mass.tolist(),
                reset_keys=sorted(env._reset_keys),
                private_render_width=env.state_dim,
                registrations={
                    suffix or "Sparse": dict(
                        swm=gym.spec(f"swm/Fetch{task}{suffix}-v3").kwargs,
                        upstream=gym.spec(f"Fetch{task}{'Dense' if suffix == 'Dense' else ''}-v4").kwargs,
                        max_episode_steps=gym.spec(f"Fetch{task}-v4").max_episode_steps,
                    )
                    for suffix in ("", "Dense", "Dict")
                },
            )
    assets = Path(upstream_fetch.__file__).parents[1] / "assets"
    for path in (assets / "fetch").rglob("*.xml"):
        record(path)
    for path in (assets / "stls" / "fetch").rglob("*.stl"):
        record(path)
    for path in Path("stable_envs").rglob("*.py"):
        if path.name in ("fetch_pickandplace.py", "_fetch_semantics.py", "fetch.py"):
            assert (
                "import stable_worldmodel" not in path.read_text() and "from stable_worldmodel" not in path.read_text()
            )
    output = dict(
        swm_pin=pin,
        versions={
            k: importlib.metadata.version(k)
            for k in ("gymnasium", "gymnasium-robotics", "mujoco", "numpy", "torch", "tensordict")
        },
        source_sha256=sources,
        symbols=symbols,
        tasks=tasks,
        historical_observations=dict(
            flat="float32 concat(upstream observation, desired_goal)",
            dict="upstream observation/achieved_goal/desired_goal container when flatten=False",
            info_state="learner-visible flat observation, never exhaustive backend State",
            integration="DEFERRED",
            owner="stable-worldmodel",
        ),
        restore_order="qpos/qvel/goal/time/reset anchor; zero external forces; mj_forward; warmstart; gripper position/quaternion cache; TimeLimit; complete PCG64",
        no_production_swm_import=True,
    )
    Path("docs/research/semantic_recovery_evidence/fetch_family/provenance.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print("FETCH_FAMILY_SOURCE_PROVENANCE_PASS")


if __name__ == "__main__":
    main()
