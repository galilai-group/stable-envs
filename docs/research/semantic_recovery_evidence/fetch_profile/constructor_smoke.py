"""Reference-only smoke: installed upstream and pinned SWM, no monkeypatches."""
import base64
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import subprocess
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import stable_worldmodel
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from gymnasium_robotics.utils import mujoco_utils

source = Path(stable_worldmodel.__file__).resolve().parents[1]
pin = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert pin == '176d54532b3c0a251a4f509b74f2fdacc143a63d'
assert not subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--'], text=True)
profile = {'python': platform.python_version(), **{p: metadata.version(p) for p in
           ('numpy', 'gymnasium', 'gymnasium-robotics', 'mujoco', 'torch', 'torchvision', 'tensordict')}}
results = {}
for label, factory in [('direct_upstream', lambda: MujocoFetchPickAndPlaceEnv(reward_type='sparse')),
                       ('pinned_swm', lambda: gym.make('swm/FetchPickAndPlace-v3'))]:
    env = factory()
    try:
        resets = []
        for seed in (0, 17, 42):
            observation, info = env.reset(seed=seed)
            u = env.unwrapped
            assert np.isfinite(u.data.qpos).all() and np.isfinite(u.data.qvel).all()
            assert np.isfinite(u.goal).all()
            resets.append({'seed': seed, 'qpos_sha256': hashlib.sha256(u.data.qpos.tobytes()).hexdigest(),
                           'qvel_sha256': hashlib.sha256(u.data.qvel.tobytes()).hexdigest(),
                           'goal': u.goal.tolist(), 'observation_kind': type(observation).__name__})
        slide_id = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, 'robot0:slide0')
        value = u.model.jnt_type[slide_id]
        assert value in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        results[label] = {'class': type(u).__name__, 'constructor': 'PASS', 'resets': resets,
                          'slide_joint_numpy_dtype': str(value.dtype), 'slide_joint_integer': int(value),
                          'numpy_scalar_equals_enum': bool(value == mujoco.mjtJoint.mjJNT_SLIDE),
                          'enum_equals_numpy_scalar': bool(mujoco.mjtJoint.mjJNT_SLIDE == value),
                          'enum_tuple_membership': True}
    finally:
        env.close()
# Verify the actual installed utility source against its wheel RECORD.
dist = metadata.distribution('gymnasium-robotics')
record = next(p for p in dist.files if str(p) == 'gymnasium_robotics/utils/mujoco_utils.py')
raw = Path(inspect.getfile(mujoco_utils)).read_bytes()
assert record.hash.mode == 'sha256'
assert base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=') == record.hash.value
print(json.dumps({'status': 'REFERENCE_PROFILE_SMOKE_PASS_NOT_CANDIDATE_ACCEPTANCE',
                  'swm_commit': pin, 'profile': profile, 'upstream_constraints': dist.requires,
                  'profile_selection': 'Four simulator package versions and torch/torchvision from pinned SWM uv.lock; existing Python interpreter',
                  'unmodified_mujoco_utils_wheel_record': True,
                  'mujoco_utils_sha256': hashlib.sha256(raw).hexdigest(), 'results': results}, indent=2))
