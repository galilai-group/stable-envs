import hashlib
import importlib.metadata as md
import json
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import mujoco
import numpy as np
import stable_worldmodel.envs  # noqa: F401
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

rows=[]
with patch.object(MujocoRenderer,'render',side_effect=AssertionError('No GL')):
    for task in ('Reach','Push','Slide','PickAndPlace'):
        for suffix in ('','Dense','Dict'):
            name=f'swm/Fetch{task}{suffix}-v3'
            for label,kwargs in [('default',{}),('output64',dict(resolution=64)),('internal_and_output64',dict(resolution=64,width=64,height=64))]:
                env=gym.make(name,**kwargs)
                try:
                    wrapper=env
                    while not hasattr(type(wrapper),'_flatten_obs'):
                        wrapper=wrapper.env
                    rows.append(dict(registration=name,configuration=label,output_size=wrapper.render_size,
                                     internal_width=env.unwrapped.width,internal_height=env.unwrapped.height))
                finally:env.close()
print(json.dumps(dict(mujoco=mujoco.__version__,numpy=np.__version__,rows=rows),indent=2))
dist=md.distribution('mujoco')
Path('/private/tmp/fetch-source-lifecycle-v1-20261003/dependency-install.json').write_text(json.dumps(dict(
    name='mujoco',version=dist.version,location=str(dist.locate_file('')),files={str(f):hashlib.sha256(dist.locate_file(f).read_bytes()).hexdigest()
    for f in dist.files if dist.locate_file(f).is_file() and str(f).endswith(('.py','.so','.dylib','METADATA'))}),indent=2)+'\n')
