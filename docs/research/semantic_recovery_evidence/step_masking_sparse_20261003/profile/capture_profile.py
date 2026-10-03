import cProfile
import hashlib
import json
import platform
import pstats
from pathlib import Path
import torch
import stable_envs as se

torch.set_num_threads(1)
with se.make('SimplePointMaze', num_envs=64, device='cpu', seed=17) as env:
    initial = env.reset(seed=17)
    actions = torch.full((64, 8, 2), 0.01)
    env.step(actions, frames=None)
    env.set_state(initial)
    profile = cProfile.Profile()
    profile.enable()
    result = env.step(actions, frames=None)
    profile.disable()
    pstats.Stats(profile).strip_dirs().sort_stats('cumulative').print_stats(35)
    print(json.dumps(dict(host=platform.node(), valid_steps=int(result['valid'].sum()),
                         goal_shape=list(result['info', 'goal'].shape),
                         goal_bytes=result['info', 'goal'].numel() * result['info', 'goal'].element_size(),
                         source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             map(Path, ['stable_envs/_stepping.py', 'stable_envs/envs/navigation.py'])}), indent=2))
