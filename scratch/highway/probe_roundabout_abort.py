"""Is the Roundabout core dump state-dependent, or is it the kernel refusing to compile?

An out-of-range index would abort only for some states and would work on the torch backend and on the
Triton backend alike for a benign one. A compile-time abort does the opposite: it kills the process on
the FIRST Triton render, whatever the state, and never touches the torch path.

  python probe_roundabout_abort.py torch     -> renders a fixed state on the readable backend
  python probe_roundabout_abort.py triton    -> same fixed state, fused backend
  python probe_roundabout_abort.py zeros     -> the most benign state there is, fused backend
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import sys
import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.envs.highway import Roundabout

mode = sys.argv[1] if len(sys.argv) > 1 else "triton"
backend = "torch" if mode == "torch" else "auto"
env = Roundabout(batch=2, device="cuda", backend=backend)
st = np.zeros((2, env.state_dim), np.float32)
if mode != "zeros":                                  # a hand-written, entirely in-distribution state
    st[:, 0], st[:, 1], st[:, 2] = 22.0, 0.0, 1.5708
    st[:, 4], st[:, 5], st[:, 6] = 0.0, 22.0, 3.1416
print(f"backend={backend} primitives={env.scene(torch.as_tensor(st, device='cuda')).n_prims} "
      f"state={st[0][:8]}", flush=True)
f = env.render(st)
torch.cuda.synchronize()
print(f"OK  frame {tuple(f.shape)}  mean {f.float().mean():.2f}", flush=True)
