"""A/B the mover-colour rules of docs/research/metaworld.md §6, ONE VARIANT PER PROCESS.

    python scratch/metaworld/diag_mover_colour.py door-close-v3 0   # mat_rgba alone
    python scratch/metaworld/diag_mover_colour.py door-close-v3 1   # mat_rgba x texture mean

One variant per process because flipping the flag in place means rebuilding the bundle, and the
raster cache holds a mujoco.Renderer bound to the OLD MjModel -- which renders the old model's
data through the new model's states and presents as the plate-constancy assertion failing at
255/255. (That is why _raster keys on the model's identity now.)
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL","egl")
sys.path.insert(0,"/mnt/vast/home/randall.balestriero/code/stable-envs")
import numpy as np
import stable_envs as se
from stable_envs.testing import conformance as C
import stable_envs.envs.metaworld as MWE

task=sys.argv[1]; flag=sys.argv[2]=="1"; size=int(sys.argv[3]) if len(sys.argv)>3 else 480
MWE.MOVER_TEXTURE_MEAN=flag
cls=se.REGISTRY[MWE._class_name(task)]
env=cls(batch=8,size=size,device="cuda")
states=env.sample_states(32,rng=np.random.default_rng(0))
official=cls.reference_render(C.reference_env(cls,size),states)
p=C.pixel_agreement(env.render(states).cpu().numpy(),official)
print("RESULT "+json.dumps({"task":task,"texture_mean":flag,
    **{k:(round(v,6) if isinstance(v,float) else v) for k,v in p.items()
       if k in ("exact_frac","mean_abs","interior_exact_frac","within8_frac")}}))
