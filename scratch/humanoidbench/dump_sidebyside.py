"""official | ours | 4x difference, through the SHIPPED class, so the tier can be judged by eye.

`docs/research/humanoidbench.md`'s images from `probe5_all_tasks.py` show the floor as it comes out of
MJWarp untouched -- dithered. What ships is the flattened floor, and that is a different picture, so it
is dumped here rather than described.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"


def main():
    task, robot = sys.argv[1], sys.argv[2]
    import imageio.v2 as iio

    import stable_envs as se
    cls = se.REGISTRY[f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}"
                      f"{''.join(p.capitalize() for p in task.split('_'))}"]
    env = cls(batch=4, device="cuda", seed=0)
    st = env.sample_states(4)
    ours = env.render(st).cpu().numpy()
    ref = cls.reference_render(cls.reference_env(env.size), st)
    strips = []
    for k in range(2):
        d = np.clip(np.abs(ours[k].astype(np.int16) - ref[k].astype(np.int16)) * 4,
                    0, 255).astype(np.uint8)
        strips.append(np.concatenate([ref[k], ours[k], d], axis=1))
    os.makedirs(f"{LOGS}/img_shipped", exist_ok=True)
    iio.imwrite(f"{LOGS}/img_shipped/{robot}_{task}.png", np.concatenate(strips, axis=0))
    print("wrote", f"{LOGS}/img_shipped/{robot}_{task}.png")


if __name__ == "__main__":
    main()
