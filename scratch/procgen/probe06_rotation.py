"""Measure the ROTATED `drawImage` path, the only thing standing between heist and bit-exactness.

`heist`'s agent calls `Entity::face_direction`, so its sprite is drawn through
`p.translate(); p.rotate(); p.drawImage()` rather than the axis-aligned fast path. With the
coordinate-coded assets in place, a crafted state per (rotation, position) reports exactly which
device pixels the rotated quad covered and which source pixel each of them sampled.

Writes `rot_samples.npz`: for each case, the 64x64 (sprite id, sx, sy) planes plus the state.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_io as S  # noqa: E402
import probe04_crafted as P4  # noqa: E402


def main():
    P4.build_tree()
    from procgen import ProcgenGym3Env
    env = ProcgenGym3Env(num=1, env_name="heist", distribution_mode="hard", rand_seed=0,
                         start_level=0, num_levels=1, use_backgrounds=False,
                         resource_root=P4.CODED + os.sep)
    env.act(np.array([4], np.int32))
    base = S.parse(env.get_state()[0])
    # a level with keys, so the rotated use_abs_coords ring keys draw too
    rng = np.random.default_rng(0)
    frames, meta = [], []
    rots = list(np.linspace(-np.pi, np.pi, 37)) + [0.0, np.pi / 2, -np.pi / 2, np.pi, 1e-7]
    for k, rot in enumerate(rots):
        st = S.parse(S.dump(base))
        st["grid"][:] = 100                        # clear the grid: only entities remain
        st["has_keys"] = [True] * len(st["has_keys"])
        for e in st["entities"]:
            if e["type"] == 0:
                e["rotation"] = float(rot)
                e["x"] = float(rng.uniform(1.0, st["main_width"] - 1))
                e["y"] = float(rng.uniform(1.0, st["main_height"] - 1))
            elif e["type"] in (1, 2, 9):
                e["will_erase"] = 0
                e["x"] = float(rng.uniform(1.0, st["main_width"] - 1))
                e["y"] = float(rng.uniform(1.0, st["main_height"] - 1))
        env.set_state([S.dump(st)])
        _, obs, _ = env.observe()
        frames.append(np.asarray(obs["rgb"][0]))
        meta.append({"rot": float(rot),
                     "ents": [{kk: e[kk] for kk in ("x", "y", "rx", "ry", "type", "image_type",
                                                    "image_theme", "render_z", "rotation",
                                                    "is_reflected", "use_abs_coords")}
                              for e in st["entities"]],
                     "main": [st["main_width"], st["main_height"]],
                     "has_keys": list(st["has_keys"])})
    np.savez_compressed(os.path.join(HERE, "rot_samples.npz"),
                        frames=np.stack(frames), meta=np.array(meta, dtype=object))
    ids, cnt = np.unique(np.stack(frames)[..., 2], return_counts=True)
    print(f"{len(frames)} cases; B histogram {dict(zip(ids.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main()
