"""Exactness over the whole encode alphabet, not just what registered envs happen to emit.

The env-rollout test in validate.py is the realistic one, but across all 172 ids it only
ever produced object types {1,2,4,5,6,7,8,9}: no Floor, no coloured Wall, no non-green
Goal, no coloured Lava. Those are all constructible and all render differently, so the
atlas has to be right for them too. Here grids are sampled uniformly from every
(type, colour, state) an ``encode()`` can produce, and the agent is dropped anywhere --
including on top of walls and lava, which a real episode never does.

Also asserts the aliasing the atlas relies on: types unseen/empty/agent all decode to a
"no object" cell and must therefore share a tile image.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import minigrid  # noqa: E402,F401
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX  # noqa: E402
from minigrid.core.grid import Grid  # noqa: E402

from calibrate import N_COLOR, atlas_index, make_obj  # noqa: E402
from mg_gpu import MiniGridGPU, batch_states  # noqa: E402

TILE = 32


def reachable_codes():
    out = []
    for name, t in OBJECT_TO_IDX.items():
        if name in ("unseen", "empty", "agent"):
            out.append((t, 0, 0))
        elif name == "door":
            out += [(t, c, s) for c in range(N_COLOR) for s in range(3)]
        elif name == "lava":
            out.append((t, COLOR_TO_IDX["red"], 0))
        else:
            out += [(t, c, 0) for c in range(N_COLOR)]
    return out


def check_alias(atlas):
    """unseen / empty / agent must all render as the bare 'no object' tile."""
    bad = 0
    for hl in range(2):
        for d in range(5):
            ref = atlas[atlas_index(hl, d, OBJECT_TO_IDX["empty"], 0, 0)]
            for t in (OBJECT_TO_IDX["unseen"], OBJECT_TO_IDX["agent"]):
                bad += int(not np.array_equal(atlas[atlas_index(hl, d, t, 0, 0)], ref))
    return bad


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    z = np.load(os.path.join(here, f"atlas_{TILE}.npz"))
    atlas = z["atlas"]
    gpu = MiniGridGPU(atlas, TILE, device="cuda")

    print("alias mismatches (unseen/agent vs empty):", check_alias(atlas))

    codes = reachable_codes()
    print("reachable encodings:", len(codes))

    rng = np.random.default_rng(0)
    env = gym.make("MiniGrid-Empty-16x16-v0")
    env.reset(seed=0)
    u = env.unwrapped

    for stw in (False, True):
        for W, H in ((9, 9), (13, 7), (16, 16)):
            states, ref_full, ref_pov = [], [], []
            for _ in range(48):
                sel = rng.integers(0, len(codes), size=(W, H))
                g = Grid(W, H)
                for i in range(W):
                    for j in range(H):
                        g.set(i, j, make_obj(*codes[sel[i, j]]))
                # feed the GPU exactly what the official pipeline would encode; the
                # unseen/agent type slots never survive a full-grid encode() and are
                # covered by check_alias instead
                enc = g.encode()
                ap = (int(rng.integers(0, W)), int(rng.integers(0, H)))
                ad = int(rng.integers(0, 4))
                ct = codes[int(rng.integers(0, len(codes)))]
                carrying = make_obj(*ct)

                u.grid, u.width, u.height = g, W, H
                u.agent_pos, u.agent_dir = ap, ad
                u.carrying = carrying
                u.see_through_walls = stw

                ref_full.append(u.get_full_render(True, TILE))
                ref_pov.append(u.get_pov_render(TILE))
                states.append({
                    "code": enc, "agent_pos": np.array(ap, np.int64), "agent_dir": ad,
                    "carrying": np.array(carrying.encode() if carrying else (1, 0, 0),
                                         np.uint8),
                    "view_size": 7, "see_through_walls": stw,
                })

            code, pos, dirs, car = batch_states(states, "cuda")
            n = len(states)
            gf = gpu.blit_triton(gpu.full_indices(code, pos, dirs, 7, stw, True))
            gp = gpu.blit_triton(gpu.pov_indices(code, pos, dirs, car, 7, stw))
            rf = torch.from_numpy(np.stack(ref_full)).cuda()
            rp = torch.from_numpy(np.stack(ref_pov)).cuda()
            ef = int((gf == rf).reshape(n, -1).all(1).sum())
            ep = int((gp == rp).reshape(n, -1).all(1).sum())
            print(f"stw={int(stw)} grid={W}x{H}  full {ef}/{n} exact "
                  f"(maxdiff {int((gf.int()-rf.int()).abs().max())})   "
                  f"pov {ep}/{n} exact "
                  f"(maxdiff {int((gp.int()-rp.int()).abs().max())})", flush=True)


if __name__ == "__main__":
    main()
