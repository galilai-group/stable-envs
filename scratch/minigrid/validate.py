"""Pixel-exactness of the GPU renderer vs the official MiniGrid renderer.

Three separate claims, tested separately:

  1. state sufficiency -- ``(grid.encode(), agent_pos, agent_dir, carrying)`` is enough:
     decode it into a *fresh* env instance and the official renderer produces the same
     pixels as the env the state came from;
  2. full-view exactness -- GPU frame == ``get_full_render`` byte for byte, with and
     without the field-of-view highlight;
  3. POV exactness -- GPU frame == ``get_pov_render`` byte for byte.

Rollouts are random-action so that doors get toggled, objects get picked up and
dropped, and the agent ends up on goals/lava -- i.e. the tile states that only ever
show up mid-episode are actually exercised.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import minigrid  # noqa: E402,F401

from mg_gpu import MiniGridGPU, batch_states, env_state, restore_state  # noqa: E402


def rollout_states(env_id, n_ep, ep_len, seed0, check_sufficiency=True, tile=32):
    env = gym.make(env_id)
    fresh = gym.make(env_id) if check_sufficiency else None
    states, frames_full, frames_nohl, frames_pov = [], [], [], []
    suff_bad = 0
    for e in range(n_ep):
        env.reset(seed=seed0 + e)
        u = env.unwrapped
        rng = np.random.default_rng(seed0 + e)
        for _ in range(ep_len):
            st = env_state(env)
            states.append(st)
            frames_full.append(u.get_full_render(True, tile))
            frames_nohl.append(u.get_full_render(False, tile))
            frames_pov.append(u.get_pov_render(tile))
            if check_sufficiency:
                fresh.reset(seed=99991 + e)
                restore_state(fresh, st)
                f = fresh.unwrapped
                suff_bad += int(not np.array_equal(f.get_full_render(True, tile),
                                                   frames_full[-1]))
                suff_bad += int(not np.array_equal(f.get_pov_render(tile),
                                                   frames_pov[-1]))
            _, _, term, trunc, _ = env.step(int(rng.integers(0, 7)))
            if term or trunc:
                env.reset(seed=seed0 + 10000 + e)
    env.close()
    if fresh is not None:
        fresh.close()
    return states, frames_full, frames_nohl, frames_pov, suff_bad


def check_env(gpu, env_id, n_ep, ep_len, seed0, device, tile=32):
    states, f_full, f_nohl, f_pov, suff_bad = rollout_states(env_id, n_ep, ep_len, seed0, tile=tile)
    code, pos, dirs, car = batch_states(states, device)
    V = states[0]["view_size"]
    stw = states[0]["see_through_walls"]

    res = {"env_id": env_id, "n_frames": len(states),
           "grid": [int(code.shape[1]), int(code.shape[2])],
           "see_through_walls": stw, "view_size": V,
           "state_sufficiency_mismatches": suff_bad}

    # the bitmask triton scan must agree with the literal torch transcription of
    # Grid.process_vis, which is what the pixel test above actually validated
    view = gpu.gather_view(code, pos, dirs, V)
    res["vis_triton_matches_torch"] = bool(
        (gpu.vis(view, V) == gpu.process_vis(view, V)).all())

    for name, ref, idx in (
        ("full_hl", f_full, gpu.full_indices(code, pos, dirs, V, stw, True)),
        ("full_nohl", f_nohl, gpu.full_indices(code, pos, dirs, V, stw, False)),
        ("pov", f_pov, gpu.pov_indices(code, pos, dirs, car, V, stw)),
    ):
        ref_t = torch.from_numpy(np.stack(ref)).to(device)
        got_torch = gpu.blit_torch(idx)
        got_triton = gpu.blit_triton(idx)
        eq_t = (got_torch == ref_t)
        res[name] = {
            "exact_frames": int(eq_t.reshape(len(states), -1).all(1).sum()),
            "pixel_match_frac": float(eq_t.double().mean()),
            "max_abs_diff": int((got_torch.int() - ref_t.int()).abs().max()),
            "triton_matches_torch": bool((got_triton == got_torch).all()),
        }
    return res


ENV_LISTS = {
    "core": [
        "MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-Random-6x6-v0",
        "MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-MultiRoom-N4-S5-v0", "MiniGrid-Fetch-8x8-N3-v0",
        "MiniGrid-GoToDoor-8x8-v0", "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-PutNear-8x8-N3-v0", "MiniGrid-RedBlueDoors-8x8-v0",
        "MiniGrid-MemoryS13-v0", "MiniGrid-LockedRoom-v0",
        "MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-Unlock-v0",
        "MiniGrid-UnlockPickup-v0", "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-ObstructedMaze-2Dlh-v0", "MiniGrid-ObstructedMaze-Full-v1",
        "MiniGrid-DistShift1-v0", "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaCrossingS9N2-v0", "MiniGrid-SimpleCrossingS9N2-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-Playground-v0",
    ],
    "babyai": [
        "BabyAI-GoToRedBallGrey-v0", "BabyAI-GoToObj-v0", "BabyAI-GoToLocal-v0",
        "BabyAI-PickupLoc-v0", "BabyAI-PutNextLocal-v0", "BabyAI-OpenDoor-v0",
        "BabyAI-UnblockPickup-v0", "BabyAI-ActionObjDoor-v0",
        "BabyAI-BossLevel-v0", "BabyAI-SynthSeq-v0", "BabyAI-MiniBossLevel-v0",
        "BabyAI-KeyCorridor-v0", "BabyAI-OneRoomS20-v0", "BabyAI-FindObjS7-v0",
        "BabyAI-GoToImpUnlock-v0", "BabyAI-UnlockPickupDist-v0",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="core")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    z = np.load(os.path.join(here, f"atlas_{args.tile}.npz"))
    gpu = MiniGridGPU(z["atlas"], args.tile, device="cuda")

    envs = ENV_LISTS[args.list] if args.list in ENV_LISTS else args.list.split(",")
    out = []
    for eid in envs:
        try:
            r = check_env(gpu, eid, args.episodes, args.steps, 1234, "cuda", tile=args.tile)
        except Exception as ex:  # noqa: BLE001
            r = {"env_id": eid, "error": f"{type(ex).__name__}: {ex}"}
        out.append(r)
        if "error" in r:
            print(f"{eid:42s} ERROR {r['error']}", flush=True)
        else:
            print(f"{eid:42s} n={r['n_frames']:4d} grid={r['grid']} "
                  f"suff_bad={r['state_sufficiency_mismatches']} "
                  f"full_hl={r['full_hl']['exact_frames']}/{r['n_frames']} "
                  f"full_nohl={r['full_nohl']['exact_frames']}/{r['n_frames']} "
                  f"pov={r['pov']['exact_frames']}/{r['n_frames']} "
                  f"triton_ok={r['full_hl']['triton_matches_torch'] and r['pov']['triton_matches_torch']}",
                  flush=True)

    ok = [r for r in out if "error" not in r]
    tot = sum(r["n_frames"] for r in ok)
    agg = {
        "n_envs": len(out), "n_env_errors": len(out) - len(ok), "total_frames": tot,
        "state_sufficiency_mismatches": sum(r["state_sufficiency_mismatches"] for r in ok),
    }
    for k in ("full_hl", "full_nohl", "pov"):
        agg[k + "_exact_frames"] = sum(r[k]["exact_frames"] for r in ok)
        agg[k + "_max_abs_diff"] = max([r[k]["max_abs_diff"] for r in ok] or [0])
        agg[k + "_min_pixel_frac"] = min([r[k]["pixel_match_frac"] for r in ok] or [1.0])
    print("\nAGG " + json.dumps(agg))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"agg": agg, "per_env": out}, f, indent=2)


if __name__ == "__main__":
    main()
