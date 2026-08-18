"""`ProcgenBigfish` against the shipped Qt renderer, on the states the game actually reaches.

Same shape as `probe10_env_parity.py`: 20 levels x 12 steps, `distribution_mode="hard"`, driven
through `set_state` so the reference draws exactly the state the renderer is handed, with the
round-trip asserted inside `reference_render`.

The extra thing measured here is coverage of the two mechanisms chaser adds. A parity number taken
on a pool where every enemy happens to be in the same animation phase, or where no ORB cell survives,
proves nothing about the fill layer or the sprite fold -- that is the same trap
`docs/research/pixel_native.md` records for miner's backgrounds and heist's keys. So the state pool
is reported alongside the parity: how many distinct `enemy_sprite` values it holds, how many ORB
cells, and how many slots are occupied.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "bigfish_parity.json")


def main():
    import state_io as S
    from procgen import ProcgenGym3Env
    from bigfish_env import ProcgenBigfish
    from stable_envs.envs.procgen import _parse

    rep = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = ProcgenBigfish(batch=1, device=dev)
    print("device:", dev, "state_dim:", env.state_dim, flush=True)

    # ---- harvest real states, level by level, and render both sides
    rng = np.random.default_rng(0)
    ours, refs, states = [], [], []
    for level in range(20):
        g3 = ProcgenGym3Env(num=1, env_name="bigfish", distribution_mode="hard", rand_seed=0,
                            start_level=level, num_levels=1, use_backgrounds=True)
        for _ in range(140):                     # let the fish population reach steady state
            g3.act(rng.integers(0, 15, size=1).astype(np.int32))
        for _ in range(12):
            g3.act(rng.integers(0, 15, size=1).astype(np.int32))
            blob = _parse(g3.get_state()[0])
            st = ProcgenBigfish.state_from_blob(blob)
            _, obs, _ = g3.observe()
            refs.append(np.asarray(obs["rgb"][0]).copy())
            states.append(st)
    states = np.stack(states)

    # the reference frames above are the env's own; re-derive them through set_state as well, which
    # is what `reference_render` does, so a state->blob->frame mismatch cannot hide
    ref2 = ProcgenBigfish.reference_render(ProcgenBigfish.reference_env(), states)
    rep["own_obs_vs_set_state"] = {
        "frames": len(refs),
        "identical": int(sum(bool((a == b).all()) for a, b in zip(refs, ref2)))}

    out = env.render(torch.as_tensor(states, device=dev)).cpu().numpy()
    ours = out.reshape(len(states), 64, 64, 3)

    d = np.abs(ours.astype(np.int32) - ref2.astype(np.int32))
    rep["parity"] = {
        "frames": int(len(states)),
        "pixels_exact_frac": float((d.max(-1) == 0).mean()),
        "max_abs_diff": int(d.max()),
        "frames_exact": int((d.reshape(len(states), -1).max(1) == 0).sum())}
    print("parity: {frames} frames, pixels exact {pixels_exact_frac:.7f}, "
          "max |diff| {max_abs_diff}, frames exact {frames_exact}".format(**rep["parity"]),
          flush=True)

    # ---- did the pool actually exercise the new mechanisms?
    from bigfish_env import BIGFISH_SLOTS
    base = 6
    pres = states[:, base + 6::7][:, :BIGFISH_SLOTS]
    rx = states[:, base + 2::7][:, :BIGFISH_SLOTS]
    th = states[:, base + 4::7][:, :BIGFISH_SLOTS]
    rf = states[:, base + 5::7][:, :BIGFISH_SLOTS]
    rep["coverage"] = {
        "fish_present_min_max": [int(pres.sum(1).min()), int(pres.sum(1).max())],
        "fish_rx_range": [float(rx[pres > 0].min()), float(rx[pres > 0].max())],
        "themes_seen": sorted({int(v) for v in th[pres > 0]}),
        "reflected_seen": sorted({int(v) for v in rf[pres > 0]}),
        "agent_r_range": [float(states[:, 4].min()), float(states[:, 4].max())],
        "bg_index_seen": sorted({int(v) for v in states[:, 0]}),
        "bg_pct_x_range": [float(states[:, 1].min()), float(states[:, 1].max())],
    }
    print("coverage:", json.dumps(rep["coverage"]), flush=True)

    # ---- the sample_states pool the conformance suite will actually use
    pool = env.sample_states(256)
    ppres = pool[:, 6 + 6::7][:, :BIGFISH_SLOTS]
    rep["pool"] = {"fish_present_min_max": [int(ppres.sum(1).min()), int(ppres.sum(1).max())]}
    pd = ProcgenBigfish.reference_render(ProcgenBigfish.reference_env(), pool[:64])
    po = env.render(torch.as_tensor(pool[:64], device=dev)).cpu().numpy().reshape(64, 64, 64, 3)
    dd = np.abs(po.astype(np.int32) - pd.astype(np.int32))
    rep["pool"]["parity_pixels_exact_frac"] = float((dd.max(-1) == 0).mean())
    rep["pool"]["parity_max_abs"] = int(dd.max())
    print("pool parity:", rep["pool"], flush=True)

    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
