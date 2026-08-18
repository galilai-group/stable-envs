"""Sanity-check the swap before spending a GPU-hour on the eval that depends on it.

Three frames per state, at the 224 px the LeWM checkpoint consumes:

  stored     the frame sitting in the official ``pusht_expert_train`` corpus (46 GB of pixels)
  reference  the same frame re-rendered by the live ``swm/PushT-v1`` env, driven to the stored state
  ours       ``stable_envs.make("PushT").render(state[:5])``

``stored`` vs ``reference`` measures how faithfully the official env can be *re-driven* from a stored
state; it is the ceiling any renderer swap can reach, and it has to be checked because ``_set_state``
runs a physics substep, so the env does not land exactly where it was asked to. ``reference`` vs ``ours``
is the swap itself. Reporting them separately keeps a state-restoration error from being mistaken for a
rendering error — the two look identical in an aggregate.

    python parity_probe.py --n 64 --out results/parity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

H5 = "/mnt/vast/home/randall.balestriero/.stable_worldmodel/datasets/pusht_expert_train.h5"


def load_rows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` random (state, stored_frame) pairs from the official corpus."""
    import h5py
    import hdf5plugin  # noqa: F401  -- the pixels column is blosc-compressed; without this the
    #                                   read fails with "can't open directory (/usr/local/lib/plugin)"
    with h5py.File(H5, "r") as f:
        total = f["state"].shape[0]
        rows = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
        states = f["state"][rows]                      # (n,7) float32
        pixels = f["pixels"][rows]                     # (n,224,224,3) uint8
    return states, pixels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    from stable_envs.testing.conformance import pixel_agreement
    from swap_renderer import _Renderer, _variation_from_env

    states, stored = load_rows(a.n, a.seed)
    print(f"[probe] {a.n} rows from {H5}", flush=True)

    # -- reference: drive the live swm env to each stored state and render it
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)

    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=a.size)
    env.reset(seed=0)
    u = env.unwrapped
    ref, ref_state = [], []
    for st in states:
        u._set_state(st)
        ref.append(u.render().copy())
        ref_state.append(u._get_obs())
    ref = np.stack(ref)
    ref_state = np.stack(ref_state)
    var = _variation_from_env(u)

    # ``_set_state`` ends with ``space.step(dt)``, so the env lands a physics substep away from the
    # state it was handed. Quantify it rather than assume it is negligible: if it is not, every frame
    # below is being compared at two different poses and no agreement number means anything.
    drift = np.abs(ref_state[:, :5] - states[:, :5])
    print(f"[probe] _set_state drift  max_pos={drift[:, :4].max():.4f} px  "
          f"max_ang={drift[:, 4].max():.6f} rad", flush=True)

    out: dict = {"n": a.n, "size": a.size, "seed": a.seed,
                 "set_state_drift_max_pos_px": float(drift[:, :4].max()),
                 "set_state_drift_max_angle_rad": float(drift[:, 4].max())}

    # -- ours, two resampling policies, at BOTH poses. Comparing against ``ref`` means rendering the
    # pose the env actually reached; comparing against ``stored`` means rendering the pose the corpus
    # recorded. Using one render for both would fold the physics drift into the rendering number.
    renders, renders_stored_pose = {}, {}
    for mode in ("direct", "resize512"):
        r = _Renderer(size=a.size, mode=mode)
        renders[mode] = r(ref_state[:, :5], var)
        renders_stored_pose[mode] = r(states[:, :5], var)

    # Distinct colours per frame. A flat-shaded vector scene has a handful; hundreds means the bytes
    # went through something lossy on their way to disk, which is the remaining candidate explanation
    # for the corpus being blurrier than any resampling of the live renderer.
    def n_colors(x):
        return float(np.mean([len(np.unique(f.reshape(-1, 3), axis=0)) for f in x[:8]]))

    out["colors_per_frame"] = {"stored": n_colors(stored), "reference": n_colors(ref),
                               "ours_direct": n_colors(renders["direct"])}
    print(f"[probe] distinct colours/frame  stored={out['colors_per_frame']['stored']:.1f}  "
          f"reference={out['colors_per_frame']['reference']:.1f}  "
          f"ours={out['colors_per_frame']['ours_direct']:.1f}", flush=True)

    pairs = {
        "stored_vs_reference": (stored, ref),
        "ours_direct_vs_reference": (renders["direct"], ref),
        "ours_resize512_vs_reference": (renders["resize512"], ref),
        "ours_direct_vs_stored": (renders_stored_pose["direct"], stored),
        "ours_resize512_vs_stored": (renders_stored_pose["resize512"], stored),
    }
    for name, (x, y) in pairs.items():
        m = pixel_agreement(x, y)
        out[name] = m
        print(f"[probe] {name:32s} exact={m['exact_frac']:.4f}  "
              f"interior_exact={m['interior_exact_frac']:.4f}  "
              f"mean_abs={m['mean_abs']:.4f}  max_abs={m['max_abs']}", flush=True)

    if a.png:
        import imageio.v2 as imageio
        k = min(4, a.n)
        strip = np.concatenate([
            np.concatenate([stored[i], ref[i], renders["direct"][i], renders["resize512"][i]], axis=1)
            for i in range(k)], axis=0)
        Path(a.png).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(a.png, strip)
        print(f"[probe] wrote {a.png}  (columns: stored | reference | ours-direct | ours-resize512)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"[probe] wrote {a.out}")


if __name__ == "__main__":
    main()
