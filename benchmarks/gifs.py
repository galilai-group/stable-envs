"""Side-by-side GIFs of one episode per env, with per-frame pixel statistics.

The source env is reset with a seed and stepped with seeded random actions, and draws with its own render ("source
native"). One stable-envs lane plays the same seed and actions on ``--device``. For envs with compact render rows, the
bridge that the frame-MSE uses (``reference_render``) also loads each stable-envs state into the source and draws it
("source via bridge"): bridge vs native checks the bridge, ours vs bridge is the frame-MSE comparison. The diff panel
is stable-envs against source native.

Per frame: MSE (mean over pixels and channels of the squared difference, 0-255 scale), RMSE, the share of pixels that
differ in any channel, and the share that differ by more than 16 levels.

usage: python -m benchmarks.gifs --envs NAME ... --out DIR [--device cuda] [--steps 40]
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _source_rows(name, size, inputs):
    from benchmarks.offline_sweep import _snv

    snv = _snv()

    cls = snv.registry.REGISTRY[name]
    ref = cls.reference_env(size=size)
    if "return_flags" in inspect.signature(cls.reference_render).parameters:
        frames, flags = cls.reference_render(ref, inputs, return_flags=True)
    else:
        frames, flags = cls.reference_render(ref, inputs), np.zeros(len(inputs), bool)
    return np.asarray(frames), np.asarray(flags, bool)


def _in_spawn(fn, *args):
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(fn, args)


def _actions(name, steps, seed):
    from benchmarks.offline_sweep import _snv

    snv = _snv()

    env = snv.make(name, num_envs=1, device="cpu")
    try:
        acts = [np.asarray(env.action_spec.sample(1, np.random.default_rng(seed + t)))[0] for t in range(steps)]
        return np.stack(acts).reshape(steps, -1)
    finally:
        env.close()


def episode(name, size, device, steps, seed):
    """Frames after each of ``steps`` seeded random actions: stable-envs, the source via the bridge (or None), the
    source played natively with the same seed and actions, the bridge's unreachable flags, and a label."""
    from benchmarks.offline_sweep import _snv

    snv = _snv()
    from benchmarks.online_ladder import _replay_frames
    from tests.helpers.reference.certification import has_reference_renderer

    actions = _actions(name, steps, seed)
    native, end = _replay_frames(name, size, seed, actions)
    n = len(native) if end is None else end + 1
    bridge = has_reference_renderer(snv.registry.REGISTRY[name])
    with snv.make(name, num_envs=1, size=size, device=device) as env:
        env.reset(seed=seed)
        ours, inputs = [], []
        for a in actions:
            out = env.step(torch.as_tensor(a[None], device=env.device), frames=slice(None))
            ours.append(out["frames"][0, -1].cpu().numpy())
            if bridge:
                inputs.extend(env.reference_inputs(out["state"]))
    ours = np.stack(ours)
    if not bridge:
        return ours[:n], None, native[:n], np.zeros(n, bool), "same seed and actions"
    bridged, flags = _in_spawn(_source_rows, name, size, inputs if isinstance(inputs[0], dict) else np.stack(inputs))
    return ours[:n], bridged[:n], native[:n], flags[:n], "same seed and actions; bridge draws our States"


def stats(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d * d).reshape(len(d), -1).mean(1)
    m = np.abs(d).max(-1)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mean_abs": np.abs(d).reshape(len(d), -1).mean(1),
        "pct_differ": (m > 0).reshape(len(d), -1).mean(1) * 100,
        "pct_over16": (m > 16).reshape(len(d), -1).mean(1) * 100,
    }


def write_gif(name, ours, bridged, native, flags, how, out: Path):
    """Panels: stable-envs | source via bridge (when there is one) | source native | |stable-envs - source native| x4."""
    ours, native = ours[..., :3], native[..., :3]
    keep = ~flags
    s = stats(ours, native)
    b = stats(bridged[..., :3], native) if bridged is not None else None
    sb = stats(ours, bridged[..., :3]) if bridged is not None else None
    diff = np.abs(ours.astype(np.int16) - native.astype(np.int16)).max(-1)
    amp = np.clip(diff * 4, 0, 255).astype(np.uint8)[..., None].repeat(3, -1)
    panels = [ours] + ([bridged[..., :3]] if bridged is not None else []) + [native, amp]
    titles = (
        ["stable-envs"]
        + (["source via bridge"] if bridged is not None else [])
        + ["source native", "|ours - native| x4"]
    )
    h, w = ours.shape[1:3]
    frames = []
    for t in range(len(ours)):
        im = Image.new("RGB", (len(panels) * w, h + 44), (0, 0, 0))
        dr = ImageDraw.Draw(im)
        for i, (p, title) in enumerate(zip(panels, titles)):
            im.paste(Image.fromarray(np.ascontiguousarray(p[t])), (i * w, 44))
            dr.text((i * w + 4, 30), title, fill=(255, 255, 255))
        dr.text((4, 2), f"{name}  t={t}  ({how})", fill=(255, 255, 255))
        line = (
            f"ours vs native: MSE {s['mse'][t]:.1f} RMSE {s['rmse'][t]:.1f} differ {s['pct_differ'][t]:.1f}%"
            f" >16 {s['pct_over16'][t]:.1f}%"
        )
        if b is not None:
            line += f" | bridge vs native: MSE {b['mse'][t]:.1f} >16 {b['pct_over16'][t]:.1f}%"
        if flags[t]:
            line += " | unreachable by bridge"
        dr.text((4, 15), line, fill=(255, 255, 0))
        frames.append(im)
    frames[0].save(out / f"{name}.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
    summary = {f"native_{k}": float(v.mean()) for k, v in s.items()}
    summary["native_mse_max"] = float(s["mse"].max())
    if b is not None:
        summary.update({f"bridge_vs_native_{k}": float(v.mean()) for k, v in b.items()})
        summary.update({f"ours_vs_bridge_{k}": float(v[keep].mean()) if keep.any() else None for k, v in sb.items()})
    summary.update(
        env=name, how=how, frames=int(len(ours)), unreachable=int(flags.sum()), frame_shape=list(ours.shape[1:])
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from benchmarks.offline_sweep import _snv

    snv = _snv()

    for name in args.envs:
        cls = snv.registry.REGISTRY[name]
        size = getattr(cls, "native_size", None) or args.size
        try:
            ours, bridged, native, flags, how = episode(name, size, args.device, args.steps, args.seed)
            if ours.shape[1:3] != native.shape[1:3]:
                raise AssertionError(f"frame shapes differ: ours {ours.shape} source {native.shape}")
            n = min(len(ours), len(native))
            bridged = bridged[:n] if bridged is not None else None
            row = write_gif(name, ours[:n], bridged, native[:n], flags[:n], how, out)
            row["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            row = {"env": name, "status": "error", "error": f"{type(e).__name__}: {e}"[:400]}
        print(json.dumps(row), flush=True)
        with open(out / "summary.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
