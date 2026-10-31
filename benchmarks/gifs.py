"""Side-by-side GIFs of one episode per env: stable-envs | source | |diff|, with per-frame pixel statistics.

Envs with compact render rows: one stable-envs lane is reset with a seed and stepped with random actions; the rows of
every step are drawn by stable-envs on ``--device`` and by the source reference renderer (the frame-MSE path), so the
diff shows rendering only. Envs without compact rows: the stable-envs lane and one source env play the same seed and
actions (the online frame check), so the diff also shows any difference in the episode.

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


def _source_rows(name, size, rows):
    import stable_envs as snv

    cls = snv.registry.REGISTRY[name]
    ref = cls.reference_env(size=size)
    if "return_flags" in inspect.signature(cls.reference_render).parameters:
        frames, flags = cls.reference_render(ref, rows, return_flags=True)
    else:
        frames, flags = cls.reference_render(ref, rows), np.zeros(len(rows), bool)
    return np.asarray(frames), np.asarray(flags, bool)


def _in_spawn(fn, *args):
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(fn, args)


def _actions(name, steps, seed):
    import stable_envs as snv

    env = snv.make(name, num_envs=1, device="cpu")
    try:
        acts = [np.asarray(env.action_spec.sample(1, np.random.default_rng(seed + t)))[0] for t in range(steps)]
        return np.stack(acts).reshape(steps, -1)
    finally:
        env.close()


def episode(name, size, device, steps, seed):
    """Return (ours, theirs, unreachable flags, how) for one episode."""
    import stable_envs as snv
    from benchmarks.online_ladder import _ours_frames, _replay_frames

    actions = _actions(name, steps, seed)
    cls = snv.registry.REGISTRY[name]
    if int(getattr(cls, "state_dim", 0) or 0) > 0:
        env = snv.make(name, num_envs=1, size=size, device=device)
        try:
            env.reset(seed=seed)
            rows = [env._pack_render_state(env.get_state())]
            for a in actions[:-1]:
                env.step(torch.as_tensor(a[None], device=env.device), frames=None)
                rows.append(env._pack_render_state(env.get_state()))
            rows = torch.as_tensor(np.concatenate([np.asarray(torch.as_tensor(r).cpu()) for r in rows])).float()
            ours = env._render(rows.to(env.device)).cpu().numpy()
        finally:
            env.close()
        theirs, flags = _in_spawn(_source_rows, name, size, rows.numpy())
        return ours, theirs, flags, "same states"
    theirs, _ = _replay_frames(name, size, seed, actions)
    ours, _ = _ours_frames(name, size, device, seed, actions)
    return ours, theirs, np.zeros(len(ours), bool), "same seed and actions"


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


def write_gif(name, ours, theirs, flags, how, out: Path):
    ours, theirs = ours[..., :3], theirs[..., :3]
    keep = ~flags
    s = stats(ours, theirs)
    diff = np.abs(ours.astype(np.int16) - theirs.astype(np.int16)).max(-1)
    amp = np.clip(diff * 4, 0, 255).astype(np.uint8)[..., None].repeat(3, -1)
    h, w = ours.shape[1:3]
    frames = []
    for t in range(len(ours)):
        im = Image.new("RGB", (3 * w, h + 30), (0, 0, 0))
        im.paste(Image.fromarray(np.ascontiguousarray(ours[t])), (0, 30))
        im.paste(Image.fromarray(np.ascontiguousarray(theirs[t])), (w, 30))
        im.paste(Image.fromarray(amp[t]), (2 * w, 30))
        dr = ImageDraw.Draw(im)
        dr.text((4, 2), f"{name}  t={t}  stable-envs | source | |diff| x4  ({how})", fill=(255, 255, 255))
        note = " unreachable by source" if flags[t] else ""
        dr.text(
            (4, 15),
            f"MSE {s['mse'][t]:.1f}  RMSE {s['rmse'][t]:.1f}  pixels differing {s['pct_differ'][t]:.1f}%"
            f"  >16 levels {s['pct_over16'][t]:.1f}%{note}",
            fill=(255, 255, 0),
        )
        frames.append(im)
    frames[0].save(out / f"{name}.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
    summary = {k: float(v[keep].mean()) if keep.any() else None for k, v in s.items()}
    summary.update(
        env=name, how=how, frames=int(len(ours)), unreachable=int(flags.sum()), frame_shape=list(ours.shape[1:])
    )
    summary["mse_max"] = float(s["mse"][keep].max()) if keep.any() else None
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
    import stable_envs as snv

    for name in args.envs:
        cls = snv.registry.REGISTRY[name]
        size = int(getattr(cls, "fixed_size", None) or args.size)
        try:
            try:
                ours, theirs, flags, how = episode(name, size, args.device, args.steps, args.seed)
            except ValueError as e:
                if "renders" not in str(e) or "only" not in str(e):
                    raise
                size = cls.default_size
                ours, theirs, flags, how = episode(name, size, args.device, args.steps, args.seed)
            if ours.shape[1:3] != theirs.shape[1:3]:
                raise AssertionError(f"frame shapes differ: ours {ours.shape} source {theirs.shape}")
            n = min(len(ours), len(theirs))
            row = write_gif(name, ours[:n], theirs[:n], flags[:n], how, out)
            row["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            row = {"env": name, "status": "error", "error": f"{type(e).__name__}: {e}"[:400]}
        print(json.dumps(row), flush=True)
        with open(out / "summary.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
