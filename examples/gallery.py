"""Media for every registered env: a state grid, a CPU/GPU side-by-side video, and variation sheets.

Per env, under ``--out/<env>/``:

  ``grid.png``            frames over the env's reset distribution at native resolution, unscaled.
  ``sidebyside.mp4``      the reference CPU renderer and the stable-envs renderer on the same recorded
                          states, frame for frame (H.264, with a GIF copy).
  ``variation_*.png``     the batched sheet holds the colour leaves measured to differ per environment
                          in one launch; the sweep sheet covers each leaf across separate calls.

Each stage is wrapped, so an env that fails one artifact still produces the others, with the reason in
``manifest.json``. Videos are encoded with ffmpeg as ``libx264 / yuv420p / baseline`` so browsers play
them, and the probed codec is recorded in the manifest.

    python examples/gallery.py --out docs/assets/media
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

import stable_envs as se

LABEL_H = 26
GAP = 4
MAX_SWEEP_PANELS = 18


def _tile(frames: np.ndarray, cols: int, pad: int = 2, pad_value: int = 235) -> np.ndarray:
    """(N,H,W,3) -> one image, row-major, with a light gutter so adjacent frames stay distinguishable."""
    n, h, w, _ = frames.shape
    rows = math.ceil(n / cols)
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), pad_value, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = f
    return canvas


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Add the caption in a band above the frame, shrinking the font until it fits."""
    band = np.full((LABEL_H, img.shape[1], 3), 245, np.uint8)
    scale = 0.42
    while scale > 0.18:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw <= img.shape[1] - 6:
            break
        scale -= 0.02
    cv2.putText(band, text, (3, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([band, img], axis=0)


def _upscale(frames: np.ndarray, k: int) -> np.ndarray:
    """Nearest-neighbour upscale, so no pixel is invented."""
    return frames if k <= 1 else np.repeat(np.repeat(frames, k, axis=-3), k, axis=-2)


def _even(vid: np.ndarray) -> np.ndarray:
    """yuv420p needs even dimensions; pad with the gutter colour rather than cropping a frame."""
    if vid.shape[1] % 2:
        vid = np.concatenate([vid, np.full((len(vid), 1, vid.shape[2], 3), 245, np.uint8)], 1)
    if vid.shape[2] % 2:
        vid = np.concatenate([vid, np.full((len(vid), vid.shape[1], 1, 3), 245, np.uint8)], 2)
    return vid


def _write_png(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img[:, :, ::-1])


# ---------------------------------------------------------------------------- video


def write_h264(path: Path, frames: np.ndarray, fps: int = 15) -> dict:
    """Encode ``(T,H,W,3)`` uint8 to a browser-playable H.264 mp4, and probe what actually landed."""
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    t, h, w, _ = frames.shape
    assert h % 2 == 0 and w % 2 == 0, f"yuv420p needs even dimensions, got {h}x{w}"
    cmd = [
        exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-movflags",
        "+faststart",
        "-crf",
        "18",
        str(path),
    ]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.decode()[-2000:]}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "frames": int(t),
        "fps": fps,
        "size": [int(h), int(w)],
        "codec": probe_codec(path),
    }


def probe_codec(path: Path) -> str:
    """Return the video stream description, from ffprobe if present, else from ffmpeg's report."""
    import imageio_ffmpeg

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        p = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,codec_tag_string",
                "-of",
                "default=nw=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if p.returncode == 0 and p.stdout.strip():
            return " ".join(p.stdout.split())
    p = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    )
    for line in p.stderr.splitlines():
        if "Stream #" in line and "Video:" in line:
            return line.strip()
    return "unknown"


def write_gif(path: Path, frames: np.ndarray, fps: int = 10, max_frames: int = 60) -> dict:
    """GIF fallback for contexts that will not run a video element (some markdown viewers)."""
    import imageio.v2 as imageio

    step = max(1, len(frames) // max_frames)
    sub = frames[::step]
    imageio.mimsave(path, list(sub), duration=1.0 / fps, loop=0)
    return {"path": str(path), "bytes": path.stat().st_size, "frames": int(len(sub))}


# ---------------------------------------------------------------------------- states


def rollout_states(cls, env, n: int) -> tuple[np.ndarray, str]:
    """Return ``n`` states and a label for how they were produced.

    Prefers a reference rollout, which works when the reference observation is the compact state.
    Otherwise continuous states are interpolated between samples and integer-coded states are sampled
    independently, since interpolating them would produce impossible states.
    """
    try:
        ref = cls.reference_env(size=env.size)
        obs, _ = ref.reset(seed=3)
        obs = np.asarray(obs, np.float64).reshape(-1)
        if obs.shape[0] < env.state_dim:
            raise ValueError(f"reference obs has {obs.shape[0]} dims, need {env.state_dim}")
        scripted = env.state_dim == 5 and "pusht" in (cls.reference_id or "").lower()
        goal = np.array([256.0, 256.0])
        states = []
        for _ in range(n):
            s = obs[: env.state_dim].copy()
            states.append(s)
            if scripted:  # aim behind the block, so the T is actually pushed
                block = s[2:4]
                away = block - goal
                action = np.clip(block + away / (np.linalg.norm(away) + 1e-6) * 45.0, 20, 492)
                action = action.astype(np.float32)
            else:
                action = ref.action_space.sample()
            obs, _, term, trunc, _ = ref.step(action)
            obs = np.asarray(obs, np.float64).reshape(-1)
            if term or trunc:
                obs, _ = ref.reset()
                obs = np.asarray(obs, np.float64).reshape(-1)
        ref.close()
        kind = "reference rollout, scripted policy" if scripted else "reference rollout, random actions"
        return np.stack(states).astype(np.float32), kind
    except Exception as e:
        print(f"  reference rollout unavailable ({type(e).__name__}: {e})", flush=True)

    way = env.sample_states(max(6, n // 20))
    if np.allclose(way, np.round(way)):
        return env.sample_states(n).astype(np.float32), "independent sampled states (integer-coded)"
    t = np.linspace(0, len(way) - 1, n)
    i0 = np.clip(t.astype(int), 0, len(way) - 2)
    f = (t - i0)[:, None]
    return (way[i0] * (1 - f) + way[i0 + 1] * f).astype(np.float32), "interpolated sampled states"


# ---------------------------------------------------------------------------- variation


def _leaf(space, path: str):
    node = space
    for part in path.split("."):
        node = node[part]
    return node


def per_env_leaves(env, states: np.ndarray, batch: int) -> tuple[dict, list[str]]:
    """Return the colour leaves that produce ``batch`` distinct frames from ``batch`` distinct values.

    A leaf can change the picture yet be read once per call or cached across the batch; such leaves
    are rejected.
    """
    rng = np.random.default_rng(0)
    one = np.repeat(states[:1], batch, axis=0)
    keep, rejected = {}, []
    for path in env.variation_space.names():
        node = _leaf(env.variation_space, path)
        shape = tuple(getattr(node, "shape", ()) or ())
        if shape != (3,):  # colours only: a per-env geometry sweep is not a sheet
            continue
        vals = rng.integers(0, 256, size=(batch, 3)).astype(np.float32)
        try:
            f = env.render(one, variation={path: vals}).cpu().numpy()
        except Exception as e:
            rejected.append(f"{path} ({type(e).__name__})")
            continue
        if len({f[i].tobytes() for i in range(batch)}) == batch:
            keep[path] = vals
        else:
            rejected.append(path)
    return keep, rejected


def sweep_panels(env, state: np.ndarray) -> list[np.ndarray]:
    """One labelled panel per (leaf, value), across the ends and middle of each leaf's range."""
    panels = []
    for path in env.variation_space.names():
        node = _leaf(env.variation_space, path)
        try:
            if hasattr(node, "n"):
                values = [(str(v), np.array(v)) for v in range(min(int(node.n), 3))]
            else:
                lo = np.broadcast_to(node.low, node.shape).astype(np.float64)
                hi = np.broadcast_to(node.high, node.shape).astype(np.float64)
                mid = np.asarray(node.value, np.float64)
                if lo.size > 3:  # a whole palette: ends only, and label it as such
                    picks = [("low", lo), ("default", mid), ("high", hi)]
                else:
                    picks = [
                        (np.array2string(lo.reshape(-1), precision=2, separator=","), lo),
                        (np.array2string(mid.reshape(-1), precision=2, separator=","), mid),
                        (np.array2string(hi.reshape(-1), precision=2, separator=","), hi),
                    ]
                values = [(t, v.astype(node.dtype)) for t, v in picks]
            for text, v in values:
                f = env.render(state[:1], variation={path: v}).cpu().numpy()[0]
                panels.append(_label(f, f"{path}={text}"))
        except Exception as e:
            print(f"  sweep skipped {path}: {type(e).__name__}: {e}", flush=True)
        if len(panels) >= MAX_SWEEP_PANELS:
            break
    return panels[:MAX_SWEEP_PANELS]


# ---------------------------------------------------------------------------- per env


def env_media(name: str, outdir: Path, video_frames: int, target_px: int) -> dict:
    cls = se.registry.REGISTRY[name]
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    man: dict = {"env": name, "notes": []}

    env = se.make(name, num_envs=1, device="cuda")
    size = env.size
    man["describe"] = env.describe()
    # Sheet geometry comes from a rendered frame: tile-based envs report their tile size as ``env.size``.
    probe = env.render(env.sample_states(1)).cpu().numpy()
    fh, fw = int(probe.shape[1]), int(probe.shape[2])
    man["frame_shape"] = [fh, fw]
    del env, probe

    span = max(fh, fw)
    grid_n = 32 if span <= 128 else (16 if span <= 256 else 8)
    cols = 8 if span <= 128 else (4 if span <= 256 else 2)
    video_scale = max(1, round(target_px / span))

    env = se.make(name, num_envs=grid_n, size=size, device="cuda")
    states = env.sample_states(grid_n)
    _write_png(d / "grid.png", _tile(env.render(states).cpu().numpy(), cols))
    man["grid"] = {"path": str(d / "grid.png"), "n": grid_n, "size": size}

    try:
        keep, rejected = per_env_leaves(env, states, grid_n)
        if keep:
            frames = env.render(np.repeat(states[:1], grid_n, axis=0), variation=keep).cpu().numpy()
            _write_png(d / "variation_batch.png", _tile(frames, cols))
            man["variation_batch"] = {
                "path": str(d / "variation_batch.png"),
                "leaves": sorted(keep),
                "batch": grid_n,
                "launches": 1,
            }
        else:
            man["notes"].append("no variation leaf varies per environment on this env")
        man["per_env_rejected"] = rejected
    except Exception as e:
        man["notes"].append(f"variation_batch failed: {type(e).__name__}: {e}")

    try:
        panels = sweep_panels(env, states)
        if panels:
            _write_png(d / "variation_sweep.png", _tile(np.stack(panels), min(6, len(panels))))
            man["variation_sweep"] = {"path": str(d / "variation_sweep.png"), "panels": len(panels)}
    except Exception as e:
        man["notes"].append(f"variation_sweep failed: {type(e).__name__}: {e}")

    traj, traj_kind = rollout_states(cls, env, video_frames)
    np.save(d / "states.npy", traj)
    man["states"] = {
        "path": str(d / "states.npy"),
        "bytes": (d / "states.npy").stat().st_size,
        "frames": int(len(traj)),
        "kind": traj_kind,
    }
    del env

    official = None
    if hasattr(cls, "reference_env") and getattr(cls, "reference_id", None) is not None:
        try:
            ref = cls.reference_env(size=size)
            official = cls.reference_render(ref, traj)
            ref.close()
        except Exception as e:
            man["notes"].append(f"reference render failed: {type(e).__name__}: {e}")

    genv = se.make(name, num_envs=len(traj), size=size, device="cuda")
    ours = genv.render(traj).cpu().numpy()
    del genv
    torch.cuda.empty_cache()

    if official is not None:
        from stable_envs.testing import conformance as C

        man["parity"] = C.pixel_agreement(ours, official)
        rows = []
        for i in np.linspace(0, len(traj) - 1, 4).astype(int):
            diff = np.abs(official[i].astype(np.int16) - ours[i].astype(np.int16)).max(-1)
            heat = np.stack([np.full_like(diff, 255), 255 - diff, 255 - diff], -1).astype(np.uint8)
            bar = np.full((fh + LABEL_H, GAP, 3), 245, np.uint8)
            rows.append(
                np.concatenate(
                    [
                        _label(official[i], f"official t={i}"),
                        bar,
                        _label(ours[i], "stable-envs"),
                        bar,
                        _label(heat, f"diff, max {int(diff.max())}"),
                    ],
                    axis=1,
                )
            )
        _write_png(d / "parity.png", np.concatenate(rows, axis=0))
        man["parity_panel"] = str(d / "parity.png")

        up_o, up_g = _upscale(official, video_scale), _upscale(ours, video_scale)
        sep = np.full((len(traj), up_o.shape[1] + LABEL_H, GAP, 3), 245, np.uint8)
        left = np.stack([_label(f, f"CPU official ({cls.reference_id})") for f in up_o])
        right = np.stack([_label(f, "GPU stable-envs, same states") for f in up_g])
        vid = _even(np.concatenate([left, sep, right], axis=2))
    else:
        up_g = _upscale(ours, video_scale)
        vid = _even(np.stack([_label(f, f"GPU stable-envs - {name}") for f in up_g]))

    man["video"] = write_h264(d / "sidebyside.mp4", vid)
    man["gif"] = write_gif(d / "sidebyside.gif", vid)
    print(f"  video codec: {man['video']['codec']}", flush=True)
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/media")
    ap.add_argument("--video-frames", type=int, default=120)
    ap.add_argument(
        "--target-px",
        type=int,
        default=288,
        help="video panel width to aim for; the upscale factor is the nearest integer",
    )
    ap.add_argument("--envs", default="", help="comma-separated subset; default is the whole registry")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Merge into the existing manifest so regenerating a subset with --envs keeps the other entries.
    manifest = {"envs": {}, "import_errors": se.import_errors()}
    if (out / "manifest.json").exists():
        try:
            manifest["envs"] = json.loads((out / "manifest.json").read_text()).get("envs", {})
        except Exception as e:
            print(f"existing manifest unreadable ({type(e).__name__}); starting fresh", flush=True)
    wanted = [n for n in sorted(se.registry.REGISTRY) if not a.envs or n in a.envs.split(",")]
    for name in wanted:
        print(f"[{name}]", flush=True)
        try:
            manifest["envs"][name] = env_media(name, out, a.video_frames, a.target_px)
        except Exception as e:
            traceback.print_exc()
            manifest["envs"][name] = {"env": name, "error": f"{type(e).__name__}: {e}"}
        torch.cuda.empty_cache()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
