"""Write the metadata snapshots and rollout videos of the environment pages in docs/environments/."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "environments" / "_data"
MEDIA = ROOT / "docs" / "assets" / "media" / "envs"

#: Snapshot keys that a run measures on its machine; the snapshot test does not compare them.
MEASURED = ("devices", "media")

FPS = 15
ROLLOUT_STEPS = 60
GRID_STEPS = 45
HOLD_STEPS = 6
MAIN_MAX_SIDE = 224
TILE_MAX_SIDE = 112
MIN_RENDER_SIZE = 128
#: A JSON list of scalars, which the snapshot writes on one line.
_FLAT_LIST = re.compile(r"\[\s*([^\[\]{}]*?)\s*\]")
RENDER_SIZE = 160

#: Render backend that ``backend="auto"`` selects on each device, by family or renderer cell.
RENDERERS = {
    "classic_control": {"cpu": "gfxdraw port (torch)", "cuda": "gfxdraw port (fused Triton kernel)"},
    "craftax": {"cpu": "Craftax pixel renderer (JAX)", "cuda": "Craftax pixel renderer (JAX)"},
    "dmcontrol-suite": {"cpu": "raycast3d (torch)", "cuda": "raycast3d (torch)"},
    "dmcontrol-trackcom": {"cpu": "cpu3d", "cuda": "mjwarp"},
    "fetch": {"cpu": "opengl (MuJoCo)", "cuda": "mjwarp"},
    "miniworld": {"cpu": "opengl (Miniworld, pyglet)", "cuda": "opengl (Miniworld) on CPU, frames copied to CUDA"},
    "navigation": {"cpu": "raster2d (torch)", "cuda": "raster2d (torch) with the Triton dot compositor"},
    "ogbench-maze": {"cpu": "raycast3d (torch)", "cuda": "raycast3d (torch)"},
    "ogbench-manipulation": {"cpu": "opengl (MuJoCo)", "cuda": "mjwarp"},
    "pusht": {"cpu": "pygame (source renderer)", "cuda": "pygame on CPU, frames copied to CUDA"},
    "pyflyt": {"cpu": "tiny (TinyRenderer)", "cuda": "nvdiffrast"},
}

#: Physics placement declared by ``execution_mode``.
PHYSICS = {
    "DELEGATED_CPU": "source simulator on CPU",
    "NATIVE_TORCH": "native torch on the env device",
    "NATIVE_JAX": "native JAX on the env device",
    "AUTO_NATIVE_MJX": "native MuJoCo on CPU; CUDA batches use native below 32 lanes, JAX/MJX otherwise",
}


def renderer_cell(cls) -> str:
    """Key of :data:`RENDERERS` for ``cls``."""
    family = cls.family
    if family == "dmcontrol":
        from stable_envs.envs.dmcontrol.trackcom import DMControlTrackCom

        return "dmcontrol-trackcom" if issubclass(cls, DMControlTrackCom) else "dmcontrol-suite"
    if family == "ogbench":
        from stable_envs.envs.ogbench.manipulation_semantics import ManipulationSemantics

        return "ogbench-manipulation" if issubclass(cls, ManipulationSemantics) else "ogbench-maze"
    return family


def _num(value):
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == int(value) and abs(value) < 2**53:
        return int(value)
    return float(f"{value:.6g}")


def _values(value):
    """A scalar when every element is equal, otherwise a flat list."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    if np.all(arr == arr[0]) or (np.isnan(arr).all()):
        return _num(arr[0])
    return [_num(v) for v in arr]


def _dtype(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _spec(spec) -> dict:
    out = {"shape": list(spec.shape), "dtype": _dtype(spec.dtype)}
    if spec.discrete_n is not None:
        out["n"] = int(spec.discrete_n)
    if spec.low is not None:
        out["low"] = _values(spec.low)
    if spec.high is not None:
        out["high"] = _values(spec.high)
    if spec.names:
        out["names"] = list(spec.names)
    return out


def _state(env) -> list[dict]:
    from stable_envs.specs import CompositeSpec

    rows = []
    for key, node in env.state_spec.items():
        if isinstance(node, CompositeSpec):
            leaves = node.items(nested=True)
            rows.append(
                {
                    "key": key,
                    "leaves": len(leaves),
                    "numel": int(sum(s.numel for _, s in leaves)),
                    "dtypes": sorted({_dtype(s.dtype) for _, s in leaves}),
                }
            )
        else:
            rows.append({"key": key, "shape": list(node.shape), "dtype": _dtype(node.dtype)})
    return rows


def _variation(env) -> list[dict]:
    sampled = {leaf.path for leaf in env.reset_leaves if leaf.sampled}
    rows = []
    for path, _, _ in env.variation_space.flat_layout():
        node = env.variation_space._walk(path)
        row = {"path": path, "type": type(node).__name__, "reset": path in sampled}
        if hasattr(node, "low"):
            row["shape"] = list(node.shape)
            row["low"] = _values(node.low)
            row["high"] = _values(node.high)
        elif hasattr(node, "n"):
            row["n"] = int(node.n)
            row["start"] = int(getattr(node, "start", 0))
        default = getattr(node, "init_value", None)
        row["default"] = None if default is None else _values(default)
        rows.append(row)
    return rows


def metadata(name: str, device: str = "cpu") -> dict:
    """Snapshot of ``name`` built from its registry entry and a one-lane env on ``device``."""
    import stable_envs as snv

    info = snv.info(name)
    cls = info["class"]
    env = cls(num_envs=1, device=device)
    try:
        action = _spec(env.action_spec) if info["has_dynamics"] else None
        out = {
            "class_name": name,
            "module": info["module"],
            "family": info["family"],
            "domain": info["domain"],
            "reference_id": info["reference_id"],
            "requires_cuda": info["requires_cuda"],
            "has_dynamics": info["has_dynamics"],
            "default_size": info["default_size"],
            "frame_shape": list(env.frame_spec.shape),
            "physics": PHYSICS.get(getattr(cls, "execution_mode", ""), None),
            "renderers": RENDERERS[renderer_cell(cls)],
            "action": action,
            "state": _state(env),
            "variation": _variation(env),
        }
    finally:
        env.close()
    return out


def comparable(snapshot: dict) -> dict:
    """The snapshot without the keys a run measures."""
    return {k: v for k, v in snapshot.items() if k not in MEASURED}


def _devices(cls) -> dict:
    devices = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        try:
            env = cls(num_envs=2, device=device)
            try:
                env.reset(seed=0)
                frame = env.render()
                assert frame.shape[0] == 2
            finally:
                env.close()
            devices[device] = "ok"
        except Exception as e:  # noqa: BLE001 — recorded in the snapshot
            devices[device] = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
    return devices


class _Policy:
    """Uniform random actions, each held for :data:`HOLD_STEPS` steps."""

    def __init__(self, env, seed: int):
        spec = env.action_spec
        self.spec = spec
        self.B = env.num_envs
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.a = None

    def _draw(self):
        s = self.spec
        shape = (self.B, *s.shape)
        if not s.dtype.is_floating_point:
            n = s.discrete_n
            if n is None:
                lo = 0 if s.low is None else int(np.min(s.low))
                hi = lo + 1 if s.high is None else int(np.max(s.high))
                n, low = hi - lo + 1, lo
            else:
                low = 0
            return torch.as_tensor(self.rng.integers(low, low + int(n), size=shape), dtype=s.dtype)
        lo = np.broadcast_to(-1.0 if s.low is None else s.low, s.shape).astype(np.float64)
        hi = np.broadcast_to(1.0 if s.high is None else s.high, s.shape).astype(np.float64)
        lo = np.where(np.isfinite(lo), lo, -1.0)
        hi = np.where(np.isfinite(hi), hi, 1.0)
        return torch.as_tensor(self.rng.uniform(lo, hi, size=shape), dtype=s.dtype)

    def __call__(self):
        if self.a is None or self.t % HOLD_STEPS == 0:
            self.a = self._draw()
        self.t += 1
        return self.a


def _frames(env, steps: int, seed: int) -> np.ndarray:
    """``(steps + 1, B, H, W, 3)`` frames of a random rollout; finished lanes hold until all finish."""
    env.reset(seed=seed)
    frames = [env.render().cpu().numpy()]
    if not env.has_dynamics:
        return np.stack(frames * (steps + 1))
    policy = _Policy(env, seed)
    done = np.zeros(env.num_envs, bool)
    for _ in range(steps):
        out = env.step(policy())
        frames.append(out["frames"][:, -1].cpu().numpy())
        for key in ("terminated", "truncated"):
            if key in out.keys():
                done |= out[key].reshape(env.num_envs, -1).any(-1).cpu().numpy().astype(bool)
        if done.all():
            env.reset()
            done[:] = False
    return np.stack(frames)


def _fit(h: int, w: int, side: int) -> tuple[int, int]:
    """Even output size within ``side``; a frame smaller than half of ``side`` grows by a whole factor."""
    longest = max(h, w)
    scale = side / longest if longest > side else max(1, side // longest)
    return max(2, int(round(h * scale / 2)) * 2), max(2, int(round(w * scale / 2)) * 2)


def _filter(h: int, w: int, oh: int, ow: int) -> str:
    return "area" if oh <= h and ow <= w else "nearest"


def _tile(frames: np.ndarray, side: int) -> np.ndarray:
    """Lay out ``(T, 4, H, W, 3)`` lanes in a 2x2 grid, each resized to fit ``side``."""
    t, b, h, w, _ = frames.shape
    th, tw = _fit(h, w, side)
    x = torch.as_tensor(frames).reshape(t * b, h, w, 3).permute(0, 3, 1, 2).float()
    mode = "area" if _filter(h, w, th, tw) == "area" else "nearest-exact"
    x = torch.nn.functional.interpolate(x, size=(th, tw), mode=mode).round().clamp(0, 255).byte()
    x = x.permute(0, 2, 3, 1).reshape(t, 2, 2, th, tw, 3).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(t, 2 * th, 2 * tw, 3).numpy()


def _encode(ffmpeg: str, frames: np.ndarray, path: Path, side: int) -> dict:
    t, h, w, _ = frames.shape
    oh, ow = _fit(h, w, side)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}"]
    cmd += [
        "-r",
        str(FPS),
        "-i",
        "-",
        "-vf",
        f"scale={ow}:{oh}:flags={'area' if _filter(h, w, oh, ow) == 'area' else 'neighbor'}",
        "-c:v",
        "libx264",
    ]
    cmd += ["-preset", "veryslow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    return {"file": path.name, "width": ow, "height": oh, "frames": t, "fps": FPS, "bytes": path.stat().st_size}


def _build(cls, num_envs: int, device: str | None):
    """Build at the default size, or at :data:`RENDER_SIZE` when the default is smaller than a clip.

    ``device=None`` keeps the constructor default of the class.
    """
    kwargs = {} if device is None else {"device": device}
    if (cls.default_size or 0) < MIN_RENDER_SIZE:
        env = None
        try:
            env = cls(num_envs=num_envs, size=RENDER_SIZE, **kwargs)
            env.reset(seed=0)
            env.render()
            return env
        except Exception:  # noqa: BLE001 — the class renders at its default size only
            if env is not None:
                env.close()
    return cls(num_envs=num_envs, **kwargs)


def _variation_rows(env, seed: int, mode: str) -> np.ndarray:
    names = env.variation_names
    paths = None if mode == "all" else [p for p in names if p.endswith("color")]
    if paths == []:
        raise ValueError("no colour leaves")
    rows = env.sample_variation(batch=env.num_envs, paths=paths, rng=np.random.default_rng(seed))
    rows[0] = env.default_variation()
    return rows


def media(name: str, ffmpeg: str, device: str | None) -> dict:
    """Render the rollout clip and the Variation grid of ``name``."""
    import stable_envs as snv

    cls = snv.info(name)["class"]
    out_dir = MEDIA / name
    result: dict = {}
    env = _build(cls, 1, device)
    try:
        result["device"] = str(env.device)
        result["render_size"] = int(env.size)
        varied = bool(env.variation_dim)
        frames = _frames(env, ROLLOUT_STEPS, seed=0)[:, 0]
        result["rollout"] = _encode(ffmpeg, frames, out_dir / "rollout.mp4", MAIN_MAX_SIDE)
    finally:
        env.close()
    if not varied:
        result["variation"] = None
        result["variation_note"] = "no Variation leaves"
        return result
    errors = []
    for mode in ("all", "color"):
        env = _build(cls, 4, device)
        try:
            env.set_variation(_variation_rows(env, 1, mode))
            frames = _frames(env, GRID_STEPS, seed=1)
            result["variation"] = _encode(
                ffmpeg, _tile(frames, TILE_MAX_SIDE), out_dir / "variation.mp4", 2 * TILE_MAX_SIDE
            )
            result["variation_leaves"] = mode
            return result
        except Exception as e:  # noqa: BLE001 — try the next, smaller set of leaves
            errors.append(f"{mode}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
        finally:
            env.close()
    result["variation"] = None
    result["variation_note"] = "; ".join(errors)
    return result


def _ffmpeg(path: str | None) -> str:
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        found = shutil.which("ffmpeg")
        if not found:
            raise SystemExit("ffmpeg not found: install imageio-ffmpeg or pass --ffmpeg") from None
        return found


def _write(name: str, update: dict) -> None:
    path = DATA / f"{name}.json"
    snapshot = json.loads(path.read_text()) if path.exists() else {}
    snapshot.update(update)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=1)
    text = _FLAT_LIST.sub(lambda m: "[" + ", ".join(v.strip() for v in m[1].split(",")) + "]", text)
    path.write_text(text + "\n")


def main(argv: list[str] | None = None) -> int:
    import stable_envs as snv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="class names (default: every registered class)")
    parser.add_argument("--family", help="only the classes of this family")
    parser.add_argument("--meta", action="store_true", help="write the metadata snapshot")
    parser.add_argument("--devices", action="store_true", help="probe each device and record the result")
    parser.add_argument("--media", action="store_true", help="render the clips")
    parser.add_argument("--device", default=None, help="device of the clips (default: the class default)")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable")
    args = parser.parse_args(argv)
    names = args.names or snv.list_envs(family=args.family)
    ffmpeg = _ffmpeg(args.ffmpeg) if args.media else ""
    failed = []
    for name in names:
        cls = snv.info(name)["class"]
        try:
            if args.meta:
                _write(name, metadata(name))
            if args.devices:
                _write(name, {"devices": _devices(cls)})
            if args.media:
                _write(name, {"media": media(name, ffmpeg, args.device)})
            print(f"ok {name}", flush=True)
        except Exception:  # noqa: BLE001 — report every class, then fail
            failed.append(name)
            print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)
    if snv.import_errors():
        print(f"import errors: {snv.import_errors()}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
