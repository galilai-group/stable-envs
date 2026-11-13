"""Capture golden frames of every installed class on one device into a new directory; never overwrite one."""

import argparse
import json
from pathlib import Path

import numpy as np

import stable_envs as se
from stable_envs.testing.conformance import sample_render_rows
from tests.helpers import golden as G


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory; move it to tests/fixtures/golden/<device> to use it")
    parser.add_argument("--device", choices=G.DEVICES, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-size", type=int, default=None, help="cap the frame size of classes that accept a size")
    parser.add_argument("--envs", nargs="*", default=None, help="class names; default every registered class")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = G.profile(args.device)
    manifest = dict(device=args.device, source_commit=args.source_commit, seed=G.SEED, n=G.N, files={})
    skipped, total = {}, 0
    for name in args.envs or sorted(se.registry.REGISTRY):
        cls = se.registry.REGISTRY[name]
        if args.device == "cpu" and getattr(cls, "requires_cuda", False):
            continue
        kw = {}
        if args.max_size and not cls.native_size and (cls.default_size or 0) > args.max_size:
            kw["size"] = args.max_size
        try:
            env = se.make(name, num_envs=G.N, device=args.device, **kw)
        except ModuleNotFoundError as e:
            skipped[name] = f"needs {e.name}"
            continue
        except (RuntimeError, ValueError) as e:
            skipped[name] = f"{type(e).__name__}: {e}"
            continue
        try:
            states = sample_render_rows(env, G.N, rng=np.random.default_rng(G.SEED))
            frames = env.render(states).cpu().numpy()
            size = int(env.size)
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {e}"
            print(f"FAIL {name}: {skipped[name]}", flush=True)
            continue
        finally:
            env.close()
        path = args.output / f"{name}.npz"
        G.save(path, np.asarray(states), frames, size)
        nbytes = path.stat().st_size
        total += nbytes
        manifest["files"][name] = {
            "file": path.name,
            "sha256": G.sha256(path),
            "size": size,
            "bytes": nbytes,
            "profile": profile,
        }
        print(f"CAPTURED {name} {frames.shape} {nbytes / 1e6:.2f} MB", flush=True)
    manifest["skipped"] = skipped
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest['files'])} classes, {total / 1e6:.1f} MB; skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
