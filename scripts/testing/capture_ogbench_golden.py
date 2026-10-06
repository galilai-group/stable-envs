"""Capture an explicit Cube/Scene baseline; never overwrite an existing fixture directory."""

import argparse
import hashlib
import json
from pathlib import Path

from stable_envs.envs.ogbench import OGBCube, OGBScene
from tests.helpers import ogbench_golden as golden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    profile = golden.profile()
    if profile["versions"] != golden.VERSIONS:
        raise RuntimeError(f"Golden capture requires {golden.VERSIONS}; found {profile}")
    args.output.mkdir(parents=True, exist_ok=False)
    profile.update(source_commit=args.source_commit, size=64, batch=257, seed=17, files={})
    for family, backend in golden.CASES:
        cls = OGBCube if family == "Cube" else OGBScene
        with cls(device="cuda", size=64, seed=17) as env:
            state = golden.fixed_states(env)
            frames = env.render(state, backend=backend).cpu().numpy()
        name = f"{family}-{backend}.npz"
        golden.save(args.output / name, state, frames)
        profile["files"][name] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        print(f"CAPTURED {name} {frames.shape}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
