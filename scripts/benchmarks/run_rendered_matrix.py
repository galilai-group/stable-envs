"""Serial composed-render benchmark runner; per-registration records in isolated profiles."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from state_ops import catalog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--profile", action="append", default=[], help="normal/fetch/manipulation/ogbench/craftax=/path/to/python"
    )
    p.add_argument("--only", default=".*")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    profiles = dict(v.split("=", 1) for v in args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for registration in catalog():
        if "Symbolic" in registration or not re.search(args.only, registration):
            continue
        family = (
            "craftax"
            if "Craftax" in registration
            else "fetch"
            if "Fetch" in registration
            else "manipulation"
            if registration in ["swm/OGBCube-v0", "swm/OGBScene-v0"]
            else "ogbench"
            if registration == "swm/OGBMaze-v0"
            else "normal"
        )
        fallback = profiles.get("manipulation", sys.executable) if family == "ogbench" else sys.executable
        py = profiles.get(family, fallback)
        name = registration.replace("/", "__")
        command = [
            py,
            str(Path(__file__).with_name("rendered_state_ops.py")),
            "--registration",
            registration,
            "--output",
            str(args.output / name),
            "--device",
            args.device,
            "--horizon",
            str(args.horizon),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["STATE_BENCH_PROFILE"] = family
        started = __import__("time").time()
        with (args.output / (name + ".log")).open("x") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.timeout)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        records.append(
            {
                "registration": registration,
                "profile": family,
                "command": command,
                "exit_code": exit_code,
                "elapsed_seconds": __import__("time").time() - started,
                "batch_policy": "B1 composed rollout; larger public batches remain covered by the earlier isolated-operation benchmark",
            }
        )
        (args.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        print(registration, exit_code, flush=True)
    raise SystemExit(int(any(r["exit_code"] != 0 for r in records)))


if __name__ == "__main__":
    main()
