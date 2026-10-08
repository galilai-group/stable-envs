"""Gate one Fetch registration, then compare strategies in isolated interpreters."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[64])
    p.add_argument("--order-offset", type=int, default=0)
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    names = [
        f"swm/Fetch{task}{suffix}-v3"
        for task in ("Reach", "Push", "Slide", "PickAndPlace")
        for suffix in ("", "Dense", "Dict")
    ]
    if args.registration not in names:
        p.error("Require an exact Fetch registration")
    args.output.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        "benchmarks/online/dmc_state_export_oscar.py",
        "--domain",
        args.registration,
        "--source-root",
        str(args.source_root.resolve()),
        "--source-manifest",
        str(args.source_manifest.resolve()),
    ]
    image_args = []
    if "PickAndPlace" not in args.registration:
        gate_dir = args.output.resolve() / "image-gates"
        subprocess.run([*common, "--benchmark", "fetch_source_frame_gate", "--output", str(gate_dir)], check=True)
        image_args = ["--image-gates", str(gate_dir)]
    modes = ["source_serial", "source_parallel", "ours"]
    offset = args.order_offset % len(modes)
    modes = modes[offset:] + modes[:offset]
    (args.output / "strategy-order.json").write_text(json.dumps(modes) + "\n")
    for mode in modes:
        command = [
            *common,
            "--benchmark",
            "fetch_source_pipeline",
            "--source-mode",
            mode,
            "--batches",
            *map(str, args.batches),
            *image_args,
            "--output",
            str(args.output / mode),
        ]
        if args.gates_only:
            command.append("--gates-only")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
