"""Run all DMC source strategies in fresh CPU/software-GL and CUDA/EGL processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[64, 256])
    p.add_argument("--order-offset", type=int, default=0)
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ["source_serial", "source_parallel", "ours"]
    n = args.order_offset % len(modes)
    modes = modes[n:] + modes[:n]
    (args.output / "strategy-order.json").write_text(json.dumps(modes) + "\n")
    for mode in modes:
        command = [sys.executable, "scripts/benchmarks/online_stepping/dmc_state_export_oscar.py",
                   "--benchmark", "dmc_source_pipeline", "--domain", args.domain,
                   "--source-root", str(args.source_root.resolve()),
                   "--source-manifest", str(args.source_manifest.resolve()), "--source-mode", mode,
                   "--batches", *map(str, args.batches), "--output", str(args.output / mode)]
        if args.gates_only:
            command.append("--gates-only")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
