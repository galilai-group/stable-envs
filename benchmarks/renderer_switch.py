"""Time the CPU and the CUDA renderer of each class at batches 1, 2, 4, ..., as ``stable-envs tune`` times them.

usage: python -m benchmarks.renderer_switch [--max-batch N] ENV [ENV ...]

Prints both per-call draw times at each batch and the smallest batch from which the CUDA renderer is faster at every
larger measured batch: the value a class ships as its ``large_renderer_min_rows``.
"""

import argparse
import json

import stable_envs as snv
from stable_envs import tuning


def sweep(name: str, largest: int) -> dict:
    """``{path: {"switch": batch or None, "rows": [(batch, cpu s, cuda s)]}}`` for class ``name`` on CUDA."""
    before = {path for path, _, _ in tuning.render_paths("cuda")}
    with snv.make(name, num_envs=1, device="cuda") as env:
        # A first draw registers the render paths.
        env.render(env.reset(seed=0))
    out = {}
    for path, pair, measure in tuning.render_paths("cuda"):
        if path in before or pair.small == pair.large:
            continue
        rows, small_lost = [], 0
        for batch in [1 << k for k in range(largest.bit_length())]:
            large = tuning.time_backend(measure, pair.large, batch, "cuda")
            # The CPU renderer only gets slower with the batch; stop timing it after it lost twice.
            small = tuning.time_backend(measure, pair.small, batch, "cuda") if small_lost < 2 else float("inf")
            small_lost = small_lost + 1 if small > large else 0
            rows.append((batch, small, large))
            times = f"{pair.small}={small * 1e6:.0f}us {pair.large}={large * 1e6:.0f}us"
            print(f"{name} {path} B={batch} {times}", flush=True)
        switch = next((b for i, (b, _, _) in enumerate(rows) if all(lg <= sm for _, sm, lg in rows[i:])), None)
        out[path] = {"switch": switch, "rows": rows}
        print(f"SWITCH {name} {path} {switch}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("envs", nargs="+")
    parser.add_argument("--max-batch", type=int, default=4096)
    args = parser.parse_args()
    found = {}
    for name in args.envs:
        found.update(sweep(name, args.max_batch))
    print(json.dumps({path: value["switch"] for path, value in found.items()}))


if __name__ == "__main__":
    main()
