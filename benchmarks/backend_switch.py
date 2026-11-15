"""Time the two CUDA simulation backends of each class at batches 1, 2, 4, ..., as ``stable-envs tune`` times them.

usage: python -m benchmarks.backend_switch [--max-batch N] ENV [ENV ...]

Prints both per-call step times at each batch and the smallest batch from which the large-batch backend is faster at
every larger measured batch: the value a class ships as its CUDA switch point.
"""

import argparse
import json

import stable_envs as snv
from stable_envs._execution import backends


def sweep(name: str, largest: int) -> dict:
    """``{path: {"switch": batch or None, "rows": [(batch, small s, large s)]}}`` for class ``name`` on CUDA."""
    before = {path for path, _, _ in backends.registered("cuda")}
    snv.make(name, num_envs=1, device="cuda").close()
    out = {}
    for path, rungs, measure in backends.registered("cuda"):
        if path in before or rungs.small == rungs.large:
            continue
        rows, small_lost = [], 0
        for batch in [1 << k for k in range(largest.bit_length())]:
            large = backends.time_backend(measure, rungs.large, batch, "cuda")
            # The small-batch backend only gets slower with the batch; stop timing it after it lost twice.
            small = backends.time_backend(measure, rungs.small, batch, "cuda") if small_lost < 2 else float("inf")
            small_lost = small_lost + 1 if small > large else 0
            rows.append((batch, small, large))
            times = f"{rungs.small}={small * 1e6:.0f}us {rungs.large}={large * 1e6:.0f}us"
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
