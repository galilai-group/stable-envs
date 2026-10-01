"""Before/after production FetchAdapter.step, including owned full State export."""

import argparse
import importlib.util
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from fetch_probe import equal

from stable_envs._adapters.fetch_pickandplace import FetchAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location("baseline_fetch", "baseline_fetch.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.set_num_threads(1)
    report = dict(status="RUNNING", host=platform.node(), affinity=sorted(os.sched_getaffinity(0)), rows=[])
    for batch in (1, 64, 256, 1024):
        old, new = baseline.FetchAdapter(batch, task="Push"), FetchAdapter(batch, task="Push")
        actions = np.random.default_rng(19).uniform(-1, 1, (4, batch, 4)).astype(np.float32)
        try:
            old.reset(seed=17)
            new.reset(seed=17)
            before = old.get_state()
            equal(before, new.get_state())

            def run(adapter):
                return [adapter.step(a) for a in actions]

            expected, got = run(old), run(new)
            for a, b in zip(expected, got):
                equal(a, b)
            samples = {"baseline": [], "optimized": []}
            for repeat in range(5):
                order = [("baseline", old), ("optimized", new)]
                for name, adapter in order if repeat % 2 == 0 else order[::-1]:
                    adapter.set_state(before)
                    start = time.perf_counter()
                    out = run(adapter)
                    elapsed = time.perf_counter() - start
                    for a, b in zip(expected, out):
                        equal(a, b)
                    samples[name].append(batch * 4 / elapsed)
            row = dict(
                batch=batch,
                workers=new._workers,
                exact="PASS",
                samples=samples,
                **{
                    name: dict(mean=statistics.mean(values), sd=statistics.stdev(values))
                    for name, values in samples.items()
                },
            )
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        finally:
            old.close()
            new.close()
    report["status"] = "PASS"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FETCH_ADAPTER_COMPARE_PASS", flush=True)


if __name__ == "__main__":
    main()
