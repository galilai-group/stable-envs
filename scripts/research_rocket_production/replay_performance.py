"""Descriptive public Rocket State/replay costs; never a correctness substitute."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    device = "cuda" if args.gpu else "cpu"
    records = []

    def sync():
        if args.gpu:
            torch.cuda.synchronize()

    def timed(fn):
        sync()
        begin = time.perf_counter_ns()
        value = fn()
        sync()
        return (time.perf_counter_ns() - begin) / 1e6, value

    def distribution(fn, count):
        fn()
        raw = [timed(fn)[0] for _ in range(count)]
        return dict(
            samples_ms=raw,
            median_ms=float(np.median(raw)),
            p95_ms=float(np.percentile(raw, 95)),
            operations_per_second=1000 / float(np.median(raw)),
        )

    for batch in args.batches:
        for label, config, options, horizon in [
            ("reset", {}, {}, 0),
            ("25_percent", {"ceiling": 500000.0}, {"accelerate_drop": False}, 300),
            ("50_percent", {"ceiling": 500000.0}, {"accelerate_drop": False}, 600),
            ("75_percent", {"ceiling": 500000.0}, {"accelerate_drop": False}, 900),
            ("maximum", {"ceiling": 500000.0}, {"accelerate_drop": False}, 1202),
            ("near_contact", {}, {}, 301),
            ("crash", {}, {}, 303),
            ("success", {"ceiling": 3.2}, {"accelerate_drop": False}, 7),
        ]:
            config = dict(config, render_mode="rgb_array")
            constructor = lambda: stable_envs.make(
                "PyFlytRocketLanding", num_envs=batch, device=device, simulator_config=config, seed=17
            )
            cold_ms, e = timed(constructor)
            try:
                e.reset(seed=17, options=options)
                action = np.zeros((batch, 7), np.float64)
                for _ in range(horizon):
                    e.step(action)
                state = e.get_state()
                buffer = io.BytesIO()
                np.savez(buffer, **{"/".join(k): v.cpu().numpy() for k, v in state.items(True, True)})
                # Repeated B16 maximum replay is expensive. Keep and disclose every
                # sample; no minimum claim of statistical tail precision is made.
                samples = 10 if batch == 1 else 3
                record = dict(
                    label=label,
                    batch=batch,
                    config=config,
                    options=options,
                    horizon=horizon,
                    physical_replay_steps=state["history", "length"].cpu().tolist(),
                    state_bytes=sum(v.numel() * v.element_size() for v in state.values(True, True)),
                    serialized_npz_bytes=buffer.tell(),
                    cold_construction_ms=cold_ms,
                    get_state=distribution(e.get_state, 20),
                    set_state=distribution(lambda: e.set_state(state), samples),
                    restore_samples=samples,
                    branches=[],
                )
                if args.gpu:
                    record["cold_render_ms"], pixels = timed(lambda: e.render(state))
                    record["warm_render"] = distribution(lambda: e.render(state), 10)
                    record["frame_sha256"] = hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest()
                for length in [1, 10, 100]:
                    for cadence in ["none", "every_fourth", "every_step", "final"] if args.gpu else ["none"]:

                        def branch():
                            e.set_state(state)
                            frame = None
                            for i in range(length):
                                output = e.step(action)
                                if cadence == "every_step" or cadence == "every_fourth" and (i + 1) % 4 == 0:
                                    frame = e.render(output["state"])
                            if cadence == "final":
                                frame = e.render()
                            return frame

                        record["branches"].append(
                            dict(length=length, cadence=cadence, includes_restore=True, timing=distribution(branch, 3))
                        )
                e.set_state(state)
                restored = e.get_state()
                assert all(torch.equal(state[k], restored[k]) for k in state.keys(True, True))
                records.append(record)
                (args.output / "result.json").write_text(
                    json.dumps(dict(status="RUNNING", records=records), indent=2) + "\n"
                )
                print("PERF", batch, label, "PASS", flush=True)
            finally:
                e.close()
            warm_ms, warm = timed(constructor)
            warm.close()
            record["warm_construction_ms"] = warm_ms
    result = dict(
        status="ROCKET_REPLAY_PERFORMANCE_COMPLETE",
        records=records,
        job_id=os.environ.get("SLURM_JOB_ID"),
        hardware=dict(
            host=platform.node(),
            machine=platform.machine(),
            platform=platform.platform(),
            gpu=torch.cuda.get_device_name(0) if args.gpu else None,
        ),
        dynamics="Official PyFlyt/PyBullet CPU, even for CUDA rendering",
        sampling="One warmup; get_state 20 samples; restore B1 10/B16 3 samples; branches 3 samples. Runtime-bounded descriptive distributions, not precise tail estimates.",
        horizon_note="The registered 1202-row capacity is exercised with source-supported ceiling=500000 and acceleration disabled; ordinary default free fall crashes around step 303.",
        rendered="GPU only; local Apple CPU measurements must not be interpreted as GPU speedups",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
