"""Profile native-batched Navigation stepping and bounded rendering separately.

Rates count actual valid transitions. Rendering uses native resolution and is
bounded independently of State-only batch size to avoid allocating huge images.
No source-speedup or CPU/GPU pixel-parity claim is made by this diagnostic.
"""

import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pstats
import statistics
import time
from pathlib import Path

import torch

import stable_envs as se


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("TwoRoom", "Piecewise", "SimplePointMaze"), required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64, 256, 1024, 4096])
    parser.add_argument("--render-max-batch", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < 1 or args.render_max_batch < 0:
        parser.error("Invalid batch/repeat limits")
    torch.set_num_threads(1)
    report = dict(status="RUNNING", env=args.env, host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "tensordict")},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path("stable_envs/envs/navigation.py"), Path("stable_envs/env.py"),
                                  Path("stable_envs/_stepping.py"), Path(__file__).relative_to(Path.cwd())]},
                  scope="native-batched component/public-flow diagnostic; no source comparison", cells=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for device in args.devices:
            for batch in args.batches:
                with se.make(args.env, num_envs=batch, device=device, seed=17) as env:
                    initial = env.get_state()
                    # Identical action values across devices; no random transfers in timing.
                    actions = torch.full((batch, 8, 2), 0.01, dtype=torch.float32, device=device)
                    operations = [("raw_step_with_state", 1), ("get_state", 1)]
                    operations += [(name, t) for t in (1, 8) for name in ("public_states_all", "public_states_none")]
                    if batch <= args.render_max_batch:
                        operations += [("render_only", 1), ("pipeline_final_frame", 1), ("pipeline_final_frame", 8)]
                    for operation, horizon in operations:
                        action = actions[:, 0] if horizon == 1 else actions

                        def call():
                            if operation == "raw_step_with_state":
                                return env._step(action)
                            if operation == "get_state":
                                return env.get_state()
                            if operation == "render_only":
                                return env.render(initial)
                            return env.step(action, states=None if operation == "public_states_none" else slice(None),
                                            frames=slice(-1, None) if operation == "pipeline_final_frame" else None)

                        env.set_state(initial)
                        call()
                        seconds, counts = [], []
                        for _ in range(args.repeats):
                            env.set_state(initial)
                            sync(device)
                            start = time.perf_counter()
                            result = call()
                            sync(device)
                            seconds.append(time.perf_counter() - start)
                            counts.append(int(result["valid"].sum().item()) if operation.startswith("public_") or operation == "pipeline_final_frame" else batch)
                            del result
                        rates = [count / t for count, t in zip(counts, seconds, strict=True)]
                        cell = dict(device=device, batch=batch, horizon=horizon, operation=operation,
                                    seconds=seconds, valid_transitions_or_items=counts,
                                    rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates),
                                    unit="frames/s" if operation == "render_only" else "States/s" if operation == "get_state" else "steps/s",
                                    selected_frames_per_call=batch if operation in ("render_only", "pipeline_final_frame") else 0)
                        if operation == "pipeline_final_frame":
                            frame_rates = [batch / t for t in seconds]
                            cell.update(frames_per_s_mean=statistics.mean(frame_rates), frames_per_s_sd=statistics.stdev(frame_rates))
                        if batch in (64, 1024) and operation == "public_states_all":
                            env.set_state(initial)
                            profile = cProfile.Profile()
                            profile.enable()
                            call()
                            sync(device)
                            profile.disable()
                            output = io.StringIO()
                            pstats.Stats(profile, stream=output).sort_stats("cumulative").print_stats(30)
                            cell["cprofile"] = output.getvalue()
                        report["cells"].append(cell)
                        save()
                    print(args.env, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"], report["error"] = "FAIL", repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
