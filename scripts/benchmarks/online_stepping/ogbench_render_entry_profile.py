"""Separate full-State validation/projection from the unchanged GPU renderer."""

import argparse
import cProfile
import hashlib
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
from stable_envs._adapters.ogbench import render_state as pixels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("Cube", "Scene"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert len(os.sched_getaffinity(0)) == 8
    torch.set_num_threads(1)
    files = [*Path("stable_envs/_adapters").rglob("*.py"),
             *Path("stable_envs/envs/ogbench").glob("*.py"),
             Path("stable_envs/_adapters/process_requests.py"), Path("stable_envs/_stepping.py"),
             Path(__file__).relative_to(Path.cwd())]
    report = dict(status="RUNNING", family=args.family, batch=64, host=platform.node(),
                  job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(),
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  scope="same complete State; diagnostic components overlap and must not be summed; no physics or source speedup",
                  timings=[], profiles={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with se.make("OGB" + args.family, num_envs=64, device="cuda", seed=17, size=64) as env:
            assert len(env._simulator.worker_info) == 8 and env.backend == "auto"
            state = env.get_state()
            host = state.cpu()
            compact = env.to_render_state(state)
            expected = env.render(state)
            operations = {
                "full_state_to_cpu": lambda: state.cpu(),
                "validate_full_cuda": lambda: env._simulator.validate(state),
                "validate_full_host": lambda: env._simulator.validate(host),
                "project_only_cuda": lambda: pixels.project(args.family, state, 64, 1),
                "public_to_render_state_cuda": lambda: env.to_render_state(state),
                "render_compact_cuda": lambda: env.render(compact),
                "render_full_cuda": lambda: env.render(state),
                "render_full_preconverted_host": lambda: env.render(host),
                "render_full_including_host_conversion": lambda: env.render(state.cpu()),
            }
            for name, fn in operations.items():
                warm = fn()
                if name.startswith("render_"):
                    torch.testing.assert_close(warm, expected, rtol=0, atol=0)
                del warm
                samples = []
                for _ in range(6):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = fn()
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - start)
                    del result
                rates = [64 / s for s in samples]
                report["timings"].append(dict(operation=name, seconds=samples,
                    seconds_mean=statistics.mean(samples), seconds_sd=statistics.stdev(samples),
                    lanes_per_s_mean=statistics.mean(rates), lanes_per_s_sd=statistics.stdev(rates),
                    frame_mse_vs_full_cuda=0 if name.startswith("render_") else None))
                save()
                print(args.family, name, statistics.mean(samples), flush=True)
            for name in ("validate_full_cuda", "validate_full_host", "render_compact_cuda", "render_full_cuda"):
                profile = cProfile.Profile()
                torch.cuda.synchronize()
                profile.enable()
                result = operations[name]()
                torch.cuda.synchronize()
                profile.disable()
                del result
                out = io.StringIO()
                pstats.Stats(profile, stream=out).sort_stats("cumulative").print_stats(35)
                report["profiles"][name] = out.getvalue()
                save()
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
