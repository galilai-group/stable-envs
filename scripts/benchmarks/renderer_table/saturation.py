"""RTX 3090 rendering throughput sweep, with held-out confirmation at chosen B.

Prepared-input rendering only. Does not pretend to measure simulation or State
projection. Existing source-corpus input hashes and renderer nonmutation are
checked. All sizes are powers of two; no parity backend is silently replaced.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .collect import candidate
from .common import array_hash, catalog, load_state, profile, state_hash, write


def fingerprint(value):
    return state_hash(value) if hasattr(value, "batch_size") else array_hash(value)


def stats(samples, batch):
    return dict(
        samples_ms=samples,
        count=len(samples),
        mean_ms=statistics.mean(samples),
        sd_ms=statistics.stdev(samples),
        median_ms=statistics.median(samples),
        frames_per_second=1000 * batch / statistics.mean(samples),
        mean_ms_per_frame=statistics.mean(samples) / batch,
        sd_ms_per_frame=statistics.stdev(samples) / batch,
    )


def time_calls(fn, batch, *, minimum=6, maximum=20000, seconds=3):
    samples = []
    for _ in range(maximum):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        value = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        del value
        if len(samples) >= minimum and sum(samples) >= seconds * 1000:
            break
    return stats(samples, batch)


def select_batch(rows, fraction=0.95):
    valid = [r for r in rows if r["status"] == "COMPLETE"]
    best = max(r["timing"]["frames_per_second"] for r in valid)
    return min(r["batch"] for r in valid if r["timing"]["frames_per_second"] >= fraction * best)


def should_stop(rows):
    # Require all requested sizes; two consecutive doublings with <5% gain.
    valid = [r for r in rows if r["status"] == "COMPLETE"]
    if len(valid) < 3 or valid[-1]["batch"] < 256:
        return False
    rates = [r["timing"]["frames_per_second"] for r in valid[-3:]]
    return rates[1] < rates[0] * 1.05 and rates[2] < rates[1] * 1.05


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("registration")
    p.add_argument("device", choices=["cuda"])
    p.add_argument("output", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--backend", default="auto")
    p.add_argument("--max-batch", type=int, default=4096)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assert "3090" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    result = dict(
        registration=a.registration,
        device=a.device,
        backend=a.backend,
        status="RUNNING",
        profile=profile(),
        scope="Prepared-input renderer throughput; excludes step, State extraction, initial upload and projection.",
        rows=[],
        stop_reason=None,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    save = lambda: write(a.output / "result.json", result)
    save()
    monitor = None
    monitor_stream = None
    try:
        row = catalog()[a.registration]
        if "Symbolic" in row["name"]:
            result.update(status="NO_RGB")
            return
        frames = []
        config = None
        if a.corpus:
            meta = json.loads((a.corpus / "result.json").read_text())
            assert meta["status"] == "COMPLETE"
            config = meta["recipes"][meta["frames"][0]["case"]].get("simulator_config")
            pool = [f for f in meta["frames"] if meta["recipes"][f["case"]].get("simulator_config") == config]
            frames = [pool[i] for i in np.linspace(0, len(pool) - 1, min(16, len(pool)), dtype=int)]
            result["corpus"] = str(a.corpus)
            result["corpus_sha256"] = hashlib.sha256((a.corpus / "result.json").read_bytes()).hexdigest()
        else:
            assert row["name"] == "PushT"
            result["scope"] += " PushT diagnostic only; not a source-matched acceptance claim."
        env = candidate(row, "cuda", 17, config)
        if "Craftax" in row["name"]:
            env.reset(17)
        template = env.get_state()
        live_hash = state_hash(template)
        states = []
        kwargs = []
        for f in frames:
            state = load_state(a.corpus / f["state_file"], "cuda", template)
            assert state_hash(state) == f["state_hash"]
            states.append(state)
            kwargs.append(
                {
                    k: torch.tensor(v, dtype=torch.float32, device="cuda") if k == "style" else v
                    for k, v in f["render_kwargs"].items()
                }
            )
        if not states:
            states = [template]
            kwargs = [{}]
        if a.backend != "auto":
            for kw in kwargs:
                kw["backend"] = a.backend
        if hasattr(env, "to_render_state"):
            prepare, render = env.to_render_state, env.render
        else:

            def prepare(state):
                env._validate_state(state)
                return env._pack_render_state(state)

            render = env._render
        inputs = [prepare(s) for s in states]
        input_hashes = [fingerprint(s) for s in inputs]
        references = [render(s, **kw).cpu() for s, kw in zip(inputs, kwargs)]
        result["input_frames"] = frames
        result["independent_states"] = len(set(state_hash(s) for s in states))
        # Log all visible GPUs with UUID; CUDA_VISIBLE_DEVICES is in the job profile.
        monitor_stream = (a.output / "nvidia-smi.csv").open("w")
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,nounits",
                "-lms",
                "200",
            ],
            stdout=monitor_stream,
            stderr=subprocess.DEVNULL,
        )

        def make_batch(batch):
            idx = [i % len(inputs) for i in range(batch)]
            value = torch.cat([inputs[i] for i in idx], dim=0)
            kw = {}
            for k, v in kwargs[0].items():
                if torch.is_tensor(v):
                    kw[k] = torch.cat([kwargs[i][k] for i in idx], dim=0)
                else:
                    assert all(x[k] == v for x in kwargs)
                    kw[k] = v
            return value, kw, idx

        batch = 64
        while batch <= a.max_batch:
            record = dict(batch=batch, status="RUNNING", start_time_ns=time.time_ns())
            result["rows"].append(record)
            save()
            value = fn = image = None
            try:
                value, kw, idx = make_batch(batch)
                before = fingerprint(value)
                fn = lambda: render(value, **kw)
                for _ in range(3):
                    image = fn()
                    torch.cuda.synchronize()
                assert image.shape[0] == batch and image.dtype == torch.uint8 and image.device.type == "cuda"
                difference = max(
                    int((image[i].cpu().to(torch.int16) - references[j][0].to(torch.int16)).abs().max())
                    for i, j in enumerate(idx)
                )
                assert difference <= 1, difference
                del image
                image = None
                torch.cuda.reset_peak_memory_stats()
                record["timing"] = time_calls(fn, batch)
                record["torch_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                assert fingerprint(value) == before
                record.update(status="COMPLETE", max_batch_difference_uint8=difference, end_time_ns=time.time_ns())
                print(a.registration, a.backend, batch, record["timing"]["frames_per_second"], flush=True)
            except torch.OutOfMemoryError:
                record.update(status="OOM", error=traceback.format_exc(), end_time_ns=time.time_ns())
                result["stop_reason"] = "OOM"
                save()
                break
            finally:
                del fn, image, value
                torch.cuda.empty_cache()
            save()
            if should_stop(result["rows"]):
                result["stop_reason"] = "THROUGHPUT_PLATEAU"
                break
            batch *= 2
        if result["stop_reason"] is None:
            result["stop_reason"] = "MAX_BATCH_NOT_PROVEN_SATURATED"
        chosen = select_batch(result["rows"])
        result["selected_batch"] = chosen
        value, kw, idx = make_batch(chosen)
        before = fingerprint(value)
        fn = lambda: render(value, **kw)
        for _ in range(3):
            fn()
            torch.cuda.synchronize()
        result["confirmation"] = time_calls(fn, chosen, minimum=10, maximum=20000, seconds=5)
        assert fingerprint(value) == before
        assert [fingerprint(s) for s in inputs] == input_hashes
        assert state_hash(env.get_state()) == live_hash
        result["nonmutation_passed"] = True
        result["status"] = "COMPLETE"
        result["selection"] = "Smallest tested B within 5% of best sweep FPS; confirmed with a separate timing sample."
    except Exception:
        result.update(status="ERROR", error=traceback.format_exc())
        raise
    finally:
        if monitor:
            monitor.terminate()
            monitor.wait(timeout=10)
            monitor_stream.close()
        save()


if __name__ == "__main__":
    main()
