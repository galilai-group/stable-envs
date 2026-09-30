"""Read-only renderer profiling; public calls and prepared-input throughput differ.

Run one registration/device/backend per fresh process. Profiling instrumentation
is never included in latency samples. No renderer or simulator is patched.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import pstats
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .collect import candidate
from .common import ROOT, array_hash, catalog, load_state, profile, public_render, state_hash, write


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure(fn, device, *, minimum=6, maximum=30, budget_s=3, warmups=2):
    """Keep every sample; sample SD (ddof=1); materialize async JAX results."""
    assert maximum >= minimum >= 2

    def call():
        value = fn()
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
        return value

    synchronize(device)
    start = time.perf_counter_ns()
    value = call()
    synchronize(device)
    first = (time.perf_counter_ns() - start) / 1e6
    for _ in range(warmups):
        value = call()
    synchronize(device)
    samples = []
    baseline_memory = torch.cuda.memory_allocated() if device == "cuda" else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(maximum):
        synchronize(device)
        start = time.perf_counter_ns()
        value = call()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
        if len(samples) >= minimum and sum(samples) >= budget_s * 1000:
            break
    return dict(
        samples_ms=samples,
        mean_ms=statistics.mean(samples),
        sd_ms=statistics.stdev(samples),
        median_ms=statistics.median(samples),
        p95_ms=float(np.percentile(samples, 95)),
        count=len(samples),
        first_call_ms=first,
        warmups=warmups,
        sd_ddof=1,
        torch_cuda_allocated_before_samples_bytes=baseline_memory,
        torch_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        memory_scope="Torch allocator only; excludes external GL/Warp/JAX allocations.",
        synchronization="CUDA device synchronized before/after each sample"
        if device == "cuda"
        else "eager CPU; block_until_ready for JAX outputs",
    ), value


def capture_profiles(fn, device, output, label):
    """cProfile host stacks plus Torch operators; GL/JAX kernels may be opaque."""
    result = {
        "scope": "Separate instrumented calls; not timing samples. Python cumulative times overlap; CUDA operator times are not disjoint wall times."
    }
    host = cProfile.Profile()
    synchronize(device)
    host.enable()
    for _ in range(3):
        fn()
        synchronize(device)
    host.disable()
    host.dump_stats(str(output / f"{label}.prof"))
    entries = []
    for (file, line, name), (primitive, calls, own, total, _) in pstats.Stats(host).stats.items():
        entries.append(
            dict(
                file=file,
                line=line,
                function=name,
                primitive_calls=primitive,
                calls=calls,
                self_ms=own * 1000,
                cumulative_ms=total * 1000,
            )
        )
    result["python_functions"] = sorted(entries, key=lambda x: x["cumulative_ms"], reverse=True)[:120]
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            with torch.profiler.record_function(label):
                fn()
                synchronize(device)
        prof.export_chrome_trace(str(output / f"{label}.trace.json"))
        result["torch_operators"] = [
            dict(
                name=e.key,
                calls=e.count,
                cpu_total_us=e.cpu_time_total,
                cpu_self_us=e.self_cpu_time_total,
                device_total_us=getattr(e, "device_time_total", 0),
                device_self_us=getattr(e, "self_device_time_total", 0),
                cpu_memory_bytes=e.cpu_memory_usage,
                device_memory_bytes=getattr(e, "device_memory_usage", 0),
            )
            for e in prof.key_averages()
        ]
        result["cuda_kernel_events_observed"] = any(str(e.device_type).endswith("CUDA") for e in prof.events())
        result["coverage_limit"] = (
            "Torch trace can miss external OpenGL, Warp and JAX kernels; wall timing includes their completed rendering."
        )
    except Exception:
        result["torch_profiler_error"] = traceback.format_exc()
    write(output / f"{label}.profile.json", result)
    return result


def projection(env):
    if hasattr(env, "to_render_state"):
        return env.to_render_state, env.render, "public named RenderState"

    def prepare(state):
        env._validate_state(state)
        return env._pack_render_state(state)

    return prepare, env._render, "private compact-render boundary (not a public RenderState API)"


def choose_frames(meta, count):
    config = meta["recipes"][meta["frames"][0]["case"]].get("simulator_config")
    frames = [f for f in meta["frames"] if meta["recipes"][f["case"]].get("simulator_config") == config]
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    return [frames[i] for i in indices], config


def batch_kwargs(kwargs, indices):
    result = {}
    assert all(k.keys() == kwargs[0].keys() for k in kwargs)
    for key, value in kwargs[0].items():
        if torch.is_tensor(value):
            result[key] = torch.cat([kwargs[i][key] for i in indices], dim=0)
        else:
            assert all(k[key] == value for k in kwargs), key
            result[key] = value
    return result


def check_image(image, batch, device):
    assert torch.is_tensor(image) and image.dtype == torch.uint8
    assert image.ndim == 4 and image.shape[0] == batch and image.shape[-1] == 3, image.shape
    assert image.device.type == device, image.device


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("registration")
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--poses", type=int, default=6)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--no-trace", action="store_true")
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    row = catalog()[a.registration]
    result = dict(
        registration=a.registration,
        name=row["name"],
        device=a.device,
        backend=a.backend,
        status="RUNNING",
        profile=profile(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measurements=[],
        batch_checks=[],
        semantic_status_unchanged=row["status"],
        scope="Render profiling only; not new source parity or semantic acceptance evidence.",
    )
    result["production_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "stable_envs").rglob("*.py"))
    }
    save = lambda: write(a.output / "result.json", result)
    save()
    try:
        if "Symbolic" in row["name"]:
            result.update(status="NO_RGB", reason="Symbolic Craftax registration has no pixel renderer.")
            return
        if a.backend == "opengl":
            import mujoco
            from OpenGL import GL

            model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
            )
            gl_owner = mujoco.Renderer(model, height=32, width=32)
            gl_owner._gl_context.make_current()
            vendor = GL.glGetString(GL.GL_VENDOR).decode()
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            software = any(k in renderer.lower() for k in ("llvmpipe", "softpipe", "swrast"))
            assert software if a.device == "cpu" else "nvidia" in vendor.lower() and not software
            result["opengl"] = dict(vendor=vendor, renderer=renderer, software=software)
        config = None
        if a.corpus:
            meta = json.loads((a.corpus / "result.json").read_text())
            assert meta["status"] == "COMPLETE"
            frames, config = choose_frames(meta, a.poses)
            result["corpus"] = str(a.corpus)
            result["corpus_sha256"] = hashlib.sha256((a.corpus / "result.json").read_bytes()).hexdigest()
        else:
            assert row["name"] == "PushT", "A measured source-state corpus is required except for diagnostic PushT."
            frames = []
            result["scope"] = (
                "DIAGNOSTIC_ONLY: candidate PushT shared renderer; no matched source or registration-specific dynamics claim."
            )
        env = candidate(row, a.device, 17, config)
        if row["name"].startswith("Craftax"):
            env.reset(0)
        live_hash = state_hash(env.get_state())
        template = env.get_state()
        states, kwargs = [], []
        for f in frames:
            state = load_state(a.corpus / f["state_file"], a.device, template)
            assert state_hash(state) == f["state_hash"]
            states.append(state)
            kwargs.append(
                {
                    k: torch.tensor(v, dtype=torch.float32, device=a.device) if k == "style" else v
                    for k, v in f["render_kwargs"].items()
                }
            )
        if not frames:
            states, kwargs = [template], [{}]
        if a.backend != "auto":
            for kw in kwargs:
                kw["backend"] = a.backend
        hashes = [state_hash(s) for s in states]
        result["input_frames"] = [
            dict(index=f["index"], seed=f["seed"], case=f["case"], step=f["step"], state_hash=f["state_hash"])
            for f in frames
        ]
        prepare, render, boundary = projection(env)
        result["prepared_boundary"] = boundary
        prepared = [prepare(s) for s in states]
        fingerprint = lambda s: state_hash(s) if hasattr(s, "batch_size") else array_hash(s)
        prepared_hashes = [state_hash(s) if hasattr(s, "batch_size") else None for s in prepared]
        prepared_copies = [s.clone() for s in prepared]
        references = []
        for state, rs, kw in zip(states, prepared, kwargs):
            public = public_render(env, state, **kw)
            image = render(rs, **kw)
            check_image(public, 1, a.device)
            check_image(image, 1, a.device)
            assert torch.max(torch.abs(public.to(torch.int16) - image.to(torch.int16))).item() <= 1
            references.append(image.clone())
        # cProfile/Torch tracing execute outside all steady-state timing samples.
        for stage in ("public_full_state", "prepare_validate", "prepared_render"):

            def one(i):
                if stage == "public_full_state":
                    return public_render(env, states[i], **kwargs[i])
                if stage == "prepare_validate":
                    return prepare(states[i])
                return render(prepared[i], **kwargs[i])

            # Warm every selected pose/style before collecting mixed-pose timing.
            for i in range(len(states)):
                one(i)
            counter = iter(range(10**9))
            fn = lambda: one(next(counter) % len(states))
            timing, _ = measure(fn, a.device, maximum=a.samples)
            result["measurements"].append(dict(stage=stage, batch=1, timing=timing))
            save()
            if not a.no_trace:
                capture_profiles(lambda: one(0), a.device, a.output, stage)
            print(a.registration, a.device, a.backend, stage, timing["mean_ms"], flush=True)
        if row["name"].startswith("Craftax"):
            from stable_envs._adapters.craftax_render import restore_draw

            adapter = env._simulator
            decoded = restore_draw(adapter, prepared[0])[0]
            fn = lambda: adapter.output_fn(decoded)
            timing, raw = measure(fn, a.device, maximum=a.samples)
            result["measurements"].append(
                dict(
                    stage="upstream_jitted_get_obs_compute",
                    batch=1,
                    timing=timing,
                    scope="Actual upstream JAX pixels on an already decoded State; excludes bridge and host conversion. Source wrapper cached render is not recomputation.",
                )
            )
            timing, _ = measure(
                lambda: torch.from_numpy((np.clip(np.asarray(raw), 0, 1) * 255).astype(np.uint8)).to(a.device),
                a.device,
                maximum=a.samples,
            )
            result["measurements"].append(dict(stage="upstream_pixels_to_torch", batch=1, timing=timing))
        if a.device == "cuda":
            timing, _ = measure(lambda: references[0].cpu(), a.device, maximum=a.samples)
            result["measurements"].append(
                dict(
                    stage="output_to_host_microbenchmark",
                    batch=1,
                    timing=timing,
                    scope="Standalone copy; not an estimate of internal transfers and not included in public render latency.",
                )
            )
        for batch in a.batches:
            indices = [i % len(prepared) for i in range(batch)]
            rs = torch.cat([prepared[i] for i in indices], dim=0)
            kw = batch_kwargs(kwargs, indices)
            before = fingerprint(rs)
            fn = lambda rs=rs, kw=kw: render(rs, **kw)
            # Expensive CPU B256 may have only three samples; keep sample count explicit.
            try:
                timing, image = measure(fn, a.device, minimum=3, maximum=a.samples, warmups=1)
            except torch.OutOfMemoryError:
                result["batch_checks"].append(dict(batch=batch, passed=False, status="OOM"))
                result["measurements"].append(
                    dict(
                        stage="prepared_batch_render",
                        batch=batch,
                        scope="OOM: no valid completed timing samples",
                        error=traceback.format_exc(),
                    )
                )
                save()
                del rs, fn
                torch.cuda.empty_cache()
                continue
            check_image(image, batch, a.device)
            assert fingerprint(rs) == before
            maximum = max(
                int((image[j : j + 1].to(torch.int16) - references[i].to(torch.int16)).abs().max())
                for j, i in enumerate(indices)
            )
            result["batch_checks"].append(
                dict(
                    batch=batch, maximum_channel_difference_from_single=maximum, tolerance_uint8=1, passed=maximum <= 1
                )
            )
            result["measurements"].append(
                dict(
                    stage="prepared_batch_render",
                    batch=batch,
                    timing=timing,
                    mean_ms_per_frame=timing["mean_ms"] / batch,
                    frames_per_second=1000 * batch / timing["mean_ms"],
                    independent_poses=len(set(indices)),
                    scope="Prepared inputs only. Real saved poses repeated to fill batch; excludes State projection and simulator allocation.",
                )
            )
            save()
            print(a.registration, a.device, a.backend, "batch", batch, timing["mean_ms"], flush=True)
            del image, rs, fn
        assert [state_hash(s) for s in states] == hashes
        assert state_hash(env.get_state()) == live_hash
        for value, copy, digest in zip(prepared, prepared_copies, prepared_hashes):
            if digest is not None:
                assert state_hash(value) == digest
            else:
                assert torch.equal(value, copy)
        result["nonmutation_passed"] = True
        if any(c.get("status") == "OOM" for c in result["batch_checks"]):
            result["status"] = "PARTIAL_BATCH_OOM"
        else:
            result["status"] = (
                "COMPLETE" if all(c["passed"] for c in result["batch_checks"]) else "BATCH_INVARIANCE_FAILED"
            )
        # Keep GL owners alive until process exit: closing a shared context can affect siblings.
    except NotImplementedError as error:
        result.update(status="UNSUPPORTED", reason=str(error))
    except Exception:
        result.update(status="ERROR", traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
