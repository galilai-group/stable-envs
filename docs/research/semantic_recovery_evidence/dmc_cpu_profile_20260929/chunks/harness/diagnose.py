"""Paired CPU DMC warmed profiler. Profiling is separate from benchmark samples."""
import cProfile
import json
import pstats
import time


def diagnose(ours, source, output):
    import torch

    result = {"note": "Warmed B64 saved-State rendering. Instrumented cumulative times overlap; unprofiled paired timing is authoritative."}
    for label, fn in (("ours", ours), ("source", source)):
        anchor = fn().clone()
        profile = cProfile.Profile()
        start = time.perf_counter()
        profile.enable()
        actual = fn()
        profile.disable()
        result[label + "_elapsed_ms"] = (time.perf_counter() - start) * 1000
        result[label + "_repeat_mse"] = float(((actual.to(torch.float64) - anchor.to(torch.float64)) / 255).square().mean())
        if label == "ours":
            assert torch.equal(anchor, actual), label + " repeatability"
        else:
            assert result[label + "_repeat_mse"] <= .0001
        profile.dump_stats(str(output / (label + ".prof")))
        stats = pstats.Stats(profile)
        result[label + "_functions"] = [
            dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], calls=v[1], self_ms=v[2] * 1000, cumulative_ms=v[3] * 1000)
            for k, v in stats.stats.items()
        ]
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")


def sweep(ours, env, output):
    import statistics
    import torch
    from stable_envs.envs.dmcontrol.suite import DMControlSuite
    from stable_envs.render import cpu3d

    anchor = ours().clone()
    fixed_camera = isinstance(env, DMControlSuite)
    owner, key = (env, "RAY_BUDGET") if fixed_camera else (cpu3d, "CPU_CHUNK")
    original = getattr(owner, key)
    values = [original, 1 << 18, 1 << 19, 1 << 20, 1 << 21] if fixed_camera else [original, 1, 2, 4, 8]
    results = []
    try:
        for value in values:
            setattr(owner, key, value)
            actual = ours()
            delta = (actual.to(torch.int16) - anchor.to(torch.int16)).abs()
            row = dict(parameter=key, value=value, max_delta=int(delta.max()), pixels_changed=int((delta != 0).sum()))
            assert row["max_delta"] <= 1, row
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                actual = ours()
                samples.append((time.perf_counter() - start) * 1000)
            row.update(samples_ms=samples, mean_ms=statistics.mean(samples), sd_ms=statistics.stdev(samples))
            results.append(row)
            (output / "chunk_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        setattr(owner, key, original)
