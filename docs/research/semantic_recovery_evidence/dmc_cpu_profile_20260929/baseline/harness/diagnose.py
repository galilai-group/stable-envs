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
        assert torch.equal(anchor, actual), label + " repeatability"
        profile.dump_stats(str(output / (label + ".prof")))
        stats = pstats.Stats(profile)
        result[label + "_functions"] = [
            dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], calls=v[1], self_ms=v[2] * 1000, cumulative_ms=v[3] * 1000)
            for k, v in stats.stats.items()
        ]
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
