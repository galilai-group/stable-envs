"""Experimental fusion of the unchanged primitive-bound math, outside production."""
import hashlib
import json
import statistics
import time


def probe(ours, source, env, batch, output):
    import torch
    from torch._dynamo.utils import counters
    from stable_envs.render import raycast3d as rays

    original = rays._rays_near
    # Save only on module state so all batches share the actual compiled function.
    if not hasattr(rays, "_cpu_bound_experiment"):
        rays._cpu_bound_experiment = torch.compile(
            original, fullgraph=True, dynamic=True,
            options={"cpp.enable_unsafe_math_opt_flag": False, "cpp.enable_floating_point_contract_flag": "off"},
        )
    compiled = rays._cpu_bound_experiment
    anchor = ours().clone()
    row = {"batch": batch, "baseline_image_sha256": hashlib.sha256(anchor.numpy().tobytes()).hexdigest()}
    checks = {"calls": 0, "different_predicates": 0}

    def checked(*args):
        expected, actual = original(*args), compiled(*args)
        checks["calls"] += 1
        checks["different_predicates"] += int((expected != actual).sum())
        assert torch.equal(expected, actual), checks
        return actual

    try:
        rays._rays_near = checked
        start = time.perf_counter()
        actual = ours()
        row["compilation_and_differential_warmup_ms"] = (time.perf_counter() - start) * 1000
        assert torch.equal(anchor, actual), "Fusion changed complete output"
        row["predicate_checks"] = checks
        rays._rays_near = compiled
        ours()  # exclude post-check dispatch transition from measured samples
        row["graphs_before_timing"] = dict(counters["stats"])
        samples = {"baseline": [], "candidate": [], "source": []}
        for repeat in range(3):
            for label in (["baseline", "candidate", "source"] if repeat % 2 == 0 else ["source", "candidate", "baseline"]):
                rays._rays_near = compiled if label == "candidate" else original
                fn = source if label == "source" else ours
                start = time.perf_counter()
                frame = fn()
                samples[label].append((time.perf_counter() - start) * 1000)
                if label != "source":
                    assert torch.equal(frame, anchor), label
                else:
                    row["paired_mse"] = float(((anchor.to(torch.float64) - frame.to(torch.float64)) / 255).square().mean())
        row["graphs_after_timing"] = dict(counters["stats"])
        assert row["graphs_before_timing"] == row["graphs_after_timing"], "Recompiled during timed samples"
        row["timings"] = {k: {"samples_ms": v, "mean_ms": statistics.mean(v), "sd_ms": statistics.stdev(v), "fps": 1000 * batch / statistics.mean(v)} for k, v in samples.items()}
        row["status"] = "PASS"
        (output / f"fusion_b{batch}.json").write_text(json.dumps(row, indent=2) + "\n")
    finally:
        rays._rays_near = original
