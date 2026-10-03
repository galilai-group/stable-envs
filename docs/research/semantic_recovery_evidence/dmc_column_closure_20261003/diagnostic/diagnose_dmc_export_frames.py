"""Run a frozen exporter gate with failure artifacts, without changing its assertions.

Invoke from the frozen source checkout with its original benchmark arguments.
This diagnostic is not an acceptance retry: even a non-reproduction leaves the
original failure unresolved. Artifacts are written next to --output.
"""

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    path = Path("scripts/benchmarks/online_stepping/dmc_state_export.py").resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("frozen_dmc_export_gate", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    original_equal = gate.equal
    output = Path(sys.argv[sys.argv.index("--output") + 1]).parent
    artifact = output / "diagnostic.json"
    if artifact.exists():
        raise FileExistsError(artifact)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="RUNNING", gate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), frame_checks=[])

    def frames_difference(a, b):
        delta = a.to(torch.float64) - b.to(torch.float64)
        return dict(shape=list(a.shape), differing_channels=int((a != b).sum().item()),
                    normalized_mse=float(delta.square().mean().item() / 255**2) if a.numel() else 0,
                    max_channel_delta=float(delta.abs().max().item()) if a.numel() else 0)

    def equal(expected, actual):
        caller = inspect.currentframe().f_back.f_locals
        context = dict(horizon=caller.get("horizon"), selection=repr(caller.get("selection")),
                       completed_cases=caller.get("cases"))
        if "frames" in expected.keys():
            report["frame_checks"].append(dict(**context, **frames_difference(expected["frames"], actual["frames"])))
        try:
            original_equal(expected, actual)
        except AssertionError:
            report.update(status="REPRODUCED", context=context)
            arrays, differences = {}, {}
            for key in expected.keys(True, True):
                name = "/".join(key) if isinstance(key, tuple) else key
                x, y = expected[key].detach().cpu().contiguous().numpy(), actual[key].detach().cpu().contiguous().numpy()
                arrays["expected/" + name], arrays["actual/" + name] = x, y
                numeric = x != y
                if x.dtype.kind == "f":
                    numeric &= ~(np.isnan(x) & np.isnan(y))
                differences[name] = dict(numeric_differences=int(numeric.sum()),
                                         byte_differences=int((x.view(np.uint8) != y.view(np.uint8)).sum()))
            report["field_differences"] = differences
            if "states" in expected.keys() and "frames" in expected.keys():
                rerenders = {}
                for label in ("reference", "candidate"):
                    env = caller[label]
                    arrays[label + "/style"] = env.get_style().detach().cpu().numpy()
                    for repeat in range(2):
                        states = expected["states"]
                        image = torch.stack([env.render(states[:, t]) for t in range(states.batch_size[1])], dim=1)
                        rerenders[f"{label}_{repeat}"] = image
                        arrays[f"rerender/{label}_{repeat}"] = image.cpu().numpy()
                report["rerenders_vs_original_reference"] = {
                    k: frames_difference(expected["frames"], v) for k, v in rerenders.items()
                }
                report["same_context_repeat"] = {
                    k: frames_difference(rerenders[k + "_0"], rerenders[k + "_1"])
                    for k in ("reference", "candidate")
                }
            np.savez_compressed(output / "failure_arrays.npz", **arrays)
            artifact.write_text(json.dumps(report, indent=2) + "\n")
            raise

    gate.equal = equal
    try:
        gate.main()
        report["status"] = "NOT_REPRODUCED"
    except BaseException as exc:
        report["error"] = repr(exc)
        if report["status"] == "RUNNING":
            report["status"] = "DIAGNOSTIC_FAILED"
        raise
    finally:
        artifact.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
