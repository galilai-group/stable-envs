"""Measure trajectory-induced DMC pixel drift through one unchanged renderer.

Default coverage is all 16 supported tasks, 10 evaluation seeds, three lanes, and public-step
horizons 1 and 8. Pass --modes source,variation,contact for the applicable varied
and source-owned contact/constraint fixtures as well. This diagnostic uses no
pixel acceptance threshold and does not replace the exact identical-State gate.

Example: python benchmarks/dmc_mjwarp_pixels.py --source-root PINNED_CHECKOUT
--output NEW_REPORT.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmarks import _common as common  # noqa: E402
from stable_envs.testing.evidence import sha256  # noqa: E402
from stable_envs.testing.metrics import normalized_mse, summarize_errors  # noqa: E402
from tests.helpers.reference.dmc_mjwarp import (  # noqa: E402
    BATCH,
    CELLS,
    HELD_OUT_SEEDS,
    HORIZONS,
    Oracle,
    candidate_identity,
    machine_identity,
    make_candidate,
    prepare_source,
    recipe_identity,
    source_modes,
    source_trial,
)
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

METRICS = ("normalized_mse", "changed_pixel_fraction", "changed_channel_fraction", "max_abs_channel_error_uint8")


def _path(key):
    return (key,) if isinstance(key, str) else key


def compatible_state(partial, template):
    """Only absent sibling-oracle physics metadata may come from the matched model."""
    source_keys, full_keys = set(partial.keys(True, True)), set(template.keys(True, True))
    if source_keys - full_keys:
        raise ValueError(f"Unexpected source State fields: {source_keys - full_keys}")
    missing = full_keys - source_keys
    if any(_path(key)[0] != "physics" for key in missing):
        raise ValueError(f"Source State lacks non-physics fields: {missing}")
    full = partial.to(template.device).clone()
    for key in missing:
        full[key] = template[key].clone()
    return full, sorted("/".join(_path(key)) for key in missing)


def pixel_metrics(actual, reference):
    """Per-frame RGB metrics, with strict uint8 inputs and no implicit rescaling."""
    mse = normalized_mse(actual, reference)
    if actual.ndim != 4:
        raise ValueError("Expected batched RGB images [B,H,W,3]")
    changed = actual != reference
    delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    return {
        "normalized_mse": mse,
        "changed_pixel_fraction": changed.any(axis=-1).mean(axis=(1, 2)),
        "changed_channel_fraction": changed.mean(axis=(1, 2, 3)),
        "max_abs_channel_error_uint8": delta.max(axis=(1, 2, 3)),
    }


def measure_pair(env, actual, reference, template):
    actual, missing_actual = compatible_state(actual, template)
    reference, missing_reference = compatible_state(reference, template)
    # Appearance is the same live renderer style for both images. Physics and
    # episode model entries must also match: only physical trajectories differ.
    for key in actual.keys(True, True):
        path = _path(key)
        if path[0] in {"physics", "model"} or path == ("task", "target_pos"):
            assert_same_state(actual[key], reference[key])
    live = env.get_state()
    image_actual = env.render(actual).clone()
    assert_same_state(env.get_state(), live)
    image_reference = env.render(reference).clone()
    assert_same_state(env.get_state(), live)
    repeated = env.render(actual).clone()
    assert_same_state(env.get_state(), live)
    assert_same_state(image_actual, repeated)
    metrics = pixel_metrics(image_actual.cpu().numpy(), image_reference.cpu().numpy())
    qpos_difference = (actual["qpos"] - reference["qpos"]).abs().amax(dim=1).cpu().numpy()
    return (
        metrics,
        qpos_difference,
        {
            "frame_shape": list(image_actual.shape[1:]),
            "candidate_supplemented_fields": missing_actual,
            "reference_supplemented_fields": missing_reference,
            "identical_state_pixels_exact": True,
            "live_state_unchanged": True,
        },
    )


def summaries(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["env"], row["mode"], row["horizon"])].append(row)
    output = []
    for (name, mode, horizon), values in sorted(groups.items()):
        worst = max(values, key=lambda row: row["normalized_mse"])
        output.append(
            {
                "env": name,
                "mode": mode,
                "horizon": horizon,
                "frames": len(values),
                "metrics": {key: summarize_errors([row[key] for row in values]) for key in METRICS},
                "worst_mse_frame": {key: worst[key] for key in ("seed", "lane", "normalized_mse")},
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=os.environ.get("SE_DMC_MJWARP_SOURCE_ROOT"))
    parser.add_argument("--output", required=True, help="new JSON report path")
    parser.add_argument("--envs", type=common.names, default=list(CELLS))
    parser.add_argument("--modes", type=common.names, default=["source"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(HELD_OUT_SEEDS))
    parser.add_argument("--size", type=common.positive_int)
    parser.add_argument("--supersample", type=common.positive_int)
    parser.add_argument("--backend", default="auto", help="same renderer request for both trajectories")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=common.positive_int, default=1)
    args = parser.parse_args()
    if set(args.envs) - set(CELLS):
        parser.error(f"Unknown DMC cells: {sorted(set(args.envs) - set(CELLS))}")
    if set(args.modes) - {"source", "boundary", "variation", "contact"}:
        parser.error("Modes must be source,boundary,variation,contact")
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - set(HELD_OUT_SEEDS):
        parser.error(
            f"Seeds must be distinct members of the frozen held-out set {HELD_OUT_SEEDS[0]}..{HELD_OUT_SEEDS[-1]}"
        )
    if Path(args.output).exists():
        parser.error("Output already exists; use a new report path")
    device = common.checked_device(args.device)
    if device.type != "cuda":
        parser.error("Trajectory pixel diagnostics require actual CUDA Warp physics")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    args.device = str(device)
    torch.cuda.set_device(device)
    common.configure_threads(args.threads)
    with redirect_stdout(sys.stderr):
        source = prepare_source(args.source_root)
        machine = machine_identity()
    import stable_envs as se

    applicable = {name: [mode for mode in args.modes if mode in source_modes(name)] for name in args.envs}
    kwargs = {key: getattr(args, key) for key in ("size", "supersample") if getattr(args, key) is not None}
    kwargs["backend"] = args.backend
    report = common.provenance(benchmark="dmc-mjwarp-trajectory-pixels", args=args, snv_version=se.__version__)
    report.update(
        schema="stable-envs-dmc-mjwarp-trajectory-pixels/v1",
        scope="trajectory-induced drift through the same renderer; no source-renderer fidelity claim or pixel cap",
        machine=machine,
        source=source,
        candidate_files=candidate_identity(),
        recipe_files=recipe_identity(),
        diagnostic_files={
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (Path(__file__).resolve(), ROOT / "benchmarks/_common.py")
        },
        batch=BATCH,
        horizons=list(HORIZONS),
        horizon_definition="endpoint after N public actions; each public action contains two control rounds",
        modes_by_environment=applicable,
        expected_frames=sum(map(len, applicable.values())) * len(args.seeds) * len(HORIZONS) * BATCH,
        metric_definitions={
            "normalized_mse": "mean squared RGB channel difference / 255^2",
            "changed_pixel_fraction": "fraction of spatial pixels with any unequal RGB channel",
            "changed_channel_fraction": "fraction of individual RGB channels that differ",
            "max_abs_channel_error_uint8": "maximum absolute RGB channel error on the 0..255 scale",
        },
        supplementation="Only physics metadata absent from sibling-oracle State is copied from the matched live configuration",
        trials=[],
        frames=[],
        errors=[],
        complete=False,
    )
    for name in args.envs:
        with make_candidate(name, **kwargs) as env:
            for mode in applicable[name]:
                oracle = Oracle(name, varied=mode == "variation")
                try:
                    for seed in args.seeds:
                        try:
                            actions, pairs = source_trial(
                                env, oracle, seed, boundary=mode == "boundary", contact=mode == "contact"
                            )
                            template = env.get_state()
                            trial = {
                                "env": name,
                                "mode": mode,
                                "seed": seed,
                                "actions": actions.tolist(),
                                "actions_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
                                "render_size": env.size,
                                "render_supersample": env.ss,
                                "render_backend_request": env.backend,
                            }
                            if mode == "contact":
                                trial["contact_evidence"] = oracle.contact_evidence
                            for horizon in HORIZONS:
                                actual, expected = pairs[horizon - 1]
                                metrics, delta, details = measure_pair(
                                    env, actual["state"], expected["state"], template
                                )
                                trial.update(details)
                                for lane in range(BATCH):
                                    report["frames"].append(
                                        {
                                            "env": name,
                                            "mode": mode,
                                            "seed": seed,
                                            "horizon": horizon,
                                            "lane": lane,
                                            "qpos_max_abs_difference": float(delta[lane]),
                                            **{key: float(value[lane]) for key, value in metrics.items()},
                                        }
                                    )
                            report["trials"].append(trial)
                        except Exception as error:
                            report["errors"].append(
                                {"env": name, "mode": mode, "seed": seed, **common.error_record(error)}
                            )
                        report["summaries"] = summaries(report["frames"])
                        common.write_json(report, args.output, pretty=True)
                    print(f"Pixel drift {name} {mode}: {len(args.seeds)} requested seeds", file=sys.stderr, flush=True)
                finally:
                    oracle.close()
    report["complete"] = True
    report["identical_state_fidelity"] = {
        "successful_trials_exact_and_observational": len(report["trials"]),
        "check": "candidate image rerendered after the reference image; live State checked after every render",
    }
    report["summaries"] = summaries(report["frames"])
    common.finish_provenance(report)
    common.write_json(report, args.output, pretty=True)
    return int(bool(report["errors"]) or len(report["frames"]) != report["expected_frames"])


if __name__ == "__main__":
    raise SystemExit(main())
