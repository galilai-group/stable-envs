"""Force mixed terminal lanes against untouched pinned-source PushT instances."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pusht_replay_gate import diagnostics, equal, import_source

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import stable_envs as se


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-source-report", type=Path, required=True)
    args = parser.parse_args()
    # Reuse the completed gate's exact source/production profile, checking
    # bytes afresh instead of inferring authority from package names.
    profile = json.loads(args.accepted_source_report.read_text())
    assert profile["status"] == "PASS_PUBLIC_STATE_SOURCE_CONTINUATION_CPU_RENDER"
    for path, digest in profile["source_files"].items():
        assert hashlib.sha256((args.source / path).read_bytes()).hexdigest() == digest
    for path, digest in profile["production_files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for discrete, source_class in enumerate(import_source(args.source)):
        name = "PushTDiscrete" if discrete else "PushT"
        for seed in (0, 17, 42):
            for selection_name, selection in (
                ("all", slice(None)),
                ("reverse", slice(None, None, -1)),
                ("sparse", slice(None, None, 3)),
                ("empty", slice(10, 10)),
            ):
                oracles = [source_class(resolution=96) for _ in range(2)]
                lane_states = []
                try:
                    for lane, oracle in enumerate(oracles):
                        start = np.array([200, 400, 250, 250, 0, 0, 0], np.float64)
                        goal = start.copy() if lane == 0 else np.array([100, 100, 400, 100, 1, 0, 0], np.float64)
                        options = dict(variation=[], state=start, goal_state=goal)
                        oracle.reset(seed=seed + lane, options=options)
                        with se.make(name, size=96, device="cpu") as env:
                            lane_states.append(env.reset(seed=seed + lane, options=options))
                    initial = torch.cat(lane_states)
                    with se.make(name, num_envs=2, size=96, device="cpu") as candidate:
                        candidate.set_state(initial)
                        actions = np.zeros((2, 8, 2), np.int64 if discrete else np.float32)
                        active = np.ones(2, np.bool_)
                        frames, observations, tasks, valid = [], [], [], []
                        for t in range(8):
                            valid.append(active.copy())
                            row = np.zeros((2, 3), np.float64)
                            for lane, oracle in enumerate(oracles):
                                if active[lane]:
                                    result = oracle.step(actions[lane, t].copy())
                                    row[lane] = result[1:4]
                                    active[lane] = not (result[2] or result[3])
                            frames.append(np.stack([e.render() for e in oracles]))
                            observations.append(np.stack([e._get_obs() for e in oracles]))
                            tasks.append(row)
                        wanted_valid = np.stack(valid, axis=1)
                        assert wanted_valid[0].tolist() == [True] + [False] * 7
                        assert wanted_valid[1].all()
                        actual = candidate.step(actions, states=selection, frames=selection)
                        equal(wanted_valid, actual["valid"].numpy())
                        expected_task = np.stack(tasks, axis=1)
                        actual_task = torch.stack(
                            [actual[k] for k in ("reward", "terminated", "truncated")], -1
                        ).numpy()
                        equal(expected_task, actual_task)
                        indices = list(range(8)[selection])
                        expected_frames = np.stack(frames, axis=1)[:, indices]
                        equal(expected_frames, actual["frames"].numpy())
                        equal(
                            np.stack(observations, axis=1)[:, indices],
                            actual["states"]["current", "observation"].numpy(),
                        )
                        endpoint = actual["state"].clone()
                        for lane, oracle in enumerate(oracles):
                            equal(diagnostics(oracle), diagnostics(candidate._simulator.envs[lane].env))
                        for index in range(len(indices)):
                            candidate.set_state(actual["states"][:, index])
                            equal(expected_frames[:, index], candidate.render().numpy())
                        candidate.set_state(endpoint)
                        # New public calls may advance a previously terminal
                        # source lane; chunk padding must not become autoreset.
                        source_next = [e.step(actions[i, 0].copy()) for i, e in enumerate(oracles)]
                        next_out = candidate.step(actions[:, 0], frames=None)
                        equal(
                            np.stack([v[0]["state"] for v in source_next]),
                            next_out["state"]["current", "observation"].numpy(),
                        )
                        for lane, oracle in enumerate(oracles):
                            equal(diagnostics(oracle), diagnostics(candidate._simulator.envs[lane].env))
                        case = f"{name}-s{seed}-{selection_name}"
                        path = args.output / (case + ".npz")
                        np.savez_compressed(
                            path,
                            source_valid=wanted_valid,
                            actual_valid=actual["valid"].numpy(),
                            source_frames=expected_frames,
                            actual_frames=actual["frames"].numpy(),
                            source_task=expected_task,
                            actual_task=actual_task,
                        )
                        cases.append(
                            dict(
                                registration=candidate.reference_id,
                                case=case,
                                selected_frames=len(indices) * 2,
                                artifact=path.name,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            )
                        )
                        print(case, "PASS", flush=True)
                finally:
                    for oracle in oracles:
                        oracle.close()
    report = dict(
        status="PASS_PUBLIC_MIXED_TERMINAL_SOURCE_GATE",
        cases=cases,
        source_files=profile["source_files"],
        production_files=profile["production_files"],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device="cpu",
        source_never_restored=True,
        registration_promotion=False,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
