"""Bounded Rocket action-chunk/selection gate against independent source lanes.

Source lanes advance normally and are never restored. The public chunk API
holds a lane after its first terminal transition within the submitted chunk;
a new call may still exercise the source's post-terminal behavior.
No throughput claim is made by this correctness script.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))
from rocket_feasibility import RocketLandingEnv as Oracle
from rocket_feasibility import capture, differences

import stable_envs


def equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        assert torch.equal(a[key].cpu(), b[key].cpu()), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = "cuda" if args.gpu else "cpu"
    source_file = Path(inspect.getfile(Oracle))
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    assert source_hash == "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"
    cases = []
    frames, references = [], []
    for name, config, options, prefixes in [
        ("flight", {}, {}, [0, 3]),
        ("contact", {}, {}, [301, 303]),
        ("success", {"ceiling": 3.2}, {"accelerate_drop": False}, [0, 7]),
        ("horizon", {"ceiling": 500000.0}, {"accelerate_drop": False}, [1198, 1201]),
    ]:
        config = dict(config, render_mode="rgb_array", render_resolution=(96, 96))
        for dtype in (np.float32, np.float64):
            for length in (1, 8):
                source = [Oracle(**config), Oracle(**config)]
                reference = stable_envs.make(
                    "PyFlytRocketLanding", num_envs=2, device="cpu", size=96, simulator_config=config
                )
                candidate = stable_envs.make(
                    "PyFlytRocketLanding", num_envs=2, device=device, size=96, simulator_config=config
                )
                try:
                    reference.reset(seed=17, options=options)
                    zero = np.zeros(7, dtype=dtype)
                    for lane, oracle in enumerate(source):
                        oracle.reset(seed=17 + lane, options=options)
                        for _ in range(prefixes[lane]):
                            oracle.step(zero.copy())
                            reference._simulator.envs[lane].step(zero.copy())
                        assert not differences(capture(oracle), capture(reference._simulator.envs[lane].env))
                    initial = reference.get_state()
                    actions = np.zeros((2, length, 7), dtype=dtype)
                    if name == "flight":
                        actions[:, :, :] = [0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1]
                    active = np.ones(2, np.bool_)
                    states, tasks, validity, source_frames = [], [], [], []
                    for t in range(length):
                        validity.append(active.copy())
                        row = np.zeros((2, 4), np.float64)
                        next_active = active.copy()
                        for lane, oracle in enumerate(source):
                            if active[lane]:
                                out = oracle.step(actions[lane, t].copy())
                                row[lane] = [out[1], out[2], out[3], out[4]["env_complete"]]
                                next_active[lane] = not (out[2] or out[3])
                        reference._simulator.step(actions[:, t], active=torch.from_numpy(active))
                        for lane, oracle in enumerate(source):
                            assert not differences(capture(oracle), capture(reference._simulator.envs[lane].env))
                        states.append(reference.get_state())
                        tasks.append(row)
                        source_frames.append(np.stack([oracle.render() for oracle in source]))
                        active = next_active
                    expected_task = np.stack(tasks, axis=1)
                    expected_valid = torch.from_numpy(np.stack(validity, axis=1))
                    for label, selection in [
                        ("none", None),
                        ("empty", slice(9, 10)),
                        ("all", slice(None)),
                        ("reverse_sparse", slice(None, None, -2)),
                    ]:
                        candidate.set_state(initial)
                        result = candidate.step(
                            actions if length > 1 else actions[:, 0], states=selection, frames=selection
                        )
                        assert torch.equal(result["valid"].cpu(), expected_valid)
                        for column, key in enumerate(("reward", "terminated", "truncated", "success")):
                            expected = expected_task[:, :, column] if length > 1 else expected_task[:, 0, column]
                            np.testing.assert_array_equal(result[key].cpu().numpy(), expected)
                        equal(result["state"], states[-1])
                        equal(candidate.get_state(), states[-1])
                        checked, errors = 0, []
                        if selection is not None:
                            indices = list(range(length)[selection])
                            assert result["states"].batch_size == torch.Size([2, len(indices)])
                            assert torch.equal(result["frame_valid"].cpu(), expected_valid[:, indices])
                            assert torch.equal(result["state_valid"].cpu(), expected_valid[:, indices])
                            for j, t in enumerate(indices):
                                equal(result["states"][:, j], states[t])
                                ours = result["frames"][:, j].cpu().numpy()
                                oracle = source_frames[t]
                                error = np.mean(((ours.astype(np.float64) - oracle) / 255) ** 2, axis=(1, 2, 3))
                                assert np.all(error <= 0.001), error
                                errors.extend(error.tolist())
                                frames.extend(ours)
                                references.extend(oracle)
                                checked += 2
                        held = result.clone()
                        candidate.step(np.zeros((2, 7), dtype=dtype), states=None, frames=None)
                        equal(held, result)
                        cases.append(
                            dict(
                                scenario=name,
                                dtype=np.dtype(dtype).name,
                                length=length,
                                selection=label,
                                source_never_restored=True,
                                exact_state_task_lifecycle=True,
                                owned_snapshots=True,
                                valid_counts=expected_valid.sum(1).tolist(),
                                frames=checked,
                                mse_mean=float(np.mean(errors)) if errors else None,
                                mse_max=max(errors) if errors else None,
                            )
                        )
                    print(name, np.dtype(dtype).name, length, "PASS", flush=True)
                finally:
                    candidate.close()
                    reference.close()
                    for oracle in source:
                        oracle.close()
                        if oracle.modified_pad_urdf:
                            Path(oracle.modified_pad_urdf).unlink(missing_ok=True)
    np.savez_compressed(args.output / "frames.npz", ours=frames, source=references)
    files = list(ROOT.glob("stable_envs/_adapters/rocket*.py")) + list(ROOT.glob("stable_envs/envs/rocket*.py"))
    files += [ROOT / "stable_envs/_stepping.py", ROOT / "stable_envs/env.py"]
    report = dict(
        status="PASS_BOUNDED_ROCKET_ONLINE_SOURCE",
        device=device,
        cases=cases,
        source_sha256=source_hash,
        job_id=os.environ.get("SLURM_JOB_ID"),
        frames=len(frames),
        size=96,
        batch=2,
        production_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="B2, T1/T8, float32/64, four scenarios and four selectors; not a performance measurement",
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
