"""Independent pinned-source public PushT checkpoint and continuation gate.

Uses public construction, disk State, set_state and step; source lanes are
never restored. CUDA means State/frame output storage, not GPU rasterization.
Each action-space registration earns its own report rows. No promotion here.
"""

import argparse
import copy
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from pusht_replay_gate import PIN, action, diagnostics, equal, import_source
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import stable_envs as se


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--shapes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() == PIN
    source_files = {}
    for relative in (
        "spaces.py",
        "utils.py",
        "envs/utils.py",
        "envs/quantizer.py",
        "envs/pusht/env.py",
        "envs/pusht/env_discrete.py",
    ):
        name = "stable_worldmodel/" + relative
        path = args.source / name
        assert path.read_bytes() == subprocess.check_output(["git", "-C", str(args.source), "show", PIN + ":" + name])
        source_files[name] = sha(path)
    report = dict(
        status="RUNNING",
        source_pin=PIN,
        source_files=source_files,
        source_never_restored=True,
        device=args.device,
        render_execution="CPU_PYGAME",
        registration_promotion=False,
        cases=[],
        versions={
            k: md.version(k) for k in ("pymunk", "pygame", "numpy", "torch", "tensordict", "opencv-python-headless")
        },
        production_files={
            str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "stable_envs/_adapters").glob("pusht_*.py"))
        },
        script_sha256=sha(Path(__file__)),
    )
    for relative in ("stable_envs/envs/pusht_semantics.py", "stable_envs/_stepping.py"):
        report["production_files"][relative] = sha(ROOT / relative)

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for discrete, source_class in enumerate(import_source(args.source)):
        variant = "discrete" if discrete else "continuous"
        name = "PushTDiscrete" if discrete else "PushT"
        configurations = [
            (f"s{seed}-b{batch}-h{history}", seed, batch, history, None)
            for seed in (0, 17, 42)
            for batch in (1, 2)
            for history in (0, 100, 1025)
        ]
        if args.shapes:
            configurations += [
                (
                    f"shape{a}-{b}",
                    42,
                    1,
                    20,
                    dict(variation=["all"], variation_values={"agent.shape": a, "block.shape": b}),
                )
                for a in range(8)
                for b in range(1, 8)
            ]
        for label, seed, batch, prefix, options in configurations:
            case = f"{variant}-{label}"
            source = [source_class(resolution=96, render_action=True) for _ in range(batch)]
            try:
                with (
                    se.make(name, num_envs=batch, size=96, device=args.device, seed=seed, render_action=True) as env,
                    se.make(name, num_envs=batch, size=96, device=args.device, seed=999, render_action=True) as branch,
                ):
                    env.reset(seed=seed, options=copy.deepcopy(options))
                    for i, oracle in enumerate(source):
                        oracle.reset(seed=seed + i, options=copy.deepcopy(options))
                    for t in range(prefix):
                        controls = np.stack([action(e, variant, t) for e in source])
                        for e, a in zip(source, controls, strict=True):
                            e.step(a.copy())
                        env.step(controls, states=None, frames=None)
                    state = env.get_state()
                    frozen = state.clone()
                    disk = args.output / (case + "-state.npz")
                    np.savez_compressed(disk, **{"/".join(k): v.cpu().numpy() for k, v in state.items(True, True)})
                    with np.load(disk, allow_pickle=False) as data:
                        loaded = TensorDict(
                            {tuple(k.split("/")): torch.tensor(data[k], device=args.device) for k in data},
                            [batch],
                            device=args.device,
                        )
                    branch.set_state(loaded)
                    for i, oracle in enumerate(source):
                        equal(diagnostics(oracle), diagnostics(branch._simulator.envs[i].env))
                    left, right, source_obs, actual_obs, source_task, actual_task = [], [], [], [], [], []
                    future_steps = 16 if options is not None else 32
                    for t in range(future_steps):
                        controls = np.stack([action(e, variant, prefix + t) for e in source])
                        expected = [e.step(a.copy()) for e, a in zip(source, controls, strict=True)]
                        actual = branch.step(controls, states=None, frames=slice(-1, None))
                        s_obs = np.stack([v[0]["state"] for v in expected])
                        a_obs = actual["state"]["current", "observation"].cpu().numpy()
                        s_task = np.array([[v[1], v[2], v[3]] for v in expected])
                        a_task = (
                            torch.stack([actual[k] for k in ("reward", "terminated", "truncated")], -1).cpu().numpy()
                        )
                        equal(s_obs, a_obs)
                        equal(s_task, a_task)
                        l = np.stack([e.render() for e in source])
                        r = actual["frames"][:, 0].cpu().numpy()
                        equal(l, r, case + "/frame")
                        for i, oracle in enumerate(source):
                            equal(diagnostics(oracle), diagnostics(branch._simulator.envs[i].env))
                        left.append(l)
                        right.append(r)
                        source_obs.append(s_obs)
                        actual_obs.append(a_obs)
                        source_task.append(s_task)
                        actual_task.append(a_task)
                    # Open-loop chunks use the same public API and stop each
                    # source lane after its own terminal transition.
                    controls = np.stack(
                        [np.stack([action(e, variant, prefix + future_steps + t) for t in range(8)]) for e in source]
                    )
                    active = np.ones(batch, np.bool_)
                    chunk_obs, chunk_tasks, chunk_frames, validity = [], [], [], []
                    for t in range(8):
                        validity.append(active.copy())
                        tasks = np.zeros((batch, 3), np.float64)
                        for i, oracle in enumerate(source):
                            if active[i]:
                                result = oracle.step(controls[i, t].copy())
                                tasks[i] = result[1:4]
                                active[i] = not (result[2] or result[3])
                        chunk_obs.append(np.stack([e._get_obs() for e in source]))
                        chunk_tasks.append(tasks)
                        chunk_frames.append(np.stack([e.render() for e in source]))
                    chunk = branch.step(controls, states=slice(None), frames=slice(None))
                    equal(np.stack(validity, axis=1), chunk["valid"].cpu().numpy())
                    equal(np.stack(chunk_obs, axis=1), chunk["states"]["current", "observation"].cpu().numpy())
                    equal(
                        np.stack(chunk_tasks, axis=1),
                        torch.stack([chunk[k] for k in ("reward", "terminated", "truncated")], -1).cpu().numpy(),
                    )
                    equal(np.stack(chunk_frames, axis=1), chunk["frames"].cpu().numpy())
                    for i, oracle in enumerate(source):
                        equal(diagnostics(oracle), diagnostics(branch._simulator.envs[i].env))
                    for key, value in frozen.items(True, True):
                        assert torch.equal(value, state[key]), key
                    witness = args.output / (case + "-witness.npz")
                    np.savez_compressed(
                        witness,
                        source_frames=left,
                        actual_frames=right,
                        source_observations=source_obs,
                        actual_observations=actual_obs,
                        source_task=source_task,
                        actual_task=actual_task,
                        chunk_source_frames=np.stack(chunk_frames, axis=1),
                        chunk_actual_frames=chunk["frames"].cpu().numpy(),
                        chunk_source_observations=np.stack(chunk_obs, axis=1),
                        chunk_actual_observations=chunk["states"]["current", "observation"].cpu().numpy(),
                        chunk_source_task=np.stack(chunk_tasks, axis=1),
                        chunk_actual_task=torch.stack([chunk[k] for k in ("reward", "terminated", "truncated")], -1)
                        .cpu()
                        .numpy(),
                        chunk_source_valid=np.stack(validity, axis=1),
                        chunk_actual_valid=chunk["valid"].cpu().numpy(),
                    )
                    report["cases"].append(
                        dict(
                            registration=env.reference_id,
                            case=case,
                            batch=batch,
                            prefix=prefix,
                            future_transitions=batch * future_steps + int(np.sum(validity)),
                            frames=batch * (future_steps + 8),
                            chunk_length=8,
                            mse=0.0,
                            state=disk.name,
                            state_sha256=sha(disk),
                            witness=witness.name,
                            witness_sha256=sha(witness),
                        )
                    )
                    save()
                    print(case, "PASS", flush=True)
            finally:
                for oracle in source:
                    oracle.close()
    report["status"] = "PASS_PUBLIC_STATE_SOURCE_CONTINUATION_CPU_RENDER"
    save()


if __name__ == "__main__":
    main()
