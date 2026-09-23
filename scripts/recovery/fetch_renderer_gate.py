"""Fetch frozen baseline comparison and CPU-State to GPU-render boundary."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import stable_envs as se


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    assert torch.cuda.is_available()
    path = Path("tests/fixtures/regression/join/step4/states/FetchPickAndPlace.npz")
    with np.load(path) as archive:
        rows = archive["states"][:4].copy()
    with se.make("FetchPickAndPlace", num_envs=4, device="cuda") as env:
        frame = env._render(rows).cpu().numpy()
        frozen = {
            "FetchPickAndPlace": {
                "rows_sha256": digest(rows.tobytes()),
                "frame_sha256": digest(frame.tobytes()),
                "shape": list(frame.shape),
            }
        }
    args.output.write_text(json.dumps(frozen, indent=2) + "\n")
    if args.baseline:
        return
    values = json.loads(args.fixture.read_text())
    state = TensorDict({}, [3], device="cuda")
    for name, spec in values.items():
        state[tuple(name.split("."))] = torch.tensor(
            spec["values"], dtype=getattr(torch, spec["dtype"]), device="cuda"
        )
    with se.make("FetchPickAndPlace", num_envs=3, device="cuda") as env:
        original = state.clone()
        live = env.get_state()
        full = env.render(state).clone()
        projected = env._render(env._pack_render_state(state)).clone()
        assert torch.equal(full, projected)
        for key in state.keys(True, True):
            assert torch.equal(state[key], original[key]), key
            assert torch.equal(live[key], env.get_state()[key]), key
        for bad in (env._pack_render_state(state), env._unpack_render_state(env._pack_render_state(state))):
            try:
                env.set_state(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("compact renderer state restored")
        env.set_variation({"object.color": [0.2, 0.8, 0.3]})
        varied = env.render(state)
        assert not torch.equal(full, varied), "object.color must affect frame"
        assert torch.equal(varied, env._render(env._pack_render_state(state)))
        for key in state.keys(True, True):
            assert torch.equal(state[key], original[key]), key
            assert torch.equal(live[key], env.get_state()[key]), key
    result = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "fixture_sha256": digest(args.fixture.read_bytes()),
        "profile_versions": {
            k: importlib.metadata.version(k)
            for k in (
                "torch",
                "numpy",
                "gymnasium",
                "gymnasium-robotics",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
                "tensordict",
            )
        },
        "semantics_claim": "Renderer-only profile; no physics parity claim for MuJoCo >=3.11",
        "full_private_equality": True,
        "snapshot_nonmutation": True,
        "simulator_nonmutation": True,
        "compact_restoration_refused": True,
        "object_color_changes_only_frame": True,
        "canonical_frame_sha256": digest(full.cpu().numpy().tobytes()),
        "cpu_renderer": "CPU_BACKEND_MISSING",
    }
    args.gate_output.write_text(json.dumps(result, indent=2) + "\n")
    print("FETCH_STATE_RENDERER_GATE_PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--gate-output", type=Path)
    main(p.parse_args())
