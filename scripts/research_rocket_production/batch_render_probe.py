"""Localize Rocket online-vs-single rendering without altering production."""

import argparse
import hashlib
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

import stable_envs as se


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = dict(render_mode="rgb_array", render_resolution=(96, 96))
    sources = [Oracle(**config) for _ in range(2)]
    with (
        se.make("PyFlytRocketLanding", num_envs=2, device="cpu", size=96, simulator_config=config) as ref,
        se.make("PyFlytRocketLanding", num_envs=2, device="cuda", size=96, simulator_config=config) as gpu,
        se.make("PyFlytRocketLanding", num_envs=1, device="cuda", size=96, simulator_config=config) as single,
    ):
        try:
            ref.reset(seed=17)
            for i, source in enumerate(sources):
                source.reset(seed=17 + i)
                for _ in range([0, 3][i]):
                    source.step(np.zeros(7, np.float32))
                    ref._simulator.envs[i].step(np.zeros(7, np.float32))
                assert not differences(capture(source), capture(ref._simulator.envs[i].env))
            initial = ref.get_state()
            a = np.tile(np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1], np.float32), (2, 1))
            states, source_frames, cpu_frames = [], [], []
            for t in range(8):
                for source, action in zip(sources, a, strict=True):
                    source.step(action.copy())
                ref.step(a, frames=None)
                states.append(ref.get_state())
                source_frames.append(np.stack([e.render() for e in sources]))
                cpu_frames.append(ref.render().numpy())
            source = np.stack(source_frames, axis=1)
            arrays = {"source": source, "cpu": np.stack(cpu_frames, axis=1)}
            np.testing.assert_array_equal(arrays["cpu"], source)
            gpu.set_state(initial)
            out = gpu.step(np.repeat(a[:, None], 8, axis=1), states=slice(None), frames=slice(None))
            arrays["online"] = out["frames"].cpu().numpy()
            comparisons = {key: [] for key in ("render_restored", "render_chunk1", "single", "reversed")}
            for t, state in enumerate(states):
                for k, v in state.items(True, True):
                    assert torch.equal(v, out["states"][:, t][k].cpu()), (t, k)
                    arrays[f"state/{t}/" + "/".join(k)] = v.numpy()
                gpu.set_state(state)
                comparisons["render_restored"].append(gpu.render().cpu().numpy())
                comparisons["render_chunk1"].append(gpu.render(chunk_size=1).cpu().numpy())
                solo = []
                for i in range(2):
                    single.set_state(state[i : i + 1])
                    solo.append(single.render()[0].cpu().numpy())
                comparisons["single"].append(np.stack(solo))
                # Check material/clip rounding independently of batching order.
                gpu.set_state(state[torch.tensor([1, 0])])
                comparisons["reversed"].append(gpu.render().cpu().numpy()[::-1].copy())
            arrays.update({k: np.stack(v, axis=1) for k, v in comparisons.items()})
            np.savez_compressed(args.output / "witness.npz", **arrays)
            scores = {
                k: np.mean(((v.astype(float) - source) / 255) ** 2, axis=(-3, -2, -1)).tolist()
                for k, v in arrays.items()
                if not k.startswith("state/")
            }
            report = dict(
                status="DIAGNOSTIC_COMPLETE",
                job_id=os.environ.get("SLURM_JOB_ID"),
                gpu=torch.cuda.get_device_name(0),
                scores=scores,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
        finally:
            for e in sources:
                e.close()
                if e.modified_pad_urdf:
                    Path(e.modified_pad_urdf).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
