"""Measure raster precision alternatives on the retained eight-action witness."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import stable_envs as se
from stable_envs.envs import rocket_render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import nvdiffrast.torch as dr

    raw = np.load(args.input, allow_pickle=False)
    source = raw["source"]
    report = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        backend_path=dr.__file__,
        gl_available=hasattr(dr, "RasterizeGLContext"),
        trials=[],
    )
    arrays = {"source": source}
    with se.make(
        "PyFlytRocketLanding",
        num_envs=2,
        device="cuda",
        size=96,
        simulator_config=dict(render_mode="rgb_array", render_resolution=(96, 96)),
    ) as env:
        renderer, _ = rocket_render._gpu_context(env, 96, 1)
        original = renderer.ctx
        contexts = [("cuda", original)]
        if hasattr(dr, "RasterizeGLContext"):
            try:
                contexts.append(("gl", dr.RasterizeGLContext(device="cuda")))
            except Exception as e:
                report["gl_error"] = repr(e)
        for label, ctx in contexts:
            renderer.ctx = ctx
            frames = []
            for t in range(8):
                prefix = f"state/{t}/"
                state = TensorDict(
                    {
                        tuple(k[len(prefix) :].split("/")): torch.from_numpy(raw[k])
                        for k in raw.files
                        if k.startswith(prefix)
                    },
                    [2],
                )
                # Render the portable numeric projection directly. Separately
                # retain whether this host can exactly replay a foreign-host
                # checkpoint; rendering must not be confounded with replay.
                if label == "cuda" and t == 0:
                    from stable_envs._adapters.rocket.replay import RocketReplay

                    get_state = RocketReplay.get_state
                    captured = []

                    def capture_replay(self):
                        data = get_state(self)
                        captured.append(data)
                        return data

                    RocketReplay.get_state = capture_replay
                    try:
                        env.set_state(state)
                        report["foreign_host_restore"] = "PASS_EXACT"
                    except ValueError as error:
                        report["foreign_host_restore"] = str(error)
                        deltas = {}
                        actual = captured[-1]
                        for key, value in state.items(True, True):
                            name = "/".join(key)
                            expected = value[0].numpy()
                            if not np.array_equal(expected, actual[name]):
                                arrays["restore_expected/" + name] = expected
                                arrays["restore_actual/" + name] = actual[name]
                                deltas[name] = float(np.max(np.abs(expected - actual[name])))
                        report["foreign_host_deltas"] = deltas
                    finally:
                        RocketReplay.get_state = get_state
                projected = env.to_render_state(state)
                frames.append(env.render(projected).cpu().numpy())
                if label == "cuda" and t == 2:
                    # Preserve the actual GPU clip inputs and mesh for numerical analysis.
                    fn = renderer.scene.clip

                    def capture(*a, **kw):
                        result = fn(*a, **kw)
                        arrays["clip"] = result[0].cpu().numpy()
                        arrays["faces"] = renderer.scene.face.cpu().numpy()
                        arrays["geom"] = renderer.scene.fgeom.cpu().numpy()
                        return result

                    renderer.scene.clip = capture
                    env.render(projected)
                    renderer.scene.clip = fn
            data = np.stack(frames, axis=1)
            arrays[label] = data
            report["trials"].append(
                dict(kind=label, mse=np.mean(((data.astype(float) - source) / 255) ** 2, axis=(-3, -2, -1)).tolist())
            )
        renderer.ctx = original
        # The averaging footprint must be centered on TinyRenderer's integer
        # sample: at scale s the matching offset is s/2, not a constant 1/2.
        for ss in (2, 4):
            r, _ = rocket_render._gpu_context(env, 96, ss)
            r.sample_offset = (ss / 2, ss / 2)
            frames = []
            for t in range(8):
                prefix = f"state/{t}/"
                state = TensorDict(
                    {
                        tuple(k[len(prefix) :].split("/")): torch.from_numpy(raw[k])
                        for k in raw.files
                        if k.startswith(prefix)
                    },
                    [2],
                )
                frames.append(env.render(state, supersample=ss).cpu().numpy())
            data = np.stack(frames, axis=1)
            arrays[f"centered_ss{ss}"] = data
            report["trials"].append(
                dict(
                    kind=f"centered_ss{ss}",
                    mse=np.mean(((data.astype(float) - source) / 255) ** 2, axis=(-3, -2, -1)).tolist(),
                )
            )
    np.savez_compressed(args.output / "witness.npz", **arrays)
    report.update(
        status="DIAGNOSTIC_COMPLETE",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(0),
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
