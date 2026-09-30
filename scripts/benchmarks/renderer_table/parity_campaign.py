"""Fresh 1,000-frame campaign, one renderer role per process.

Raw per-frame timings and images are retained. No device substitution, oracle
image cache substitution, or weakened state witness is permitted.
"""

import argparse
import hashlib
import inspect
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from . import collect
from .common import catalog, load_state, profile, public_render, rgb, state_hash, timed, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("registration")
    p.add_argument("role", choices=["source_cpu", "source_gpu", "ours_cpu", "ours_gpu"])
    p.add_argument("output", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--backend", default="auto")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    row = catalog()[a.registration]
    device = "cuda" if a.role.endswith("gpu") else "cpu"
    result = dict(
        registration=a.registration,
        role=a.role,
        backend=a.backend,
        profile=profile(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        requested_frames=a.frames,
        status="RUNNING",
        frames=[],
    )
    write(a.output / "campaign.json", result)
    try:
        if a.role.startswith("source") and (
            row["name"].startswith(("DMC", "Fetch", "OGB")) or row["name"] == "PointMazeLargeNavigate"
        ):
            import mujoco
            from OpenGL import GL

            model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
            )
            gl_probe = mujoco.Renderer(model, height=32, width=32)
            gl_probe._gl_context.make_current()
            result["gl_vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
            result["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
            software = any(s in result["gl_renderer"].lower() for s in ("llvmpipe", "softpipe", "swrast"))
            assert software if device == "cpu" else "nvidia" in result["gl_vendor"].lower() and not software
            write(a.output / "campaign.json", result)
        if a.role.startswith("source"):
            original_bridge = collect.Bridge

            def audited_bridge(row, seed, options):
                bridge = original_bridge(row, seed, options)
                source = getattr(bridge.source, "unwrapped", bridge.source)
                paths = {inspect.getsourcefile(type(source).__init__), inspect.getsourcefile(type(source).render)}
                write(
                    a.output / "source-paths.json",
                    {
                        "class": str(type(source)),
                        "sources": {
                            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths if path
                        },
                    },
                )
                return bridge

            collect.Bridge = audited_bridge
            if "Craftax" in row["name"] and device == "cuda":
                import jax

                assert jax.default_backend() == "gpu", jax.devices()
                original_candidate = collect.candidate
                collect.candidate = lambda r, d, seed, config=None: original_candidate(r, "cuda", seed, config)
            collected = collect.cpu(row, a.output, target_frames=a.frames, source_only=True)
            result["collected_frames"] = len(collected["frames"])
        else:
            corpus = json.loads((a.corpus / "result.json").read_text())
            assert corpus["status"] == "COMPLETE"
            result["corpus_sha256"] = hashlib.sha256((a.corpus / "result.json").read_bytes()).hexdigest()
            env, active_config = None, object()
            for f in corpus["frames"]:
                config = corpus["recipes"][f["case"]].get("simulator_config")
                if config != active_config:
                    if env is not None:
                        env.close()
                    env = collect.candidate(row, device, 0, config)
                    active_config = config
                    if row["name"].startswith("Craftax"):
                        env.reset(0)
                    baseline = state_hash(env.get_state())
                state = load_state(a.corpus / f["state_file"], device, env.get_state())
                assert state_hash(state) == f["state_hash"]
                kw = {
                    k: torch.tensor(v, dtype=torch.float32, device=device) if k == "style" else v
                    for k, v in f["render_kwargs"].items()
                }
                if a.backend != "auto":
                    kw["backend"] = a.backend
                timing, output = timed(
                    lambda: public_render(env, state, **kw), gpu=device == "cuda", samples=1, warmups=1
                )
                image = rgb(output).copy()
                assert state_hash(state) == f["state_hash"] and state_hash(env.get_state()) == baseline
                assert torch.is_tensor(output) and output.device.type == device
                filename = f"frame-{f['index']:04d}.npy"
                np.save(a.output / filename, image, allow_pickle=False)
                result["frames"].append(
                    dict(
                        index=f["index"],
                        case=f["case"],
                        seed=f["seed"],
                        step=f["step"],
                        state_hash=f["state_hash"],
                        timing=timing,
                        image_file=filename,
                        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    )
                )
                write(a.output / "campaign.json", result)
                print(a.registration, a.role, f["index"], flush=True)
            if not corpus["frames"]:
                result["unavailable"] = corpus.get("unavailable", {})
            else:
                assert len(result["frames"]) == a.frames
        result["status"] = "COMPLETE"
    except NotImplementedError as e:
        result.update(status="UNSUPPORTED", reason=str(e))
    except Exception:
        result.update(status="ERROR", traceback=traceback.format_exc())
        raise
    finally:
        write(a.output / "campaign.json", result)


if __name__ == "__main__":
    main()
