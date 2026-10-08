"""Independent render-model and supersampling ablations; never changes State.

The OpenGL arm is a research control, not a production backend or oracle replay.
It compiles the accepted candidate's assets and consumes only its RenderState.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .collect import candidate
from .common import catalog, load_state, profile, rgb, state_hash, write
from .config_fixes import mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("name", choices=["OGBCube", "OGBScene"])
    p.add_argument("mode", choices=["opengl", "production_opengl", "ss1", "ss2", "ss4"])
    p.add_argument("device", choices=["cpu", "cuda"])
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    row = next(r for r in catalog().values() if r["name"] == a.name)
    corpus = a.campaign / "run-6751001_3/results" / a.name
    metadata = json.loads((corpus / "cpu/result.json").read_text())
    env = candidate(row, a.device, 17)
    baseline = state_hash(env.get_state())
    cache = {}
    result = dict(name=a.name, mode=a.mode, device=a.device, status="RUNNING", profile=profile(), frames=[])
    write(a.output / "result.json", result)
    family = a.name[3:]
    for f in metadata["frames"]:
        # The expensive supersampling ablation uses one frame per case plus
        # Scene's reported worst frame; the independent GL control uses all.
        if (
            a.mode not in ("opengl", "production_opengl")
            and f["index"] != 114
            and any(x["case"] == f["case"] for x in result["frames"])
        ):
            continue
        state = load_state(corpus / "cpu" / f["state_file"], a.device, env.get_state())
        assert state_hash(state) == f["state_hash"]
        rs = env.to_render_state(state)
        if a.mode == "opengl":
            import mujoco
            from OpenGL import GL

            from stable_envs._adapters.ogbench.compatibility import MODEL_VALUES, CompatibleCube, CompatibleScene

            r = rs[0].cpu()
            key = hashlib.sha256(b"".join(r["appearance", k].numpy().tobytes() for k in MODEL_VALUES)).hexdigest()
            if key not in cache:
                model_env = (CompatibleCube if family == "Cube" else CompatibleScene)(seed=0)
                for k in MODEL_VALUES:
                    model_env.variation_space._walk(k)._value = r["appearance", k].numpy().copy()
                model_env._mjcf_model = model_env.modify_mjcf_model(model_env.build_mjcf_model())
                model_env.compile_model_and_data()
                size = int(r["renderer", "size"].item())
                renderer = mujoco.Renderer(model_env.model, height=size, width=size)
                cache[key] = model_env, renderer
                renderer._gl_context.make_current()
                result["gl_vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
                result["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
                result["offsamples"] = int(model_env.model.vis.quality.offsamples)
            model_env, renderer = cache[key]

            def draw():
                model_env.data.qpos[:] = r["pose", "qpos"].numpy()
                model_env.data.mocap_pos[:] = r["pose", "mocap_pos"].numpy()
                model_env.data.mocap_quat[:] = r["pose", "mocap_quat"].numpy()
                model_env.variation_space._walk("cube.color")._value = r["appearance", "cube.color"].numpy().copy()
                model_env._cube_colors_from_variation()
                for ids in model_env._cube_target_geom_ids_list:
                    model_env.model.geom_rgba[ids, 3] = 0.2
                if family == "Scene":
                    model_env.variation_space._walk("lock_color")._value = r["appearance", "lock_color"].numpy().copy()
                    model_env._cur_button_states = r["pose", "buttons"].numpy().copy()
                    model_env._apply_button_states()
                mujoco.mj_forward(model_env.model, model_env.data)
                renderer.update_scene(model_env.data, camera="front_pixels")
                return renderer.render().copy()
        elif a.mode == "production_opengl":

            def draw():
                return rgb(env.render(rs, backend="opengl")).copy()

        else:
            rs["renderer", "supersample"][:] = int(a.mode[-1])

            def draw():
                return rgb(env.render(rs)).copy()

        draw()  # Compile/upload/warm-up is outside timing.
        timings = []
        for _ in range(3):
            if a.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            image = draw()
            if a.device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter_ns() - start) / 1e6)
        repeated = draw()
        repeat_delta = np.abs(image.astype(np.int16) - repeated.astype(np.int16))
        assert state_hash(state) == f["state_hash"] and state_hash(env.get_state()) == baseline
        sources = {d: np.load(corpus / d / f["oracle_file"], allow_pickle=False) for d in ["cpu", "source_gpu"]}
        np.savez_compressed(a.output / f"frame-{f['index']:04d}.npz", ours=image, **sources)
        result["frames"].append(
            dict(
                index=f["index"],
                case=f["case"],
                seed=f["seed"],
                step=f["step"],
                state_hash=f["state_hash"],
                ms=timings,
                repeat_mse=mse(image, repeated),
                repeat_max_channel_delta=int(repeat_delta.max()),
                repeat_changed_channels=int(np.count_nonzero(repeat_delta)),
                mse_cpu=mse(image, sources["cpu"]),
                mse_gpu=mse(image, sources["source_gpu"]),
            )
        )
        write(a.output / "result.json", result)
        print(a.name, a.mode, f["index"], result["frames"][-1]["mse_gpu"], flush=True)
    result["status"] = "PASS"
    write(a.output / "result.json", result)
    # Keep all GL contexts alive until process exit: mixing context destruction
    # and surviving dm_control contexts can invalidate later texture uploads.


if __name__ == "__main__":
    main()
