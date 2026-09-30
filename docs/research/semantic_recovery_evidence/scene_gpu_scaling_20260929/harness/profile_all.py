"""Paired saved-State rendering: equal workload, allocation and output destination.

All reset/restoration, upload, projection, batching and JIT compilation occur
outside timers. Source functions always recompute images; cached wrapper pixels
are never treated as rendering. No production implementation is changed.
"""

import argparse
import hashlib
import inspect
import runpy
import json
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .bridges import DMC
from .collect import candidate
from .common import catalog, load_state, profile, rgb, state_hash, write
from .special import bridge_for


def statistics_for(samples, batch):
    mean = statistics.mean(samples)
    return dict(
        count=len(samples),
        samples_ms=samples,
        mean_ms=mean,
        sd_ms=statistics.stdev(samples),
        mean_ms_per_frame=mean / batch,
        sd_ms_per_frame=statistics.stdev(samples) / batch,
        frames_per_second=1000 * batch / mean,
    )


def sample_pairs(ours, source, batch, sync, minimum=7, maximum=15, seconds=1, quality=None):
    """Alternate order to reduce thermal/order bias; keep both outputs materialized."""
    samples = {"ours": [], "source": []}
    funcs = {"ours": ours, "source": source}
    for i in range(maximum):
        images = {}
        for label in ("ours", "source") if i % 2 == 0 else ("source", "ours"):
            sync()
            start = time.perf_counter_ns()
            result = funcs[label]()
            sync()
            samples[label].append((time.perf_counter_ns() - start) / 1e6)
            if quality is not None:
                images[label] = result
            del result
        if quality is not None:
            quality(images["ours"], images["source"])
        images.clear()
        if i + 1 >= minimum and max(map(sum, samples.values())) >= seconds * 1000:
            break
    return {k: statistics_for(v, batch) for k, v in samples.items()}


def source_instance(row, frame, recipe, state, env):
    """Install the saved State in the pinned source, with exact recapture."""
    name = row["name"]
    if name == "PyFlytRocketLanding":
        from PyFlyt.gym_envs.rocket_envs.rocket_base_env import RocketBaseEnv

        from stable_envs._adapters.rocket import RocketAdapter

        adapter = RocketAdapter(1, "cpu", frame["seed"], recipe.get("simulator_config"))
        adapter.set_state(state)
        assert state_hash(adapter.get_state()) == frame["state_hash"]

        # This is the real PyFlyt source renderer, not our Torch implementation.
        def upstream_render():
            return RocketBaseEnv.render(adapter.envs[0].env)[..., :3]

        upstream_render.__wrapped__ = RocketBaseEnv.render
        return adapter, upstream_render, lambda: state_hash(adapter.get_state())
    bridge = bridge_for(row, frame["seed"], recipe["reset_options"])
    if name in {"OGBCube", "OGBScene"}:
        from stable_envs._adapters.ogbench_manipulation_compat import CUBE_SAMPLES, MODEL_VALUES

        from .manipulation_source_replay import install, instrument_model_values

        raw = bridge.source
        raw.family = name[3:]
        raw.supported_variations = set(raw.variation_space.names())
        if raw.family == "Scene":
            raw.supported_variations -= set(CUBE_SAMPLES)

        # Source reset replaces its Variation instance. Install only a read-only
        # path accessor on its class so numeric capture survives that replacement.
        def walk(space, path):
            node = space
            for part in path.split("."):
                node = node[part]
            return node

        type(raw.variation_space)._walk = walk
        instrument_model_values(raw)
        for _ in range(recipe.get("reset_count", 1)):
            bridge.reset(frame["seed"])
        values = {k: v[0].numpy().copy() for k, v in state.items(True, True)}
        for namespace, actual in [("compiled", raw._model_values), ("mjcf", raw._mjcf_values)]:
            for k in MODEL_VALUES:
                assert np.array_equal(actual[k], values["model", namespace, k]), (namespace, k)
        # Capture schema stores source model settings as float64. This casts
        # source-owned bookkeeping only; no candidate model values are installed.
        raw._model_values = {k: np.asarray(v, dtype=np.float64).copy() for k, v in raw._model_values.items()}
        raw._mjcf_values = {k: np.asarray(v, dtype=np.float64).copy() for k, v in raw._mjcf_values.items()}
        install(raw, values)
        captured = torch.stack([env._simulator._row(raw)])
        assert state_hash(captured) == frame["state_hash"], [
            (
                k,
                str(state[k].dtype),
                str(captured[k].dtype),
                float((state[k].to(torch.float64) - captured[k].to(torch.float64)).abs().max())
                if state[k].numel()
                else 0,
            )
            for k in state.keys(True, True)
            if state[k].dtype != captured[k].dtype or state[k].numpy().tobytes() != captured[k].numpy().tobytes()
        ]
        capture = lambda: state_hash(torch.stack([env._simulator._row(raw)]))
    else:
        for _ in range(recipe.get("reset_count", 1)):
            bridge.reset(frame["seed"])
        if name.startswith("Fetch"):
            import mujoco

            from stable_envs._adapters.fetch_pickandplace import FetchAdapter, state_fields

            # Reuse numeric State installation only; source model/rendering stays upstream-owned.
            adapter = object.__new__(FetchAdapter)
            adapter.num_envs, adapter.device, adapter._closed = 1, torch.device("cpu"), False
            adapter.envs, adapter._mujoco = [bridge.source], mujoco
            model = bridge.source.unwrapped.model
            adapter.fields = state_fields(model.nq, model.nv)
            adapter._gripper = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")]
            adapter.set_state(state)
            capture = lambda: state_hash(adapter.get_state())
        else:
            bridge.ref.set_state(state)
            capture = bridge.snapshot
    assert capture() == frame["state_hash"], (name, frame["index"], "source recapture differs")
    return bridge, bridge.render, capture


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("registration")
    p.add_argument("device", choices=["cpu", "cuda"])
    p.add_argument("output", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--backend", default="auto")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 64, 128, 256])
    p.add_argument("--poses", type=int, default=16)
    p.add_argument("--selected-result", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if a.device == "cuda":
        assert "3090" in torch.cuda.get_device_name()
    if a.selected_result and a.selected_result.exists() and a.device == "cuda":
        prior = json.loads(a.selected_result.read_text())
        assert prior["registration"] == a.registration
        if prior["status"] == "COMPLETE":
            a.batches = sorted(set(a.batches + [prior["selected_batch"]]))
    row = catalog()[a.registration]
    result = dict(
        registration=a.registration,
        device=a.device,
        backend=a.backend,
        status="RUNNING",
        profile=profile(),
        measurements=[],
        frames=[],
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Saved States; no rollouts or setup timed; equal batch and output destination; single CPU thread.",
        source_batch_method="Repeated upstream render, cycling the same saved poses; no cached images.",
    )
    save = lambda: write(a.output / "result.json", result)
    save()
    owners, functions, captures = [], [], []
    try:
        if "Symbolic" in row["name"]:
            result["status"] = "NO_RGB"
            return
        if row["name"] == "PushT":
            result.update(
                status="UNESTABLISHED", reason="No accepted source-matched saved-State corpus; no speedup claimed."
            )
            return
        meta = json.loads((a.corpus / "result.json").read_text())
        result["corpus"] = str(a.corpus)
        result["corpus_sha256"] = hashlib.sha256((a.corpus / "result.json").read_bytes()).hexdigest()
        config = meta["recipes"][meta["frames"][0]["case"]].get("simulator_config")
        pool = [f for f in meta["frames"] if meta["recipes"][f["case"]].get("simulator_config") == config]
        frames = [pool[i] for i in np.linspace(0, len(pool) - 1, min(a.poses, len(pool)), dtype=int)]
        env = candidate(row, a.device, 17, config)
        craftax = row["name"].startswith("Craftax")
        if craftax:
            env.reset(17)
        template = env.get_state()
        live_hash = state_hash(template)
        if hasattr(env, "to_render_state"):
            prepare, render = env.to_render_state, env.render
        else:

            def prepare(state):
                env._validate_state(state)
                return env._pack_render_state(state)

            render = env._render
        inputs, kwargs, source_images, states = [], [], [], []
        for f in frames:
            state = load_state(a.corpus / f["state_file"], "cpu", template)
            assert state_hash(state) == f["state_hash"]
            states.append(state)
            inputs.append(prepare(state.to(a.device)))
            kw = {
                k: torch.as_tensor(v, device=a.device, dtype=torch.float32) if k == "style" else v
                for k, v in f["render_kwargs"].items()
            }
            if a.backend != "auto":
                kw["backend"] = a.backend
            kwargs.append(kw)
            recipe = meta["recipes"][f["case"]]
            if craftax:
                from tests.helpers.reference.craftax_source import assert_engine, oracle, restore_oracle

                source = oracle(row["name"])
                source.reset(seed=f["seed"])
                restore_oracle(source, env, state)
                assert_engine(source, env, state)
                owners.append(source)
            else:
                owner, fn, capture = source_instance(row, f, recipe, state, env)
                owners.append(owner)
                functions.append(fn)
                captures.append(capture)
                source_images.append(rgb(fn()).copy())
            result["frames"].append(
                dict(index=f["index"], case=f["case"], state_hash=f["state_hash"], state_file=f["state_file"])
            )
        input_hashes = [
            state_hash(x) if hasattr(x, "batch_size") else hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
            for x in inputs
        ]
        result["independent_states"] = len(set(state_hash(s) for s in states))
        if craftax:
            import jax
            import jax.numpy as jnp

            # Real upstream get_obs, not SWM render() returning _last_obs.
            upstream = env._simulator.upstream
            scalar = jax.jit(upstream.get_obs)
            source_images = [(np.clip(np.asarray(scalar(o._state)), 0, 1) * 255).astype(np.uint8) for o in owners]

            def batched_pixels(s):
                return (jnp.clip(jax.vmap(upstream.get_obs)(s), 0, 1) * 255).astype(jnp.uint8)

            pixels = jax.jit(batched_pixels)
            result["source_batch_method"] = (
                "jax.jit(jax.vmap(upstream.get_obs)); uint8 conversion; compilation excluded; checked against scalar source."
            )
        else:
            # Verify no source-State drift during a render (full recapture).
            assert [f() for f in captures] == [f["state_hash"] for f in frames]
        result["source_anchor_mse"] = []
        for f, image in zip(frames, source_images):
            anchor = np.load(a.corpus / f["oracle_file"], allow_pickle=False)
            anchor_mse = float(np.mean(((image.astype(np.float64) - anchor.astype(np.float64)) / 255) ** 2))
            result["source_anchor_mse"].append(anchor_mse)
            if row["name"] == "PyFlytRocketLanding":
                assert np.array_equal(image, anchor), (
                    "PyFlyt source renderer differs from saved upstream image",
                    anchor_mse,
                )
        result["source_render_paths"] = {}
        for fn in functions or [env._simulator.upstream.get_obs]:
            path = inspect.getsourcefile(inspect.unwrap(fn))
            if path:
                result["source_render_paths"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        is_gl = (
            row["name"] in DMC
            or row["name"].startswith("Fetch")
            or row["name"] in {"OGBCube", "OGBScene", "PointMazeLargeNavigate"}
        )
        if is_gl:
            from OpenGL import GL

            if row["name"] in DMC:
                physics = owners[0].source.env.physics
                with physics.contexts.gl.make_current() as ctx:
                    vendor, renderer_name = ctx.call(
                        lambda: (GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode())
                    )
            else:
                import mujoco

                probe = mujoco.Renderer(
                    mujoco.MjModel.from_xml_string(
                        '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
                    ),
                    height=32,
                    width=32,
                )
                owners.append(probe)
                probe._gl_context.make_current()
                vendor, renderer_name = GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()
            software = any(k in renderer_name.lower() for k in ["llvmpipe", "softpipe", "swrast"])
            assert software if a.device == "cpu" else "nvidia" in vendor.lower() and not software
            result["opengl"] = dict(vendor=vendor, renderer=renderer_name)
        result["source_render_device"] = a.device if is_gl or craftax else "cpu"
        result["source_gpu_supported"] = is_gl or craftax
        # Warm both complete renderer/context sets before fixing the reference.
        # Source software GL can settle edge rasterization after other contexts
        # initialize, while its complete numeric simulator State stays identical.
        if not craftax:
            for inp, kw in zip(inputs, kwargs):
                render(inp, **kw)
            warm_first = [rgb(fn()).copy() for fn in functions]
            warm_second = [rgb(fn()).copy() for fn in functions]
            result["source_warmup_drift_mse"] = [
                float(np.mean(((x.astype(np.float64) - y.astype(np.float64)) / 255) ** 2))
                for x, y in zip(source_images, warm_second)
            ]
            result["source_warm_repeat_max_uint8"] = max(
                int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()) for x, y in zip(warm_first, warm_second)
            )
            result["source_warm_repeat_mse"] = [
                float(np.mean(((x.astype(np.float64) - y.astype(np.float64)) / 255) ** 2))
                for x, y in zip(warm_first, warm_second)
            ]
            if row["name"] == "DMCManipulatorBringBall":
                result["source_pixel_variation"] = (
                    "Software GL floor-edge variation observed at unchanged full State; repeated MSE measured, not asserted bit-exact."
                )
                assert max(result["source_warm_repeat_mse"]) <= 0.0001
            else:
                assert result["source_warm_repeat_max_uint8"] <= 1
            assert [f() for f in captures] == [f["state_hash"] for f in frames]
            source_images = warm_second
        mse = []
        for inp, kw, img in zip(inputs, kwargs, source_images):
            ours = rgb(render(inp, **kw))
            assert ours.shape == img.shape
            mse.append(float(np.mean(((ours.astype(np.float64) - img.astype(np.float64)) / 255) ** 2)))
        result["paired_mse"] = dict(samples=mse, mean=statistics.mean(mse), sd=statistics.stdev(mse), maximum=max(mse))
        save()
        sync = torch.cuda.synchronize if a.device == "cuda" else lambda: None
        for batch in a.batches:
            indices = [i % len(inputs) for i in range(batch)]
            value = torch.cat([inputs[i] for i in indices], dim=0)
            kw = {}
            for k, v in kwargs[0].items():
                if torch.is_tensor(v):
                    kw[k] = torch.cat([kwargs[i][k] for i in indices], dim=0)
                else:
                    assert all(x[k] == v for x in kwargs)
                    kw[k] = v
            if craftax:
                batched = jax.tree.map(lambda *v: jnp.stack(v), *[owners[i]._state for i in indices])
                compiled_image = pixels(batched)
                compiled_image.block_until_ready()
                check = np.asarray(compiled_image)
                difference = max(
                    int(np.abs(check[i].astype(np.int16) - source_images[j].astype(np.int16)).max())
                    for i, j in enumerate(indices)
                )
                assert difference <= 1, ("upstream vmap changes pixels", difference)

                def original(destination):
                    image = pixels(batched)
                    if destination == "cpu":
                        return torch.from_numpy(np.asarray(image).copy())
                    return torch.utils.dlpack.from_dlpack(image).contiguous()
            else:
                batch_fns = [functions[i] for i in indices]

                def original(destination):
                    # Every lane executes source rendering. Stack/copies and upload count.
                    images = np.stack([rgb(fn()) for fn in batch_fns])
                    return torch.from_numpy(images).to(destination).contiguous()

            def ours(destination, value=value, kw=kw):
                return render(value, **kw).to(destination).contiguous()

            for destination in ["cpu", "cuda"] if a.device == "cuda" else ["cpu"]:
                our_fn, src_fn = lambda: ours(destination), lambda: original(destination)
                # Warm both and validate batched output before measurements.
                oi, si = our_fn(), src_fn()
                sync()
                assert oi.shape == si.shape and oi.dtype == si.dtype == torch.uint8
                assert oi.device.type == si.device.type == destination
                # Compare each batch lane against its independently rendered image.
                oh, sh = oi.cpu().numpy(), si.cpu().numpy()
                np.save(a.output / f"ours_b{batch}_{destination}_first16.npy", oh[:16], allow_pickle=False)
                np.save(a.output / f"source_b{batch}_{destination}_first16.npy", sh[:16], allow_pickle=False)
                ours_single = [rgb(render(x, **k)) for x, k in zip(inputs, kwargs)]
                assert (
                    max(
                        int(np.abs(oh[i].astype(np.int16) - ours_single[j].astype(np.int16)).max())
                        for i, j in enumerate(indices)
                    )
                    <= 1
                )
                source_repeat_max = max(
                    int(np.abs(sh[i].astype(np.int16) - source_images[j].astype(np.int16)).max())
                    for i, j in enumerate(indices)
                )
                source_repeat_mse = [
                    float(np.mean(((sh[i].astype(np.float64) - source_images[j].astype(np.float64)) / 255) ** 2))
                    for i, j in enumerate(indices)
                ]
                if row["name"] == "DMCManipulatorBringBall":
                    assert max(source_repeat_mse) <= 0.0001
                else:
                    assert source_repeat_max <= 1
                batch_mse = [
                    float(np.mean(((oh[i].astype(np.float64) - sh[i].astype(np.float64)) / 255) ** 2))
                    for i in range(min(batch, len(inputs)))
                ]
                actual_mse = dict(
                    mean=statistics.mean(batch_mse),
                    sd=statistics.stdev(batch_mse) if len(batch_mse) > 1 else 0.0,
                    maximum=max(batch_mse),
                    samples=batch_mse,
                )
                del oi, si, oh, sh
                repeated_mse = []

                def record_quality(ours_image, source_image):
                    n = min(batch, len(inputs))
                    left = ours_image[:n].cpu().numpy().astype(np.float64)
                    right = source_image[:n].cpu().numpy().astype(np.float64)
                    repeated_mse.extend(np.mean(((left - right) / 255) ** 2, axis=(1, 2, 3)).tolist())

                timing = sample_pairs(
                    our_fn,
                    src_fn,
                    batch,
                    sync,
                    quality=record_quality if row["name"] == "DMCManipulatorBringBall" else None,
                )
                if repeated_mse:
                    actual_mse = dict(
                        mean=statistics.mean(repeated_mse),
                        sd=statistics.stdev(repeated_mse),
                        maximum=max(repeated_mse),
                        samples=repeated_mse,
                    )
                result["measurements"].append(
                    dict(
                        batch=batch,
                        output_device=destination,
                        paired_mse=actual_mse,
                        source_repeat_max_uint8=source_repeat_max,
                        source_repeat_mse_max=max(source_repeat_mse),
                        **timing,
                        speedup=timing["source"]["mean_ms"] / timing["ours"]["mean_ms"],
                    )
                )
                save()
                print(a.registration, a.device, batch, destination, result["measurements"][-1]["speedup"], flush=True)
                if batch in [256, 4096] and destination == "cpu":
                    runpy.run_path(str(Path(__file__).with_name("diagnose.py")))["diagnose"](our_fn, src_fn, a.output / f"profile_b{batch}", batch)
                if batch in [256, 1024, 4096, 16384]:
                    module = runpy.run_path(str(Path(__file__).with_name("telemetry.py")))
                    if batch == 256 and destination == "cpu":
                        module["probe_tools"](a.output / "tool_inventory")
                    telemetry = {}
                    order = [("ours", our_fn), ("source", src_fn)] if batch in [256, 4096] else [("source", src_fn), ("ours", our_fn)]
                    for label, fn in order:
                        telemetry[label] = module["run_phase"](f"{label}_b{batch}_{destination}", fn, sync, a.output / "telemetry", duration_s=12, interval_ms=200)
                    result["measurements"][-1]["telemetry"] = telemetry
                    save()
            del value
        if not craftax:
            assert [f() for f in captures] == [f["state_hash"] for f in frames]
        assert [
            state_hash(x) if hasattr(x, "batch_size") else hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
            for x in inputs
        ] == input_hashes
        assert state_hash(env.get_state()) == live_hash
        result["status"] = "COMPLETE"
        result["state_nonmutation_passed"] = True
    except Exception:
        result.update(status="ERROR", error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
