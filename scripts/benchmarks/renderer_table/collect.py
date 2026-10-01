"""Generate matched CPU trajectories, then time public GPU rendering of that corpus.

Every invocation owns a fresh output directory. Failures are retained as ERROR;
they are never converted into unsupported measurements by the collector.
"""

import argparse
import hashlib
import inspect
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from .bridges import CLASSIC, DMC, NAV, action_sequence, checkpoints
from .common import (
    PIN,
    array_hash,
    catalog,
    load_state,
    profile,
    public_render,
    quality,
    rgb,
    save_state,
    state_arrays,
    state_hash,
    timed,
    unavailable,
    write,
)
from .special import bridge_for as Bridge


def appearance_options(source):
    """One combined registered color/light case, explicitly recorded in the recipe."""
    from tests.helpers.reference.variation import swm_variation_space

    values = {}
    try:
        space = swm_variation_space(source)
    except AttributeError:
        return {}
    for path in space.names():
        leaf = space
        for part in path.split("."):
            leaf = leaf[part]
        if ("color" in path or path.startswith("visuals.")) and np.asarray(leaf.value).size == 3:
            values[path] = np.asarray(leaf.low + (leaf.high - leaf.low) * 0.35, dtype=leaf.dtype)
    return {"variation": [], "variation_values": values} if values else {}


def candidate(row, device, seed, config=None):
    import stable_envs as se

    kwargs = {"size": 224} if row["name"].startswith("Fetch") or row["name"] in DMC else {}
    if row["name"] == "PyFlytRocketLanding":
        kwargs = {
            "size": 480,
            "simulator_config": config or {"render_mode": "rgb_array", "render_resolution": (480, 480)},
        }
    return se.make(row["name"], num_envs=1, device=device, seed=seed, **kwargs)


def render_kwargs(env, bridge):
    if env.__class__.__name__.startswith("Fetch"):
        # Fetch's simulator State excludes appearance. Persist the actual reset
        # style alongside it; a fresh GPU env otherwise renders every saved State
        # with default colors, including explicitly varied source frames.
        return {"style": env.get_style().detach().clone()}
    if env.__class__.__name__.startswith("Craftax") or env.__class__.__name__ in {
        "OGBCube",
        "OGBScene",
        "PyFlytRocketLanding",
    }:
        return {}
    from tests.helpers.reference.variation import style_from_swm_env, variation_map

    if variation_map(env) is None:
        return {}
    return {"style": torch.as_tensor(style_from_swm_env(env, bridge.source))[None].to(env.device)}


def kwargs_json(kwargs):
    return {k: v.cpu().numpy().tolist() if torch.is_tensor(v) else v for k, v in kwargs.items()}


def cpu(row, output, *, target_frames=None, source_only=False):
    result = dict(row, phase="cpu", oracle_commit=PIN, profile=profile(), frames=[], recipes=[], status="RUNNING")
    write(output / "result.json", result)
    name = row["name"]
    if "Craftax" in name and "Symbolic" in name:
        bridge, env = Bridge(row, 0, {}), candidate(row, "cpu", 0)
        try:
            bridge.reset(0)
            env.reset(0)
            result["semantic_witness"] = bridge.proof(env)
            result["unavailable"] = {}
            for label, fn in [("original", bridge.render), ("cpu", lambda: public_render(env, env.get_state()))]:
                try:
                    fn()
                except NotImplementedError as e:
                    result["unavailable"][label] = unavailable(str(e), "NOT IMPLEMENTED")
                else:
                    raise AssertionError("symbolic registration unexpectedly returned pixels")
            result["status"] = "COMPLETE"
            write(output / "result.json", result)
            return result
        finally:
            bridge.close()
            env.close()
    if name == "PyFlytRocketLanding":
        varied = {}
    else:
        probe = Bridge(row, 17, {})
        varied = appearance_options(probe.source)
        probe.close()
    if name.startswith("Fetch") and varied:
        varied["variation_values"] = {
            k: v for k, v in varied["variation_values"].items() if k in {"object.color", "table.color"}
        }
        if name.startswith("FetchReach"):
            varied["variation_values"].pop("object.color", None)
    dtypes = ["float64"] if name in DMC else ["float32", "float64"]
    if name.startswith("Craftax"):
        dtypes = ["int64"]
    if name in {"CartPole", "Acrobot", "MountainCar"}:
        dtypes = ["int64"]
    plans = [(s, d, {}) for s in [0, 17, 42] for d in dtypes]
    if name.startswith("Fetch"):
        # Source defaults otherwise sample five visual leaves that the accepted
        # candidate reset does not sample. Use the same explicit supported reset
        # configuration and let both public renderers read their actual model/style.
        plans = [(s, d, {"variation": []}) for s in [0, 17, 42] for d in dtypes]
    if varied:
        plans.append((17, dtypes[0], varied))
    if name == "PyFlytRocketLanding":
        from .special import rocket_plans

        plans = rocket_plans(row)
    if name in {"OGBCube", "OGBScene"}:
        from .special import manipulation_plans

        plans = manipulation_plans(row, varied)
    if name.startswith("FetchPickAndPlace"):
        green, blue = np.array([0.2, 0.8, 0.3]), np.array([0.1, 0.3, 0.8])
        appearances = [
            {},
            {"object.color": green},
            {"object.color": blue},
            {"table.color": green},
            {"table.color": blue},
            {"object.color": green, "table.color": blue},
        ]
        plans = [
            (
                seed,
                "float64",
                dict(variation=[], variation_values=values, _marks=[0], _label="reset-appearance-" + str(i)),
            )
            for seed in [0, 17, 42]
            for i, values in enumerate(appearances)
        ]
        result["sampling_limitation"] = {
            "scope": "RESET_ONLY",
            "reason": "Upstream PickAndPlace render calls mj_forward and changes the continuation-relevant gripper position/quaternion caches after steps. Dynamic frames are NONCOMPARABLE under the required render nonmutation rule. No restore, cache repair, or pose teleport is used.",
            "action_dtype_coverage": "No actions in measured reset frames; separate source-render mutation probes cover post-action states.",
            "diagnostic": "fetch_mutation_probe.py; retained per-registration, three-seed zero/random action diagnostics",
        }
    frame_count = sum(len(o.get("_marks", checkpoints(name))) for _, _, o in plans)
    samples = max(1, math.ceil(100 / frame_count))
    if name.startswith("Craftax"):
        samples = 10  # Terminal checkpoints can shorten the planned trajectory.
    if target_frames is not None:
        if target_frames < 1:
            raise ValueError("target_frames must be positive")
        # Retain every original stress recipe first, including contact, terminal,
        # appearance, and high-ceiling cases. Extend with new seeds rather than
        # counting repeated timings of one image as independent frames.
        original_plans = plans
        plans = [
            (
                seed + epoch * 1009,
                dtype,
                options
                if epoch == 0 or name.startswith("FetchPickAndPlace")
                else {**options, "_marks": list(range(101))},
            )
            for epoch in range(target_frames)
            for seed, dtype, options in original_plans
        ]
        samples = 1
        result["requested_frames"] = target_frames
        result["sampling"] = (
            "Original sparse stress recipes, then dense steps 0..100 with seeds incremented by 1009 per sweep. "
            "PickAndPlace stays reset-only because source rendering mutates continuation caches after steps."
        )
    result["candidate_render_process"] = "separate" if source_only else "shared"
    for case, (seed, dtype, options) in enumerate(plans):
        if target_frames is not None and len(result["frames"]) >= target_frames:
            break
        bridge = Bridge(row, seed, options)
        env = candidate(row, "cpu", seed, options.get("_config"))
        try:
            source_options = {k: v for k, v in options.items() if not k.startswith("_")}
            reset_kw = {"options": source_options} if "options" in inspect.signature(env.reset).parameters else {}
            if name in NAV and source_options.get("variation_values"):
                # These accepted environments expose appearance through public
                # set_variation/render(style), not through their semantic reset options.
                reset_kw = {
                    "options": dict(
                        source_options,
                        variation_values={
                            k: v
                            for k, v in source_options["variation_values"].items()
                            if k in getattr(env, "reset_value_paths", ())
                        },
                    )
                }
            # DMC reset has no options API; apply only mapped appearance values.
            if options and not reset_kw:
                from tests.helpers.reference.variation import style_from_swm_env

                bridge.reset(seed)
                env.set_variation(style_from_swm_env(env, bridge.source))
                env.reset(seed)
            else:
                bridge.reset(seed)
                env.reset(seed, **reset_kw)
                if name in NAV and source_options.get("variation_values"):
                    from tests.helpers.reference.variation import style_from_swm_env

                    env.set_variation(style_from_swm_env(env, bridge.source))
            for _ in range(options.get("_reset_count", 1) - 1):
                bridge.reset(seed)
                env.reset(seed, **reset_kw)
            marks = options.get("_marks", checkpoints(name))
            actions = action_sequence(bridge.source, name, seed, dtype, max(marks))
            if name == "PyFlytRocketLanding":
                actions = np.zeros((max(marks), 1, 7), dtype=dtype)
                if options["_label"] in ["mixed", "all"]:
                    actions[:] = np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1], dtype=dtype)
            np.save(output / f"actions-{case}.npy", actions, allow_pickle=False)
            recipe = dict(
                case=case,
                seed=seed,
                action_dtype=str(actions.dtype),
                reset_options=source_options,
                reset_count=options.get("_reset_count", 1),
                simulator_config=options.get("_config"),
                label=options.get("_label", "trajectory"),
                action_generator="numpy.default_rng(90000+seed).uniform(action_space.low, high) or integers(0,n)",
                actions_file=f"actions-{case}.npy",
                actions_hash=array_hash(actions),
                camera=bridge.camera,
                checkpoints=marks,
                constructor={"num_envs": 1, "seed": seed},
            )
            result["recipes"].append(recipe)
            for step in range(max(marks) + 1):
                done = False
                if step:
                    if "_interaction" in options:
                        from .special import contact_action

                        actions[step - 1] = contact_action(bridge, options["_interaction"], step - 1, dtype)
                    action = actions[step - 1]
                    bridge.step(action)
                    out = env.step(action, frames=None)
                    if name in CLASSIC or name.startswith(("Fetch", "Craftax")):
                        done = bool(out["terminated"].any() or out["truncated"].any())
                if step not in marks and not done:
                    continue
                proof = bridge.proof(env)
                state = env.get_state()
                state_before, oracle_before = state_hash(state), bridge.snapshot()
                kw = render_kwargs(env, bridge)
                timing, original = timed(bridge.render, samples=samples, warmups=1 if target_frames else 10)
                original = rgb(original)
                assert bridge.snapshot() == oracle_before, "oracle render mutated continuation State"
                index = len(result["frames"])
                key = f"frame-{index:04d}"
                np.save(output / (key + "-oracle.npy"), original, allow_pickle=False)
                save_state(output / (key + "-state.npz"), state)
                record = dict(
                    index=index,
                    case=case,
                    step=step,
                    seed=seed,
                    action_dtype=str(actions.dtype),
                    prefix_hash=array_hash(actions[:step]),
                    proof=proof,
                    state_file=key + "-state.npz",
                    state_hash=state_before,
                    oracle_file=key + "-oracle.npy",
                    camera=bridge.camera,
                    shape=list(original.shape),
                    oracle_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                    render_kwargs=kwargs_json(kw),
                    original_timing=timing,
                    original_backend="Mesa llvmpipe software OpenGL"
                    if name in DMC or name in {"PointMazeLargeNavigate", "OGBCube", "OGBScene"}
                    else "CPU software raster",
                    oracle_render_nonmutation=True,
                )
                if not source_only:
                    try:
                        timing, image = timed(lambda: public_render(env, state, **kw), samples=samples)
                        image = rgb(image)
                        record.update(cpu_timing=timing, cpu_quality=quality(image, original))
                        np.save(output / (key + "-cpu.npy"), image, allow_pickle=False)
                    except NotImplementedError as e:
                        record["cpu_timing"] = unavailable(str(e))
                assert state_hash(env.get_state()) == state_before, "candidate renderer mutated live State"
                assert state_hash(state) == state_before, "candidate renderer mutated supplied State"
                bridge.proof(env)
                record["candidate_render_nonmutation"] = True
                result["frames"].append(record)
                write(output / "result.json", result)
                print(name, case, step, "matched", flush=True)
                if done or (target_frames is not None and len(result["frames"]) >= target_frames):
                    break
            np.save(output / f"actions-{case}.npy", actions, allow_pickle=False)
            recipe["actions_hash"] = array_hash(actions)
            if "_interaction" in options:
                recipe["interaction"] = options["_interaction"]
                recipe["action_generator"] = (
                    "special.contact_action: accepted source-observation controller, commands shared exactly with candidate"
                )
            write(output / "result.json", result)
        finally:
            env.close()
            bridge.close()
    if target_frames is not None:
        assert len(result["frames"]) == target_frames, "Insufficient distinct matched frames"
    result["status"] = "COMPLETE"
    write(output / "result.json", result)
    return result


def gpu(row, corpus, output):
    source = json.loads((corpus / "result.json").read_text())
    assert source["status"] == "COMPLETE"
    result = dict(
        row,
        phase="gpu",
        profile=profile(),
        frames=[],
        status="RUNNING",
        corpus_result_sha256=hashlib.sha256((corpus / "result.json").read_bytes()).hexdigest(),
    )
    write(output / "result.json", result)
    env = candidate(row, "cuda", 0)
    if row["name"].startswith("Craftax"):
        env.reset(0)
    try:
        if "Symbolic" in row["name"]:
            try:
                public_render(env, env.get_state())
            except NotImplementedError as e:
                result["unavailable"] = {"gpu": unavailable(str(e), "NOT IMPLEMENTED")}
            else:
                raise AssertionError("symbolic registration unexpectedly returned pixels")
        before = state_hash(env.get_state())
        active_config = None
        active_case, replay_step = None, 0
        for f in source["frames"]:
            config = source["recipes"][f["case"]].get("simulator_config")
            if config != active_config:
                env.close()
                env = candidate(row, "cuda", 0, config)
                before = state_hash(env.get_state())
                active_config = config
            state = load_state(corpus / f["state_file"], "cuda", template=env.get_state())
            assert state_hash(state) == f["state_hash"]
            device_proof = None
            if row["name"].startswith("Craftax"):
                recipe = source["recipes"][f["case"]]
                if active_case != f["case"]:
                    env.reset(recipe["seed"])
                    active_case, replay_step = f["case"], 0
                actions = np.load(corpus / recipe["actions_file"], allow_pickle=False)
                while replay_step < f["step"]:
                    env.step(actions[replay_step], frames=None)
                    replay_step += 1
                replayed = env.get_state()
                cpu_leaves, gpu_leaves = state_arrays(state), state_arrays(replayed)
                assert cpu_leaves.keys() == gpu_leaves.keys()
                differences = {
                    key: dict(
                        max_absolute=float(
                            np.max(np.abs(cpu_leaves[key].astype(np.float64) - gpu_leaves[key].astype(np.float64)))
                        ),
                        count=int(np.count_nonzero(cpu_leaves[key] != gpu_leaves[key])),
                    )
                    for key in cpu_leaves
                    if key != "meta/signature" and not np.array_equal(cpu_leaves[key], gpu_leaves[key])
                }
                device_proof = dict(
                    kind="EXACT_REPLAY_EXCEPT_PLATFORM_SIGNATURE",
                    excluded_leaf="meta/signature",
                    reason="The public contract includes CPU/GPU platform in its compatibility signature. Each State was reached by public reset/step; signatures are never edited.",
                    cpu_leaf_hashes={k: array_hash(v) for k, v in cpu_leaves.items()},
                    gpu_leaf_hashes={k: array_hash(v) for k, v in gpu_leaves.items()},
                    actual_gpu_state_hash=state_hash(replayed),
                    action_prefix_hash=array_hash(actions[:replay_step]),
                    actual_gpu_state_file=f"frame-{f['index']:04d}-gpu-state.npz",
                    differences=differences,
                )
                state = replayed
                save_state(output / device_proof["actual_gpu_state_file"], state)
                before = state_hash(env.get_state())
                if differences:
                    reason = (
                        "CPU/GPU public replay has unequal numeric State leaves: "
                        + ", ".join(differences)
                        + ". The public compatibility signature also prohibits cross-device State restoration; no State fields were patched."
                    )
                    result["frames"].append(
                        dict(
                            index=f["index"],
                            state_hash=f["state_hash"],
                            camera=f["camera"],
                            shape=f["shape"],
                            gpu_timing=unavailable(reason, "NONCOMPARABLE"),
                            gpu_to_cpu_timing=unavailable(reason, "NONCOMPARABLE"),
                            device_replay_proof=device_proof,
                            supplied_and_live_state_nonmutation=True,
                            rendering_attempted=False,
                        )
                    )
                    write(output / "result.json", result)
                    print(row["name"], f["index"], "GPU NONCOMPARABLE", differences, flush=True)
                    continue
            supplied_before = state_hash(state)
            kwargs = {
                k: torch.tensor(v, dtype=torch.float32, device="cuda") if k == "style" else v
                for k, v in f["render_kwargs"].items()
            }
            timing, image = timed(
                lambda: public_render(env, state, **kwargs),
                gpu=True,
                samples=100 if device_proof else max(1, math.ceil(100 / len(source["frames"]))),
            )
            assert torch.is_tensor(image) and image.device.type == "cuda", "public GPU output was not native CUDA"
            host_timing, host_image = timed(
                lambda: public_render(env, state, **kwargs),
                gpu=True,
                to_host=True,
                samples=100 if device_proof else max(1, math.ceil(100 / len(source["frames"]))),
            )
            image = rgb(image)
            assert np.array_equal(image, rgb(host_image))
            original = np.load(corpus / f["oracle_file"], allow_pickle=False)
            assert hashlib.sha256(original.tobytes()).hexdigest() == f["oracle_sha256"]
            assert state_hash(state) == supplied_before and state_hash(env.get_state()) == before
            record = dict(
                index=f["index"],
                state_hash=f["state_hash"],
                camera=f["camera"],
                shape=list(image.shape),
                gpu_timing=timing,
                gpu_to_cpu_timing=host_timing,
                gpu_quality=quality(image, original),
                supplied_and_live_state_nonmutation=True,
            )
            if device_proof:
                record["device_replay_proof"] = device_proof
            np.save(output / f"frame-{f['index']:04d}-gpu.npy", image, allow_pickle=False)
            result["frames"].append(record)
            write(output / "result.json", result)
            print(row["name"], f["index"], "GPU measured", flush=True)
    finally:
        env.close()
    result["status"] = "COMPLETE"
    write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registration")
    parser.add_argument("phase", choices=["cpu", "gpu"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--target-frames", type=int)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    row = catalog()[args.registration]
    try:
        if args.phase == "cpu":
            cpu(row, args.output, target_frames=args.target_frames, source_only=args.source_only)
        else:
            gpu(row, args.corpus, args.output)
    except Exception:
        write(args.output / "error.json", dict(status="ERROR", traceback=traceback.format_exc(), command=sys.argv))
        raise


if __name__ == "__main__":
    main()
