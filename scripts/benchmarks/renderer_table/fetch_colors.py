"""Fetch color regression against pinned SWM and immutable matched-frame corpora.

Run CPU mode in the accepted simulator profile; GPU modes on Oscar. Each mode
requires a new output directory and never changes the input corpus or oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.benchmarks.renderer_table.collect import candidate, kwargs_json, render_kwargs
from scripts.benchmarks.renderer_table.common import (
    catalog,
    load_state,
    profile,
    public_render,
    rgb,
    state_hash,
    write,
)


def mse(a, b):
    return float(np.mean(((a.astype(np.float64) - b.astype(np.float64)) / 255) ** 2))


def source_colors(source):
    """Read the actual source model, not its potentially inactive variation leaves."""
    wrapper = source
    while not hasattr(type(wrapper), "_get_geoms_for_material"):
        wrapper = wrapper.env
    m = source.unwrapped.model
    groups = {}
    for name, attr, material in [
        ("table.color", "_table_geoms", "table_mat"),
        ("floor.color", "_floor_geoms", "floor_mat"),
        ("object.color", "_object_geoms", "block_mat"),
    ]:
        geoms = getattr(wrapper, attr, None)
        if geoms is None:
            geoms = wrapper._get_geoms_for_material(m, material)
            if name == "object.color" and not geoms:
                geoms = wrapper._get_geoms_for_material(m, "puck_mat")
        for geom in geoms:
            rgba = m.geom_rgba[geom]
            mid = int(m.geom_matid[geom])
            effective = m.mat_rgba[mid] if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1]) else rgba
            if name in groups:
                np.testing.assert_array_equal(groups[name], effective[:3])
            groups[name] = effective[:3].copy()
    return groups


def cpu(output):
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    from scripts.recovery.fetch_public_gate import compare_output, full_compare

    results = []
    for row in catalog().values():
        if row["family"] != "fetch":
            continue
        has_object = "Reach" not in row["name"]
        options = [
            {"variation": []},
            {"variation": [], "variation_values": {"table.color": [0.35] * 3}},
            {"variation": [], "variation_values": {"table.color": [0.1, 0.8, 0.3]}},
            {"variation": ["table.color"]},
        ]
        if has_object:
            options.extend(
                [
                    {"variation": [], "variation_values": {"object.color": [0.2, 0.4, 0.9]}},
                    {
                        "variation": [],
                        "variation_values": {
                            "table.color": [0.35] * 3,
                            "object.color": [0.35] * 3,
                        },
                    },
                    {"variation": ["table.color", "object.color"]},
                ]
            )
        # Each first activation independently, then successive partial/no-color
        # resets to catch style replacement and source material persistence.
        for case, first in enumerate(options):
            env = candidate(row, "cpu", 17)
            refs = [gym.make(row["registration"]) for _ in range(2)]
            env.close()
            import stable_envs as se

            env = se.make(row["name"], num_envs=2, device="cpu", seed=17, size=224)
            try:
                sequence = [first, options[1], options[0]]
                if has_object:
                    sequence += [options[4], options[0]]
                for reset_index, selected in enumerate(sequence):
                    env.reset(seed=17, options=selected)
                    for lane, ref in enumerate(refs):
                        ref.reset(seed=17 + lane, options=selected)
                    full_compare(env.get_state(), refs)
                    colors = [source_colors(ref) for ref in refs]
                    for lane, colors_row in enumerate(colors):
                        for name, expected in colors_row.items():
                            np.testing.assert_allclose(
                                env.resolve_style()[name][lane].cpu().numpy(),
                                expected,
                                atol=3e-8,
                                rtol=0,
                                err_msg=f"{row['name']} case={case} reset={reset_index} lane={lane} {name}",
                            )
                    # Saved render kwargs survive later live appearance changes.
                    saved = render_kwargs(env, None)
                    saved_values = saved["style"].clone()
                    roundtrip = torch.tensor(kwargs_json(saved)["style"], dtype=torch.float32)
                    assert torch.equal(roundtrip, saved_values)
                    env._style.values.add_(0.001)
                    assert torch.equal(saved["style"], saved_values)
                    env.set_variation(saved_values)
                    for action in np.random.default_rng(71).uniform(-1, 1, (3, 2, 4)):
                        compare_output(env.step(action, capture="states"), [ref.step(action[lane]) for lane, ref in enumerate(refs)])
                        full_compare(env.get_state(), refs)
                    results.append(dict(registration=row["registration"], case=case, reset=reset_index, colors=colors))
            finally:
                env.close()
                for ref in refs:
                    ref.close()
        print(row["registration"], "CPU colors and continuation PASS", flush=True)
    write(output / "result.json", dict(status="PASS", profile=profile(), checks=results))


def gpu(mode, corpus, output):
    results = []
    for row in catalog().values():
        if row["family"] != "fetch":
            continue
        inputs = corpus / row["name"]
        source = json.loads((inputs / "cpu/result.json").read_text())
        assert source["status"] == "COMPLETE"
        target = output / row["name"]
        target.mkdir()
        frames = []
        for recipe in source["recipes"]:
            env = candidate(row, "cuda", recipe["seed"])
            try:
                env.reset(seed=recipe["seed"], options=recipe["reset_options"])
                # baseline_styled executes the baseline production code; only
                # its omitted benchmark appearance transfer is corrected here.
                kwargs = {"style": env.get_style().clone()} if mode == "baseline_styled" else render_kwargs(env, None)
                style = kwargs_json(kwargs)
                before = state_hash(env.get_state())
                for frame in (f for f in source["frames"] if f["case"] == recipe["case"]):
                    index = frame["index"]
                    state = load_state(inputs / "cpu" / frame["state_file"], "cuda", env.get_state())
                    assert state_hash(state) == frame["state_hash"]
                    # The renderer profile uses MuJoCo 3.11 while the accepted
                    # CPU simulator uses 3.8.1. Its fresh reset need not agree.
                    # Render the exact archived CPU State, never the GPU env's
                    # incidental reset State; reset above configures style only.
                    restored_kwargs = {"style": torch.tensor(style["style"], device="cuda", dtype=torch.float32)}
                    image = rgb(public_render(env, state, **restored_kwargs))
                    again = rgb(public_render(env, state, **restored_kwargs))
                    np.testing.assert_array_equal(image, again)
                    assert state_hash(env.get_state()) == before and state_hash(state) == frame["state_hash"]
                    original_cpu = np.load(inputs / "cpu" / frame["oracle_file"], allow_pickle=False)
                    assert hashlib.sha256(original_cpu.tobytes()).hexdigest() == frame["oracle_sha256"]
                    original_gpu = np.load(inputs / "source_gpu" / frame["oracle_file"], allow_pickle=False)
                    old = np.load(inputs / "gpu" / f"frame-{index:04d}-gpu.npy", allow_pickle=False)
                    np.save(target / f"frame-{index:04d}.npy", image, allow_pickle=False)
                    frames.append(
                        dict(
                            index=index,
                            case=frame["case"],
                            seed=frame["seed"],
                            step=frame["step"],
                            state_hash=frame["state_hash"],
                            render_kwargs=style,
                            source_cpu_sha256=frame["oracle_sha256"],
                            source_gpu_sha256=hashlib.sha256(original_gpu.tobytes()).hexdigest(),
                            image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                            old_vs_source_cpu=mse(old, original_cpu),
                            old_vs_source_gpu=mse(old, original_gpu),
                            new_vs_source_cpu=mse(image, original_cpu),
                            new_vs_source_gpu=mse(image, original_gpu),
                            source_cpu_vs_source_gpu=mse(original_cpu, original_gpu),
                        )
                    )
            finally:
                env.close()
        result = dict(registration=row["registration"], name=row["name"], status="PASS", frames=frames)
        write(target / "result.json", result)
        results.append(result)
        print(row["registration"], len(frames), "frames PASS", flush=True)
    write(output / "result.json", dict(status="PASS", mode=mode, profile=profile(), registrations=results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "baseline_styled", "fixed"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "cpu":
        cpu(args.output)
    else:
        gpu(args.mode, args.corpus, args.output)


if __name__ == "__main__":
    main()
