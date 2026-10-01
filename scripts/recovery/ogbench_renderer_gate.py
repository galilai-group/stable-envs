"""Immutable Cube/Scene fixture generation and Oscar-only renderer acceptance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from stable_envs.env import Env
from stable_envs.envs.ogbench import OGBCube, OGBScene
from stable_envs.envs.ogbench import manipulation as frozen
from stable_envs.envs.ogbench import manipulation_renderer as renderer
from stable_envs.render import _resources
from stable_envs.testing import resource_guard

ROOT = Path(__file__).resolve().parents[2]
CLASSES = {"Cube": OGBCube, "Scene": OGBScene}


def save_state(path, state):
    keys = list(state.keys(True, True))
    np.savez_compressed(path, **{f"v{i}": state[k].detach().cpu().numpy() for i, k in enumerate(keys)})
    path.with_suffix(".json").write_text(json.dumps([list(k) for k in keys], indent=2) + "\n")


def load_state(path):
    keys = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as data:
        values = {tuple(k): torch.from_numpy(data[f"v{i}"].copy()) for i, k in enumerate(keys)}
    return TensorDict(values, [len(next(iter(values.values())))])


def equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    assert all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a.keys(True, True))


def generate(folder):
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for family, cls in CLASSES.items():
        with cls(seed=17) as env:

            def record(label, seed, kind):
                stem = f"{family}-{label}"
                full = env.get_state()
                rs = env.to_render_state(full)
                save_state(folder / (stem + "-full.npz"), full)
                save_state(folder / (stem + "-render.npz"), rs)
                cases.append(
                    dict(
                        family=family,
                        label=label,
                        seed=seed,
                        kind=kind,
                        full=stem + "-full.npz",
                        render=stem + "-render.npz",
                    )
                )

            for seed, task in [(0, 1), (17, 2), (42, 5)]:
                env.reset(seed, {"task_id": task, "variation": []})
                record(f"default-{seed}", seed, "default")
            env.reset(17, {"task_id": 1, "variation": []})
            visual_base = env.get_state()
            record("visual-base", 17, "default")
            variants = {
                "agent": {"agent.color": np.array([0.2, 0.7, 0.3])},
                "cube": {"cube.color": np.array([[0.1, 0.8, 0.3]])},
                "floor": {"floor.color": np.array([[0.7, 0.15, 0.2], [0.12, 0.6, 0.8]])},
                "camera": {"camera.angle_delta": np.array([[4.0, -3.0]])},
                "light": {"light.intensity": np.array([0.9])},
                "size-low": {"cube.size": np.array([0.01])},
                "size-high": {"cube.size": np.array([0.03])},
            }
            if family == "Scene":
                variants["locks"] = {"lock_color": np.array([[0.2, 0.8, 0.3], [0.1, 0.2, 0.9]])}
            for label, values in variants.items():
                options = {"task_id": 1, "variation": [], "variation_values": values}
                # Repeated application honors SWM's inverted agent dirty predicate.
                env.reset(17, options)
                env.reset(17, options)
                if label == "locks":
                    locked = env.get_state()
                    locked["scene", "cur_button_states"][0] = torch.tensor([0, 0])
                    env.set_state(locked)
                if not label.startswith("size-"):
                    equal(visual_base["physics"], env.get_state()["physics"])
                record(label, 17, "variation")
                cases[-1]["comparison_base"] = "buttons-00" if label == "locks" else "visual-base"
                cases[-1]["physics_unchanged"] = not label.startswith("size-")
            if family == "Scene":
                for bits in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                    env.reset(17, {"task_id": 1, "variation": []})
                    state = env.get_state()
                    state["scene", "cur_button_states"][0] = torch.tensor(bits)
                    env.set_state(state)
                    record("buttons-" + "".join(map(str, bits)), 17, "buttons")
            for interaction in ["cube"] if family == "Cube" else ["button", "drawer", "window"]:
                env.reset(17, {"task_id": 1, "variation": []})
                u = env._simulator.envs[0]
                if interaction == "cube":
                    target = u.data.joint("object_joint_0").qpos[:3].copy()
                    target[2] = 0.035
                elif interaction == "button":
                    target = u.data.site_xpos[u._button_site_ids[0]].copy()
                    target[2] = 0.06
                else:
                    target = u.data.site_xpos[getattr(u, "_" + interaction + "_site_id")].copy()
                contacts = set()
                for i in range(50):
                    goal = target.copy()
                    if i < 12:
                        goal[2] += 0.09
                    if i >= 32:
                        goal[{"drawer": 1, "window": 0, "cube": 2}.get(interaction, 2)] += {
                            "drawer": -0.12,
                            "window": 0.18,
                            "cube": 0.15,
                        }.get(interaction, 0.0)
                    action = np.zeros(5, dtype=np.float64 if i % 2 else np.float32)
                    action[:3] = np.clip((goal - u.data.site_xpos[u._pinch_site_id]) / 0.05, -1, 1)
                    action[4] = 1 if interaction == "button" or i >= 24 else -1
                    env.step(action[None], capture="states")
                    for c in u.data.contact:
                        pair = tuple(u.model.body(int(u.model.geom_bodyid[g])).name for g in (c.geom1, c.geom2))
                        if any("ur5e/" in b for b in pair) and any("ur5e/" not in b and b != "world" for b in pair):
                            contacts.add(" / ".join(pair))
                    if i in (19, 49):
                        record(f"contact-{interaction}-{i + 1}", 17, "contact")
                assert contacts
                cases[-1]["contacts"] = sorted(contacts)
    (folder / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("OGB_RENDER_FIXTURES_PASS", len(cases), flush=True)


def owned_oracle():
    path = ROOT / "docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py"
    spec = importlib.util.spec_from_file_location("gpu_ogb_owned_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_frame(family, rs, serial, out):
    """Independent SWM construction; frozen draw algorithm and constants unchanged."""
    oracle = owned_oracle()
    raw_cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    raw = raw_cls(entropy_seed=17)
    values = {
        name: rs["appearance", name][0].numpy().copy()
        for name in ["cube.size", "agent.color", "camera.angle_delta", "light.intensity", "cube.color", "floor.color"]
    }
    if family == "Scene":
        values["lock_color"] = rs["appearance", "lock_color"][0].numpy().copy()
    raw.reset(seed=17, options={"task_id": 1, "variation": [], "variation_values": values})
    if family == "Scene":
        raw._cur_button_states = rs["pose", "buttons"][0].numpy().copy()
        raw._apply_button_states()
    raw.data.qpos[:] = rs["pose", "qpos"][0].numpy()
    raw.data.mocap_pos[:] = rs["pose", "mocap_pos"][0].numpy().reshape(1, 3)
    raw.data.mocap_quat[:] = rs["pose", "mocap_quat"][0].numpy().reshape(1, 4)
    mujoco.mj_forward(raw.model, raw.data)
    # Source frames retain the frozen visual-tier residuals; they are never relabelled exact.
    source = raw.render(camera="front_pixels").copy()
    np.save(out / f"{serial}-source.npy", source)
    m = raw.model
    b = frozen._Bundle()
    b.unwrapped = raw
    b.size = int(rs["renderer", "size"][0, 0])
    b.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
    b.movers = frozen._movers(m)
    b.nq, b.nv, b.nmocap = m.nq, m.nv, m.nmocap
    b.nbuttons = 2 if family == "Scene" else 0
    b.state_names = tuple(str(i) for i in range(b.nq + 3 + b.nbuttons))
    b.pristine = {n: getattr(m, n).copy() for n in ["geom_group", "geom_rgba", "mat_rgba"]}
    b.button_tables = frozen._button_tables(raw, m, b.nbuttons)
    b.var_defaults = frozen._variation_defaults(raw)
    b.cube_geoms = frozen._cube_geoms(raw, m)
    b.grid_tex = frozen._grid_texture(m, b.var_defaults["floor.color"])
    token = "immutable-swm-baseline-" + family + "-" + str(serial)
    frozen._BUNDLE_CACHE[token] = b
    obj = object.__new__(frozen.OGBManipBase)
    obj.CELL = token
    obj._b = b
    obj.state_dim = b.nq + 3 + b.nbuttons
    obj.state_names = b.state_names
    Env.__init__(obj, num_envs=1, size=b.size, supersample=1, device="cuda", backend="mjwarp")
    columns = [rs["pose", "qpos"], rs["pose", "mocap_pos"]]
    if b.nbuttons:
        columns.append(rs["pose", "buttons"])
    assert torch.equal(rs["pose", "mocap_quat"], torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
    frame = frozen.OGBManipBase._render(obj, torch.cat(columns, -1)).cpu().numpy()[0]
    obj.close()
    np.save(out / f"{serial}-baseline.npy", frame)
    # Reference renderers obey the same process-lifetime GL ownership rule.
    return dict(
        source_exact_fraction=float(np.mean(np.all(source == frame, axis=-1))),
        source_max_abs=int(np.abs(source.astype(int) - frame.astype(int)).max()),
    )


def run_family(family, fixtures, out):
    assert torch.cuda.is_available(), "Oscar CUDA GPU required"
    assert hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() == renderer.FROZEN_SHA256
    patched = resource_guard.install()
    assert patched, "GPU context lifetime must be instrumented"
    context_before = resource_guard.take()
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in json.loads((fixtures / "cases.json").read_text()) if c["family"] == family]
    result = dict(
        family=family,
        status="FAIL",
        versions={
            n: version(n)
            for n in ["mujoco", "ogbench", "numpy", "gymnasium", "dm-control", "torch", "mujoco-warp", "warp-lang"]
        },
        gpu=torch.cuda.get_device_name(),
        cases=[],
        resources_before=_resources.counts(),
        context_instrumentation=patched,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    env = CLASSES[family](seed=42, device="cuda")
    states = []
    frames = []
    try:
        for index, case in enumerate(cases):
            serial = f"{index:03d}-{case['label']}"
            full = load_state(fixtures / case["full"])
            rs = load_state(fixtures / case["render"])
            before_np = np.random.get_state()
            # Live simulator deliberately has a different history.
            live = env.get_state()
            full_before = full.clone()
            rs_before = rs.clone()
            with (
                patch.object(env._simulator.cls, "step", side_effect=AssertionError("render stepped physics")),
                patch.object(env._simulator.cls, "reset", side_effect=AssertionError("render reset physics")),
            ):
                a = env.render(full).cpu().numpy()[0]
                b = env.render(rs).cpu().numpy()[0]
                c = env._render(rs).cpu().numpy()[0]
            np.save(out / f"{serial}-full.npy", a)
            np.save(out / f"{serial}-render.npy", b)
            assert np.array_equal(a, b) and np.array_equal(b, c), "full/RenderState/repeated exact equality"
            equal(live, env.get_state())
            equal(full_before, full)
            equal(rs_before, rs)
            after_np = np.random.get_state()
            assert (
                before_np[0] == after_np[0]
                and np.array_equal(before_np[1], after_np[1])
                and before_np[2:] == after_np[2:]
            )
            source_metrics = baseline_frame(family, rs, serial, out)
            baseline = np.load(out / f"{serial}-baseline.npy")
            comparison = dict(
                exact=bool(np.array_equal(a, baseline)),
                different_channels=int(np.count_nonzero(a != baseline)),
                max_abs=int(np.abs(a.astype(int) - baseline.astype(int)).max()),
            )
            entry = dict(
                case,
                comparison=comparison,
                source_visual_residual=source_metrics,
                nonmutation=True,
                full_private_exact=True,
            )
            result["cases"].append(entry)
            (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            assert comparison["exact"], f"frozen baseline differs: {serial} {comparison}"
            states.append(rs)
            frames.append(a)
            print("OGB_CASE_PASS", family, case["label"], flush=True)
        batch = torch.cat(states)
        expected = np.stack(frames)
        a = env.render(batch, chunk_size=1).cpu().numpy()
        b = env.render(batch, chunk_size=3).cpu().numpy()
        reverse = torch.arange(len(batch) - 1, -1, -1)
        c = env.render(batch[reverse], chunk_size=2).cpu().numpy()[::-1]
        assert np.array_equal(a, expected) and np.array_equal(b, expected) and np.array_equal(c, expected), (
            "batch/chunk/order"
        )
        with CLASSES[family](num_envs=3, seed=0, device="cuda") as public_batch:
            full_batch = torch.cat([load_state(fixtures / c["full"]) for c in cases[:3]])
            live_before = public_batch.get_state()
            packed = public_batch.to_render_state(full_batch)
            full_frames = public_batch.render(full_batch).cpu().numpy()
            packed_frames = public_batch.render(packed).cpu().numpy()
            assert np.array_equal(full_frames, packed_frames) and np.array_equal(full_frames, expected[:3])
            equal(live_before, public_batch.get_state())
        rigs = len(frozen._PLATE_RIGS)
        keepalive = len(frozen._KEEPALIVE)
        independent_before = resource_guard.take()
        with CLASSES[family](seed=0, device="cuda") as other:
            live = other.get_state()
            frame = other.render(states[0]).cpu().numpy()[0]
            equal(live, other.get_state())
            assert np.array_equal(frame, frames[0]), "independent renderer"
        assert len(frozen._PLATE_RIGS) == rigs and len(frozen._KEEPALIVE) == keepalive, (
            "new instance leaked a plate/context"
        )
        assert resource_guard.delta(independent_before)["context_created"] == 0
        context_delta = resource_guard.delta(context_before)
        assert context_delta["context_freed"] == 0 and context_delta["renderer_closed"] == 0, context_delta
        assert not _resources.release_failures(), _resources.release_failures()
        # All varied cases must affect pixels compared with the same task/seed default baseline.
        changes = {}
        for i, case in enumerate(cases):
            if case["kind"] == "variation":
                base_i = next(j for j, c in enumerate(cases) if c["label"] == case["comparison_base"])
                changes[case["label"]] = bool(np.any(frames[i] != frames[base_i]))
        assert all(changes.values()), changes
        result.update(
            status="PASS",
            batch_order_exact=True,
            chunk_size_exact=True,
            independent_environment_exact=True,
            visual_changes=changes,
            resources_after=_resources.counts(),
            context_lifetime=resource_guard.totals(),
            no_context_released=True,
            process_lifetime_plate_rigs=rigs,
            process_lifetime_renderer_refs=keepalive,
            resource_policy="Frozen process-lifetime EGL rigs retained; per-instance close releases only instance tables. Repeated/independent calls add no rig for existing configurations.",
        )
        print(f"OGB_{family.upper()}_GPU_GATE_PASS", flush=True)
    except BaseException:
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        env.close()
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", type=Path)
    p.add_argument("--family", choices=CLASSES)
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.generate:
        generate(args.generate)
    else:
        run_family(args.family, args.fixtures, args.output)


if __name__ == "__main__":
    main()
