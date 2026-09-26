"""Source-reachable Rocket renderer preflight. Never changes production or frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle(out):
    from stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding import RocketLandingEnv

    from stable_envs.envs import rocket_landing as rl

    rows, styles, frames, cached, records = [], [], [], [], []

    def record(e, label, config, options, seed, dtype):
        row = e.state[:13].copy()
        st = rl._default_style_row()
        vis = {v[1]: np.array(v[7]) for v in e.env.getVisualShapeData(e.env.drones[0].Id)}
        pad = e.env.getVisualShapeData(e.landing_pad_id)[0][7]
        for name, rgba in [
            ("rocket.body_color", vis[-1]),
            ("rocket.fin_color", vis[2]),
            ("rocket.leg_color", vis[6]),
            ("rocket.booster_color", vis[1]),
            ("pad.color", pad),
        ]:
            offset, width = rl.style_layout()[name]
            st[offset : offset + width] = np.asarray(rgba)[:3] * 255
        st[rl.style_layout()["flame.ignited"][0]] = int(np.allclose(vis[9], rl.IGNITION_RGBA))
        frame = e.render().copy()
        assert np.array_equal(frame, e.render())
        link = e.env.getLinkState(e.env.drones[0].Id, 0)
        lag = rl.lagged_pose(row)[2][0]
        rows.append(row)
        styles.append(st)
        frames.append(frame)
        cached.append(np.r_[link[0], link[1]])
        records.append(
            dict(
                case=label,
                seed=seed,
                action_dtype=dtype,
                config=config,
                options=options,
                public_steps=e.step_count,
                height=float(e.lin_pos[2]),
                terminated=bool(e.termination),
                truncated=bool(e.truncation),
                env_complete=bool(e.info["env_complete"]),
                fatal_collision=bool(e.info["fatal_collision"]),
                contact=float(e.landing_pad_contact),
                lag_position_error=float(np.max(np.abs(lag - link[0]))),
                reference_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            )
        )

    for seed in (0, 17, 42):
        for dtype in ("float32", "float64"):
            for scenario, config, options in [
                ("fall", {}, {}),
                ("success", {"ceiling": 3.2}, {"accelerate_drop": False}),
                ("variation", {}, {"variation": ["all"]}),
            ]:
                e = RocketLandingEnv(render_mode="rgb_array", **config)
                try:
                    e.reset(seed=seed, options=options)
                    record(e, scenario + "_reset", config, options, seed, dtype)
                    for i in range(310 if scenario == "fall" else 12):
                        action = (
                            np.zeros(7, dtype=dtype)
                            if scenario != "variation"
                            else np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1], dtype=dtype)
                        )
                        e.step(action)
                        if (
                            (scenario == "fall" and e.step_count in (1, 300, 301, 302, 303, 304, 310))
                            or (scenario == "success" and e.step_count in (1, 6, 7, 8, 12))
                            or (scenario == "variation" and e.step_count in (1, 12))
                        ):
                            record(e, scenario + f"_{e.step_count}", config, options, seed, dtype)
                finally:
                    e.close()
                    if e.modified_pad_urdf:
                        Path(e.modified_pad_urdf).unlink(missing_ok=True)
    np.savez_compressed(
        out / "source-frames.npz",
        state13=np.stack(rows),
        style=np.stack(styles),
        frames=np.stack(frames),
        cached_link0=np.stack(cached),
    )
    report = dict(
        records=records,
        source_path=inspect.getfile(RocketLandingEnv),
        source_sha256=digest(inspect.getfile(RocketLandingEnv)),
        versions={k: md.version(k) for k in ["PyFlyt", "pybullet", "numpy", "gymnasium"]},
        renderer_sha256=digest(rl.__file__),
        source_calls_only=True,
        teleport_or_raw_physics=False,
    )
    (out / "source.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ROCKET_SOURCE_RENDERER_PREFLIGHT_CAPTURED", len(records), flush=True)


def gpu(out, source):
    import torch

    from stable_envs.envs import rocket_landing as rl
    from stable_envs.testing.conformance import pixel_agreement, ssim

    inp = np.load(source / "source-frames.npz", allow_pickle=False)
    meta = json.loads((source / "source.json").read_text())
    assert digest(rl.__file__) == meta["renderer_sha256"]
    e = rl.PyFlytRocketLanding(num_envs=1, device="cuda", size=480)
    try:
        states = torch.tensor(inp["state13"], dtype=torch.float32, device="cuda")
        styles = torch.tensor(inp["style"], dtype=torch.float32, device="cuda")
        original_states = states.clone()
        original_styles = styles.clone()
        frames = e._render(states, style=styles).cpu().numpy()
        singles = np.concatenate(
            [e._render(states[i : i + 1], style=styles[i : i + 1]).cpu().numpy() for i in range(len(states))]
        )
        assert np.array_equal(frames, singles)
        assert torch.equal(states, original_states) and torch.equal(styles, original_styles)
        records = []
        limits = rl.PyFlytRocketLanding.visual_fidelity
        for i, r in enumerate(meta["records"]):
            metrics = pixel_agreement(frames[i : i + 1], inp["frames"][i : i + 1])
            metrics["ssim"] = ssim(frames[i : i + 1], inp["frames"][i : i + 1])
            records.append(
                dict(
                    **r,
                    metrics=metrics,
                    within_frozen_limits=(
                        metrics["mean_abs"] <= limits["mean_abs_max"]
                        and metrics["within8_frac"] >= limits["within8_min"]
                        and metrics["ssim"] >= limits["ssim_min"]
                    ),
                )
            )
        groups = {}
        for group in sorted({r["case"] for r in records}):
            idx = [i for i, r in enumerate(records) if r["case"] == group]
            metrics = pixel_agreement(frames[idx], inp["frames"][idx])
            metrics["ssim"] = ssim(frames[idx], inp["frames"][idx])
            groups[group] = dict(
                count=len(idx),
                metrics=metrics,
                within_frozen_limits=(
                    metrics["mean_abs"] <= limits["mean_abs_max"]
                    and metrics["within8_frac"] >= limits["within8_min"]
                    and metrics["ssim"] >= limits["ssim_min"]
                ),
            )
        np.savez_compressed(out / "frozen-renderer-frames.npz", frames=frames)
        result = dict(
            status="RENDERER_PREFLIGHT_PASS"
            if all(v["within_frozen_limits"] for v in groups.values())
            else "RENDERER_PREFLIGHT_BLOCKER",
            records=records,
            groups=groups,
            frozen_limits={k: limits[k] for k in ["mean_abs_max", "within8_min", "ssim_min"]},
            batch_single_exact=True,
            inputs_unchanged=True,
            renderer_sha256=digest(rl.__file__),
            source_archive_sha256=digest(source / "source-frames.npz"),
            job_id=os.environ.get("SLURM_JOB_ID"),
            gpu=torch.cuda.get_device_name(0),
            versions={k: md.version(k) for k in ["torch", "nvdiffrast", "numpy", "scipy", "pybullet"]},
            scope="Frozen renderer preflight on authoritative reachable states. No public production State adapter is implemented or certified.",
        )
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["status"], json.dumps(groups), flush=True)
    finally:
        e.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["oracle", "gpu"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    oracle(a.output) if a.mode == "oracle" else gpu(a.output, a.source)
    (a.output / "SHA256SUMS").write_text(
        "".join(digest(p) + "  " + p.name + "\n" for p in sorted(a.output.iterdir()) if p.is_file())
    )


if __name__ == "__main__":
    main()
