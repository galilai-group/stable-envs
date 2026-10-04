"""Source-only feasibility probe for PushT's immediate render projection.

The destination constructs matching reset geometry, then never steps/replays.
Its drawing call suppresses stale reset contacts and draws captured numeric
contact segments instead. This is an audit experiment, not production wiring
or GPU renderer acceptance. The task source is never modified on disk.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as md
import json
import subprocess
from pathlib import Path

import numpy as np
import pymunk
from pusht_replay_gate import PIN, action, diagnostics, equal, import_source
from pymunk.space_debug_draw_options import SpaceDebugColor


class ContactSegments(pymunk.SpaceDebugDrawOptions):
    def __init__(self):
        super().__init__()
        self.flags = self.DRAW_COLLISION_POINTS
        self.segments = []

    def draw_segment(self, a, b, color):
        self.segments.append([*a, *b, *color])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert md.version("pymunk") == "7.2.0"
    source_files = {}
    for name in (
        "spaces.py",
        "utils.py",
        "envs/utils.py",
        "envs/quantizer.py",
        "envs/pusht/env.py",
        "envs/pusht/env_discrete.py",
    ):
        relative = "stable_worldmodel/" + name
        data = (args.source / relative).read_bytes()
        assert data == subprocess.check_output(["git", "-C", str(args.source), "show", PIN + ":" + relative])
        source_files[relative] = hashlib.sha256(data).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    classes = import_source(args.source)
    cases = []
    for discrete, cls in enumerate(classes):
        variant = "discrete" if discrete else "continuous"
        configurations = [(f"seed{seed}", seed, None) for seed in (0, 17, 42)]
        configurations += [
            (
                f"shape{agent}-{block}",
                42,
                dict(variation=["all"], variation_values={"agent.shape": agent, "block.shape": block}),
            )
            for agent in range(8)
            for block in range(1, 8)
        ]
        for label, seed, options in configurations:
            source = cls(resolution=96, render_action=True)
            destination = cls(resolution=96, render_action=True)
            try:
                source.reset(seed=seed, options=copy.deepcopy(options))
                destination.reset(seed=seed, options=copy.deepcopy(options))
                original_draw = destination.space.debug_draw
                numeric_segments = np.empty((0, 8), np.float64)

                def draw_from_cache(draw_options):
                    old = draw_options.flags
                    draw_options.flags = old & ~draw_options.DRAW_COLLISION_POINTS
                    try:
                        original_draw(draw_options)
                    finally:
                        draw_options.flags = old
                    for row in numeric_segments:
                        draw_options.draw_segment(row[:2], row[2:4], SpaceDebugColor(*row[4:]))

                frames, reconstructed, poses, contact_rows, actions = [], [], [], [], []
                stale_mse, stale_first = [], None
                for t in range(32):
                    control = action(source, variant, t)
                    source.step(control.copy())
                    actions.append(control.astype(np.float64))
                    before = diagnostics(source)
                    cache = ContactSegments()
                    source.space.debug_draw(cache)
                    numeric_segments = np.asarray(cache.segments, np.float64).reshape(-1, 8)
                    contact_rows.append(numeric_segments.copy())
                    bodies = []
                    for name in ("agent", "block"):
                        a, b = getattr(source, name), getattr(destination, name)
                        bodies.append([*a.position, a.angle])
                        b.angle, b.position = a.angle, a.position
                        destination.space.reindex_shapes_for_body(b)
                    poses.append(bodies)
                    destination.latest_action = copy.deepcopy(source.latest_action)
                    reference = source.render()
                    destination.space.debug_draw = original_draw
                    stale = destination.render()
                    error = float(np.mean(((stale.astype(np.float64) - reference) / 255) ** 2))
                    stale_mse.append(error)
                    if error and stale_first is None:
                        stale_first = (t, stale.copy())
                    destination.space.debug_draw = draw_from_cache
                    actual = destination.render()
                    equal(reference, actual, f"{variant}/{label}/frame{t}")
                    equal(before, diagnostics(source), "projection/render must not mutate source")
                    frames.append(reference)
                    reconstructed.append(actual)
                capacity = max(map(len, contact_rows))
                contacts = np.zeros((32, capacity, 8), np.float64)
                for t, row in enumerate(contact_rows):
                    contacts[t, : len(row)] = row
                file = args.output / f"{variant}-{label}.npz"
                np.savez_compressed(
                    file,
                    source=frames,
                    projected=reconstructed,
                    body_poses=poses,
                    contacts=contacts,
                    contact_counts=np.array(list(map(len, contact_rows))),
                    actions=actions,
                    pose_only_first=(stale_first[1] if stale_first else np.zeros((0,), np.uint8)),
                )
                cases.append(
                    dict(
                        variant=variant,
                        case=label,
                        seed=seed,
                        frames=32,
                        destination_steps_after_reset=0,
                        projected_exact=True,
                        source_nonmutation=True,
                        max_contact_segments=capacity,
                        pose_only_mse_mean=float(np.mean(stale_mse)),
                        pose_only_mse_max=max(stale_mse),
                        pose_only_first_different_step=None if stale_first is None else stale_first[0],
                        artifact=file.name,
                        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    )
                )
                print(variant, label, "PASS", flush=True)
            finally:
                source.close()
                destination.close()
    result = dict(
        status="PASS_SOURCE_RENDER_PROJECTION_FEASIBILITY",
        source_files=source_files,
        versions={name: md.version(name) for name in ("pymunk", "pygame", "numpy", "opencv-python-headless")},
        source_pin=PIN,
        cases=cases,
        frames=sum(c["frames"] for c in cases),
        public_acceptance=False,
        gpu_acceptance=False,
        recipe="Matching reset geometry + current body poses + previous action marker + numeric contact-segment cache; no destination step/replay after reset",
        limitation="Source-only rendering experiment; production State cache/projection and renderer implementation still required",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
