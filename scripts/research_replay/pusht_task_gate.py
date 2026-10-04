"""Verify the migrated private PushT task against pinned source and replay witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from pusht_replay_gate import diagnostics, equal, import_source, rebuild
from pusht_reset_anchor_gate import decode, reconstruct

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stable_envs._adapters.pusht_task import PushT, PushTDiscrete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = import_source(args.source.resolve())
    migrated = (PushT, PushTDiscrete)
    report = dict(
        status="RUNNING",
        cases=[],
        quantizer_centers=0,
        public_acceptance=False,
        production_files={
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((ROOT / "stable_envs/_adapters").glob("pusht_*.py"))
        },
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        a, b = original[1](resolution=64), migrated[1](resolution=64)
        try:
            for radial in range(16):
                for angular in range(16):
                    indices = np.array([radial, angular], np.int64)
                    equal(a.quantizer.dequantize(indices), b.quantizer.dequantize(indices))
                    report["quantizer_centers"] += 1
        finally:
            a.close()
            b.close()
        for cohort in ("pusht_replay", "pusht_reset_anchors"):
            directory = args.evidence / cohort
            records = json.loads((directory / "result.json").read_text())
            assert records["status"].startswith("PASS_")
            for row in records["cases"]:
                index = 0 if row["variant"] == "continuous" else 1
                file = directory / row["artifact"]
                assert hashlib.sha256(file.read_bytes()).hexdigest() == row["artifact_sha256"]
                with np.load(file, allow_pickle=False) as data:
                    if cohort == "pusht_replay":
                        options = None
                        if row["scenario"] != "default":
                            start = np.array([250, 220, 250, 250, 0, 0, 0], np.float64)
                            options = dict(
                                variation=[],
                                state=start,
                                goal_state=start.copy()
                                if row["scenario"] == "success"
                                else np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
                            )
                        source = rebuild(original[index], row["seed"], options, data["prefix"], data["dtype_codes"])
                        candidate = rebuild(migrated[index], row["seed"], options, data["prefix"], data["dtype_codes"])
                        actions, codes = data["suffix"], data["suffix_dtype_codes"]
                    else:
                        path = directory / row["anchor"]
                        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["anchor_sha256"]
                        saved = decode(json.loads(path.read_text()))
                        source = reconstruct(original[index], saved, dict(resolution=64))
                        candidate = reconstruct(migrated[index], saved, dict(resolution=64))
                        actions = data["controls"]
                        codes = [3 if index else 1 if i % 2 == 0 else 2 for i in range(len(actions))]
                    try:
                        equal(diagnostics(source), diagnostics(candidate))
                        equal(source.render(), candidate.render())
                        equal(candidate.render(), data["source_frames"][0])
                        frame_index = 1
                        for step, (control, code) in enumerate(zip(actions, codes, strict=True)):
                            typed = control.astype({1: np.float32, 2: np.float64, 3: np.int64}[int(code)])
                            a, b = source.step(typed.copy()), candidate.step(typed.copy())
                            equal(a, b)
                            equal(diagnostics(source), diagnostics(candidate))
                            equal(b[0]["state"], data["source_observations"][step])
                            np.testing.assert_array_equal(
                                [b[1], b[2], b[3], b[4]["n_contacts"]], data["source_tasks"][step]
                            )
                            if cohort == "pusht_replay" or step % 10 == 0:
                                equal(source.render(), candidate.render())
                                equal(candidate.render(), data["source_frames"][frame_index])
                                frame_index += 1
                        report["cases"].append(
                            dict(
                                cohort=cohort,
                                artifact=row["artifact"],
                                source_artifact_sha256=row["artifact_sha256"],
                                future_steps=len(actions),
                                exact=True,
                            )
                        )
                        save()
                        print(cohort, row["artifact"], "PASS", flush=True)
                    finally:
                        source.close()
                        candidate.close()
        report["status"] = "PASS_PRIVATE_TASK_MIGRATION"
    except BaseException:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
