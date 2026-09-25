"""Replay the existing Rocket witnesses with controlled native-cache additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))
import rocket_feasibility as probe  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=[
            "baseline",
            "skip_reconstruction",
            "links",
            "colliders",
            "velocities",
            "interpolation",
            "activation",
            "pre_contacts",
            "empty_contacts",
        ],
        required=True,
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    masks = {
        "baseline": 0,
        "skip_reconstruction": 0,
        "links": 1,
        "colliders": 3,
        "velocities": 7,
        "interpolation": 15,
        "activation": 31,
        "pre_contacts": 31,
        "empty_contacts": 31,
    }
    capture_original, restore_original = probe.capture, probe.restore
    cache = {}
    contact_cache = {}
    contact_comparisons = []
    retained = {}
    temporary = args.output / "native-transfer.txt"

    def operation(env, name):
        os.environ["SE_ROCKET_CACHE_OPERATION"] = name
        os.environ["SE_ROCKET_CACHE_FILE"] = str(temporary.resolve())
        os.environ["SE_ROCKET_CACHE_MASK"] = str(masks[args.mode])
        try:
            env.env.getPhysicsEngineParameters()
        finally:
            del os.environ["SE_ROCKET_CACHE_OPERATION"]
        assert temporary.exists(), "Research hook not present in loaded PyBullet"

    def capture(env):
        state = capture_original(env)
        operation(env, "export")
        cache[id(state)] = temporary.read_text()
        contact_path = Path(str(temporary) + ".contacts")
        if contact_path.exists():
            contact_cache[id(state)] = contact_path.read_text()
        return state

    def restore(env, state, mode, state_id=None, filename=None):
        if args.mode in {"pre_contacts", "empty_contacts"}:
            temporary.write_text(cache[id(state)])
            os.environ["SE_ROCKET_CACHE_OPERATION"] = "import"
            os.environ["SE_ROCKET_PRE_CONTACT_CACHE_IMPORT"] = "1"
        if args.mode == "empty_contacts":
            os.environ["SE_ROCKET_SYNC_EMPTY_CONTACTS"] = "1"
        if args.mode == "skip_reconstruction":
            os.environ["SE_ROCKET_SKIP_RECONSTRUCTION"] = "1"
        try:
            restore_original(env, state, mode, state_id, filename)
        finally:
            os.environ.pop("SE_ROCKET_SKIP_RECONSTRUCTION", None)
            os.environ.pop("SE_ROCKET_PRE_CONTACT_CACHE_IMPORT", None)
            os.environ.pop("SE_ROCKET_SYNC_EMPTY_CONTACTS", None)
            os.environ.pop("SE_ROCKET_CACHE_OPERATION", None)
        native = cache[id(state)]
        digest = hashlib.sha256(native.encode()).hexdigest()
        retained[digest] = native
        if masks[args.mode]:
            temporary.write_text(native)
            operation(env, "import")
        operation(env, "export")
        contact_path = Path(str(temporary) + ".contacts")
        if contact_path.exists():
            before = contact_cache[id(state)]
            after = contact_path.read_text()
            contact_comparisons.append({"route": mode, "exact": before == after, "source": before, "restored": after})

    probe.capture, probe.restore = capture, restore
    sys.argv = ["rocket_feasibility", "--output", str(args.output / "feasibility")]
    probe.main()
    import rocket_snapshot_controls as controls

    sys.argv = ["rocket_snapshot_controls", "--output", str(args.output / "controls")]
    controls.main()
    for digest, native in retained.items():
        (args.output / ("native-" + digest + ".txt")).write_text(native)
    temporary.unlink(missing_ok=True)
    result = {
        "mode": args.mode,
        "contact_comparisons": contact_comparisons,
        "mask": masks[args.mode],
        "ordinary_step_patch": False,
        "retained_numeric_engine_checkpoints": len(retained),
        "public_state_implemented": False,
        "note": "Same committed witness collectors/actions; native instrumentation is research-only.",
    }
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
