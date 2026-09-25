"""Time DMC's explicit integration-state signature as the closest native subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from dm_control.mujoco import engine
from state_ops import authoritative, catalog, engine_chain, measure, profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for registration, row in catalog().items():
        if row["family"] != "dmcontrol":
            continue
        recorded = json.loads((a.cpu / registration.replace("/", "__") / "result.json").read_text())
        assert recorded["status"] == "COMPLETE"
        source = authoritative(registration)
        try:
            source.reset(seed=17)
            actions = [np.asarray(v, dtype=np.float64) for v in recorded["actions_lane_zero"]]
            for action in actions[:2] + actions[-1:]:
                source.step(action)
            physics = next(obj.physics for obj in engine_chain(source) if getattr(obj, "physics", None) is not None)
            signature = mujoco.mjtState.mjSTATE_INTEGRATION
            snapshot = physics.get_state(sig=signature)
            native = dict(
                classification="NATIVE_PARTIAL",
                mechanism="dm_control.Physics.get_state/set_state(sig=mjSTATE_INTEGRATION)",
                like_for_like=False,
                omits="Task, RNG, wrapper lifecycle and mutable model configuration. Integration signature includes warm-start state.",
                signature=int(signature),
                numeric_payload_bytes=snapshot.nbytes,
                default_signature_payload_bytes=physics.get_state().nbytes,
                get=measure(lambda: physics.get_state(sig=signature)),
                set=measure(lambda: physics.set_state(snapshot, sig=signature)),
            )
            rows.append(dict(registration=registration, native=native))
        finally:
            source.close()
    sourcepath = Path(engine.__file__)
    result = dict(
        status="COMPLETE",
        profile=profile(),
        rows=rows,
        source_path=str(sourcepath),
        source_sha256=hashlib.sha256(sourcepath.read_bytes()).hexdigest(),
        note="The original default-signature get/set remains a narrower lower bound. This explicit integration signature is the closest tested native engine-state subset; no exhaustive equivalence.",
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DMC_NATIVE_INTEGRATION_COMPLETE", len(rows))


if __name__ == "__main__":
    main()
