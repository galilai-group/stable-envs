"""The replay obligation the coordinator set for `CraftaxClassic` when W1 took it partial -> full.

The frozen Step-4 digest for this configuration is a digest of a SMALLER PICTURE: a (64,112,144,3)
world window rendered from a 127-slot state. The cell now draws the whole SWM observation,
(64,144,144,3) from a 149-slot state, so `examples/fidelity_step4.py` cannot replay it -- it raises,
by design, rather than reshaping a 127-wide corpus into a different number of 149-wide states:

    ValueError: CraftaxClassic.render: CraftaxClassic expects state_dim 149, got 127

That raise is the correct behaviour and is recorded as such. What CAN be held to the frozen digest is
the obligation the coordinator wrote down: **the world rows must stay byte-identical at light 1.0**.

This script is that check and only that check. It reads the FROZEN corpus
(`examples/results/step4/states/CraftaxClassic.npz`, read-only), zero-extends each 127-slot row into
the 149-slot layout -- which is an empty inventory, awake, with the day-cycle key at zero -- forces
`light_level` to 1.0 so the noise field is not drawn, renders at the frozen size, takes the world
rows, and digests them with the SAME function the frozen artifact used
(`conformance.frames_digest`, which hashes shape and dtype before the bytes, so a plain sha256 of
the same pixels is a different number and would look like a failure).

    python scratch/step7/closure/craftax_world_row_replay.py --device cuda --out DIR/x.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
FROZEN_STATES = ROOT / "examples" / "results" / "step4" / "states" / "CraftaxClassic.npz"
FROZEN_SHARD = ROOT / "examples" / "results" / "step4" / "conformance" / "swm.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    import torch                                                            # noqa: F401
    import stable_envs as se
    from stable_envs.envs.craftax import OBS_H, OBS_W, N_STRIP
    from stable_envs.testing import conformance as C

    frozen_row = json.loads(FROZEN_SHARD.read_text())["configs"]["CraftaxClassic"]["oracle"]
    size = int(frozen_row["size"])
    z = np.load(FROZEN_STATES, allow_pickle=False)
    old = np.asarray(z["states"], np.float32)                               # (64, 127)

    env = se.make("CraftaxClassic", device=a.device)
    light = OBS_H * OBS_W * 2 + 2 + N_STRIP
    new = np.zeros((old.shape[0], env.state_dim), np.float32)
    new[:, :old.shape[1]] = old
    new[:, light] = 1.0

    frames = env.render(torch.as_tensor(new, device=a.device), size=size).cpu().numpy()
    w = OBS_H * size
    world = np.ascontiguousarray(frames[:, :w])
    strip = frames[:, w:]

    rec = {
        "what_this_is": "CraftaxClassic's Step-4 replay obligation under the partial->full coverage "
                        "change: the WORLD ROWS at light_level 1.0 must reproduce the frozen digest",
        "device": a.device, "host": platform.node(), "machine": platform.machine(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_sha": os.environ.get("STABLE_ENVS_GIT_COMMIT"),
        "frozen_states": str(FROZEN_STATES.relative_to(ROOT)),
        "frozen_states_state_dim": int(old.shape[1]),
        "cell_state_dim": int(env.state_dim),
        "size": size,
        "frozen_frames_shape": frozen_row["frames_shape"],
        "full_frame_shape": list(frames.shape),
        "world_rows_shape": list(world.shape),
        "frozen_digest_ours": frozen_row["digest_ours"],
        "world_rows_digest": C.frames_digest(world),
        "world_rows_mean_pixel": float(world.mean()),
        "frozen_mean_pixel": frozen_row["mean_pixel"],
        "strip_rows_are_black": bool(strip.max() == 0),
        "strip_rows_shape": list(strip.shape),
    }
    rec["world_rows_identical_to_frozen"] = rec["world_rows_digest"] == rec["frozen_digest_ours"]
    rec["verdict"] = ("WORLD ROWS IDENTICAL" if rec["world_rows_identical_to_frozen"]
                      else "WORLD ROWS MOVED")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
