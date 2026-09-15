"""Assemble the seven Step-7 cells' COMMITTED corpora into one ``<Config>.npz`` states directory.

``examples/fidelity_step4.corpus_for`` loads ``<states-dir>/<config_id>.npz`` and reads a ``meta``
key out of it; two of the Step-7 corpora were written without that key (recorded as a Step-8 freeze
carry-over in the dashboard) and PFRocketLanding's reference corpus is a multi-array file whose
states live under ``state13``. This script copies -- never redraws -- the committed arrays into the
one schema the replay harness reads, and records for each cell which committed file it came from and
the sha256 of the array, so the Gate-A run is provably over the corpus that is in the repository.

Nothing here samples, selects or judges. If a source file is missing the cell is reported missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
S7 = ROOT / "examples" / "results" / "step7"

#: cell -> (committed file, key holding the states, how many rows to take, note)
SOURCES = {
    "CraftaxClassic": (S7 / "craftax-classic-full/corpus/states.npz", "states", None,
                       "W1's frozen 64-state corpus (three daylight bands, 30 sleeping rows)"),
    "CraftaxFull": (S7 / "craftax-full/corpus/states.npz", "states", None,
                    "W2's frozen 64-state corpus (stratified over level, band, sleeping, light_map)"),
    "DMCManipulatorBringBall": (S7 / "dmc-manipulator/corpus/states/DMCManipulatorBringBall.npz",
                                "states", None, "W3's corpus at the declared 3.11 pin"),
    "DMCQuadrupedWalk": (S7 / "dmc-quadruped/phase1/corpus/states/DMCQuadrupedWalk.npz",
                         "states", None, "W4's corpus"),
    "OGBCube": (S7 / "ogb-manipulation/states/OGBCube.npz", "states", None, "W5's re-measured corpus"),
    "OGBScene": (S7 / "ogb-manipulation/states/OGBScene.npz", "states", None, "W5's re-measured corpus"),
    "PFRocketLanding": (S7 / "rocket-landing/reference/states.npz", "state13", 64,
                        "W6's reference corpus; the first 64 of its 200 lagged-rollout states, "
                        "which is the corpus size every other cell is judged on"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="states directory to write <Config>.npz into")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for cell, (src, key, take, note) in SOURCES.items():
        if not src.is_file():
            report[cell] = {"status": "missing", "source": str(src)}
            print(f"[{cell:26s}] MISSING {src}")
            continue
        z = np.load(src, allow_pickle=False)
        states = np.ascontiguousarray(np.asarray(z[key], np.float32))
        if take is not None:
            states = np.ascontiguousarray(states[:take])
        sha = hashlib.sha256(states.tobytes()).hexdigest()
        meta = {}
        if "meta" in z.files:
            try:
                meta = json.loads(str(z["meta"]))
            except Exception:                                               # noqa: BLE001
                meta = {}
        meta.update({"env": cell, "state_dim": int(states.shape[1]), "n_states": int(states.shape[0]),
                     "copied_from": str(src.relative_to(ROOT)), "copied_key": key,
                     "rows_taken": "all" if take is None else take,
                     "array_sha256": sha, "note": note,
                     "assembled_by": "scratch/step7/closure/stage_corpora.py"})
        np.savez(out / f"{cell}.npz", states=states,
                 meta=np.array(json.dumps(meta, sort_keys=True)))
        report[cell] = {"status": "staged", "source": str(src.relative_to(ROOT)),
                        "shape": list(states.shape), "array_sha256": sha}
        print(f"[{cell:26s}] {states.shape} sha {sha[:16]} <- {src.relative_to(ROOT)}")
    (out / "_staged.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {out}/_staged.json")


if __name__ == "__main__":
    main()
