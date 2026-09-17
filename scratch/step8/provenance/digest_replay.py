"""Frozen-corpus DIGEST replay for one cell, for comparing two interpreters on the same card.

    python scratch/step8/provenance/digest_replay.py --cells OGBCube,OGBScene,FetchReach,DMCCheetahRun \
        --device cuda --out <file>.json

For each cell: load the committed frozen corpus (the path and the array sha256 come from
``examples/results/step8/certification/corpus_index.json`` where the cell has an entry, and from
``examples/results/step4/states/<cell>.npz`` otherwise), build the cell through the PUBLIC path
(``se.make``), render the corpus, and record ``conformance.frames_digest``.

Deliberately narrow. It is not a certification and does not judge anything: a bump evaluation asks
"did any pixel move", and a digest answers exactly that, in one number per cell, under any
interpreter -- including one where a fidelity bound could not be judged because the reference moved
with the renderer. The corpus sha256 is re-checked on load so the two arms are provably the same
states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CORPUS_INDEX = ROOT / "examples/results/step8/certification/corpus_index.json"
STEP4_STATES = ROOT / "examples/results/step4/states"


def corpus_for(cell: str) -> tuple[Path, str | None, str]:
    index = json.loads(CORPUS_INDEX.read_text()).get("cells", {}) if CORPUS_INDEX.is_file() else {}
    if cell in index:
        rec = index[cell]
        return ROOT / rec["source"], rec.get("array_sha256"), "corpus_index.json"
    return STEP4_STATES / f"{cell}.npz", None, "examples/results/step4/states (frozen Step-4 corpus)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=64)
    a = ap.parse_args()

    import numpy as np
    import stable_envs as se
    from stable_envs.testing import conformance as C
    sys.path.insert(0, str(ROOT / "examples"))
    from fidelity_step4 import provenance                                   # the shared record

    out = {"what_this_is": "frozen-corpus digest replay; no threshold applied, nothing judged",
           "provenance": provenance(a.device), "device": a.device, "cells": {}}
    for cell in a.cells.split(","):
        cell = cell.strip()
        rec: dict = {}
        try:
            path, want_sha, how = corpus_for(cell)
            rec["corpus"] = {"path": str(path.relative_to(ROOT)), "from": how,
                             "declared_array_sha256": want_sha}
            with np.load(path) as z:
                key = "states" if "states" in z.files else z.files[0]
                states = np.asarray(z[key])
            rec["corpus"].update({
                "key": key, "shape": list(states.shape),
                "array_sha256": hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()})
            rec["corpus"]["sha256_matches_declared"] = (
                want_sha is None or rec["corpus"]["array_sha256"] == want_sha)
            states = states[: a.n_states]
            env = se.make(cell, batch=len(states), device=a.device)
            frames = env.render(states)
            rec.update({"status": "measured", "n_states": int(len(states)),
                        "size": int(getattr(env, "size", 0)),
                        "digest_ours": C.frames_digest(frames)})
            if getattr(env, "style_dim", 0):
                import torch
                rec["digest_ours_default_style"] = C.frames_digest(
                    env.render(states, style=torch.as_tensor(env.default_style())
                               .to(frames.device).expand(len(states), -1).contiguous())
                    if hasattr(env, "default_style") else frames)
        except BaseException as e:                                          # noqa: BLE001
            rec.update({"status": "error", "why": f"{type(e).__name__}: {str(e)[:400]}"})
        out["cells"][cell] = rec
        print(cell, rec.get("status"), rec.get("digest_ours") or rec.get("why"), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
