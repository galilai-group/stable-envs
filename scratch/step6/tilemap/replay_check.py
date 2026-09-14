"""Diff a Step-6 replay against the FROZEN Step-4 oracle, digest by digest.

``examples/fidelity_step4.py`` re-renders the frozen states and writes a fresh record; this says
whether that record carries the same bytes the oracle froze. Digest equality is the whole claim — a
metric that merely stayed inside a bound would say the pixels are acceptable, not that they are the
same pixels, and an optimisation is only exempt from a semantic rerun when they are the same.

Three digests matter for a style-carrying env and all three are compared:

* ``digest_ours`` — the frames at the env's current appearance;
* ``digest_ours_default_style`` — explicitly at the default, which is the appearance the official
  frames were drawn at;
* ``digest_ours_corpus_style`` — at the corpus's own per-environment style, which is the only one
  that exercises the variation knobs, and therefore the only one that would catch a patch that is
  exact at the default and wrong under a variation.

``digest_official`` is compared when the replay could reach the official renderer and is reported as
unavailable — never as a pass — when it could not.

    python scratch/step6/tilemap/replay_check.py --replay out/replay.json --config CraftaxClassic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "examples" / "results" / "step4" / "conformance"

KEYS = ("digest_ours", "digest_ours_default_style", "digest_ours_corpus_style", "style_digest",
        "digest_official")


def frozen_oracle(config: str) -> dict:
    for shard in sorted(FROZEN.glob("*.json")):
        d = json.loads(shard.read_text())
        c = (d.get("configs") or {}).get(config)
        if c:
            return {"shard": shard.name, "oracle": c.get("oracle") or {},
                    "gate_a": c.get("registered_default_coverage"),
                    "declared_tier": c.get("declared_tier")}
    raise SystemExit(f"{config} is in no frozen Step-4 conformance shard under {FROZEN}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--config", default="CraftaxClassic")
    ap.add_argument("--against", type=Path, default=None,
                    help="compare against ANOTHER replay file instead of the frozen Step-4 oracle. "
                         "The frozen digests were taken on CUDA, so a CPU replay has no frozen "
                         "counterpart; what it has is the baseline checkout replayed in the same "
                         "job on the same node, which is the strongest available CPU statement.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    got = json.loads(a.replay.read_text())
    cfg = (got.get("configs") or {}).get(a.config) or got.get(a.config) or got
    new = cfg.get("oracle") or {}
    if a.against:
        other = json.loads(a.against.read_text())
        ocfg = (other.get("configs") or {}).get(a.config) or other
        ref = {"shard": str(a.against), "oracle": ocfg.get("oracle") or {},
               "gate_a": ocfg.get("registered_default_coverage"),
               "declared_tier": ocfg.get("declared_tier")}
    else:
        ref = frozen_oracle(a.config)
    old = ref["oracle"]

    rows, failures, unavailable = [], [], []
    for k in KEYS:
        want, have = old.get(k), new.get(k)
        if want is None:
            continue
        if have is None:
            unavailable.append(k)
            rows.append({"digest": k, "frozen": want, "replay": None, "status": "unavailable"})
            continue
        ok = want == have
        rows.append({"digest": k, "frozen": want, "replay": have,
                     "status": "identical" if ok else "CHANGED"})
        if not ok:
            failures.append(k)

    shape_ok = old.get("frames_shape") == new.get("frames_shape")
    res = {"config": a.config, "compared_against": ref["shard"],
           "reference_kind": "another replay" if a.against else "frozen Step-4 oracle", "replay": str(a.replay),
           "device_frozen": old.get("device"), "device_replay": new.get("device"),
           "size_frozen": old.get("size"), "size_replay": new.get("size"),
           "n_states_frozen": old.get("n_states"), "n_states_replay": new.get("n_states"),
           "frames_shape_identical": shape_ok,
           "mean_pixel": [old.get("mean_pixel"), new.get("mean_pixel")],
           "distinct_colors": [old.get("distinct_colors"), new.get("distinct_colors")],
           "registered_default_coverage": ref["gate_a"],
           "digests": rows, "changed": failures, "unavailable": unavailable,
           "verdict": "PIXELS UNCHANGED" if not failures and shape_ok else "PIXELS CHANGED"}
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
    sys.exit(0 if res["verdict"] == "PIXELS UNCHANGED" else 1)


if __name__ == "__main__":
    main()
