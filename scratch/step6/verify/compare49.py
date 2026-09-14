"""Diff a Step-6 replay of the frozen corpus against the frozen Step-4 oracle, one row per config.

Not a new oracle and not a new bound. It reads ``examples/results/step4/conformance/<shard>.json``
(read-only) and the shards ``examples/fidelity_step4.py`` just wrote under
``examples/results/step6/verify/replay/``, and prints what moved. Gate A comes from
``scratch/step4/certify.py``, which owns the bounds; this file never judges a threshold.

Three classes of field, kept apart because conflating them is how a declaration change gets reported
as a pixel change (the distinction is ``scratch/step6/reacher-correctness/compare_replay.py``'s and
is reused verbatim):

  * PIXEL    ``digest_ours`` / ``digest_ours_default_style`` -- the frames at the OFFICIAL default
             appearance. A difference here means the picture moved, full stop.
  * OFFICIAL ``digest_official`` -- the reference renderer's own output. Moves when the dependency
             pin or the machine moves, never because of us.
  * STYLE    ``digest_ours_corpus_style`` / ``style_digest`` / ``style_dim`` -- taken under a SAMPLED
             appearance, so a new variation leaf moves them without moving a default pixel.

Every check's status is compared name by name, so ``measured -> blocked_environment`` is visible as
itself rather than as an absent number.

    python scratch/step6/verify/compare49.py --replay <dir> --frozen <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: The five frozen shard names, plus ``dmcvis`` -- the four visual-tier DM Control cells split
#: into a shard of their own so that they can carry the provenance of the 3.11 reference pair
#: their declaration names. certify.py reads the reference version off the SHARD, so copying
#: those rows into the 3.12 document instead would have the certifier block the very numbers it
#: was handed -- which it did, the first time this was assembled.
SHARDS = ("dmc", "dmcvis", "fetch", "ogb", "raster", "swm")

PIXEL_KEYS = ("digest_ours", "digest_ours_default_style")
OFFICIAL_KEYS = ("digest_official",)
STYLE_KEYS = ("digest_ours_corpus_style", "style_digest", "style_dim")
INFO_KEYS = ("mean_pixel", "distinct_colors", "frames_shape", "size", "supersample", "batch")

#: Deltas the coordinator's brief names IN ADVANCE, with the reason each is already explained. A row
#: that matches one of these is reported as ``expected-known`` and is NOT a regression. A row that
#: does not match one of these and still moved is a finding. The predicate is deliberately narrow:
#: "Reacher may change its DECLARATION" does not license Reacher changing a pixel.
KNOWN = {
    "DMCReacherEasy": "declaration only (style_dim 17->18, visibility leaf; reacher-correctness)",
    "DMCReacherHard": "declaration only (style_dim 17->18, visibility leaf; reacher-correctness)",
    "DMCFingerTurnEasy": "declaration only (style_dim 17->18, visibility leaf; reacher-correctness)",
    "DMCFingerTurnHard": "declaration only (style_dim 17->18, visibility leaf; reacher-correctness)",
    "DMCCheetahRun": "MJWarp trackcom: visual checks BLOCKED under the 3.12 reference; digest does "
                     "not reproduce at BASELINE either (texture-fidelity verdict B + MJWARP-1)",
    "DMCHopperHop": "MJWarp trackcom: as DMCCheetahRun",
    "DMCWalkerWalk": "MJWarp trackcom: as DMCCheetahRun",
    "DMCHumanoidStand": "MJWarp trackcom: as DMCCheetahRun",
    "AntmazeTeleportExplore": "documented bake nondeterminism; digest does not reproduce at BASELINE",
}
#: Which of the known rows are allowed to move a DEFAULT PIXEL digest. The four reacher/finger cells
#: are NOT in here: their patch was certified pixel-identical at the default appearance over all 16
#: DM Control cells, so a moved default pixel there would be a real regression.
KNOWN_PIXEL_MOVERS = {"DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidStand",
                      "AntmazeTeleportExplore"}


def load(d: Path, suffix: str = "") -> dict:
    out = {}
    for sh in SHARDS:
        p = d / f"{sh}{suffix}.json"
        if not p.is_file():
            p = d / "conformance" / f"{sh}.json"
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        for cid, rec in doc["configs"].items():
            out[cid] = (sh, rec, doc.get("provenance", {}))
    return out


def checks(rec: dict) -> dict:
    out = {}
    for grp in ("mandatory", "family_specific"):
        for k, v in (rec.get(grp) or {}).items():
            out[k] = v.get("status")
    return out


def row(cid, frozen, replay):
    fsh, f, fprov = frozen
    rsh, r, rprov = replay
    fo, ro = (f.get("oracle") or {}), (r.get("oracle") or {})
    d = {"config": cid, "shard": rsh,
         "replay_status": r.get("status"), "frozen_status": f.get("status"),
         "why": str(r.get("why") or "")[:400]}
    d["pixel_changed"] = [k for k in PIXEL_KEYS if ro.get(k) != fo.get(k)]
    d["official_changed"] = [k for k in OFFICIAL_KEYS if ro.get(k) != fo.get(k)]
    d["style_changed"] = [k for k in STYLE_KEYS if ro.get(k) != fo.get(k)]
    d["declaration_changed"] = (r.get("signature", {}).get("describe_digest")
                                != f.get("signature", {}).get("describe_digest"))
    d["digest_ours_frozen"] = fo.get("digest_ours")
    d["digest_ours_replay"] = ro.get("digest_ours")
    d["mean_pixel_frozen"] = fo.get("mean_pixel")
    d["mean_pixel_replay"] = ro.get("mean_pixel")
    fc, rc = checks(f), checks(r)
    moved = {}
    for k in sorted(set(fc) | set(rc)):
        if fc.get(k) != rc.get(k):
            moved[k] = f"{fc.get(k)}->{rc.get(k)}"
    d["checks_total_frozen"] = len(fc)
    d["checks_total_replay"] = len(rc)
    d["checks_moved"] = moved
    d["checks_identical"] = not moved

    known = cid in KNOWN
    unexplained = []
    if d["replay_status"] != "measured":
        unexplained.append(f"status {d['replay_status']}")
    if d["pixel_changed"] and cid not in KNOWN_PIXEL_MOVERS:
        unexplained.append("default-appearance pixels moved")
    for k, v in moved.items():
        # Only the LOSS of a number is a finding. A check that the frozen run measured and this one
        # did not is a number that went missing; a check that did not exist in the frozen run at all
        # (``None->...``, e.g. reacher-correctness's new check_task_object_visibility) is coverage
        # this branch ADDED, and a check that went anything->measured is coverage it recovered.
        if v.startswith("None->") or v.endswith("->measured"):
            continue
        if not known:
            unexplained.append(f"{k} {v}")
    if d["official_changed"] and not known:
        unexplained.append("digest_official moved")
    d["unexplained"] = unexplained
    gained = {k for k, v in moved.items() if v.startswith("None->")}
    d["checks_gained"] = sorted(gained)
    d["checks_lost"] = {k: v for k, v in moved.items() if k not in gained}
    if (not unexplained and not d["pixel_changed"] and not d["checks_lost"]
            and not d["declaration_changed"]):
        d["verdict"] = "identical"
    elif not unexplained:
        d["verdict"] = "expected-known" if known else "declaration/style only"
    else:
        d["verdict"] = "REGRESSION?"
    d["known_note"] = KNOWN.get(cid, "")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, type=Path)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--frozen", type=Path,
                    default=ROOT / "examples" / "results" / "step4" / "conformance")
    ap.add_argument("--gate-a", type=Path, default=None,
                    help="certification.json written by scratch/step4/certify.py over the replay")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    frozen = load(a.frozen.parent if a.frozen.name == "conformance" else a.frozen)
    replay = load(a.replay, a.suffix)
    gate = {}
    if a.gate_a and a.gate_a.is_file():
        gate = {c: v.get("gate_a") for c, v in
                json.loads(a.gate_a.read_text())["configs"].items()}
    fgate = {}
    fcert = ROOT / "examples" / "results" / "step4" / "certification.json"
    if fcert.is_file():
        fgate = {c: v.get("gate_a") for c, v in json.loads(fcert.read_text())["configs"].items()}

    rows, missing = [], []
    for cid in sorted(frozen):
        if cid not in replay:
            missing.append(cid)
            continue
        d = row(cid, frozen[cid], replay[cid])
        d["gate_a_frozen"] = fgate.get(cid)
        d["gate_a_replay"] = gate.get(cid)
        # A Gate A that MOVED is not automatically a finding: FAIL->PASS is a fixed failure and is
        # the point of the step. Only a move INTO FAIL is a regression; a move into BLOCKED is a
        # judgement this interpreter cannot make and is reported as coverage lost, not as a pass.
        if d["gate_a_replay"] and d["gate_a_frozen"] and d["gate_a_replay"] != d["gate_a_frozen"]:
            if d["gate_a_replay"] == "FAIL":
                d["unexplained"].append(f"gate_a {d['gate_a_frozen']}->FAIL")
                d["verdict"] = "REGRESSION?"
            elif d["gate_a_replay"] == "BLOCKED" and cid not in KNOWN:
                d["unexplained"].append(f"gate_a {d['gate_a_frozen']}->BLOCKED (coverage lost)")
                d["verdict"] = "REGRESSION?"
            elif d["verdict"] == "identical":
                d["verdict"] = "gate A improved"
        rows.append(d)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"compare49{a.suffix}.json").write_text(json.dumps(
        {"n_frozen": len(frozen), "n_replayed": len(replay), "not_replayed": missing,
         "rows": rows}, indent=1))

    hdr = ("| config | shard | status | digest_ours | official | style | declaration | checks | "
           "Gate A (frozen->replay) | verdict |")
    L = [hdr, "|" + "---|" * 10]
    for d in rows:
        L.append("| {c} | {s} | {st} | {dg} | {of} | {sy} | {dc} | {ck} | {ga} | {v} |".format(
            c=d["config"], s=d["shard"], st=d["replay_status"],
            dg="identical" if not d["pixel_changed"] else "**CHANGED** " + ",".join(d["pixel_changed"]),
            of="identical" if not d["official_changed"] else "CHANGED",
            sy="-" if not d["style_changed"] else ",".join(d["style_changed"]),
            dc="changed" if d["declaration_changed"] else "-",
            ck=("%d/%d identical" % (d["checks_total_frozen"], d["checks_total_frozen"])
                + (f" +{len(d['checks_gained'])} new" if d["checks_gained"] else ""))
               if not d["checks_lost"]
               else "; ".join(f"{k} {v}" for k, v in d["checks_lost"].items()),
            ga=f"{d['gate_a_frozen']}->{d['gate_a_replay']}",
            v=d["verdict"]))
    if missing:
        for cid in missing:
            L.append(f"| {cid} | {frozen[cid][0]} | NOT REPLAYED | - | - | - | - | - | - | **MISSING** |")
    (a.out / f"compare49{a.suffix}.md").write_text("\n".join(L) + "\n")

    from collections import Counter
    c = Counter(d["verdict"] for d in rows)
    print("\n".join(L))
    print(f"\n{len(rows)} replayed of {len(frozen)} frozen; verdicts {dict(c)}; "
          f"not replayed: {missing or 'none'}")
    bad = [d["config"] for d in rows if d["verdict"] == "REGRESSION?"]
    print("UNEXPLAINED:", bad or "none")
    for d in rows:
        if d["unexplained"]:
            print(f"  {d['config']}: {d['unexplained']}")


if __name__ == "__main__":
    main()
