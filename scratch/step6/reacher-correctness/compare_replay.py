"""Replay a Step-6 fidelity run against the FROZEN Step-4 oracle, field by field.

Reads ``examples/results/step4/conformance/<shard>.json`` (read-only) and a Step-6 replay written
by ``examples/fidelity_step4.py``, and prints, per configuration, what moved.

The split matters and is the whole point of running this: a patch is allowed to change what an env
DECLARES and is not allowed to change what it DRAWS at the default appearance.

  * ``digest_ours`` and ``digest_ours_default_style`` are the frames at the OFFICIAL appearance.
    These must be byte-identical or the patch changed the picture.
  * ``digest_official`` is the reference renderer's own output. Identical unless the machine or the
    dependency pin moved.
  * ``digest_ours_corpus_style`` and ``style_digest`` are taken under ``uniform_style``, which draws
    from the variation space -- so they change BY CONSTRUCTION when a leaf is added, and changing
    them is not evidence of a pixel change. Reported separately and never mixed in.
  * ``style_dim`` / ``describe_digest`` record the declaration, which a new leaf legitimately moves.

    python scratch/step6/reacher-correctness/compare_replay.py FROZEN.json REPLAY.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

#: Digests taken at the DEFAULT appearance. A difference here is a pixel change, full stop.
PIXEL_KEYS = ("digest_ours", "digest_ours_default_style", "digest_official", "mean_pixel",
              "distinct_colors", "frames_shape")
#: Digests taken under a SAMPLED appearance. A new variation leaf moves them without moving a pixel
#: at the default, so they are reported and never asserted on.
STYLE_KEYS = ("digest_ours_corpus_style", "style_digest", "style_dim")


def main() -> int:
    frozen = json.loads(Path(sys.argv[1]).read_text())["configs"]
    replay = json.loads(Path(sys.argv[2]).read_text())["configs"]
    bad = 0
    for cid in sorted(replay):
        r, f = replay[cid], frozen.get(cid)
        if f is None:
            print(f"{cid:24s} NOT IN THE FROZEN ORACLE")
            continue
        if r.get("status") != "measured":
            print(f"{cid:24s} {r.get('status')}: {str(r.get('why'))[:160]}")
            bad += 1
            continue
        ro, fo = r.get("oracle") or {}, f.get("oracle") or {}
        px = [k for k in PIXEL_KEYS if ro.get(k) != fo.get(k)]
        sty = [k for k in STYLE_KEYS if ro.get(k) != fo.get(k)]
        decl = (r.get("signature", {}).get("describe_digest")
                != f.get("signature", {}).get("describe_digest"))
        # every check that the frozen run measured and this one did not reproduce
        moved = []
        for grp in ("mandatory", "family_specific"):
            for k, v in (r.get(grp) or {}).items():
                fv = (f.get(grp) or {}).get(k)
                if v.get("status") == "error":
                    moved.append(f"{k}=ERROR")
                elif fv and fv.get("status") == "measured" and v.get("status") != "measured":
                    moved.append(f"{k}:{fv.get('status')}->{v.get('status')}")
        verdict = "PIXELS MOVED" if px else ("checks moved" if moved else "pixel-identical")
        if px or moved:
            bad += 1
        print(f"{cid:24s} {verdict:16s} declaration_changed={decl} "
              f"style_keys_changed={sty or '-'} checks={moved or '-'}")
        for k in px:
            print(f"    {k}: frozen={fo.get(k)!r} replay={ro.get(k)!r}")
        tov = ((r.get("family_specific") or {}).get("check_task_object_visibility") or {})
        if tov.get("status") == "measured":
            print(f"    task objects: {json.dumps(tov['evidence'].get('objects', {}))}")
    print(f"\n{len(replay)} config(s); {bad} with something to look at")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
