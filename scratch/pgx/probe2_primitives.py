"""Enumerate the primitives PGX actually EMITS, with inherited attributes resolved.

Reading ``pgx/_src/dwg/*.py`` gives the intent; this gives the document. Every constant a renderer
needs -- draw order, stroke widths, resolved colours, the translate stack -- is read off the emitted
XML rather than off the source, because the source is one ``svgwrite`` default away from a picture
that looks right and is not the reference's.

Prints, per game: the canvas, and every drawable in document order with its resolved
(fill, stroke, stroke-width, transform).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import jax
import numpy as np
import pgx

SVG = "{http://www.w3.org/2000/svg}"
DRAWABLE = ("rect", "circle", "line", "polygon", "path", "text", "image")
#: SVG presentation attributes that INHERIT from an ancestor <g>.
INHERIT = ("fill", "stroke", "stroke-width", "font-size", "font-family")


def mid_state(env_id: str, steps: int, seed: int = 0):
    env = pgx.make(env_id)
    init, step = jax.jit(env.init), jax.jit(env.step)
    key = jax.random.PRNGKey(seed)
    key, sk = jax.random.split(key)
    st = init(sk)
    for _ in range(steps):
        if bool(st.terminated) or bool(st.truncated):
            break
        legal = np.asarray(st.legal_action_mask)
        if not legal.any():
            break
        key, sk = jax.random.split(key)
        a = int(jax.random.choice(sk, np.flatnonzero(legal)))
        st = step(st, a)
    return st


def walk(node, inherited: dict, tf: list, out: list) -> None:
    attrs = dict(node.attrib)
    inh = dict(inherited)
    for k in INHERIT:
        if k in attrs:
            inh[k] = attrs[k]
    my_tf = list(tf)
    if "transform" in attrs:
        my_tf = my_tf + [attrs["transform"]]
    tag = node.tag.replace(SVG, "")
    if tag in DRAWABLE:
        rec = {"tag": tag, "transform": " | ".join(my_tf)}
        rec.update({k: v for k, v in attrs.items() if k != "transform"})
        for k in INHERIT:
            if k not in attrs and k in inh:
                rec[k + "*"] = inh[k]          # * marks INHERITED, not written on the element
        out.append(rec)
    for ch in node:
        walk(ch, inh, my_tf, out)


def dump(env_id: str, steps: int, limit: int = 24) -> None:
    st = mid_state(env_id, steps)
    svg = st.to_svg()
    root = ET.fromstring(svg)
    out: list = []
    walk(root, {}, [], out)
    print("=" * 110)
    print(f"{env_id}   canvas={root.attrib.get('width')} x {root.attrib.get('height')}   "
          f"primitives={len(out)}")
    print("=" * 110)
    kinds: dict = {}
    for r in out:
        kinds[r["tag"]] = kinds.get(r["tag"], 0) + 1
    print(f"  counts: {kinds}")
    shown = 0
    seen_sig = set()
    for i, r in enumerate(out):
        # show every distinct SIGNATURE (tag + styling), plus the first few of each
        sig = (r["tag"], r.get("fill", r.get("fill*")), r.get("stroke", r.get("stroke*")),
               r.get("stroke-width", r.get("stroke-width*")), r["transform"])
        if sig in seen_sig and shown > limit:
            continue
        seen_sig.add(sig)
        shown += 1
        body = {k: v for k, v in r.items() if k != "tag"}
        print(f"  [{i:4d}] {r['tag']:8s} {body}")
        if shown > 60:
            print(f"  ... ({len(out)-i-1} more)")
            break
    print()


if __name__ == "__main__":
    print(f"pgx {pgx.__version__}")
    for eid, steps in (("tic_tac_toe", 5), ("connect_four", 8), ("othello", 6),
                       ("go_9x9", 10), ("go_19x19", 12), ("hex", 8)):
        try:
            dump(eid, steps)
        except Exception as e:  # noqa: BLE001
            print(f"{eid}: FAILED {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc(file=sys.stdout)
