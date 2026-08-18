"""What does PGX actually EMIT, and can it be rasterized reproducibly?

Three questions, before any renderer is written:

1. Which SVG element types does each game's drawing use? A game whose document is rects, circles,
   lines and polygons with flat fills is a Scene2D in another notation. A game that emits <text> is
   font-dependent, and a font is per-installation content rather than a constant of the family.
2. Is there a rasterizer at all? PGX ships none -- ``save_svg`` writes vector text -- so "the official
   renderer" has to be named, and its determinism is a precondition for every parity number.
3. Is the raster non-uniform? A blank frame would let parity be measured against black and pass.
"""
from __future__ import annotations

import collections
import hashlib
import io
import re
import sys

import jax
import numpy as np
import pgx

ENVS = sorted(pgx.available_envs())


def svg_of(env_id: str, steps: int = 6, seed: int = 0) -> str:
    env = pgx.make(env_id)
    init = jax.jit(env.init)
    step = jax.jit(env.step)
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
    return st.to_svg()


def census() -> None:
    print("=" * 100)
    print("SVG ELEMENT CENSUS  (element -> count in one mid-game document)")
    print("=" * 100)
    rows = []
    for eid in ENVS:
        try:
            s = svg_of(eid)
        except Exception as e:  # noqa: BLE001
            print(f"{eid:26s} FAILED {type(e).__name__}: {str(e)[:60]}")
            continue
        tags = collections.Counter(re.findall(r"<(\w[\w-]*)", s))
        for junk in ("svg", "defs", "g", "?xml"):
            tags.pop(junk, None)
        has_text = tags.get("text", 0)
        has_img = tags.get("image", 0) + len(re.findall(r"xlink:href", s))
        rows.append((eid, len(s), has_text, has_img, dict(tags)))
        print(f"{eid:26s} bytes={len(s):7d} text={has_text:4d} img={has_img:4d}  {dict(tags)}")
    print()
    clean = [r[0] for r in rows if r[2] == 0]
    print(f"TEXT-FREE games ({len(clean)}): {clean}")
    print(f"TEXT-BEARING games ({len(rows)-len(clean)}): {[r[0] for r in rows if r[2] > 0]}")


def rasterizers(env_id: str = "othello") -> None:
    print()
    print("=" * 100)
    print(f"RASTERIZER AVAILABILITY + DETERMINISM  (on {env_id})")
    print("=" * 100)
    s = svg_of(env_id)

    def try_resvg():
        import resvg_py
        png = resvg_py.svg_to_bytes(svg_string=s)
        return bytes(png)

    def try_skia():
        import skia
        stream = skia.MemoryStream(s.encode())
        dom = skia.SVGDOM.MakeFromStream(stream)
        w, h = 400, 400
        surface = skia.Surface(w, h)
        with surface as canvas:
            canvas.clear(skia.ColorWHITE)
            dom.setContainerSize(skia.Size(w, h))
            dom.render(canvas)
        img = surface.makeImageSnapshot()
        return img.tobytes()

    def try_cairo():
        import cairosvg
        return cairosvg.svg2png(bytestring=s.encode())

    for name, fn in (("resvg_py", try_resvg), ("skia", try_skia), ("cairosvg", try_cairo)):
        try:
            a = fn()
            b = fn()
            ha = hashlib.sha256(a).hexdigest()[:16]
            hb = hashlib.sha256(b).hexdigest()[:16]
            print(f"{name:12s} OK  bytes={len(a):8d}  digest={ha}  twice_identical={ha == hb}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:12s} UNAVAILABLE  {type(e).__name__}: {str(e)[:90]}")


def decode_and_check(env_id: str = "othello") -> None:
    """Rasterize with resvg, decode to an array, and prove the frame is not blank."""
    print()
    print("=" * 100)
    print("FRAME IS NOT BLANK  (the 'parity against black' trap)")
    print("=" * 100)
    import resvg_py
    from PIL import Image
    for eid in ("othello", "connect_four", "go_9x9", "tic_tac_toe", "hex"):
        try:
            s = svg_of(eid)
            png = bytes(resvg_py.svg_to_bytes(svg_string=s))
            arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
            uniq = len(np.unique(arr.reshape(-1, 3), axis=0))
            print(f"{eid:16s} shape={arr.shape} unique_colors={uniq:5d} "
                  f"min={arr.min()} max={arr.max()} blank={uniq <= 1}")
        except Exception as e:  # noqa: BLE001
            print(f"{eid:16s} FAILED {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    print(f"pgx {pgx.__version__}   jax {jax.__version__}   python {sys.version.split()[0]}")
    print(f"jax devices: {jax.devices()}")
    census()
    rasterizers()
    decode_and_check()
