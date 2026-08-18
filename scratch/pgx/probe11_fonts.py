"""Does a PINNED FONT make the 14 text-bearing PGX ids reproducible?

The question is whether "the reference" can be defined as
``resvg + skip_system_fonts=True + one pinned font file``. If it can, the decoder is a constant of
the family again and the text ids are not disqualified by the argument that a font is
per-installation content. Measured on chess, which is the prize and the hardest case (16 <text>
elements plus 32 <image> piece references).

Four things have to hold, and any one of them failing is the answer:
  1. resvg must rasterize the document at all with system fonts skipped;
  2. the result must be identical across calls and across PROCESSES;
  3. it must be identical to the SYSTEM-font render, or else pinning changes the picture and the
     "official appearance" becomes a thing we invented;
  4. the glyph set must be finite and small, or an atlas is not available and every game needs a text
     rasterizer at render time.
"""
from __future__ import annotations

import glob
import hashlib
import io
import re
import sys

import jax
import numpy as np
import pgx
import resvg_py
from PIL import Image

FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts"]


def mid_state(env_id, steps=6, seed=0):
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
        key, sk = jax.random.split(key)
        try:
            st = step(st, a, sk)
        except TypeError:
            st = step(st, a)
    return st


def raster(svg, **kw):
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, **kw))
    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


def find_fonts():
    out = []
    for d in FONT_DIRS:
        out += glob.glob(f"{d}/**/*.ttf", recursive=True)
        out += glob.glob(f"{d}/**/*.otf", recursive=True)
    return sorted(out)


def main():
    fonts = find_fonts()
    print(f"system font files found: {len(fonts)}")
    for f in fonts[:12]:
        print(f"   {f}")
    print()

    for eid in ("chess", "shogi", "2048"):
        try:
            st = mid_state(eid)
            svg = st.to_svg()
        except Exception as e:  # noqa: BLE001
            print(f"{eid}: state/svg FAILED {type(e).__name__}: {e}")
            continue
        n_text = len(re.findall(r"<text", svg))
        n_img = len(re.findall(r"<image", svg))
        texts = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
        print(f"=== {eid}: {n_text} <text>, {n_img} <image>, distinct strings={sorted(set(texts))[:20]}")

        try:
            a = raster(svg)
            b = raster(svg)
            print(f"   system fonts    : shape={a.shape} stable_in_process={np.array_equal(a, b)} "
                  f"sha={hashlib.sha256(a.tobytes()).hexdigest()[:16]}")
        except Exception as e:  # noqa: BLE001
            print(f"   system fonts    : FAILED {type(e).__name__}: {e}")
            a = None

        try:
            c = raster(svg, skip_system_fonts=True)
            d = raster(svg, skip_system_fonts=True)
            same = None if a is None else int(np.abs(a.astype(np.int16) - c.astype(np.int16)).max())
            print(f"   skip_system     : stable={np.array_equal(c, d)} "
                  f"sha={hashlib.sha256(c.tobytes()).hexdigest()[:16]} vs_system_maxdiff={same}")
        except Exception as e:  # noqa: BLE001
            print(f"   skip_system     : FAILED {type(e).__name__}: {e}")

        if fonts:
            pin = [f for f in fonts if "DejaVuSerif.ttf" in f] or [f for f in fonts if f.endswith(".ttf")]
            try:
                e1 = raster(svg, skip_system_fonts=True, font_files=[pin[0]])
                e2 = raster(svg, skip_system_fonts=True, font_files=[pin[0]])
                same = None if a is None else int(np.abs(a.astype(np.int16) - e1.astype(np.int16)).max())
                print(f"   pinned {pin[0].split('/')[-1]:24s}: stable={np.array_equal(e1, e2)} "
                      f"sha={hashlib.sha256(e1.tobytes()).hexdigest()[:16]} vs_system_maxdiff={same}")
            except Exception as ex:  # noqa: BLE001
                print(f"   pinned font     : FAILED {type(ex).__name__}: {ex}")
        print()

    # MinAtar: what does its renderer actually need?
    print("=== minatar render path ===")
    try:
        import matplotlib          # noqa: F401
        print("   matplotlib IS importable")
    except Exception as e:  # noqa: BLE001
        print(f"   matplotlib NOT importable: {type(e).__name__}: {e}")
    try:
        st = mid_state("minatar-breakout", steps=4)
        svg = st.to_svg()
        print(f"   minatar svg bytes={len(svg)} <image>={len(re.findall('<image', svg))} "
              f"<text>={len(re.findall('<text', svg))}")
        print(f"   observation shape={np.asarray(st.observation).shape} "
              f"dtype={np.asarray(st.observation).dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"   minatar FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
