"""The constants a renderer needs, each read off the reference rather than off the source.

1. The resolved ColorSet per game per theme -- taken from the EMITTED document, so a svgwrite default
   or a theme branch cannot be transcribed wrong.
2. Whether an arbitrary board array can be written back into a pgx State, which is what
   ``reference_render`` has to do and what makes the parity number a comparison of one scene.
3. How resvg sizes its output, and whether ``to_svg(scale=k)`` gives an exactly k-times canvas -- the
   parity test renders at several sizes and a half-pixel canvas would silently resample.
4. Whether the raster is stable across PROCESSES, not just across calls.
"""
from __future__ import annotations

import hashlib
import io
import re
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np
import pgx

GAMES = {"tic_tac_toe": 9, "connect_four": 42, "othello": 64, "go_9x9": 81, "go_19x19": 361}


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
        st = step(st, int(jax.random.choice(sk, np.flatnonzero(legal))))
    return st


def colors() -> None:
    print("=" * 100)
    print("RESOLVED COLOURS PER GAME PER THEME  (from the emitted document)")
    print("=" * 100)
    for eid in GAMES:
        st = mid_state(eid, 8)
        for theme in ("light", "dark"):
            svg = st.to_svg(color_theme=theme)
            found = sorted(set(re.findall(r'(?:fill|stroke)="([^"]+)"', svg)))
            root = ET.fromstring(svg)
            print(f"{eid:12s} {theme:5s} canvas={root.attrib.get('width')}x{root.attrib.get('height')} "
                  f"colours={found}")
    print()


def state_writeback() -> None:
    """Can an arbitrary board be written into a State and read back by the renderer?"""
    print("=" * 100)
    print("STATE WRITE-BACK  (reference_render must be able to place the env AT a state)")
    print("=" * 100)
    rng = np.random.default_rng(0)
    for eid, n in GAMES.items():
        st = mid_state(eid, 8)
        x = st._x
        board = np.asarray(x.board)
        print(f"{eid:12s} _x.board shape={board.shape} dtype={board.dtype} "
              f"values={sorted(set(board.reshape(-1).tolist()))[:8]}")
        # write a random legal-alphabet board back
        if eid.startswith("go"):
            alpha = np.array([-1, 0, 1])
        elif eid == "connect_four":
            alpha = np.array([-1, 0, 1])
        elif eid == "othello":
            alpha = np.array([-1, 0, 1])
        else:
            alpha = np.array([-1, 0, 1])
        newb = rng.choice(alpha, size=board.shape).astype(board.dtype)
        try:
            st2 = st.replace(_x=x.replace(board=jnp.asarray(newb)))
            back = np.asarray(st2._x.board)
            ok = np.array_equal(back, newb)
            svg2 = st2.to_svg()
            print(f"{'':12s}   writeback ok={ok}  svg_bytes={len(svg2)}")
        except Exception as e:  # noqa: BLE001
            print(f"{'':12s}   WRITEBACK FAILED {type(e).__name__}: {e}")
        # othello/go also carry a turn/colour that flips what is drawn
        for fld in ("turn", "color", "_size"):
            if hasattr(x, fld):
                print(f"{'':12s}   _x.{fld} = {np.asarray(getattr(x, fld))}")
    print()


def resvg_sizing() -> None:
    print("=" * 100)
    print("RESVG SIZING  (does to_svg(scale=k) give an exactly k-times raster?)")
    print("=" * 100)
    import resvg_py
    from PIL import Image
    for eid in GAMES:
        st = mid_state(eid, 8)
        base = None
        for k in (1.0, 2.0, 0.5):
            svg = st.to_svg(scale=k)
            root = ET.fromstring(svg)
            png = bytes(resvg_py.svg_to_bytes(svg_string=svg))
            arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
            if base is None:
                base = arr.shape[0] / k
            print(f"{eid:12s} scale={k:<4} declared={root.attrib.get('width')} "
                  f"raster={arr.shape[:2]} exact_multiple={abs(arr.shape[0] - base * k) < 1e-9}")
        print()


def cross_process_digest() -> None:
    """Printed so a second run of this script can be compared against the first."""
    print("=" * 100)
    print("RASTER DIGEST  (compare across two independent processes)")
    print("=" * 100)
    import resvg_py
    for eid in GAMES:
        st = mid_state(eid, 8)
        png = bytes(resvg_py.svg_to_bytes(svg_string=st.to_svg()))
        print(f"{eid:12s} sha256={hashlib.sha256(png).hexdigest()}")
    print()


if __name__ == "__main__":
    colors()
    state_writeback()
    resvg_sizing()
    cross_process_digest()
