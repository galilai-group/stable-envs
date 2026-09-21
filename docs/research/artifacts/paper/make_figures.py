"""Build the paper's figures from artifacts that already exist in the repository.

Nothing here measures anything or draws a new plot: every panel is a crop, a rescale or a
concatenation of a PNG that `examples/gallery.py` already wrote. The provenance of each
output is recorded in `figures/SOURCES.txt`, so a reader can check a figure the same way
they would check a number.

    python examples/paper/make_figures.py            # from the repository root
"""
from __future__ import annotations

import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "figures")

SRC_LOGO = "docs/assets/brand/logo.png"
SRC_PUSHT = "docs/assets/media/PushT/parity.png"
SRC_MINIGRID = "docs/assets/media/MiniGridDoorKey/parity.png"


def _open(rel: str) -> Image.Image:
    return Image.open(os.path.join(ROOT, rel))


def _flatten(im: Image.Image) -> Image.Image:
    """Composite onto white; pdflatex renders a PNG alpha channel as black."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    lines = []

    logo = _flatten(_open(SRC_LOGO))
    logo.save(os.path.join(OUT, "logo.png"))
    lines.append(f"logo.png            <- {SRC_LOGO} (alpha flattened onto white)")

    # Figure 1: official | ours | |diff|, on identical states, for one Scene2D env and one
    # tile-atlas env, laid out in a row so the figure is wide and short. Both panels come
    # from examples/gallery.py and both are three columns, so they read the same way.
    # ONE band per environment, not two: the figure sits in a 3.25in column, so a second
    # PushT timestep would buy nothing and cost every panel a third of its width.
    p = _flatten(_open(SRC_PUSHT))
    band = p.height // 4
    blocks = [p.crop((0, 0, p.width, band))]
    # The MiniGrid panel's own label strip (the first 24 px of its 282 px band) is dropped;
    # the PushT headers to its left name the same three columns.
    mg = _flatten(_open(SRC_MINIGRID)).crop((0, 24, 776, 286))
    mg = mg.resize((round(mg.width * band / mg.height), band), Image.LANCZOS)
    blocks.append(mg)
    gap = 8
    width = sum(b.width for b in blocks) + gap * (len(blocks) - 1)
    parity = Image.new("RGB", (width, band), (255, 255, 255))
    x = 0
    for b in blocks:
        parity.paste(b, (x, 0))
        x += b.width + gap
    parity.save(os.path.join(OUT, "parity.png"))
    lines.append(
        f"parity.png          <- {SRC_PUSHT} (row 1 of 4) beside "
        f"{SRC_MINIGRID} (row 1 of 4, label strip removed, rescaled to height {band})")

    with open(os.path.join(OUT, "SOURCES.txt"), "w") as f:
        f.write("Figures are crops/rescales of committed repository artifacts.\n\n")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
