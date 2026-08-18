"""Does resvg render <text> AT ALL? The three font configurations agreeing is only explicable two
ways -- text always drawn identically, or text never drawn -- and they have opposite consequences."""
import io
import numpy as np, resvg_py
from PIL import Image

def raster(svg, **kw):
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, **kw))
    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))

BOX = ('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="60">'
       '<rect width="120" height="60" fill="white"/>{}</svg>')
CASES = {
    "text only (Serif)":  '<text x="10" y="40" font-size="36px" font-family="Serif" fill="black">Ag8</text>',
    "text only (Courier)":'<text x="10" y="40" font-size="36px" font-family="Courier" fill="black">Ag8</text>',
    "text, no family":    '<text x="10" y="40" font-size="36px" fill="black">Ag8</text>',
    "control: a rect":    '<rect x="10" y="10" width="40" height="30" fill="black"/>',
}
for name, body in CASES.items():
    svg = BOX.format(body)
    for label, kw in (("default", {}), ("skip_system_fonts", {"skip_system_fonts": True}),
                      ("pinned DejaVuSerif",
                       {"skip_system_fonts": True,
                        "font_files": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]}),
                      ("serif_family=DejaVu Serif", {"serif_family": "DejaVu Serif"})):
        try:
            a = raster(svg, **kw)
            ink = int((a.reshape(-1, 3).min(1) < 250).sum())
            print(f"{name:22s} {label:26s} non-white pixels = {ink:6d}")
        except Exception as e:
            print(f"{name:22s} {label:26s} FAILED {type(e).__name__}: {e}")
    print()
