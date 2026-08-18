"""The asset bill: how much data would a GPU renderer have to carry to draw *any* RoboCasa kitchen?

A per-episode "scene id" is only compact if the decoder for that id is small. RoboCasa's decoder is
the asset library itself -- every object mesh, every fixture, every material texture that a layout
can draw from. This walks the installed asset tree and counts it, so the "scene id + variation"
proposal can be priced honestly rather than asserted.

Also counts the combinatorics: layouts x styles x object-instance choices, which is what a
"small variation vector" would have to index into.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

MESH_EXT = {".stl", ".obj", ".msh", ".ply", ".dae", ".glb", ".gltf"}
TEX_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".exr", ".webp", ".ktx"}
XML_EXT = {".xml", ".mjcf"}


def walk(root):
    n = Counter()
    b = Counter()
    per_dir = Counter()
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            kind = ("mesh" if ext in MESH_EXT else
                    "texture" if ext in TEX_EXT else
                    "xml" if ext in XML_EXT else "other")
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            n[kind] += 1
            b[kind] += sz
            rel = os.path.relpath(dirpath, root).split(os.sep)[0]
            per_dir[rel] += sz
    return n, b, per_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import robocasa
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    pkg = os.path.dirname(robocasa.__file__)
    assets = os.path.join(pkg, "models", "assets")

    rec = {"assets_root": assets}
    n, b, per_dir = walk(assets)
    rec["file_counts"] = dict(n)
    rec["bytes_by_kind_GB"] = {k: v / 1e9 for k, v in b.items()}
    rec["total_GB"] = sum(b.values()) / 1e9
    rec["top_subdirs_GB"] = {k: v / 1e9 for k, v in per_dir.most_common(12)}

    # object catalogue: robocasa objects live as one dir per instance with a model.xml
    objroot = os.path.join(assets, "objects")
    cats = {}
    if os.path.isdir(objroot):
        for group in sorted(os.listdir(objroot)):
            g = os.path.join(objroot, group)
            if not os.path.isdir(g):
                continue
            cats[group] = {}
            for cat in sorted(os.listdir(g)):
                c = os.path.join(g, cat)
                if os.path.isdir(c):
                    insts = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
                    cats[group][cat] = len(insts)
    rec["object_catalogue"] = cats
    rec["n_object_categories"] = sum(len(v) for v in cats.values())
    rec["n_object_instances"] = sum(sum(v.values()) for v in cats.values())

    # fixtures
    fxroot = os.path.join(assets, "fixtures")
    fx = {}
    if os.path.isdir(fxroot):
        for group in sorted(os.listdir(fxroot)):
            g = os.path.join(fxroot, group)
            if os.path.isdir(g):
                fx[group] = sum(1 for _r, _d, f in os.walk(g) for x in f if x.endswith(".xml"))
    rec["fixture_xml_counts"] = fx
    rec["n_fixture_xml"] = sum(fx.values())

    # textures / styles
    texroot = os.path.join(assets, "textures")
    rec["n_texture_files"] = sum(1 for _r, _d, f in os.walk(texroot) for x in f
                                 if os.path.splitext(x)[1].lower() in TEX_EXT) if os.path.isdir(texroot) else 0
    genroot = os.path.join(assets, "generative_textures")
    rec["generative_textures_present"] = os.path.isdir(genroot)
    if os.path.isdir(genroot):
        rec["n_generative_texture_files"] = sum(
            1 for _r, _d, f in os.walk(genroot) for x in f
            if os.path.splitext(x)[1].lower() in TEX_EXT)

    rec["n_layouts"] = sum(1 for t in LayoutType if t.value >= 0)
    rec["n_styles"] = sum(1 for t in StyleType if t.value >= 0)
    rec["layout_style_pairs"] = rec["n_layouts"] * rec["n_styles"]

    print(json.dumps(rec, indent=2, default=str))
    with open(os.path.join(args.out, "assets.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)


if __name__ == "__main__":
    main()
