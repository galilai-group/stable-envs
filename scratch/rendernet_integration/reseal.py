"""Rewrite a sealed checkpoint through the current ``rendernet.seal``, keeping its measured header.

One-off: the first seal embedded ``torch.__version__``, which is a ``str`` SUBCLASS, so the file could
not be read back by its own ``weights_only=True`` loader. ``seal`` now JSON-normalises every header
field; this re-writes the existing file through it rather than re-training, so the declaration stays
attached to the weights it was measured on.
"""
from __future__ import annotations

import argparse

import torch

from stable_envs.render import rendernet as RN

ap = argparse.ArgumentParser()
ap.add_argument("path")
a = ap.parse_args()

ck = torch.load(a.path, map_location="cpu", weights_only=False)
RN.seal(a.path, cell=ck["cell"], render=ck["render"], net=ck["net"],
        state_dict=ck["state_dict"], visual_fidelity=ck["visual_fidelity"],
        variation_fidelity=ck["variation_fidelity"], baseline=ck["baseline"],
        provenance=ck["provenance"])
back = RN.load(ck["cell"], device="cpu", path=a.path)
print("re-sealed and re-read", a.path)
print(" cell", back.cell)
print(" render", ck["render"])
print(" visual_fidelity", back.visual_fidelity)
print(" variation_fidelity", back.variation_fidelity)
