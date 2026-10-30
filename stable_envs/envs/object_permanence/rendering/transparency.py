"""Explicit straight-ray alpha-surface semantics, separate from opaque contracts."""

CONTRACT = "stable-envs/transparent-pixel-rays-v1"
MATERIAL = "stable-envs/constant-alpha-surfaces-v1"
SPEC = {
    "id": CONTRACT,
    "camera": "fixed perspective or orthographic; existing pixel-center axial rays",
    "surfaces": "two-sided; constant alpha; zero-alpha faces omitted; no refraction",
    "events": "unique (binary64 depth, semantic object ID); shared triangle edges counted once",
    "order": "ascending exact depth, then semantic ID; lowest face index within an event",
    "termination": "first alpha-one surface, otherwise background; no transmittance cutoff",
    "rgb": "front-to-back alpha over in linear simplified-shader color, then uint8 quantization",
    "depth_ids": "nearest nonzero-alpha surface, zero for no hit",
    "masks": "overlapping per-object geometric support before first opaque surface; no perceptual threshold",
    "capacity": "runtime memory bound only; overflow raises, never truncates silently",
}
