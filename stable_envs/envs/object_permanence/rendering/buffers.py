"""Output contracts shared by recorded opaque, alpha and optical rendering."""

import torch

DTYPES = dict(
    rgb=torch.uint8,
    depth=torch.float32,
    ids=torch.int16,
    masks=torch.bool,
    weights=torch.float32,
    path_depth=torch.float32,
    layer_depth=torch.float32,
    layer_ids=torch.int16,
)


def outputs(modes, out, *, batch, height, width, objects, device, allowed, layers=16):
    """Validate every caller buffer before writing; otherwise allocate owned fields."""
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(allowed):
        raise ValueError("invalid output modes")
    shapes = {}
    for mode in modes:
        shape = (batch, height, width)
        if mode == "rgb":
            shape += (3,)
        elif mode in ("masks", "weights", "path_depth"):
            shape = (batch, objects, height, width)
        elif mode in ("layer_depth", "layer_ids"):
            shape += (layers,)
        shapes[mode] = shape
    if out is None:
        return {m: torch.empty(shapes[m], dtype=DTYPES[m], device=device) for m in modes}
    if set(out) != set(modes):
        raise ValueError("out must contain exactly the requested modes")
    for mode, value in out.items():
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != shapes[mode]
            or value.dtype != DTYPES[mode]
            or value.device != device
            or not value.is_contiguous()
        ):
            raise ValueError("out must match shape, dtype, device and be contiguous")
    return out
