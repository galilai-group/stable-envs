"""Optional learned correction of rendered frames.

Weights are specific to a scene, camera, model, and render configuration. Loading
checks that identity and the declared fidelity against the sealed checkpoint.
Corrected images are approximations, not pixel-exact reference renders.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _resources

#: Where sealed correctors live. One file per cell, named for the env class it is keyed to.
CACHE_ROOT = os.path.expanduser(os.environ.get("STABLE_ENVS_RENDERNET_CACHE", "~/.cache/stable-envs/rendernet"))

#: Bumped when the checkpoint layout changes in a way an older loader would misread. A file written
#: under a different format string is refused rather than best-effort parsed.
FORMAT = "stable-envs/rendernet/1"

#: The command that produces a missing checkpoint, quoted in every "there is none" error.
BUILD_CMD = "python examples/rendernet/train_corrector.py {env}"

# Fix the network batch shape so cuDNN/Inductor algorithm choices cannot vary by request.
# Pad short calls and discard padding after inference to preserve pixel reproducibility.
NET_BATCH = int(os.environ.get("STABLE_ENVS_RENDERNET_BATCH", 128))

# Compilation mode changes numerical results; keep it consistent when producing and replaying data.
# Capture CUDA graphs explicitly after compilation and autotuning.
COMPILE_MODE = os.environ.get("STABLE_ENVS_RENDERNET_COMPILE", "max-autotune-no-cudagraphs")

#: The frames are in [0,255]; 1/255 keeps a zero-initialised residual head starting exactly at the
#: bilinear baseline, which is what makes every trained number a departure from a measured point.
_SCALE = 1.0 / 255.0


class Corrector(nn.Module):
    """Map (B,r,r,3) uint8 frames to (B,out,out,3) uint8 frames.

    Add a learned residual to a bilinear resize. Zero-initialized heads start at
    that resize exactly; preprocessing and quantization belong to the model.
    """

    out_size: int = 256
    # Resize only the trunk input; the bilinear anchor uses the original frame.
    pre_size: int = 0

    def forward_float(self, x: torch.Tensor) -> torch.Tensor:
        """Return differentiable float NCHW predictions without output clamping."""
        base = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        if self.pre_size and x.shape[-1] != self.in_size:
            x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        return base + self.residual(x)

    def forward(self, u8: torch.Tensor) -> torch.Tensor:
        x = u8.permute(0, 3, 1, 2).to(self.dtype).mul_(_SCALE)
        x = x.contiguous(memory_format=torch.channels_last)
        # Quantize in the model dtype to avoid a full-resolution float32 temporary.
        y = self.forward_float(x).mul_(255.0).round_().clamp_(0, 255)
        return y.to(torch.uint8).permute(0, 2, 3, 1)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _conv(cin: int, cout: int, k: int = 3) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, k, padding=k // 2)


def _act() -> nn.Module:
    """SiLU keeps gradients available for negative trunk activations."""
    return nn.SiLU(inplace=True)


class ESPCN(Corrector):
    """Low-resolution convolutions followed by a subpixel upsample."""

    def __init__(
        self,
        in_size: int = 128,
        out_size: int = 256,
        width: int = 32,
        depth: int = 4,
        unshuffle: int = 1,
        pre_size: int = 0,
    ):
        super().__init__()
        if in_size <= 0 or out_size <= 0 or out_size % in_size:
            raise ValueError(f"out_size {out_size} must be a positive multiple of in_size {in_size}")
        self.out_size, self.in_size, self.unshuffle = out_size, in_size, unshuffle
        self.pre_size = pre_size
        up = (out_size // in_size) * unshuffle
        layers = [_conv(3 * unshuffle * unshuffle, width), _act()]
        for _ in range(depth - 2):
            layers += [_conv(width, width), _act()]
        self.trunk = nn.Sequential(*layers)
        self.head = _conv(width, 3 * up * up)
        self.shuffle = nn.PixelShuffle(up)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.unshuffle > 1:
            x = F.pixel_unshuffle(x, self.unshuffle)
        return self.shuffle(self.head(self.trunk(x)))


class UNetTiny(Corrector):
    """One downsample/upsample pair with a skip connection and subpixel output."""

    def __init__(
        self,
        in_size: int = 128,
        out_size: int = 256,
        width: int = 32,
        depth: int = 2,
        unshuffle: int = 1,
        pre_size: int = 0,
    ):
        super().__init__()
        if in_size <= 0 or out_size <= 0 or out_size % in_size:
            raise ValueError(f"out_size {out_size} must be a positive multiple of in_size {in_size}")
        self.out_size, self.in_size, self.unshuffle = out_size, in_size, unshuffle
        self.pre_size = pre_size
        up = (out_size // in_size) * unshuffle
        w2 = width * 2
        self.stem = nn.Sequential(_conv(3 * unshuffle * unshuffle, width), _act())
        self.down = nn.Sequential(
            nn.Conv2d(width, w2, 3, stride=2, padding=1),
            _act(),
            *sum([[_conv(w2, w2), _act()] for _ in range(depth)], []),
        )
        self.up = nn.Sequential(_conv(w2, width * 4), nn.PixelShuffle(2), _act())
        self.fuse = nn.Sequential(_conv(2 * width, width), _act())
        self.head = _conv(width, 3 * up * up)
        self.shuffle = nn.PixelShuffle(up)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.unshuffle > 1:
            x = F.pixel_unshuffle(x, self.unshuffle)
        s = self.stem(x)
        d = self.up(self.down(s))
        return self.shuffle(self.head(self.fuse(torch.cat([s, d], 1))))


# name -> (class, trunk resolution, width, depth, unshuffle, input render resolution).
ZOO: dict[str, tuple] = {
    #  name                        class     trunk width depth unshuffle render
    "espcn_112t128_w48_d4_u2": (ESPCN, 128, 48, 4, 2, 112),
    "espcn_112t128_w64_d6_u2": (ESPCN, 128, 64, 6, 2, 112),
    "espcn_112t128_w96_d6_u2": (ESPCN, 128, 96, 6, 2, 112),
    "espcn_112t128_w128_d4_u2": (ESPCN, 128, 128, 4, 2, 112),
    "unet_112t128_w48_d2_u2": (UNetTiny, 128, 48, 2, 2, 112),
    "unet_112t128_w64_d3_u2": (UNetTiny, 128, 64, 3, 2, 112),
    "espcn_96t128_w96_d6_u2": (ESPCN, 128, 96, 6, 2, 96),
}


def render_size(net: str) -> int:
    """Return the input frame resolution required by a named architecture."""
    cls, in_size, width, depth, un, pre = ZOO[net]
    return pre or in_size


def build(net: str, out_size: int = 256) -> Corrector:
    if net not in ZOO:
        raise KeyError(f"unknown corrector architecture {net!r}; known: {sorted(ZOO)}")
    cls, in_size, width, depth, un, pre = ZOO[net]
    return cls(in_size=in_size, out_size=out_size, width=width, depth=depth, unshuffle=un, pre_size=pre)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# Compare both scene identity and the render configuration expected by the weights.
HEADER_KEYS = ("format", "cell", "render", "net", "visual_fidelity", "variation_fidelity", "baseline", "provenance")


def slug(cell: dict) -> str:
    """Return the environment class name identifying its checkpoint."""
    return str(cell["env_class"])


def path_for(cell_or_name) -> str:
    name = cell_or_name if isinstance(cell_or_name, str) else slug(cell_or_name)
    return os.path.join(CACHE_ROOT, f"{name}.pt")


def state_dict_digest(sd: dict) -> str:
    """Hash sorted parameter names, shapes, and float32 values to bind a header to weights."""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def header_of(ck: dict) -> dict:
    return {k: ck[k] for k in HEADER_KEYS if k in ck}


def available() -> dict[str, dict]:
    """List sealed-checkpoint sidecars without allocating or loading weights.

    Unreadable sidecars warn and are omitted. Loading subsequently verifies each
    sidecar against the authoritative header inside its checkpoint.
    """
    out: dict[str, dict] = {}
    if not os.path.isdir(CACHE_ROOT):
        return out
    for fn in sorted(os.listdir(CACHE_ROOT)):
        if not fn.endswith(".json"):
            continue
        name = fn[:-5]
        if not os.path.exists(os.path.join(CACHE_ROOT, f"{name}.pt")):
            continue
        try:
            with open(os.path.join(CACHE_ROOT, fn)) as f:
                hdr = json.load(f)
            if not isinstance(hdr, dict) or not isinstance(hdr.get("cell", {}), dict):
                raise ValueError("expected an object header with an object cell")
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.warn(f"Ignoring unreadable RenderNet sidecar {fn}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        if hdr.get("format") == FORMAT and hdr.get("cell", {}).get("env_class") == name:
            out[name] = hdr
    return out


def missing_message(env_class: str) -> str:
    return (
        f"no sealed render corrector for {env_class} under {CACHE_ROOT}. This repository ships "
        f"code, not weights, exactly as it ships no compiled RoboCasa kitchen and no MiniGrid "
        f"atlas; build one on this machine with\n\n    "
        + BUILD_CMD.format(env=env_class)
        + "\n\nIt trains on the cell's own states and its own official reference renderer, "
        "measures the corrected path, and seals only if the result beats the render it "
        "replaces. See stable_envs/render/rendernet.py."
    )


class CellMismatch(RuntimeError):
    """The checkpoint identity does not match the requested environment."""


def _describe_cell(c: dict) -> str:
    return ", ".join(f"{k}={c[k]!r}" for k in sorted(c))


def check_cell(want: dict, got: dict) -> None:
    """Require exact scene, camera, and model identity; no partial matching."""
    if want == got:
        return
    diff = sorted(set(want) | set(got))
    lines = [
        f"  {k}: checkpoint {got.get(k, '<absent>')!r} vs env {want.get(k, '<absent>')!r}"
        for k in diff
        if want.get(k) != got.get(k)
    ]
    raise CellMismatch(
        "this render corrector was fitted on a different cell and applying it here would be worse "
        "than not correcting at all -- measured at 2.6x worse than a plain resize on the wrong "
        "camera, and it fails by painting plausible texture that is not in the official frame, which "
        "the pixel MEAN does not catch. A corrector is per (scene, camera).\n"
        + "\n".join(lines)
        + f"\n  checkpoint cell: {_describe_cell(got)}"
        + f"\n  env cell:        {_describe_cell(want)}"
    )


class _Runner(_resources.Closeable):
    """Own one CUDA graph and its static input/output buffers at a fixed batch size.

    The parent LoadedCorrector owns the weights. Reset the graph before releasing
    buffers referenced by its private memory pool.
    """

    resource_kind = "rendernet.Runner"

    def __init__(self, module: Corrector, n: int, r: int, s: int, device, mode: str):
        self.n, self.mode = int(n), str(mode)
        dev = torch.device(device)
        self.sin = torch.empty((self.n, r, r, 3), dtype=torch.uint8, device=dev)
        self.sout = torch.empty((self.n, s, s, 3), dtype=torch.uint8, device=dev)
        fn = module
        if self.mode != "eager":
            # Compilation occurs on the first call; propagate errors without changing modes.
            fn = torch.compile(module, mode=self.mode, fullgraph=True, dynamic=False)

        def step():
            with torch.no_grad():
                self.sout.copy_(fn(self.sin))

        for _ in range(3):  # compile, autotune, and warm the allocator
            step()
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            step()
        self._opened()

    def _release(self) -> None:
        """Release the graph before its static buffers; tolerate runtimes without reset()."""
        g, self.graph = getattr(self, "graph", None), None
        if g is not None:
            reset = getattr(g, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:  # noqa: BLE001
                    pass
        self.sin = None
        self.sout = None

    def __call__(self, low: torch.Tensor) -> torch.Tensor:
        """Return a view of the first m output rows for (m <= n,r,r,3) input.

        Repeat the last input row into padding so graph replay cannot read pixels
        left by a prior call.
        """
        self._check_open("__call__")
        m = int(low.shape[0])
        self.sin[:m].copy_(low)
        if m < self.n:
            self.sin[m:].copy_(low[-1:].expand(self.n - m, *low.shape[1:]))
        self.graph.replay()
        return self.sout[:m]


class LoadedCorrector(_resources.Closeable):
    """A verified checkpoint, its render configuration, and a lazily built runner.

    Owns weights and the runner. When a cache shares this object across environments,
    the cache closes it; environments only drop their references. Input resolution,
    texture LOD, and culling settings must match the checkpoint's render configuration.
    """

    resource_kind = "rendernet.LoadedCorrector"

    def __init__(
        self,
        module: Corrector,
        header: dict,
        path: str,
        device="cuda",
        mode: str | None = None,
        net_batch: int | None = None,
    ):
        self.module, self.header, self.path = module, header, path
        self.device = torch.device(device)
        self.mode = COMPILE_MODE if mode is None else str(mode)
        self.net_batch = int(NET_BATCH if net_batch is None else net_batch)
        self._runner: _Runner | None = None
        self.cell = dict(header["cell"])
        r = header["render"]
        self.render_size = int(r["render_size"])
        self.out_size = int(r["out_size"])
        self.tex_lod = int(r["tex_lod"])
        self.backface_cull = bool(r["backface_cull"])
        self.net = str(header["net"])
        self.visual_fidelity = dict(header.get("visual_fidelity") or {})
        self.variation_fidelity = dict(header.get("variation_fidelity") or {})
        self.baseline = dict(header.get("baseline") or {})
        self.provenance = dict(header.get("provenance") or {})
        self._opened()

    def _release(self) -> None:
        """Close the runner and release the weights."""
        run, self._runner = self._runner, None
        if run is not None:
            run.close()
        self.module = None

    def __repr__(self) -> str:
        return (
            f"LoadedCorrector({self.cell.get('env_class')}, {self.net}, "
            f"{self.render_size}px tex_lod={self.tex_lod} -> {self.out_size}px, "
            f"{n_params(self.module) / 1e3:.0f}k params, {self.mode}, batch {self.net_batch})"
        )

    def runner(self) -> _Runner:
        """Build the compiled CUDA runner on first use."""
        self._check_open("runner")
        if self._runner is None:
            self._runner = _Runner(
                self.module, self.net_batch, self.render_size, self.out_size, self.device, self.mode
            )
        return self._runner

    def __call__(self, low: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, r, r, 3) uint8`` -> ``(B, S, S, 3) uint8``, optionally into ``out``."""
        if tuple(low.shape[1:]) != (self.render_size, self.render_size, 3):
            raise ValueError(
                f"{self!r} was fitted on {self.render_size}x{self.render_size} renders and got "
                f"{tuple(low.shape)}. The input resolution is part of what the weights encode; a "
                f"corrector handed a different one produces a plausible wrong picture rather than an "
                f"error."
            )
        b = int(low.shape[0])
        s = self.out_size
        if out is None:
            out = torch.empty((b, s, s, 3), dtype=torch.uint8, device=low.device)
        run = self.runner()
        for lo in range(0, b, self.net_batch):
            hi = min(lo + self.net_batch, b)
            out[lo:hi] = run(low[lo:hi])
        return out


def _read(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"{path} is not a readable sealed corrector ({type(e).__name__}: {e}). It is loaded with "
            f"weights_only=True on purpose: a checkpoint is data, and a cache directory that can "
            f"execute code on load is a worse problem than a missing corrector."
        ) from e


def load(
    cell: dict,
    device="cuda",
    dtype=torch.float16,
    mode: str | None = None,
    net_batch: int | None = None,
    path: str | None = None,
) -> LoadedCorrector:
    """Load a sealed corrector for exactly the requested environment.

    Validate format, scene identity, weight digest, and sidecar agreement before
    transferring the model to the requested device. Refuse mismatches.
    """
    p = path_for(cell) if path is None else str(path)
    if not os.path.exists(p):
        raise FileNotFoundError(missing_message(slug(cell)))
    ck = _read(p)
    if ck.get("format") != FORMAT:
        raise ValueError(
            f"{p} declares format {ck.get('format')!r}; this loader reads {FORMAT!r}. "
            f"Re-seal it with {BUILD_CMD.format(env=slug(cell))}"
        )
    check_cell(cell, dict(ck.get("cell") or {}))
    sd = ck["state_dict"]
    want = ck.get("state_dict_sha256")
    got = state_dict_digest(sd)
    if want != got:
        raise ValueError(
            f"{p}: the header was measured on different weights from the ones in the file "
            f"(sha256 {want} declared, {got} present). Every number in visual_fidelity describes a "
            f"specific checkpoint; a header that does not belong to its weights is a declaration "
            f"about something else."
        )
    side = p[:-3] + ".json"
    if os.path.exists(side):
        with open(side) as f:
            hdr = json.load(f)
        embedded = header_of(ck)
        if json.loads(json.dumps(embedded, sort_keys=True)) != {k: hdr.get(k) for k in embedded}:
            raise ValueError(
                f"{side} disagrees with the header inside {os.path.basename(p)}. The json is a "
                f"convenience copy and the checkpoint is authoritative; a divergence means one of "
                f"them was edited, and a declared floor is not something to edit."
            )
    m = build(ck["net"], out_size=int(ck["render"]["out_size"]))
    m.load_state_dict({k: v for k, v in sd.items()})
    m = m.to(device=device, dtype=dtype).eval().to(memory_format=torch.channels_last)
    for prm in m.parameters():
        prm.requires_grad_(False)
    return LoadedCorrector(m, header_of(ck), p, device=device, mode=mode, net_batch=net_batch)


def unsealed(
    module: Corrector,
    *,
    cell: dict,
    render: dict,
    net: str,
    device="cuda",
    dtype=torch.float16,
    mode: str | None = None,
    net_batch: int | None = None,
) -> LoadedCorrector:
    """Build an unsealed candidate for training and measurement.

    Environment registration uses load(), which requires a verified declaration.
    """
    m = module.to(device=device, dtype=dtype).eval().to(memory_format=torch.channels_last)
    for prm in m.parameters():
        prm.requires_grad_(False)
    hdr = {"format": FORMAT, "cell": dict(cell), "render": dict(render), "net": str(net)}
    return LoadedCorrector(m, hdr, "<unsealed>", device=device, mode=mode, net_batch=net_batch)


def seal(
    path: str,
    *,
    cell: dict,
    render: dict,
    net: str,
    state_dict: dict,
    visual_fidelity: dict,
    variation_fidelity: dict,
    baseline: dict,
    provenance: dict,
) -> str:
    """Write weights and matching checkpoint/JSON headers with the same identity and digest."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = {k: v.detach().to(torch.float32).cpu() for k, v in state_dict.items()}

    # Plain JSON types keep weights_only loading safe and sidecars comparable.
    # Record exotic metadata as strings, without retaining arbitrary Python objects.
    def plain(d):
        return json.loads(json.dumps(dict(d), default=str, sort_keys=True))

    ck: dict[str, Any] = {
        "format": FORMAT,
        "cell": plain(cell),
        "render": plain(render),
        "net": str(net),
        "visual_fidelity": plain(visual_fidelity),
        "variation_fidelity": plain(variation_fidelity),
        "baseline": plain(baseline),
        "provenance": plain(provenance),
        "state_dict": sd,
        "state_dict_sha256": state_dict_digest(sd),
    }
    torch.save(ck, path)
    with open(path[:-3] + ".json", "w") as f:
        json.dump(header_of(ck), f, indent=1, sort_keys=True)
    return path
