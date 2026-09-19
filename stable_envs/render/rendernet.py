"""A LEARNED render corrector for the visual tier: a cheap frame in, the official frame approximated.

WHAT THIS IS, IN ONE SENTENCE
-----------------------------

A small convolutional network (441 k parameters, ~17 us a frame) that maps a **cheap MJWarp render at
112 px** to the **256 px frame MuJoCo's own renderer would have drawn**, correcting two unrelated
defects at once: the resolution the speed lever gave away, and ``mujoco_warp``'s missing texture
coordinates on BOX/CYLINDER primitives, which draws a marble counter as one flat colour
(``render/mjwarp.py``, "the one gap this module cannot close"). Measured on held-out states, end to
end including inference, ``RoboCasaCounterToSink`` goes from 5,754 f/s at mean |diff| 6.72 to
**14,829 f/s at 2.65** — faster AND closer, on every declared quantity at once.

**IT CHANGES THE DECODER, NOT THE CONTRACT.** The corrector is a fixed checkpoint applied to a render
that is already a pure function of ``(state, style)``, so ``render`` is still ``f(get_state())``,
still batch-invariant, still deterministic, still reproducible bit-for-bit by anyone holding the
weights. What changes is what the tier's frame IS: no longer *MuJoCo's renderer approximated*, but **a
learned approximation of MuJoCo's renderer, with declared error**. The detail it restores on textured
surfaces is SYNTHESISED — right statistics, wrong pixels — and that difference is not visible in the
picture, only in the declaration. An env whose frames come from this path says so in its docstring, in
``describe()``, and in ``visual_fidelity["decoder"]``; see ``docs/concepts/fidelity-tiers.md``.

A CORRECTOR IS PER (SCENE, CAMERA), AND A MISMATCHED ONE IS WORSE THAN NOTHING
------------------------------------------------------------------------------

This is the property the whole module is built around, because it fails SILENTLY. Measured, zero-shot,
with the ``RoboCasaCounterToSink`` corrector applied to cells it was not trained for:

============================================  ==========  ==========  ============
target                                         corrector    bilinear   verdict
============================================  ==========  ==========  ============
same kitchen, camera moved 40 cm                    5.94        6.14   no better than a resize
same kitchen, wrist camera                         11.35        4.39   2.6x WORSE than doing nothing
a different kitchen                                33.19       32.85   mean ties, SSIM 0.763 vs 0.792
another HumanoidBench task                         11.37        9.63   worse than doing nothing
============================================  ==========  ==========  ============

On a different kitchen it paints marble-like speckle onto plain cabinet fronts that carry no texture
in the official frame. **That failure is caught by SSIM and within-8 and NOT by the mean** — a
corrector can lower a frame's average error while making more of its pixels wrong. So the answer here
is not a warning in a docstring, it is a lock: every checkpoint is KEYED to the cell it was fitted on
(scene, camera, and the hash of the compiled MJCF), the key is verified against the env at load, and a
mismatch RAISES. There is no fallback to a neighbour's weights and no "close enough" comparison,
because a loud refusal is recoverable and a quiet wrong picture is not.

WHERE THE WEIGHTS LIVE, AND HOW TO GET THEM
--------------------------------------------

**This repository is code only.** Weights are cached exactly as the RoboCasa scene cache is
(``~/.cache/stable-envs/robocasa/``) and MiniGrid's tile atlas is: built once, on the machine that
needs them, under ``~/.cache/stable-envs/rendernet/`` (override with
``STABLE_ENVS_RENDERNET_CACHE``), one sealed checkpoint per cell::

    python examples/rendernet/train_corrector.py RoboCasaCounterToSink

which generates its own training pairs from the env and its official reference renderer, fits the
network, MEASURES the corrected path, and seals the result only if it beats the render it replaces.
With no checkpoint for a cell, nothing is registered for it and the plain visual-tier class is
unaffected — the absence of weights removes an env, it never changes one.

WHAT A SEALED CHECKPOINT CARRIES
---------------------------------

The declaration travels WITH the weights, and that is deliberate: ``visual_fidelity`` is a property of
the (renderer, checkpoint) pair, not of the class, so a checkpoint that arrives without its measured
numbers cannot be registered at all. ``examples/rendernet/train_corrector.py`` is the only thing that
writes one, it measures before it seals, and it refuses to seal a corrector that does not beat the
plain render on mean |diff|, within-8 AND SSIM, at the default appearance and under the env's own
declared variation space. A corrector that is worse than doing nothing cannot be produced by the
supported path, which is the mechanical form of the table above.

The sidecar ``.json`` is a convenience copy for reading; the authoritative header is inside the
``.pt``, and ``load`` refuses if the two disagree. Editing a declared floor in the json to make
something pass therefore does not work, which is the point.
"""

from __future__ import annotations

import hashlib
import json
import os
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

#: THE ONLY BATCH THE NETWORK IS EVER GIVEN. Not a chunk size — a constant, and the reason is
#: correctness before it is speed.
#:
#: A convolution's numerics depend on the kernel that runs it, and the kernel depends on the SHAPE:
#: cuDNN picks an algorithm per shape, and Inductor's autotuner benchmarks per shape and picks the
#: fastest tile configuration. Two configurations sum a 3x3xC window in different orders, fp16 carries
#: 11 bits of mantissa, and the network's last act is to round to 1/255 — so a network handed 7 frames
#: and the same network handed 8 are not guaranteed to agree on frame 0 to the byte. Every batch is
#: therefore padded up to exactly this many rows and the padding is discarded, which makes ONE compiled
#: kernel, ONE autotune decision and ONE CUDA graph serve every call. ``test_batch_invariance``,
#: ``test_frame_i_is_independent_of_batch_size`` and the fresh-process digest then hold by construction
#: rather than by luck, which is the same argument ``humanoidbench.render`` makes for pinning MJWarp's
#: world count and pays the same price: a batch of 1 costs a batch of this many.
NET_BATCH = int(os.environ.get("STABLE_ENVS_RENDERNET_BATCH", 128))

#: How the network is compiled. ``"max-autotune-no-cudagraphs"`` is the shipped setting and the
#: declared numbers were measured through it.
#:
#: **Inductor is allowed here and is not allowed anywhere else in this repository**, which is worth
#: stating because the rule it looks like it breaks is a real one. ``docs/concepts/bit-exact-triton.md``
#: keeps the rasterizer and the ray-casters off Inductor because it fuses floating-point operations,
#: and those are PARITY-tier paths where a 1-ulp change disqualifies the claim. This path is visual
#: tier and its output is a network's approximation of another renderer: fp fusion here is not a defect,
#: it is arithmetic, and the fidelity it produces is re-measured rather than assumed.
#:
#: CUDA graphs are captured explicitly rather than through Inductor's own (hence
#: ``-no-cudagraphs``), for the reason ``mjwarp.gl_rounding`` records about capture scope: a graph is
#: the thing that actually runs, so it has to be captured where it can be seen.
#:
#: ``"eager"`` is the escape hatch for a machine without a working Triton, and it is a DIFFERENT
#: configuration: same weights, slightly different arithmetic, so the declared numbers are only
#: approximately its numbers and two processes must agree on this setting to reproduce each other's
#: frames. It is an environment variable and not a per-call argument so that it cannot vary within a
#: process or between the two halves of a comparison.
COMPILE_MODE = os.environ.get("STABLE_ENVS_RENDERNET_COMPILE", "max-autotune-no-cudagraphs")

#: The frames are in [0,255]; 1/255 keeps a zero-initialised residual head starting exactly at the
#: bilinear baseline, which is what makes every trained number a departure from a measured point.
_SCALE = 1.0 / 255.0


# --------------------------------------------------------------------------------- the networks


class Corrector(nn.Module):
    """``(B, r, r, 3) uint8`` -> ``(B, out, out, 3) uint8``. A bilinear upsample plus a residual.

    Every model here ends with a bilinear skip and a ZERO-INITIALISED head, so at initialisation the
    module is EXACTLY the bilinear baseline. That is not a training trick, it is the honesty property:
    a corrector that moves the frame away from the render in order to invent detail has to pay for it
    in a residual that can be looked at, and every measured number is a departure from a point that
    was already measured.

    The uint8 unpack and repack are inside the module because a separate kernel over a
    ``(256, 256, 256, 3)`` tensor costs more than the convolutions do.
    """

    out_size: int = 256
    #: Resolution the RENDER arrives at, which may differ from ``in_size``, the resolution the trunk
    #: runs at: the cheapest render on the measured frontier is 112 px and 112 does not divide 256.
    #: The 112 -> 128 resize costs 1-2 us and buys the whole sub-pixel path, and the bilinear SKIP is
    #: taken from the 112 px original, so nothing is resampled twice on the anchor.
    pre_size: int = 0

    def forward_float(self, x: torch.Tensor) -> torch.Tensor:
        """TRAINING path: float NCHW in [0,1] at the render resolution -> float NCHW at ``out_size``.

        Rounding and clamping are the only difference from ``forward``. Keeping them out of here is
        what makes the loss differentiable; keeping them IN ``forward`` is what makes the tensor the
        metric is computed on the tensor a consumer receives.
        """
        base = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        if self.pre_size and x.shape[-1] != self.in_size:
            x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        return base + self.residual(x)

    def forward(self, u8: torch.Tensor) -> torch.Tensor:
        x = u8.permute(0, 3, 1, 2).to(self.dtype).mul_(_SCALE)
        x = x.contiguous(memory_format=torch.channels_last)
        # STAY IN THE MODEL'S DTYPE. Writing ``(base + res).float()`` before quantising materialises
        # an fp32 (B, 3, 256, 256) tensor -- 805 MB at B=1024 -- for a step fp16 performs exactly
        # (every integer to 2048 is representable). Measured at 3.3 us a frame, 18 % of the budget,
        # and invisible until the end-to-end timing disagreed with the standalone one by that much.
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
    """SiLU, and the reason is measured rather than stylistic.

    The first sweep used ``ReLU``, and every configuration did the same thing: held-out mean |diff|
    fell 7.77 -> 4.26 over six epochs and then went back to 7.79, which is the bilinear baseline to
    three digits. The trunk had died, and with a zero-initialised head the failure is invisible in the
    loss shape because the model has a perfectly good fallback to fall back TO. SiLU has no zero-
    gradient region, so a unit pushed negative can come back.
    """
    return nn.SiLU(inplace=True)


class ESPCN(Corrector):
    """Convolutions at the INPUT resolution, one sub-pixel convolution at the end (Shi et al. 2016).

    The right baseline, because every operation happens at the cheap resolution and the only thing
    that touches 256 px is the final ``PixelShuffle``. ``unshuffle`` optionally folds a further 2x of
    resolution into channels before the trunk, which is where the throughput headroom is: at 128 px
    and C=32 one 3x3 conv moves 2.1 GB per layer at batch 1024, and trading resolution for channels
    leaves the arithmetic identical while cutting the traffic 4x.
    """

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
        if out_size % in_size:
            raise ValueError(f"out_size {out_size} must be a multiple of in_size {in_size}")
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
    """One downsample, one upsample, a skip. Receptive field per unit of memory traffic.

    The texture gap is a LARGE-SCALE defect — a whole counter is one colour — so the network has to
    see far enough to know which surface it is standing on. ``ESPCN``'s receptive field is
    ``2*depth+1`` input pixels; this doubles it for one strided layer, with the coarse branch running
    at a quarter of the traffic.
    """

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


#: ``name -> (class, trunk resolution, width, depth, unshuffle, render resolution)``. Named so a
#: timing table and a checkpoint cannot drift apart: the architecture is stored in the checkpoint by
#: NAME and rebuilt from this table, so a file can never be loaded into a differently shaped network.
#: Only the configurations that were measured end to end are here — the frontier study is
#: ``scratch/altrender/rendernet/README.md``, and a name it swept but did not recommend is not a name
#: this loader accepts.
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
    """The resolution the RENDERER must produce for this architecture."""
    cls, in_size, width, depth, un, pre = ZOO[net]
    return pre or in_size


def build(net: str, out_size: int = 256) -> Corrector:
    if net not in ZOO:
        raise KeyError(f"unknown corrector architecture {net!r}; known: {sorted(ZOO)}")
    cls, in_size, width, depth, un, pre = ZOO[net]
    return cls(in_size=in_size, out_size=out_size, width=width, depth=depth, unshuffle=un, pre_size=pre)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------------------- sealed checkpoints

#: Fields of the header that a loader compares, in the order they are reported when one disagrees.
#: ``cell`` is the identity lock; ``render`` is how the cheap frame must be produced, and getting it
#: wrong feeds the network an input it was never fitted on, which is the same failure as a mismatched
#: cell with none of its visibility.
HEADER_KEYS = ("format", "cell", "render", "net", "visual_fidelity", "variation_fidelity", "baseline", "provenance")


def slug(cell: dict) -> str:
    """The filename a cell's checkpoint is looked up under. A cell IS an env class in this repo."""
    return str(cell["env_class"])


def path_for(cell_or_name) -> str:
    name = cell_or_name if isinstance(cell_or_name, str) else slug(cell_or_name)
    return os.path.join(CACHE_ROOT, f"{name}.pt")


def state_dict_digest(sd: dict) -> str:
    """sha256 over the parameter tensors, in sorted key order. What ties a header to its weights.

    The point of the digest is not tamper-proofing, it is that the sidecar json and the checkpoint
    cannot drift: a declaration is only meaningful for the weights it was measured on, and this is
    what lets ``load`` say the two belong together.
    """
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
    """``env class name -> the sidecar header`` for every sealed corrector on this machine.

    Reads the ``.json`` sidecars only, so listing does not deserialise weights: this runs at env-module
    import time, once per process, and a directory of checkpoints must not cost a GPU allocation to
    enumerate. The header is not TRUSTED here — ``load`` re-reads it out of the ``.pt`` and refuses a
    disagreement — it is only used to decide which classes to build.
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
        except Exception:  # noqa: BLE001
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
    """A corrector was asked to correct a cell it was not fitted on. Always fatal, never a fallback.

    Separate from ``ValueError`` so a caller can catch exactly this and nothing else — and so that a
    ``except Exception`` somewhere upstream that swallows it is visible as the mistake it is. The
    measured cost of the fallback this refuses is in the module docstring: on the wrong camera the
    corrector is 2.6x worse than not correcting at all, and it looks fine.
    """


def _describe_cell(c: dict) -> str:
    return ", ".join(f"{k}={c[k]!r}" for k in sorted(c))


def check_cell(want: dict, got: dict) -> None:
    """The identity lock. Every field of the checkpoint's cell must equal the env's, exactly.

    Exact equality over the WHOLE dict rather than a subset, and no version tolerance, because every
    field in it is something that changes the picture: the kitchen decides which room is drawn, the
    camera decides where from, and the MJCF hash decides whether it is even the same compiled scene —
    a rebuilt kitchen can move geometry that the fitted weights have baked assumptions about, and the
    live-qpos cache already re-measures on that hash for the same reason.
    """
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
    """One compiled, graph-captured invocation of a corrector at ONE fixed shape.

    Three layers, each removing a different cost, and the order matters:

    * **fp16 + channels_last.** The convolutions are the arithmetic; anything else is traffic.
    * **``torch.compile``.** Most of this network is not convolution. The bilinear anchor, the add,
      the ``*255``, the round, the clamp, the cast to uint8 and the NCHW -> NHWC permute are seven
      passes over a full-resolution tensor, and Inductor lowers ``upsample_bilinear2d`` to a pointwise
      gather so it can fuse ALL of them, including the layout change, into the epilogue of the final
      ``PixelShuffle``. That is where the win is: the arithmetic was never the 16.6 us.
    * **An explicit CUDA graph.** The remaining launches become one.

    The static input and output buffers are what a graph requires — a replay reads and writes the
    same addresses every time — and they are also what makes the caller's copies contiguous memcpys
    instead of strided gathers.

    **Ownership.** The graph, the two static buffers and the compiled callable are exclusive to this
    runner; the ``Corrector`` module it was built from belongs to the ``LoadedCorrector`` and is not
    released here. ``close()`` resets the graph before dropping the buffers, because a captured graph
    holds a private memory pool that references those exact addresses — freeing the tensors while the
    graph still exists returns memory the graph would replay into.
    """

    resource_kind = "rendernet.Runner"

    def __init__(self, module: Corrector, n: int, r: int, s: int, device, mode: str):
        self.n, self.mode = int(n), str(mode)
        dev = torch.device(device)
        self.sin = torch.empty((self.n, r, r, 3), dtype=torch.uint8, device=dev)
        self.sout = torch.empty((self.n, s, s, 3), dtype=torch.uint8, device=dev)
        fn = module
        if self.mode != "eager":
            try:
                fn = torch.compile(module, mode=self.mode, fullgraph=True, dynamic=False)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"could not compile the render corrector with mode={self.mode!r} "
                    f"({type(e).__name__}: {e}). This is not silently downgraded to eager, because "
                    f"eager is a DIFFERENT configuration -- the same weights with slightly different "
                    f"arithmetic -- and two processes that disagree about it do not reproduce each "
                    f"other's frames. Set STABLE_ENVS_RENDERNET_COMPILE=eager to choose it "
                    f"deliberately, everywhere, including in whatever wrote your dataset."
                ) from e

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
        """Reset the graph, then drop it and the static buffers. Idempotent.

        ``CUDAGraph.reset()`` releases the graph's private memory pool; it is reached through
        ``getattr`` because it is not part of the API this repository pins a version of, and a torch
        without it must still close cleanly — dropping the last reference frees the same pool, just at
        the collector's convenience rather than here.
        """
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
        """``(m <= n, r, r, 3)`` in, a VIEW of the static output's first ``m`` rows out.

        The tail of the static input is filled with a repeat of the last row rather than left as
        whatever the pool held, so that a replay is a function of the requested frames alone. Worlds
        are independent in this network — it is a convolution over each frame separately — but a
        buffer that carries the previous call's pixels makes "the same states twice" depend on what
        ran in between, and that is exactly the property ``check_determinism`` exists to hold.
        """
        self._check_open("__call__")
        m = int(low.shape[0])
        self.sin[:m].copy_(low)
        if m < self.n:
            self.sin[m:].copy_(low[-1:].expand(self.n - m, *low.shape[1:]))
        self.graph.replay()
        return self.sout[:m]


class LoadedCorrector(_resources.Closeable):
    """A sealed checkpoint, verified against one cell, ready to be applied in a render path.

    Holds the network AND the render configuration it was fitted on, because those are one object: a
    corrector trained on a 112 px render with texture LOD 2 that is handed a 256 px render, or a
    render with full-resolution textures, is being fed an input outside its training distribution and
    will produce a confident wrong picture with no error. ``render`` on the env reads ``render_size``,
    ``tex_lod`` and ``backface_cull`` from HERE and nowhere else.

    **Ownership.** It owns the network's weights and, once something has actually rendered through
    it, the ``_Runner`` those weights were compiled and captured into. ``close()`` closes the runner
    and drops the module. It owns no context of any kind. A corrector is normally held in a
    process-scoped cache keyed by env class (``envs/robocasa._CORRECTOR_CACHE``) and shared by every
    env of that cell, so THE CACHE closes it, not an env: an env's ``close`` drops its reference and
    nothing more. See ``_resources`` for why that distinction is the whole design.
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
        """Close the compiled runner (if one was ever built) and drop the weights. Idempotent."""
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
        """The compiled, captured invocation. Built on FIRST USE, not at load.

        Compilation and autotuning take tens of seconds and allocate; doing that inside ``__init__``
        would make constructing an env — which the conformance suite does per module, and which a
        listing does for nothing — pay for a render nobody asked for.
        """
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
    """The sealed corrector for ``cell``, verified against it. Raises rather than approximating.

    Four things are checked and each of them has produced a wrong picture somewhere: the format string
    (an older layout read by a newer loader), the sidecar against the embedded header (a declaration
    edited without the weights), the weight digest against the header (a header edited without the
    weights), and the CELL (the whole point — see the module docstring).
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
    """A corrector that has NOT been sealed yet, for the one caller that cannot have been: the
    training script, which has to run the corrected path in order to measure the declaration that a
    sealed checkpoint is required to carry.

    Deliberately not reachable from an env — ``corrector()`` goes through ``load`` and its four checks
    — so that "measure a candidate" and "use a corrector" stay different verbs. A candidate has no
    declaration, and an env with no declaration cannot be registered.
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
    """Write a checkpoint AND its sidecar. The only supported way to produce one.

    Kept in the library rather than in the training script so that the file layout has exactly one
    writer and one reader, in one place, and so that ``load``'s four checks are checks against
    something this function guarantees rather than against a convention.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = {k: v.detach().to(torch.float32).cpu() for k, v in state_dict.items()}

    # EVERY HEADER FIELD GOES THROUGH JSON ON THE WAY IN. Two things follow and both are load-bearing.
    # ``load`` reads with ``weights_only=True`` — a cache directory that can execute code on load is a
    # worse problem than a missing corrector — and that refuses any type outside a small allowlist, so
    # one stray ``torch.__version__`` (a str SUBCLASS) in the provenance makes the whole checkpoint
    # unreadable by its own loader. And the sidecar has to be byte-comparable with what is embedded,
    # which it can only be if the embedded copy is already plain JSON types. ``default=str`` means an
    # exotic value is recorded as text rather than silently dropped.
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
