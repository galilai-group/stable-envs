"""Render correctors: ``(B, r, r, 3) uint8`` cheap render -> ``(B, 256, 256, 3)`` uint8.

The shape of the problem is DLSS's, not a super-resolution benchmark's. Three things follow from
that and they set every decision in this file.

**The target is the OFFICIAL renderer, not a high-resolution version of our own output.** So the
network is asked to correct two unrelated defects at once: the resolution the speed lever gave away,
and ``mujoco_warp``'s missing texgen for textures painted on BOX/CYLINDER primitives, which draws a
marble counter as one flat colour and is most of the shipped tier's error before any speed lever is
applied. The second defect has no resolution component at all -- a full-resolution render has it too.

**The budget is microseconds.** A RoboCasa room camera renders at 18,871 f/s at 112 px with texture
LOD 2, and its stored-frame target is 14,343 f/s, so the corrector gets 16.7 us per frame -- 17.1 ms
for a batch of 1024. Everything here is fp16, ``channels_last``, and meant to be replayed from a CUDA
graph; the uint8 unpack and repack are inside the network because a separate kernel over a
(1024, 256, 256, 3) tensor costs 0.2 ms on its own.

**Spatial work is the currency.** A 3x3 conv at 128 px moves 1024 x 128 x 128 x C x 2 bytes per layer
in and the same out; at C=32 that is 2.1 GB of traffic per layer, which is 0.5-0.7 ms of an H200's
bandwidth before a single multiply. So the wide variants front the network with a ``PixelUnshuffle``
that trades resolution for channels -- the arithmetic is identical, the traffic is 4x smaller, and
the receptive field per layer is 2x larger in output pixels.

Every model here ends with a **bilinear skip**: the output is the upsampled input plus a learned
residual. That is not a training trick, it is the honesty property. A corrector that has to move the
frame away from the cheap render in order to hallucinate has to pay for it in the residual, the
residual is a tensor that can be looked at, and at initialisation (final conv zeroed) the model is
EXACTLY the bilinear baseline -- so training can only be measured as a departure from a known point.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: ImageNet-free normalisation. The frames are already in [0,255]; 1/255 keeps the residual head's
#: outputs O(1) so a zero-initialised final conv starts exactly at the bilinear baseline.
_SCALE = 1.0 / 255.0


class Corrector(nn.Module):
    """Common surface: ``forward(uint8 NHWC) -> uint8 NHWC`` at ``out_size``.

    Subclasses implement ``residual(x)`` where ``x`` is float NCHW in [0,1] at the input resolution
    and the return is float NCHW in [0,1]-ish at ``out_size``, to be ADDED to the bilinear upsample.
    """

    out_size: int = 256
    #: Resolution the RENDER arrives at. May differ from ``in_size``, the resolution the trunk runs
    #: at, because the cheapest render on this frontier is 112 px and 112 does not divide 256. The
    #: resize 112 -> 128 costs 1-2 us and buys the whole sub-pixel path; the bilinear SKIP is taken
    #: from the 112 px original, so nothing is resampled twice on the anchor.
    pre_size: int = 0
    #: Whether a depth plane is concatenated to the trunk input. Set by ``build``.
    use_depth: bool = False

    @property
    def geom(self):
        """The ``GeomTable`` submodule, or None.

        A PROPERTY and not an attribute, and that is a bug fix rather than a style choice. A class
        attribute ``geom = None`` shadows ``nn.Module.__getattr__``, which is the only path that
        reaches ``_modules`` -- so ``m.geom = GeomTable(...)`` registered the submodule correctly and
        ``self.geom`` still read None. The network then built its first convolution for 13 input
        channels and was fed 4, which at least raised; had the shapes happened to agree it would have
        trained on zeros.
        """
        return self._modules.get("geom_table")

    def aux_dim(self) -> int:
        return (1 if self.use_depth else 0) + (self.geom.dim if self.geom is not None else 0)

    def _augment(self, xin: torch.Tensor, depth, ids) -> torch.Tensor:
        """Concatenate the auxiliary planes to the trunk input, at the trunk's resolution.

        **The id map is resampled NEAREST and nothing else.** A bilinear resize of an integer geom id
        produces ids that do not exist, which index arbitrary rows of the table -- a silent, plausible
        corruption exactly at the object boundaries where the aux channels are supposed to help.
        """
        planes = [xin]
        h = xin.shape[-1]
        if self.use_depth:
            if depth is None:
                raise ValueError("this corrector was built with a depth channel and got none")
            d = depth if depth.shape[-1] == h else F.interpolate(
                depth, size=(h, h), mode="bilinear", align_corners=False)
            planes.append(d.to(xin.dtype))
        if self.geom is not None:
            if ids is None:
                raise ValueError("this corrector was built with a geom table and got no id map")
            i = ids if ids.shape[-1] == h else F.interpolate(
                ids[:, None].float(), size=(h, h), mode="nearest")[:, 0]
            planes.append(self.geom(i).to(xin.dtype))
        return torch.cat(planes, 1).contiguous(memory_format=torch.channels_last)

    def forward(self, u8: torch.Tensor, depth=None, ids=None) -> torch.Tensor:
        x = u8.permute(0, 3, 1, 2).to(self.dtype).mul_(_SCALE)
        x = x.contiguous(memory_format=torch.channels_last)
        if depth is not None and depth.dtype == torch.uint8:
            depth = depth[:, None].to(self.dtype).mul_(_SCALE)
        y = self.forward_float(x, depth, ids)
        y = y.mul_(255.0).round_().clamp_(0, 255)
        return y.to(torch.uint8).permute(0, 2, 3, 1)

    def forward_float(self, x: torch.Tensor, depth=None, ids=None) -> torch.Tensor:
        """Training path: float NCHW in [0,1] at render res -> float NCHW at ``out_size``.

        Rounding and clamping are the only difference from ``forward``; keeping them out of the
        training graph is what makes the loss differentiable, and keeping them IN ``forward`` is what
        makes the evaluated tensor the one a consumer receives.

        **The bilinear skip is taken from the RGB frame alone**, whatever the aux channels say. The
        anchor has to stay the picture the renderer produced, so that a corrector given richer inputs
        is still measured as a departure from the same known point.
        """
        base = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear",
                             align_corners=False)
        if self.pre_size and x.shape[-1] != self.in_size:
            x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear",
                              align_corners=False)
        if self.aux_dim():
            x = self._augment(x, depth, ids)
        return base + self.residual(x)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _coords(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate two constant screen-position planes, cached per (shape, dtype, device).

        This is a DELIBERATE MEMORISATION HANDLE and it exists to be measured, not to be shipped by
        default. A fixed camera in a fixed scene puts the marble counter at the same pixels for every
        state, so a corrector told where it is can learn "paint marble here" and never look at the
        image. Whether that is what a plain convolutional corrector is ALREADY doing is the central
        question of this investigation, and the cheapest way to answer it is to build the version
        that certainly does, and compare: if coordinates buy a large in-distribution gain and a large
        out-of-distribution loss, then the mechanism is screen-space lookup and the plain model's
        gain is on the same axis.
        """
        n, _, h, w = x.shape
        key = (h, w, x.dtype, x.device)
        c = getattr(self, "_coord_cache", None)
        if c is None or self._coord_key != key:
            ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
            xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
            c = torch.cat([ys.expand(1, 1, h, w), xs.expand(1, 1, h, w)], 1)
            c = c.contiguous(memory_format=torch.channels_last)
            self._coord_cache, self._coord_key = c, key
        return torch.cat([x, c.expand(n, 2, h, w)], 1)


class GeomTable(nn.Module):
    """``(B, H, W) int geom id -> (B, F, H, W)``: a per-pixel row of a per-geom table.

    One module, two meanings, and the difference is the whole aux experiment (see ``aux.py``):

    * ``learned=True`` -- the table is fitted with the network. Maximum capacity and **structurally
      unable to transfer**, because geom 742 is a different object in a different kitchen.
    * ``learned=False`` -- the table is ``aux.geom_features`` of whatever model is loaded, held as a
      buffer and frozen. Every column is something MuJoCo knows and MJWarp failed to draw, so the
      table is DEFINED for a scene the corrector has never seen. Swapping in the new scene's table at
      evaluation time is what makes a cross-kitchen test meaningful rather than nonsense.

    The gather is an embedding lookup either way, which at 1024 x 112 x 112 x 9 half-precision is
    231 MB of traffic -- under 0.1 ms, and it is inside the timed region in every measurement here.
    """

    def __init__(self, table: torch.Tensor | None = None, n_rows: int = 0, dim: int = 0,
                 learned: bool = False):
        super().__init__()
        self.learned = learned
        if learned:
            self.emb = nn.Embedding(int(n_rows), int(dim))
            nn.init.normal_(self.emb.weight, std=0.05)
            with torch.no_grad():
                self.emb.weight[0].zero_()               # row 0 is background
            self.dim = int(dim)
        else:
            if table is None:
                raise ValueError("a frozen GeomTable needs its table")
            self.register_buffer("table", table.float())
            self.dim = int(table.shape[1])

    def set_table(self, table: torch.Tensor) -> None:
        """Point a FROZEN table at another scene's features. Refused on a learned table, because
        there is no correspondence between two scenes' embedding rows and pretending otherwise is
        how a cross-scene number gets fabricated."""
        if self.learned:
            raise RuntimeError("a learned per-geom embedding has no meaning in another scene; "
                               "train a descriptor table if you want to cross scenes")
        self.table = table.to(self.table.dtype).to(self.table.device)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        w = self.emb.weight if self.learned else self.table
        flat = F.embedding(ids.reshape(ids.shape[0], -1).long(), w.to(ids.device))
        out = flat.view(ids.shape[0], ids.shape[1], ids.shape[2], self.dim)
        return out.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)


def _conv(cin, cout, k=3):
    return nn.Conv2d(cin, cout, k, padding=k // 2)


def _act():
    """SiLU, and the reason is measured rather than stylistic.

    The first sweep used ``ReLU(inplace=True)`` and every configuration did the same thing: held-out
    mean |diff| fell 7.77 -> 4.26 over six epochs and then went back to **7.79, which is the bilinear
    baseline to three digits** -- the residual had collapsed to identically zero. That is what a
    trunk of dead ReLUs produces, and with a zero-initialised head the failure is invisible in the
    loss shape because the model has a perfectly good fallback to fall back TO. SiLU has no zero
    gradient region, so a unit that is pushed negative can come back; gradient clipping in
    ``train.py`` handles the spike that pushed it. Elementwise cost is not the constraint here -- the
    trunk is bandwidth bound either way, and the re-timing is in ``speed.json``.
    """
    return nn.SiLU(inplace=True)


class ESPCN(Corrector):
    """Convolutions at the INPUT resolution, one sub-pixel convolution at the end.

    The classical shape (Shi et al. 2016), which is the right baseline because every operation
    happens at the cheap resolution and the only thing that ever touches 256 px is the final
    ``PixelShuffle``. ``unshuffle`` optionally folds a further 2x of resolution into channels before
    the trunk, which is where all of the throughput headroom is.
    """

    def __init__(self, in_size: int = 128, out_size: int = 256, width: int = 32, depth: int = 4,
                 unshuffle: int = 1, in_ch: int = 3, pre_size: int = 0, coord: bool = False):
        super().__init__()
        if out_size % in_size:
            raise ValueError(f"out_size {out_size} must be a multiple of in_size {in_size}")
        self.out_size, self.in_size, self.unshuffle = out_size, in_size, unshuffle
        self.pre_size, self.coord = pre_size, coord
        up = (out_size // in_size) * unshuffle          # total PixelShuffle factor after the trunk
        c0 = in_ch * unshuffle * unshuffle + 2 * int(coord)
        layers = [_conv(c0, width), _act()]
        for _ in range(depth - 2):
            layers += [_conv(width, width), _act()]
        self.trunk = nn.Sequential(*layers)
        self.head = _conv(width, 3 * up * up)
        self.shuffle = nn.PixelShuffle(up)
        nn.init.zeros_(self.head.weight)                # start EXACTLY at the bilinear baseline
        nn.init.zeros_(self.head.bias)

    def residual(self, x):
        if self.unshuffle > 1:
            x = F.pixel_unshuffle(x, self.unshuffle)
        if self.coord:
            x = self._coords(x)
        return self.shuffle(self.head(self.trunk(x)))


class UNetTiny(Corrector):
    """One downsample, one upsample, a skip. Buys receptive field per unit of traffic.

    The texgen gap is a LARGE-SCALE defect -- a whole counter is one colour -- so a corrector needs
    to see far enough to know which surface it is standing on. ``ESPCN``'s receptive field is
    ``2*depth+1`` input pixels; this doubles that for one extra strided layer, and the coarse branch
    runs at a quarter of the traffic.
    """

    def __init__(self, in_size: int = 128, out_size: int = 256, width: int = 32, depth: int = 2,
                 unshuffle: int = 1, in_ch: int = 3, pre_size: int = 0, coord: bool = False):
        super().__init__()
        self.out_size, self.in_size, self.unshuffle = out_size, in_size, unshuffle
        self.pre_size, self.coord = pre_size, coord
        up = (out_size // in_size) * unshuffle
        c0 = in_ch * unshuffle * unshuffle + 2 * int(coord)
        w2 = width * 2
        self.stem = nn.Sequential(_conv(c0, width), _act())
        self.down = nn.Sequential(nn.Conv2d(width, w2, 3, stride=2, padding=1), _act(),
                                  *sum([[_conv(w2, w2), _act()] for _ in range(depth)], []))
        self.up = nn.Sequential(_conv(w2, width * 4), nn.PixelShuffle(2), _act())
        self.fuse = nn.Sequential(_conv(2 * width, width), _act())
        self.head = _conv(width, 3 * up * up)
        self.shuffle = nn.PixelShuffle(up)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x):
        if self.unshuffle > 1:
            x = F.pixel_unshuffle(x, self.unshuffle)
        if self.coord:
            x = self._coords(x)
        s = self.stem(x)
        d = self.up(self.down(s))
        return self.shuffle(self.head(self.fuse(torch.cat([s, d], 1))))


#: Named configurations, so a timing table and a training run cannot drift apart.
#: ``pre`` is the resolution the RENDER arrives at; 0 means "same as the trunk".
ZOO = {
    #  name                        class     trunk width depth unshuffle pre
    "espcn_128_w16_d3":     (ESPCN,    128, 16, 3, 1, 0),
    "espcn_128_w24_d4":     (ESPCN,    128, 24, 4, 1, 0),
    "espcn_128_w32_d4":     (ESPCN,    128, 32, 4, 1, 0),
    "espcn_128_w32_d6":     (ESPCN,    128, 32, 6, 1, 0),
    "espcn_128_w48_d4_u2":  (ESPCN,    128, 48, 4, 2, 0),
    "espcn_128_w64_d4_u2":  (ESPCN,    128, 64, 4, 2, 0),
    "espcn_128_w64_d6_u2":  (ESPCN,    128, 64, 6, 2, 0),
    "espcn_128_w96_d6_u2":  (ESPCN,    128, 96, 6, 2, 0),
    "espcn_128_w96_d8_u2":  (ESPCN,    128, 96, 8, 2, 0),
    "espcn_128_w128_d6_u2": (ESPCN,    128, 128, 6, 2, 0),
    "unet_128_w32_d2_u2":   (UNetTiny, 128, 32, 2, 2, 0),
    "unet_128_w48_d2_u2":   (UNetTiny, 128, 48, 2, 2, 0),
    "unet_128_w64_d3_u2":   (UNetTiny, 128, 64, 3, 2, 0),
    "unet_128_w96_d3_u2":   (UNetTiny, 128, 96, 3, 2, 0),
    # 112 px in -- the cheapest render on the whole frontier, resized to a trunk that divides 256.
    "espcn_112t128_w48_d4_u2":  (ESPCN,    128, 48, 4, 2, 112),
    "espcn_112t128_w64_d4_u2":  (ESPCN,    128, 64, 4, 2, 112),
    "espcn_112t128_w64_d6_u2":  (ESPCN,    128, 64, 6, 2, 112),
    "espcn_112t128_w96_d6_u2":  (ESPCN,    128, 96, 6, 2, 112),
    "espcn_112t128_w96_d4_u2":  (ESPCN,    128, 96, 4, 2, 112),
    "espcn_112t128_w128_d4_u2": (ESPCN,    128, 128, 4, 2, 112),
    "unet_112t128_w48_d2_u2":   (UNetTiny, 128, 48, 2, 2, 112),
    "unet_112t128_w64_d3_u2":   (UNetTiny, 128, 64, 3, 2, 112),
    # 96 px in: one step further down the ladder, for the cells that need the most speed.
    "espcn_96t128_w64_d6_u2":   (ESPCN,    128, 64, 6, 2, 96),
    "espcn_96t128_w96_d6_u2":   (ESPCN,    128, 96, 6, 2, 96),
    # 256 px in, 256 px out: the "fix only the texgen gap, change no resolution" trade.
    "espcn_256_w32_d4_u2":  (ESPCN,    256, 32, 4, 2, 0),
    "espcn_256_w64_d4_u4":  (ESPCN,    256, 64, 4, 4, 0),
    "espcn_256_w96_d6_u4":  (ESPCN,    256, 96, 6, 4, 0),
    "unet_256_w64_d3_u4":   (UNetTiny, 256, 64, 3, 4, 0),
    # The resolution axis of the frontier, with the TRUNK held fixed at 128/u2 so that the only
    # thing that changes between these three is how much the renderer was asked to resolve.
    "unet_96t128_w64_d3_u2":  (UNetTiny, 128, 64, 3, 2, 96),
    "unet_144t128_w64_d3_u2": (UNetTiny, 128, 64, 3, 2, 144),
    "unet_112t128_w96_d3_u2": (UNetTiny, 128, 96, 3, 2, 112),
}

#: Configurations that additionally receive two constant screen-position planes. Separated from the
#: zoo above by NAME rather than by a flag so a checkpoint can never be loaded into the wrong one.
for _n in ("unet_112t128_w48_d2_u2", "unet_112t128_w64_d3_u2", "espcn_112t128_w96_d6_u2"):
    ZOO[_n + "_coord"] = ZOO[_n]


def has_coord(name: str) -> bool:
    return name.endswith("_coord")


def render_size(name: str) -> int:
    """The resolution the RENDERER must produce for this corrector."""
    cls, in_size, width, depth, un, pre = ZOO[name]
    return pre or in_size


def aux_dim_of(aux: dict | None) -> int:
    """Extra input channels a spec asks for, without building anything."""
    if not aux:
        return 0
    n = 1 if aux.get("depth") else 0
    g = aux.get("geom")
    if g == "learned":
        n += int(aux["dim"])
    elif g == "descriptor":
        n += int(np.asarray(aux["table"]).shape[1])
    return n


def build(name: str, out_size: int = 256, in_ch: int = 3, aux: dict | None = None) -> Corrector:
    """``aux`` is ``{"depth": bool, "geom": "learned"|"descriptor"|None, "dim": k, "table": (R,F)}``.

    It is stored in the checkpoint and rebuilt from it, so a corrector cannot be loaded into a shape
    that expects different inputs -- which on this problem would not raise, it would run and produce
    a plausible wrong picture.
    """
    cls, in_size, width, depth, un, pre = ZOO[name]
    geom = None
    if aux and aux.get("geom") == "learned":
        geom = GeomTable(n_rows=int(aux["n_rows"]), dim=int(aux["dim"]), learned=True)
    elif aux and aux.get("geom") == "descriptor":
        geom = GeomTable(table=torch.as_tensor(np.asarray(aux["table"], np.float32)))
    m = cls(in_size=in_size, out_size=out_size, width=width, depth=depth, unshuffle=un,
            in_ch=in_ch + aux_dim_of(aux), pre_size=pre, coord=has_coord(name))
    if geom is not None:
        m.add_module("geom_table", geom)
    m.use_depth = bool(aux and aux.get("depth"))
    return m


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def gflops_per_frame(name: str, out_size: int = 256, in_ch: int = 3,
                     aux: dict | None = None) -> float:
    """Multiply-accumulates x2 for one frame, counted from the ACTUAL output shapes.

    Taken from a forward hook rather than from the config, because the arithmetic is the whole
    question and a hand-derived formula for the UNet's up path is exactly the kind of thing that is
    wrong by 2x without anything complaining.
    """
    m = build(name, out_size, in_ch, aux)
    total = [0]

    def hook(mod, inp, out):
        total[0] += (mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
                     * out.numel() / out.shape[0])

    hs = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    r = render_size(name)
    kw = {}
    if m.use_depth:
        kw["depth"] = torch.zeros(1, r, r, dtype=torch.uint8)
    if m.geom is not None:
        kw["ids"] = torch.zeros(1, r, r, dtype=torch.int32)
    with torch.no_grad():
        m(torch.zeros(1, r, r, in_ch, dtype=torch.uint8), **kw)
    for h in hs:
        h.remove()
    return 2 * total[0] / 1e9
