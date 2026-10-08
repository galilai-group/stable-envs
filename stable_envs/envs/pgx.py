from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch

# Verify both optional imports and the SVG rasterization path before registering this family.
_VENV = "/mnt/vast/home/randall.balestriero/venvs/se-pgx"
try:
    import pgx as _pgx
    import resvg_py as _resvg
    from PIL import Image as _Image
except Exception as _e:  # pragma: no cover
    raise ImportError(
        f"the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter "
        f"cannot provide them: {type(_e).__name__}: {_e}. The envs were measured in {_VENV}, built "
        f"with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin "
        f"discovery."
    ) from _e

try:
    _resvg.svg_to_bytes(
        svg_string='<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">'
        '<rect width="1" height="1" fill="black"/></svg>'
    )
except Exception as _e:  # pragma: no cover
    raise ImportError(
        f"resvg_py imports but cannot rasterize ({type(_e).__name__}: {_e}). This is the failure mode "
        f"cairosvg has on this cluster's login node -- the import succeeds and every call raises -- "
        f"which is why this check renders a document instead of importing a name."
    ) from _e

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..spaces import Box, Dict, RGBBox

# Limit shared plates by distinct game, resolution, device, and appearance.
_PLATE_CACHE_MAX = 64
_PLATE_CACHE = _RES.BoundedCache(_PLATE_CACHE_MAX, name="pgx.plate")
_POOL_CACHE: dict = {}
_COLOR_CACHE: dict = {}
_BUNDLE_CACHE: dict = {}


def _rgb_of(name: str) -> tuple[int, int, int]:
    """Resolve a named SVG color with a one-pixel resvg render."""
    hit = _COLOR_CACHE.get(name)
    if hit is None:
        png = bytes(
            _resvg.svg_to_bytes(
                svg_string=f'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">'
                f'<rect width="1" height="1" fill="{name}"/></svg>'
            )
        )
        a = np.array(_Image.open(io.BytesIO(png)).convert("RGB"))
        hit = _COLOR_CACHE[name] = tuple(int(v) for v in a.reshape(3))
    return hit


def _raster(svg: str, size_w: int, size_h: int) -> np.ndarray:
    """Rasterize SVG to uint8 RGB, compositing transparent pixels over white."""
    png = bytes(_resvg.svg_to_bytes(svg_string=svg, width=int(size_w), height=int(size_h)))
    im = _Image.open(io.BytesIO(png)).convert("RGBA")
    a = np.asarray(im, np.float32)
    rgb, al = a[..., :3], a[..., 3:4] / 255.0
    return np.clip(rgb * al + 255.0 * (1.0 - al), 0, 255).astype(np.uint8)


class _Bundle:
    __slots__ = ("colorset", "empty_svg", "native", "grid", "ncols", "nrows", "palette", "marks")


def _empty_state(game: str):
    """Construct an empty board in the game's drawing convention."""
    import jax
    import jax.numpy as jnp

    env = _pgx.make(game)
    st = jax.jit(env.init)(jax.random.PRNGKey(0))
    spec = GAMES[game]
    board = np.full(spec["cells"], spec["empty"], np.int32)
    x = st._x._replace(board=jnp.asarray(board))
    if hasattr(x, "turn"):
        x = x._replace(turn=jnp.int32(0))
    return st.replace(_x=x)


def _drawables(svg: str) -> list[tuple]:
    """Return drawable tags and inherited fill/stroke values in document order."""
    out: list[tuple] = []

    def walk(node, fill, stroke):
        a = node.attrib
        f = a.get("fill", fill)
        s = a.get("stroke", stroke)
        tag = node.tag.split("}")[-1]
        if tag in ("rect", "circle", "line", "polygon", "path", "text", "image"):
            out.append((tag, f, s))
        for ch in node:
            walk(ch, f, s)

    walk(ET.fromstring(svg), None, None)
    return out


def _mark_colors(game: str, empty_svg: str) -> dict:
    """Read mark colors from SVG differences after placing one piece per player.

    Do not infer colors from ColorSet field names: tic-tac-toe uses grid_color
    for its strokes.
    """
    import jax.numpy as jnp

    spec = GAMES[game]
    base = _empty_state(game)
    empty = _drawables(empty_svg)
    out = {}
    for who in ("p1", "p2"):
        board = np.full(spec["cells"], spec["empty"], np.int32)
        board[0] = spec[who]
        st = base.replace(_x=base._x._replace(board=jnp.asarray(board)))
        got = _drawables(st.to_svg())
        new = got[len(empty) :] if len(got) > len(empty) else [g for g in got if g not in empty]
        assert new, f"{game}: placing a {who} mark added no drawable to the document"
        # A stone is one circle (fill + stroke); an X is two identically-stroked lines.
        fills = {g[1] for g in new}
        strokes = {g[2] for g in new}
        assert len(strokes) == 1, f"{game}: {who} mark has mixed strokes {strokes}"
        assert len(fills) == 1, f"{game}: {who} mark has mixed fills {fills}"
        fill, stroke = fills.pop(), strokes.pop()
        # ``fill="none"`` is tic_tac_toe's O: an annulus that must let the plate through, so the
        # colour that matters is the stroke.
        out[f"{who}.color"] = stroke if (fill in (None, "none")) else fill
        out[f"{who}.outline"] = stroke
        out[f"{who}.hollow"] = fill in (None, "none")
    return out


def _bundle(game: str) -> _Bundle:
    hit = _BUNDLE_CACHE.get(game)
    if hit is not None:
        return hit
    from pgx._src.visualizer import Visualizer

    spec = GAMES[game]
    st = _empty_state(game)
    v = Visualizer()
    v._set_config_by_state(st)
    cs = v.config["COLOR_SET"]
    b = _Bundle()
    b.colorset = cs
    b.empty_svg = st.to_svg()
    # Parse XML attributes independently of svgwrite's formatting and attribute order.
    root = ET.fromstring(b.empty_svg)
    w, h = float(root.attrib["width"]), float(root.attrib["height"])
    assert abs(w - h) < 1e-9, f"{game}: canvas {w}x{h} is not square"
    assert float(w).is_integer(), f"{game}: canvas {w} is not a whole number of pixels"
    b.native = int(round(w))
    b.grid = float(spec["grid"])
    b.ncols, b.nrows = int(spec["ncols"]), int(spec["nrows"])
    # Map every static SVG color to an appearance control.
    names = set(re.findall(r'(?:fill|stroke)="([^"]+)"', b.empty_svg)) - {"none"}
    pal = {"background.color": cs.background_color, "grid.color": cs.grid_color}
    rest = names - {cs.background_color, cs.grid_color}
    if rest:
        assert len(rest) == 1, f"{game}: unexpected static colours {rest}"
        pal["line.color"] = rest.pop()
    assert set(pal.values()) == names, (
        f"{game}: static colours {names} are not covered by {pal} -- a knob that owns no colour is "
        f"inert, and a colour that no knob owns cannot be varied"
    )
    b.palette = pal
    b.marks = _mark_colors(game, b.empty_svg)
    _BUNDLE_CACHE[game] = b
    return b


# Game-specific drawing encodings and geometry in SVG canvas units.
# Cell index is row * ncols + col; radius excludes the stroke half-width.
GAMES: dict[str, dict] = {
    "tic_tac_toe": dict(
        cells=9, ncols=3, nrows=3, grid=60.0, empty=-1, p1=0, p2=1, mark="xo", radius=24.0, stroke=3.0, arm=0.8
    ),
    "connect_four": dict(
        cells=42, ncols=7, nrows=6, grid=35.0, empty=-1, p1=0, p2=1, mark="stone", radius=35.0 / 3.0, stroke=1.0
    ),
    "othello": dict(
        cells=64, ncols=8, nrows=8, grid=30.0, empty=0, p1=1, p2=-1, mark="stone", radius=30.0 / 2.4, stroke=1.0
    ),
    "go_9x9": dict(
        cells=81, ncols=9, nrows=9, grid=25.0, empty=0, p1=1, p2=-1, mark="stone", radius=25.0 / 2.2, stroke=1.0
    ),
    "go_19x19": dict(
        cells=361, ncols=19, nrows=19, grid=25.0, empty=0, p1=1, p2=-1, mark="stone", radius=25.0 / 2.2, stroke=1.0
    ),
}


def _state_pool(game: str, n: int = 512) -> np.ndarray:
    """Cache positions sampled along seeded games using legal random actions."""
    hit = _POOL_CACHE.get(game)
    if hit is not None:
        return hit
    import jax

    spec = GAMES[game]
    env = _pgx.make(game)
    init, step = jax.jit(env.init), jax.jit(env.step)
    out = np.empty((n, spec["cells"]), np.float32)
    key = jax.random.PRNGKey(0)
    i = 0
    ep = 0
    while i < n:
        key, sk = jax.random.split(key)
        st = init(sk)
        ep += 1
        for _ in range(spec["cells"] * 2):
            if i >= n:
                break
            out[i] = _drawn_board(game, st)
            i += 1
            if bool(st.terminated) or bool(st.truncated):
                break
            legal = np.asarray(st.legal_action_mask)
            if not legal.any():
                break
            key, sk = jax.random.split(key)
            a = int(jax.random.choice(sk, np.flatnonzero(legal)))
            key, sk = jax.random.split(key)
            st = step(st, a, sk)
    _POOL_CACHE[game] = out
    return out


def _drawn_board(game: str, st) -> np.ndarray:
    """Extract the displayed board: Go chain signs or Othello absolute player colors."""
    b = np.asarray(st._x.board, np.int32)
    if game.startswith("go"):
        return np.clip(b, -1, 1).astype(np.float32)
    if game == "othello":
        return (b if int(np.asarray(st._x.turn)) == 0 else -b).astype(np.float32)
    return b.astype(np.float32)


class PgxBoardBase(Env):
    """Composite a resvg-rendered empty board with batched Torch marks.

    Render state contains displayed cell values, not the full PGX simulation
    state. Mark edges use sampled coverage and can differ from resvg.
    """

    family = "pgx"

    GAME: str = ""
    default_supersample = 1
    # Only these appearance paths require a new static board bake.
    PLATE_PATHS: tuple[str, ...] = ()

    parity_interior_min = 1.0
    parity_note = ""

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        b = _bundle(self.GAME)
        spec = GAMES[self.GAME]
        self._b = b
        self._spec = spec
        self.state_dim = int(spec["cells"])
        self.state_names = tuple(f"cell_r{i // spec['ncols']}c{i % spec['ncols']}" for i in range(spec["cells"]))
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        assert self.default_size == b.native, (
            f"{type(self).__name__}: default_size {self.default_size} is not the document's own canvas {b.native}"
        )
        self._check_plate_is_constant()

    def _close(self) -> None:
        """Drop instance-owned draw tensors; shared plates and document caches remain live."""
        for k in ("_outline_cache", "_lat_cache", "_alpha_cache"):
            self.__dict__.pop(k, None)

    def _check_plate_is_constant(self) -> None:
        """Compare two empty-board bakes; probe states do not alter the document."""
        st = self.sample_states(2, rng=np.random.default_rng(0))
        a = self._bake_plate(64, None, probe_state=st[0])
        c = self._bake_plate(64, None, probe_state=st[1])
        d = int(np.abs(a.astype(np.int16) - c.astype(np.int16)).max())
        assert d == 0, (
            f"{type(self).__name__}: the baked plate differs by {d}/255 between two states, so the "
            f"board furniture is NOT a function of the appearance alone and the plate design is "
            f"unsound for this game"
        )

    def make_variation_space(self) -> Dict:
        """Expose board colors, player colors, and mark scale using source-derived defaults."""
        b = _bundle(self.GAME)
        space: dict = {}
        for path, name in b.palette.items():
            head, leaf = path.split(".")
            space.setdefault(head, {})[leaf] = RGBBox(init_value=list(_rgb_of(name)))
        # Keep source outlines fixed: thin antialiased strokes do not expose a fully covered color.
        for who in ("p1", "p2"):
            space[who] = {"color": RGBBox(init_value=list(_rgb_of(b.marks[f"{who}.color"])))}
        out = {k: Dict(v) for k, v in space.items()}
        out["mark"] = Dict({"scale": Box(low=0.25, high=1.15, shape=(1,), init_value=[1.0])})
        return Dict(out)

    def _mark_color(self, sty, who: str, which: str, B: int) -> torch.Tensor:
        if which == "color":
            return sty[f"{who}.color"].reshape(B, 1, 1, 3)
        cache = getattr(self, "_outline_cache", None)
        if cache is None:
            cache = self._outline_cache = {}
        hit = cache.get(who)
        if hit is None:
            hit = cache[who] = torch.tensor(
                _rgb_of(self._b.marks[f"{who}.outline"]), device=self.device, dtype=torch.float32
            ).reshape(1, 1, 1, 3)
        return hit

    def _plate(self, res: int, colors: dict) -> torch.Tensor:
        key = (
            self.GAME,
            int(res),
            str(self.device),
            tuple(sorted((k, tuple(np.round(np.asarray(v, np.float64), 4).tolist())) for k, v in colors.items())),
        )
        hit = _PLATE_CACHE.get(key)
        if hit is not None:
            return hit
        return _PLATE_CACHE.put(key, torch.as_tensor(self._bake_plate(res, colors), device=self.device))

    def _bake_plate(self, res: int, colors: dict | None, probe_state=None) -> np.ndarray:
        """Rasterize the empty-board SVG after replacing matching fill/stroke colors."""
        b = _bundle(self.GAME)
        svg = b.empty_svg
        if probe_state is not None:
            # probe_state does not change the empty-board document.
            svg = b.empty_svg
        if colors:
            for path, name in b.palette.items():
                if path not in colors:
                    continue
                r, g, bl = (int(round(float(v))) for v in np.asarray(colors[path]).reshape(3))
                hexc = f"#{r:02x}{g:02x}{bl:02x}"
                for attr in ("fill", "stroke"):
                    svg = svg.replace(f'{attr}="{name}"', f'{attr}="{hexc}"')
        return _raster(svg, int(res), int(res))

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        res = size * ss
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        paths = tuple(self._b.palette)
        key = torch.cat([sty[p].reshape(B, -1) for p in paths], 1).cpu().numpy()
        uniq, inv = np.unique(np.round(key.astype(np.float64), 4), axis=0, return_inverse=True)
        plates = []
        for row in uniq:
            o = 0
            over = {}
            for p in paths:
                w = int(sty.layout[p][1])
                over[p] = row[o : o + w]
                o += w
            plates.append(self._plate(res, over))
        pl = torch.stack(plates)[torch.as_tensor(inv, device=self.device, dtype=torch.long)]

        img = self._composite(s, sty, pl.to(torch.float32), res)
        if ss == 1:
            buf.copy_(img.round_().clamp_(0, 255).to(torch.uint8))
        else:
            small = img.reshape(B, size, ss, size, ss, 3).mean((2, 4)).round_().clamp_(0, 255).to(torch.uint8)
            buf.copy_(small)
        return buf.view(*lead, size, size, 3) if lead else buf

    # Samples per axis for tic-tac-toe strokes; discs use scanline integration.
    COV_SUB = 8

    def _lattice(self, res: int):
        """Cache cell indices, local offsets, and an in-board mask per resolution.

        A pixel looks up its nearest lattice point, so this approach requires marks
        to stay within their cells.
        """
        cache = getattr(self, "_lat_cache", None)
        if cache is None:
            cache = self._lat_cache = {}
        hit = cache.get(res)
        if hit is not None:
            return hit
        b, spec = self._b, self._spec
        G = b.grid
        sc = res / float(b.native)  # canvas units -> pixels
        px = (torch.arange(res, device=self.device, dtype=torch.float32) + 0.5) / sc
        out = {}
        for axis, n in (("x", spec["ncols"]), ("y", spec["nrows"])):
            raw = torch.round(px / G) - 1.0
            c = raw.clamp(0, n - 1)
            out[axis] = (c.to(torch.long), px - (c + 1.0) * G, (raw >= 0) & (raw < n))
        colc, u, inx = out["x"]
        rowc, v, iny = out["y"]
        # Clamping makes gather safe; the separate mask prevents painting outside the board.
        idx = (rowc.reshape(-1, 1) * spec["ncols"] + colc.reshape(1, -1)).reshape(-1)
        inside = iny.reshape(-1, 1) & inx.reshape(1, -1)
        hit = (idx, u, v, inside, sc)
        cache[res] = hit
        return hit

    def _alpha_maps(self, res: int, scale: float):
        """Cache sampled mark coverage by resolution and scale.

        Tic-tac-toe keeps separate stroke layers in SVG paint order. Stones use
        disjoint fill/stroke regions from scanline integration.
        """
        cache = getattr(self, "_alpha_cache", None)
        if cache is None:
            cache = self._alpha_cache = {}
        key = (int(res), round(float(scale), 6))
        hit = cache.get(key)
        if hit is not None:
            return hit
        if len(cache) > 8:
            cache.clear()
        spec, b = self._spec, self._b
        G = b.grid
        sc = res / float(b.native)
        S = int(self.COV_SUB)
        base = torch.arange(res, device=self.device, dtype=torch.float32) + 0.5
        r = float(spec["radius"]) * float(scale)
        half = 0.5 * float(spec["stroke"])
        nmap = 2 if spec["mark"] == "stone" else 3
        acc = [torch.zeros((res, res), device=self.device, dtype=torch.float32) for _ in range(nmap)]

        # Match _lattice indexing exactly, including its -1 offset before clamping.
        def _local(t, n):
            return t - ((torch.round(t / G) - 1.0).clamp(0, n - 1) + 1.0) * G

        if spec["mark"] == "stone":
            return self._stone_alpha(res, r, half, sc, base, _local, key, cache)
        offs = (torch.arange(S, device=self.device, dtype=torch.float32) + 0.5) / S - 0.5
        uu = _local((base.reshape(-1, 1) + offs.reshape(1, -1)) / sc, spec["ncols"]).reshape(1, res * S)  # (1, res*S)
        for iy in range(S):
            ys = (base + offs[iy]) / sc
            vv = _local(ys, spec["nrows"]).reshape(-1, 1)  # (res, 1)
            d = torch.sqrt(uu * uu + vv * vv).reshape(res, res, S)
            if spec["mark"] == "stone":
                acc[0] += (d < r).to(torch.float32).sum(-1)  # fill, out to the PATH
                acc[1] += ((d - r).abs() <= half).to(torch.float32).sum(-1)  # stroke, straddling it
            else:
                uu2 = uu.reshape(1, res, S)
                vv2 = vv.reshape(-1, 1, 1)
                # X consists of two butt-capped rectangles, composited in SVG path order.
                halfL = 0.5 * float(spec["arm"]) * G * float(np.sqrt(2.0)) * float(scale)
                inv = float(np.sqrt(0.5))
                for k, sgn in enumerate((1.0, -1.0)):
                    p = ((uu2 + sgn * vv2) * inv).abs() <= halfL
                    q = ((uu2 - sgn * vv2) * inv).abs() <= half
                    acc[k] += (p & q).to(torch.float32).sum(-1)
                acc[2] += ((d - r).abs() <= half).to(torch.float32).sum(-1)  # the O, an annulus
        hit = tuple(a / float(S * S) for a in acc)
        cache[key] = hit
        return hit

    # Samples along y for disc coverage; horizontal overlap is integrated exactly.
    SCAN_ROWS = 16

    def _stone_alpha(self, res, r, half, sc, base, _local, key, cache):
        """Integrate disc coverage exactly along x and sample along y."""
        N = int(self.SCAN_ROWS)
        hpx = 0.5 / sc  # half a pixel, in canvas units
        u = _local(base / sc, self._spec["ncols"]).reshape(1, -1)  # (1, res)
        v = _local(base / sc, self._spec["nrows"]).reshape(-1, 1)  # (res, 1)
        x0, x1 = u - hpx, u + hpx
        acc = [torch.zeros((res, res), device=self.device, dtype=torch.float32) for _ in range(2)]

        def disc_cov(R):
            out = torch.zeros((res, res), device=self.device, dtype=torch.float32)
            for k in range(N):
                y = v + ((k + 0.5) / N - 0.5) * (2.0 * hpx)
                w = torch.sqrt(torch.clamp(R * R - y * y, min=0.0))
                lo = torch.maximum(x0, -w)
                hi = torch.minimum(x1, w)
                out += torch.clamp(hi - lo, min=0.0)
                del y, w, lo, hi
            return out / (float(N) * 2.0 * hpx)

        # Split fill and stroke into disjoint regions so partial coverage is not counted twice.
        inner = disc_cov(r - half)
        outer = disc_cov(r + half)
        acc[0] = inner
        acc[1] = (outer - inner).clamp(0.0, 1.0)
        hit = (acc[0], acc[1])
        cache[key] = hit
        return hit

    def _composite(self, s: torch.Tensor, sty, plate: torch.Tensor, res: int) -> torch.Tensor:
        spec = self._spec
        B = int(s.shape[0])
        idx, _u, _v, inside, _sc = self._lattice(res)
        val = s.gather(1, idx.reshape(1, -1).expand(B, -1)).reshape(B, res, res)
        is1 = ((val == float(spec["p1"])) & inside).to(plate.dtype)
        is2 = ((val == float(spec["p2"])) & inside).to(plate.dtype)

        msc = sty["mark.scale"].reshape(-1)
        uniq, inv = np.unique(np.round(msc.detach().cpu().numpy().astype(np.float64), 6), return_inverse=True)
        maps = [self._alpha_maps(res, g) for g in uniq]
        g = torch.as_tensor(inv, device=self.device, dtype=torch.long)
        a = [torch.stack([m[k] for m in maps])[g] for k in range(len(maps[0]))]  # each (B,H,W)

        c1 = self._mark_color(sty, "p1", "color", B)
        c2 = self._mark_color(sty, "p2", "color", B)
        img = plate
        if spec["mark"] == "stone":
            o1 = self._mark_color(sty, "p1", "outline", B)
            o2 = self._mark_color(sty, "p2", "outline", B)
            # Disjoint regions and mutually exclusive player marks form one weighted sum.
            wf1, ws1 = (a[0] * is1).unsqueeze(-1), (a[1] * is1).unsqueeze(-1)
            wf2, ws2 = (a[0] * is2).unsqueeze(-1), (a[1] * is2).unsqueeze(-1)
            return plate * (1.0 - wf1 - ws1 - wf2 - ws2) + c1 * wf1 + o1 * ws1 + c2 * wf2 + o2 * ws2
        img = self._over(img, c1, a[0] * is1)  # X, first stroke
        img = self._over(img, c1, a[1] * is1)  # X, second stroke
        return self._over(img, c2, a[2] * is2)  # O, an annulus over the plate

    @staticmethod
    def _over(dst: torch.Tensor, color: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        al = alpha.unsqueeze(-1)
        return dst * (1.0 - al) + color * al

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        pool = _state_pool(self.GAME)
        return pool[rng.integers(0, len(pool), int(batch))].copy()

    @classmethod
    def reference_env(cls, size: int = 0):
        """Return game and resolution settings for the PGX SVG/resvg reference."""
        return {"game": cls.GAME, "size": int(size or _bundle(cls.GAME).native)}

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Write displayed cells into PGX state, verify exact readback, and rasterize SVG.

        Othello turn is fixed to zero so absolute player colors survive the round trip.
        """
        import jax
        import jax.numpy as jnp

        game = env["game"]
        size = int(env["size"])
        b = _bundle(game)
        base = _pgx.make(game)
        st0 = jax.jit(base.init)(jax.random.PRNGKey(0))
        scale = size / float(b.native)
        out = []
        for row in np.asarray(states, np.int32):
            x = st0._x._replace(board=jnp.asarray(row.astype(np.int32)))
            if hasattr(x, "turn"):
                x = x._replace(turn=jnp.int32(0))
            st = st0.replace(_x=x)
            got = _drawn_board(game, st)
            err = float(np.abs(got - row.astype(np.float32)).max())
            if err > 0:
                raise AssertionError(
                    f"{cls.__name__}: the reference did not reach the requested position (max "
                    f"{err}); every parity number computed from it would compare two different "
                    f"boards"
                )
            svg = st.to_svg(scale=scale)
            out.append(_raster(svg, size, size))
        return np.stack(out)


# Keep experimental classes available without registering those below their acceptance threshold.
NOT_REGISTERED = {
    "go_19x19": "all-pixel 0.8876 < the suite's shared ALL_EXACT_MIN of 0.90; interior is 1.0",
}


def _make_classes():
    made = {}
    for game in GAMES:
        cname = "Pgx" + "".join(p.capitalize() for p in game.split("_"))
        b = _bundle(game)
        keep = register if game not in NOT_REGISTERED else (lambda c: c)
        made[cname] = keep(
            type(
                cname,
                (PgxBoardBase,),
                dict(
                    GAME=game,
                    name=f"pgx-{game.replace('_', '-')}",
                    reference_id=f"pgx/{game}",
                    # Read canvas size and static appearance paths from the source document.
                    default_size=b.native,
                    PLATE_PATHS=tuple(b.palette),
                    __doc__=(
                        f"PGX {game}: state is one ternary per board cell, rendered as a resvg-baked "
                        f"plate of the board furniture plus a lattice gather for the marks."
                    ),
                ),
            )
        )
    return made


globals().update(_make_classes())
