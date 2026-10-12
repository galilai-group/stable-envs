from __future__ import annotations

import math

import numpy as np
import torch
from tensordict import TensorDict

from contrib.render import tilemap as tm
from stable_envs._core import stepgraph as _stepgraph
from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.spaces import Box, Dict, Discrete, RGBBox

VIEW_SIZE = 7  #: agent_view_size; 7 for every registered MiniGrid/BabyAI env
WALL_CODE = (tm.T_WALL, tm.C_GREY, 0)  #: Grid.slice pads out-of-bounds with a grey Wall()

# Empty cells and empty hands share a code, so dropping needs no extra encoding.
EMPTY_CODE = tm.REACHABLE.index((tm.T_EMPTY, 0, 0))
CARRY_NONE = EMPTY_CODE
WALL_CODE_IDX = tm.REACHABLE.index(WALL_CODE)

_RULES: dict[str, dict] = {}


def _transition_tables(device) -> dict:
    """Cache per-code transition rules read from source objects and the Actions enum."""
    key = str(device)
    if key not in _RULES:
        from minigrid.core.actions import Actions
        from minigrid.core.constants import DIR_TO_VEC

        objs = [tm.make_obj(*c) for c in tm.REACHABLE]
        L = lambda v: torch.tensor(v, dtype=torch.long, device=device)
        B = lambda f: torch.tensor([bool(f(o)) for o in objs], dtype=torch.bool, device=device)
        _RULES[key] = {
            "type": L([c[0] for c in tm.REACHABLE]),
            "color": L([c[1] for c in tm.REACHABLE]),
            "state": L([c[2] for c in tm.REACHABLE]),
            # All codes decoding to None count as empty for movement.
            "overlap": B(lambda o: o is None or o.can_overlap()),
            "pickup": B(lambda o: o is not None and o.can_pickup()),
            "none": B(lambda o: o is None),
            #: (colour, state) -> code, for writing a door back after a toggle
            "door": L([[tm.REACHABLE.index((tm.T_DOOR, c, s)) for s in range(tm.N_STATE)] for c in range(tm.N_COLOR)]),
            "dir_vec": L([list(v) for v in DIR_TO_VEC]),
            "actions": {
                n: int(getattr(Actions, n)) for n in ("left", "right", "forward", "pickup", "drop", "toggle", "done")
            },
        }
    return _RULES[key]


def _code_table(device) -> torch.Tensor:
    """Return the int64 code-to-(type, colour, state) table on ``device``."""
    return torch.tensor(tm.REACHABLE, dtype=torch.long, device=device)


def _code_lookup(device) -> torch.Tensor:
    """Map (type, colour, state) to code; normalize irrelevant channels by object type."""
    out = torch.full((tm.N_TYPE, tm.N_COLOR, tm.N_STATE), 0, dtype=torch.long, device=device)
    for i, (t, c, s) in enumerate(tm.REACHABLE):
        out[t, c, s] = i
        if t in (tm.T_UNSEEN, tm.T_EMPTY, tm.T_AGENT, tm.T_LAVA):
            out[t] = i  # colour/state are not encoded for these types
    return out


class _TileEnv(Env):
    """Share MiniGrid tile atlases, palette handling and batched rendering."""

    family = "minigrid"

    grid_w: int = 8
    grid_h: int = 8
    see_through_walls: bool = False
    has_lava: bool = False
    tile_size_default: int = 32

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.atlas = tm.MiniGridAtlas.get(self._tile(self.size), device=self.device)
        self.code_table = _code_table(self.device)
        self.code_lookup = _code_lookup(self.device)
        self._pool: np.ndarray | None = None
        self._default_t: torch.Tensor | None = None

    def _close(self) -> None:
        """Drop instance sampling/style caches; the shared atlas remains available."""
        self._pool = None
        self._default_t = None

    def _tile(self, size: int) -> int:
        t, r = divmod(int(size), self.grid_h)
        if r or t < 3:
            raise ValueError(
                f"{type(self).__name__}: size {size} is not a whole tile size for a {self.grid_w}x"
                f"{self.grid_h} grid; use a multiple of {self.grid_h} that is at least {3 * self.grid_h}"
            )
        return t

    def make_variation_space(self) -> Dict:
        """Expose source palette entries, chrome, visibility highlighting and optional lava.

        Chrome rows are background, grid line and agent. Highlighting blends these
        entries, so they need not appear literally in the image. Tile size belongs
        to output resolution, not Variation."""
        space = {
            "palette": Box(low=0, high=255, shape=(tm.N_COLOR, 3), dtype=np.uint8, init_value=tm.official_colors()),
            #: rows: background, grid line, agent
            "chrome": Box(
                low=0,
                high=255,
                shape=(3, 3),
                dtype=np.uint8,
                init_value=np.array([tm.DEFAULT_BG, tm.DEFAULT_GRIDLINE, tm.DEFAULT_AGENT], np.uint8),
            ),
            "highlight": Discrete(2, init_value=1),
            "highlight_overlay": Box(
                low=np.zeros(4, np.float32),
                high=np.array([255.0, 255.0, 255.0, 1.0], np.float32),
                shape=(4,),
                dtype=np.float32,
                init_value=[*tm.DEFAULT_HIGHLIGHT, tm.DEFAULT_HIGHLIGHT_ALPHA],
            ),
        }
        if self.has_lava:
            space["lava_color"] = RGBBox(init_value=tm.DEFAULT_LAVA)
        return Dict(space)

    def _values(self, row: np.ndarray) -> dict:
        """Unpack one style row, truncating colours before atlas recolouring."""
        d = self.variation_space.unpack(row)
        chrome = np.trunc(np.asarray(d["chrome"], np.float64)).reshape(3, 3)
        ov = np.asarray(d["highlight_overlay"], np.float64).reshape(4)
        out = {
            "palette": np.trunc(np.asarray(d["palette"], np.float64)).reshape(tm.N_COLOR, 3),
            "background_color": chrome[0],
            "grid_line_color": chrome[1],
            "agent_color": chrome[2],
            "highlight_color": np.trunc(ov[:3]),
            "highlight_alpha": float(ov[3]),
        }
        if self.has_lava:
            out["lava_color"] = np.trunc(np.asarray(d["lava_color"], np.float64))
        return out

    def _shared_style(self, sty) -> bool:
        """Check shared appearance on device, transferring only the Boolean result."""
        v = sty.values
        return v.shape[0] == 1 or bool((v == v[:1]).all())

    def _is_default_style(self, sty) -> bool:
        if self._default_t is None:
            self._default_t = torch.as_tensor(self.default_style(), device=self.device, dtype=torch.float32)
        return bool((sty.values[0] == self._default_t).all())

    def _label_palettes(self, sty) -> torch.Tensor:
        """Build per-row normal/highlight palettes as float64 ``(B, 2*N_LABEL, 3)``.

        Preserve truncation and uint8-style wraparound before applying the overlay."""
        B, dev = sty.batch, self.device
        pal = torch.trunc(sty["palette"].reshape(B, tm.N_COLOR, 3).double())
        chrome = torch.trunc(sty["chrome"].reshape(B, 3, 3).double())
        p = torch.zeros(B, tm.N_LABEL, 3, dtype=torch.float64, device=dev)
        p[:, tm.LBL_BG] = chrome[:, 0]
        p[:, tm.LBL_GRID] = chrome[:, 1]
        p[:, tm.LBL_AGENT] = chrome[:, 2]
        p[:, tm.LBL_LAVA] = (
            torch.trunc(sty["lava_color"].double())
            if "lava_color" in sty
            else torch.tensor(tm.DEFAULT_LAVA, dtype=torch.float64, device=dev).expand(B, 3)
        )
        p[:, tm.LBL_PAL : tm.LBL_PAL + tm.N_COLOR] = pal
        p[:, tm.LBL_PAL_HALF : tm.LBL_PAL_HALF + tm.N_COLOR] = torch.trunc(pal / 2)
        p[:, tm.LBL_PAL_45 : tm.LBL_PAL_45 + tm.N_COLOR] = torch.trunc(0.45 * pal)
        ov = sty["highlight_overlay"].double()
        # Match source uint8 subtraction: darker highlight colours wrap modulo 256.
        diff = torch.remainder(torch.trunc(ov[:, :3]).unsqueeze(1) - p, 256.0)
        ph = torch.floor(torch.clamp(p + ov[:, 3].view(B, 1, 1) * diff, 0, 255))
        return torch.cat([p, ph], 1)

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
        """Gather compact render rows into uint8 frames, preserving leading batch dimensions."""
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}: supersample={supersample} is not available. MiniGrid's own "
                f"3x supersampling is already baked into the atlas by Grid.render_tile; sampling "
                f"again on top of it would render something the reference never draws."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        size = int(size or self.size)
        tile = self._tile(size)
        atlas = self.atlas if tile == self.atlas.T else tm.MiniGridAtlas.get(tile, device=self.device)
        sty = self.resolve_style(style, variation, batch=s.shape[0])
        idx = self._tile_indices(s, sty)
        flat_out = None
        if out is not None:
            if not out.is_contiguous():
                raise ValueError(
                    f"{type(self).__name__}.render: `out` must be contiguous — the "
                    f"kernel writes through a flat view of it"
                )
            # Contiguity makes this reshape a view into the caller’s output buffer.
            flat_out = out.reshape(-1, self.grid_h * tile, self.grid_w * tile, 3)
        if self._shared_style(sty):
            # A shared style needs only one recoloured atlas; defaults use source RGB directly.
            rgb = (
                atlas.official
                if self._is_default_style(sty)
                else atlas.rgb(self._values(sty.values[0].detach().cpu().numpy()))
            )
            img = tm.blit(rgb, idx, tile, out=flat_out)
        else:
            img = self._blit_styled(atlas, idx, sty, tile, flat_out)
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def _blit_styled(self, atlas, idx: torch.Tensor, sty, tile: int, out) -> torch.Tensor:
        """Recolour gathered 3× subpixels, then downsample in source float64 order.

        Per-row palettes avoid allocating an atlas per environment. The two mean
        operations must keep their axis order before uint8 conversion."""
        B, H, W = idx.shape
        dev = self.device
        f = tm.SUBDIVS
        px = f * tile
        pal = self._label_palettes(sty)  # (B, 2*N_LABEL, 3)
        geom = idx.long() % tm.N_GEOM
        hl = idx.long() // tm.N_GEOM
        if out is None:
            out = torch.empty(B, H * tile, W * tile, 3, dtype=torch.uint8, device=dev)
        # Bound the float64 subpixel working set per row of tiles.
        step = max(1, int((256 << 20) // (px * W * px * 3 * 8)))
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            bi = torch.arange(b0, b1, device=dev).view(-1, 1, 1)
            for j in range(H):
                lab = atlas.labels[geom[b0:b1, j]].long()  # (b, W, px, px)
                key = lab + hl[b0:b1, j].view(-1, W, 1, 1) * tm.N_LABEL
                key = key.permute(0, 2, 1, 3).reshape(b1 - b0, px, W * px)
                col = pal[bi, key]  # (b, px, W*px, 3) f64
                col = col.reshape(b1 - b0, tile, f, W * tile, f, 3).mean(4).mean(2)
                out[b0:b1, j * tile : (j + 1) * tile] = col.to(torch.uint8)
        return out

    def _tile_indices(self, s: torch.Tensor, sty) -> torch.Tensor:
        raise NotImplementedError

    def _decode_cells(self, codes: torch.Tensor) -> torch.Tensor:
        """Round and wrap cell codes into the render alphabet, returning integer triples."""
        return self.code_table[torch.round(codes).long() % tm.N_CODE]

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample compact render rows from a cached pool of source rollouts."""
        rng = rng or self.rng
        if self._pool is None:
            self._pool = self._build_pool(max(256, batch))
        if batch <= len(self._pool):
            return self._pool[rng.choice(len(self._pool), batch, replace=False)].copy()
        return self._pool[rng.integers(0, len(self._pool), batch)].copy()

    def _build_pool(self, n: int) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _official(cls):
        import gymnasium as gym
        import minigrid  # noqa: F401  (registers the ids)

        e = gym.make(cls.reference_id)
        e.reset(seed=0)
        cls._check_source(e.unwrapped)
        return e

    @classmethod
    def _check_source(cls, u) -> None:
        """Verify the declared visibility rule against the installed source."""
        if bool(u.see_through_walls) != cls.see_through_walls:
            raise AssertionError(
                f"{cls.__name__} declares see_through_walls={cls.see_through_walls} but "
                f"{cls.reference_id} uses {bool(u.see_through_walls)}; the visibility mask, and so "
                f"the highlight, would be wrong"
            )


class MiniGridTopDown(_TileEnv):
    """Render the full grid with highlighted field of view and advance grid dynamics.

    Public State includes the carried object, which the compact render row omits.
    Box contents, rewards and episode counters are not modelled; simulation
    comparisons require empty boxes. Cells beyond the grid act as walls."""

    # Actions: left, right, forward, pickup, drop, toggle, done.
    # Flooring a sample from [0, 7) selects one of these seven actions.
    N_ACTIONS = 7
    action_dim = 1
    action_low = 0.0
    action_high = 7.0
    use_step_graph = True
    step_reference_checkable = True
    step_reference_exact = True

    step_reference_hold = 1

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._rules = _transition_tables(self.device)
        self._A = self._rules["actions"]
        z = lambda v: torch.tensor(int(v), dtype=torch.long, device=self.device)
        self._empty, self._wall, self._none = z(EMPTY_CODE), z(WALL_CODE_IDX), z(CARRY_NONE)
        self._carry = torch.full((self.B,), CARRY_NONE, dtype=torch.long, device=self.device)
        self._act = torch.zeros(self.B, self.action_dim, device=self.device, dtype=torch.float32)
        self._step_graph = None
        self._step_graph_error = ""

    def _close(self) -> None:
        super()._close()
        self._step_graph = None

    def _set_render_state(self, states, variation=None, style=None) -> MiniGridTopDown:
        """Install a compact render row and empty the hands; full State restores carrying separately."""
        super()._set_render_state(states, variation=variation, style=style)
        self._carry.fill_(CARRY_NONE)
        return self

    def _restore_state(self, state: TensorDict) -> None:
        self._set_render_state(self._pack_render_state(state))
        self._carry.copy_(state["agent", "carrying"].reshape(-1).long())

    def _hidden_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """Include the live carried object when exporting full State."""
        return {("agent", "carrying"): self._carry[: self._live_rows].reshape(-1, 1).to(torch.float32)}

    def sim_state(self) -> torch.Tensor:
        """Return float64 ``(B, state_dim + 1)`` rows with carrying appended for comparison."""
        return torch.cat([self._render_state.to(torch.float64), self._carry.to(torch.float64).unsqueeze(1)], 1)

    def set_sim_state(self, v) -> MiniGridTopDown:
        v = torch.as_tensor(v, device=self.device, dtype=torch.float64).reshape(self.B, -1)
        self._set_render_state(v[:, : self.state_dim])
        self._carry.copy_(torch.round(v[:, self.state_dim]).long())
        return self

    def _advance(self) -> None:
        """Advance owned buffers at fixed addresses for CUDA graph replay."""
        self._advance_into(self._render_state, self._carry, self._act)

    def _step_state(self) -> TensorDict:
        """Snapshot live grid, pose and carrying without aliasing simulator buffers."""
        n = self._live_rows
        N = self.grid_w * self.grid_h
        rows = self._render_state[:n].clone()
        return TensorDict(
            {
                "grid": rows[:, :N],
                "agent": {
                    "position": rows[:, N : N + 2],
                    "direction": rows[:, N + 2 : N + 3],
                    "carrying": self._carry[:n].reshape(-1, 1).to(torch.float32),
                },
            },
            batch_size=[n],
            device=self.device,
        )

    def _advance_into(self, s, carry, act) -> None:
        """Apply masked grid transitions in place to ``s`` and ``carry``; actions are read-only.

        Actions are floored and clamped to 0..6. Off-grid cells act as walls and
        remain unwritten. Box toggles empty the cell; contents are not represented."""
        R, A = self._rules, self._A
        W, H, N = self.grid_w, self.grid_h, self.grid_w * self.grid_h
        a = act[:, 0].floor().clamp(0, self.N_ACTIONS - 1).long()
        codes = torch.round(s[:, :N]).long() % tm.N_CODE
        ax = torch.round(s[:, N]).long() % W
        ay = torch.round(s[:, N + 1]).long() % H
        ad = torch.round(s[:, N + 2]).long() % 4

        d = R["dir_vec"][ad]  # front_pos = agent_pos + dir_vec
        fx, fy = ax + d[:, 0], ay + d[:, 1]
        inb = (fx >= 0) & (fx < W) & (fy >= 0) & (fy < H)
        fi = (fx.clamp(0, W - 1) * H + fy.clamp(0, H - 1)).unsqueeze(1)
        held = codes.gather(1, fi).squeeze(1)
        cell = torch.where(inb, held, self._wall)
        ctype, ccol, cstate = R["type"][cell], R["color"][cell], R["state"][cell]

        ad_new = torch.where(a == A["left"], (ad - 1) % 4, torch.where(a == A["right"], (ad + 1) % 4, ad))
        move = (a == A["forward"]) & R["overlap"][cell]
        ax_new, ay_new = torch.where(move, fx, ax), torch.where(move, fy, ay)

        pick = (a == A["pickup"]) & R["pickup"][cell] & (carry == CARRY_NONE)
        drop = (a == A["drop"]) & R["none"][cell] & (carry != CARRY_NONE) & inb
        tog = a == A["toggle"]
        # A locked door opens only with a matching key; unlocked doors toggle.
        keyed = (R["type"][carry] == tm.T_KEY) & (R["color"][carry] == ccol)
        unlock = tog & (ctype == tm.T_DOOR) & (cstate == 2) & keyed
        flip = tog & (ctype == tm.T_DOOR) & (cstate != 2)
        openbox = tog & (ctype == tm.T_BOX)

        new = torch.where(pick, self._empty, cell)
        new = torch.where(drop, carry, new)
        new = torch.where(unlock, R["door"][ccol, 0], new)
        new = torch.where(flip, R["door"][ccol, 1 - cstate], new)
        new = torch.where(openbox, self._empty, new)
        new = torch.where(inb, new, held)  # off-grid: write the cell back unchanged

        s[:, :N] = codes.scatter(1, fi, new.unsqueeze(1)).to(torch.float32)
        s[:, N] = ax_new.to(torch.float32)
        s[:, N + 1] = ay_new.to(torch.float32)
        s[:, N + 2] = ad_new.to(torch.float32)
        carry.copy_(torch.where(pick, cell, torch.where(drop, self._none, carry)))

    @torch.no_grad()
    def _step(self, action) -> TensorDict:
        """Advance from ``(B, 1)`` actions and return State, including carrying.

        Reward, termination and truncation are not computed. The transition continues
        without automatic reset, including when a source episode would have ended."""
        self._act.copy_(
            torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, self.action_dim)
        )
        if self.use_step_graph and self.device.type == "cuda":
            if self._step_graph is None and not self._step_graph_error:
                self._step_graph, self._step_graph_error = _stepgraph.capture(
                    self._advance, (self._render_state, self._carry)
                )
                if self._step_graph is None:
                    self.use_step_graph = False
            if self._step_graph is not None:
                self._step_graph.replay()
                return self._step_result(self._step_state())
        self._advance()
        return self._step_result(self._step_state())

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        """Encode source grid, agent pose and carrying for transition comparisons."""
        u = ref.unwrapped
        lut = _code_lookup(torch.device("cpu")).numpy()
        enc = u.grid.encode()
        codes = lut[enc[:, :, 0], enc[:, :, 1], enc[:, :, 2]].reshape(-1)
        carry = CARRY_NONE
        if u.carrying is not None:
            e = u.carrying.encode()
            carry = int(lut[e[0], e[1], e[2]])
        return np.concatenate([codes, u.agent_pos, [u.agent_dir], [carry]]).astype(np.float64)

    @classmethod
    def reference_step(cls, ref, a) -> None:
        """Step the unwrapped source, bypassing TimeLimit bookkeeping."""
        ref.unwrapped.step(int(min(max(math.floor(float(a[0])), 0), cls.N_ACTIONS - 1)))

    @classmethod
    def step_reference_reset(cls, ref, seed: int) -> np.ndarray:
        """Reset the source and reject boxes whose contents this State cannot encode."""
        ref.reset(seed=seed)
        u = ref.unwrapped
        if any(o is not None and o.type == "box" and o.contains is not None for o in u.grid.grid):
            raise NotImplementedError(
                f"{cls.__name__}: {cls.reference_id} builds a Box with contents, and Box.toggle "
                f"replaces the box by them. The contents are invisible to the renderer and so are not "
                f"in the state, which makes that one transition unrepresentable here."
            )
        return cls.reference_sim_state(ref)

    @classmethod
    def _check_source(cls, u) -> None:
        super()._check_source(u)
        if (u.width, u.height) != (cls.grid_w, cls.grid_h):
            raise AssertionError(
                f"{cls.__name__} declares a {cls.grid_w}x{cls.grid_h} grid but {cls.reference_id} "
                f"builds {u.width}x{u.height}; every rendered frame would be the wrong shape"
            )

    def _tile_indices(self, s: torch.Tensor, sty) -> torch.Tensor:
        B, W, H = s.shape[0], self.grid_w, self.grid_h
        dev = self.device
        tcs = self._decode_cells(s[:, : W * H]).reshape(B, W, H, 3)  # [b, i, j, 3]
        ax = torch.round(s[:, W * H + 0]).long() % W
        ay = torch.round(s[:, W * H + 1]).long() % H
        ad = torch.round(s[:, W * H + 2]).long() % 4

        vis = self._visibility(tcs, ax, ay, ad)  # (B,V,V) [b,vi,vj]
        offs = tm.view_offsets(VIEW_SIZE, dev)[ad]  # (B,V,V,2) [vj,vi]
        wx = ax.view(B, 1, 1) + offs[..., 0]
        wy = ay.view(B, 1, 1) + offs[..., 1]
        keep = (wx >= 0) & (wx < W) & (wy >= 0) & (wy < H) & vis.permute(0, 2, 1)
        flat = torch.arange(B, device=dev).view(B, 1, 1) * (H * W) + wy.clamp(0, H - 1) * W + wx.clamp(0, W - 1)
        # Send masked view cells to a scratch slot: no compaction or host sync,
        # so the scatter remains CUDA-graph capturable.
        scratch = B * H * W
        flat = torch.where(keep, flat, torch.full_like(flat, scratch))
        acc = torch.zeros(scratch + 1, dtype=torch.long, device=dev)
        acc.scatter_(0, flat.reshape(-1), torch.ones_like(flat.reshape(-1)))
        # Mask highlighting per row; a batch may contain both settings.
        hl = acc[:scratch].reshape(B, H, W) * (sty["highlight"] > 0.5).long().view(B, 1, 1)

        img = tcs.permute(0, 2, 1, 3)  # [b, j, i, 3]
        d = torch.full((B, H, W), tm.NO_AGENT, dtype=torch.long, device=dev)
        d[torch.arange(B, device=dev), ay, ax] = ad
        return tm.atlas_flat_index(hl, d, img[..., 0], img[..., 1], img[..., 2]).to(torch.int32)

    def _visibility(self, tcs, ax, ay, ad):
        """Gather the rotated agent view and apply the source visibility scan."""
        B, W, H, V = tcs.shape[0], self.grid_w, self.grid_h, VIEW_SIZE
        dev = self.device
        offs = tm.view_offsets(V, dev)[ad]
        wx = ax.view(B, 1, 1) + offs[..., 0]
        wy = ay.view(B, 1, 1) + offs[..., 1]
        oob = (wx < 0) | (wx >= W) | (wy < 0) | (wy >= H)
        flat = torch.arange(B, device=dev).view(B, 1, 1) * (W * H) + wx.clamp(0, W - 1) * H + wy.clamp(0, H - 1)
        view = tcs.reshape(B * W * H, 3)[flat.reshape(-1)].reshape(B, V, V, 3)
        wall = torch.tensor(WALL_CODE, dtype=view.dtype, device=dev)
        view = torch.where(oob.unsqueeze(-1), wall, view).permute(0, 2, 1, 3)  # [b, vi, vj]
        if self.see_through_walls:
            return torch.ones(B, V, V, dtype=torch.bool, device=dev)
        t, st = view[..., 0], view[..., 2]
        transp = ~((t == tm.T_WALL) | ((t == tm.T_DOOR) & (st != 0)))  # see_behind()
        return tm.visibility_scan(transp, V)

    def _build_pool(self, n: int) -> np.ndarray:
        env = self._official()
        u = env.unwrapped
        lut = _code_lookup(torch.device("cpu")).numpy()
        out, ep = [], 0
        while len(out) < n:
            env.reset(seed=ep)
            rng = np.random.default_rng(ep)
            for _ in range(32):
                enc = u.grid.encode()
                codes = lut[enc[:, :, 0], enc[:, :, 1], enc[:, :, 2]].reshape(-1)
                out.append(np.concatenate([codes, u.agent_pos, [u.agent_dir]]))
                if len(out) >= n:
                    break
                _, _, te, tr, _ = env.step(int(rng.integers(0, 7)))
                if te or tr:
                    break
            ep += 1
        env.close()
        return np.stack(out).astype(np.float32)

    @classmethod
    def reference_env(cls, size: int | None = None):
        env = cls._official()
        env.unwrapped._se_tile = (int(size or cls.default_size)) // cls.grid_h
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore grid and agent pose, then call the source full-view renderer."""
        from minigrid.core.grid import Grid

        u = env.unwrapped
        out = []
        for st in np.asarray(states):
            codes = np.round(st[: cls.grid_w * cls.grid_h]).astype(np.int64) % tm.N_CODE
            grid = Grid(cls.grid_w, cls.grid_h)
            for k, code in enumerate(codes):
                grid.set(k // cls.grid_h, k % cls.grid_h, tm.make_obj(*tm.REACHABLE[code]))
            u.grid, u.width, u.height = grid, cls.grid_w, cls.grid_h
            u.agent_pos = (int(round(st[-3])) % cls.grid_w, int(round(st[-2])) % cls.grid_h)
            u.agent_dir = int(round(st[-1])) % 4
            u.carrying = None  # provably not read by get_full_render
            out.append(u.get_full_render(True, u._se_tile))
        return np.stack(out)


class MiniGridPOVBase(_TileEnv):
    """Render a 7×7 observation window after occlusion.

    The unseen code determines highlighting; carrying occupies the centre-bottom
    cell. This window cannot determine the next observation, so stepping is absent.
    The reference bridge requires a source with ``see_through_walls=True`` to avoid
    recomputing occlusion over an already masked view."""

    grid_w = grid_h = VIEW_SIZE
    see_through_walls = True

    def _tile_indices(self, s: torch.Tensor, sty) -> torch.Tensor:
        B, V = s.shape[0], VIEW_SIZE
        dev = self.device
        tcs = self._decode_cells(s).reshape(B, V, V, 3).permute(0, 2, 1, 3)  # [b, vj, vi, 3]
        t, c, st = tcs[..., 0], tcs[..., 1], tcs[..., 2]
        hl = (t != tm.T_UNSEEN).long() * (sty["highlight"] > 0.5).long().view(B, 1, 1)
        d = torch.full((B, V, V), tm.NO_AGENT, dtype=torch.long, device=dev)
        d[:, V - 1, V // 2] = 3  # the POV agent always points up
        return tm.atlas_flat_index(hl, d, t, c, st).to(torch.int32)

    def _build_pool(self, n: int) -> np.ndarray:
        env = self._official()
        lut = _code_lookup(torch.device("cpu")).numpy()
        out, ep = [], 0
        while len(out) < n:
            obs, _ = env.reset(seed=ep)
            rng = np.random.default_rng(ep)
            for _ in range(32):
                im = obs["image"]
                out.append(lut[im[:, :, 0], im[:, :, 1], im[:, :, 2]].reshape(-1))
                if len(out) >= n:
                    break
                obs, _, te, tr, _ = env.step(int(rng.integers(0, 7)))
                if te or tr:
                    break
            ep += 1
        env.close()
        return np.stack(out).astype(np.float32)

    @classmethod
    def reference_env(cls, size: int | None = None):
        env = cls._official()
        env.unwrapped._se_tile = (int(size or cls.default_size)) // VIEW_SIZE
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Render a restored observation through the source POV path.

        Put the agent at centre-bottom facing up so the view transform is identity.
        Restore carrying there because gen_obs_grid overwrites that cell."""
        from minigrid.core.grid import Grid

        u = env.unwrapped
        V = VIEW_SIZE
        out = []
        for st in np.asarray(states):
            codes = np.round(st).astype(np.int64) % tm.N_CODE
            grid = Grid(V, V)
            for k, code in enumerate(codes):
                grid.set(k // V, k % V, tm.make_obj(*tm.REACHABLE[code]))
            u.grid, u.width, u.height = grid, V, V
            u.agent_pos, u.agent_dir = (V // 2, V - 1), 3
            u.carrying = grid.get(V // 2, V - 1)
            out.append(u.get_pov_render(u._se_tile))
        return np.stack(out)


def _cell_names(w: int, h: int) -> tuple[str, ...]:
    return tuple(f"cell[{i},{j}]" for i in range(w) for j in range(h))


def topdown(
    class_name: str, env_id: str, w: int, h: int, stw: bool, lava: bool = False, tile: int = 32, doc: str = ""
) -> type:
    """Register a full-grid class with carrying in State but outside the render projection."""
    n = w * h
    cls = type(
        class_name,
        (MiniGridTopDown,),
        {
            "__doc__": doc or f"{env_id}: {w}x{h} top-down view.",
            "__module__": __name__,
            "name": env_id,
            "reference_id": env_id,
            "grid_w": w,
            "grid_h": h,
            "see_through_walls": stw,
            "has_lava": lava,
            "state_dim": n + 3,
            "state_names": _cell_names(w, h) + ("agent_x", "agent_y", "agent_dir"),
            # Carrying belongs to State but is invisible in the full-grid render projection.
            "state_layout": (
                (("grid",), n, 0),
                (("agent", "position"), 2, n),
                (("agent", "direction"), 1, n + 2),
                (("agent", "carrying"), 1, None),
            ),
            "default_size": h * tile,
            "sim_state_names": _cell_names(w, h) + ("agent_x", "agent_y", "agent_dir", "carrying"),
            # Compare grid, position, direction and carrying separately; use Euclidean position error.
            "step_reference_groups": (
                ("cells", tuple(range(n)), "abs"),
                ("agent", (n, n + 1), "euclidean"),
                ("dir", (n + 2,), "abs"),
                ("carrying", (n + 3,), "abs"),
            ),
            # Declare step on each generated class for simulator capability discovery.
            "step": MiniGridTopDown.step,
        },
    )
    return register(cls)


def pov(class_name: str, env_id: str, tile: int = 32, doc: str = "") -> type:
    """Register a POV class whose reference source has ``see_through_walls=True``."""
    cls = type(
        class_name,
        (MiniGridPOVBase,),
        {
            "__doc__": doc or f"{env_id}: agent-centric {VIEW_SIZE}x{VIEW_SIZE} view.",
            "__module__": __name__,
            "name": env_id + "-pov",
            "reference_id": env_id,
            "state_dim": VIEW_SIZE * VIEW_SIZE,
            "state_names": tuple(f"view[{i},{j}]" for i in range(VIEW_SIZE) for j in range(VIEW_SIZE)),
            "default_size": VIEW_SIZE * tile,
        },
    )
    return register(cls)


MiniGridEmpty = topdown(
    "MiniGridEmpty",
    "MiniGrid-Empty-8x8-v0",
    8,
    8,
    stw=True,
    doc="Empty room, 8x8. The simplest layout in the family: walls, one goal, one agent.",
)
MiniGridDoorKey = topdown(
    "MiniGridDoorKey",
    "MiniGrid-DoorKey-8x8-v0",
    8,
    8,
    stw=False,
    doc="Locked door and key, 8x8. Exercises all three door states and occluded visibility.",
)
MiniGridFourRooms = topdown(
    "MiniGridFourRooms",
    "MiniGrid-FourRooms-v0",
    19,
    19,
    stw=False,
    doc="Four rooms joined by gaps, 19x19 — the large-grid case, 608x608 at the default tile size.",
)
MiniGridLavaCrossing = topdown(
    "MiniGridLavaCrossing",
    "MiniGrid-LavaCrossingS9N2-v0",
    9,
    9,
    stw=False,
    lava=True,
    doc="Lava rivers with crossings, 9x9. The only registered env carrying the lava_color knob.",
)
MiniGridKeyCorridor = topdown(
    "MiniGridKeyCorridor",
    "MiniGrid-KeyCorridorS4R3-v0",
    10,
    10,
    stw=False,
    doc="Locked rooms off a corridor, 10x10: locked doors, keys and a ball behind them.",
)
BabyAIGoToLocal = topdown(
    "BabyAIGoToLocal",
    "BabyAI-GoToLocal-v0",
    8,
    8,
    stw=False,
    doc="BabyAI's instruction-following room, 8x8: keys, balls and boxes in six colours.",
)
MiniGridPOVFetch = pov(
    "MiniGridPOVFetch",
    "MiniGrid-Fetch-8x8-N3-v0",
    doc="Agent-centric 7x7 view of the Fetch room — MiniGrid's standard partial observation, 224x224 from 49 numbers.",
)


def register_all(ids=None, tile: int = 32) -> list[str]:
    """Register instantiable MiniGrid/BabyAI IDs, or a supplied subset, as top-down classes.

    Called explicitly; missing assets and construction failures are skipped.
    A fixed state-dimension probe can miss wrapped coordinates when its offset
    is a multiple of the grid size."""
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.minigrid_env import MiniGridEnv as _MGE

    if ids is None:
        ids = sorted(k for k in gym.registry if k.startswith("MiniGrid-") or k.startswith("BabyAI-"))
    added = []
    for eid in ids:
        cname = "MG_" + eid.replace("-", "_").replace(".", "_")
        try:
            e = gym.make(eid)
            e.reset(seed=0)
            u = e.unwrapped
            if not isinstance(u, _MGE):
                continue
            w, h, stw = u.width, u.height, bool(u.see_through_walls)
            lava = bool((u.grid.encode()[:, :, 0] == tm.T_LAVA).any())
            e.close()
        except Exception:  # noqa: BLE001  e.g. missing WFC assets
            continue
        topdown(cname, eid, w, h, stw, lava=lava, tile=tile)
        added.append(cname)
    return added
