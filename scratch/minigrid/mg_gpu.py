"""Batched GPU renderer for MiniGrid / BabyAI: frames as a pure gather from a tile atlas.

The whole family draws a frame as a grid of tiles, and ``Grid.render_tile``'s own cache
key proves the tile image is a function of ``(obj.encode(), agent_dir, highlight,
tile_size)`` and nothing else (calibrate.py checks this against pixels). So a frame is

    frame[b, j*T:(j+1)*T, i*T:(i+1)*T] = ATLAS[index(highlight, dir, type, colour, state)]

and the renderer is exact by construction: the atlas *is* the official renderer's output.
The only real work is computing the index tensor, which means reproducing
``Grid.slice`` + ``rotate_left`` (a gather through a traced offset table) and
``Grid.process_vis`` (a 2*V*V-step boolean scan, batched).

State consumed, per environment:
    code       (B, W, H, 3) uint8   -- grid.encode()
    agent_pos  (B, 2)        int64
    agent_dir  (B,)          int64
    carrying   (B, 3)        uint8  -- encode of the carried object, (1,0,0) if none;
                                      only affects the agent-POV frame
"""
from __future__ import annotations

import numpy as np
import torch

N_TYPE, N_COLOR, N_STATE, N_DIR, N_HL = 11, 6, 3, 5, 2
NO_AGENT = 4          # agent_dir slot meaning "no agent on this tile"
T_WALL, C_GREY = 2, 5  # Grid.slice pads out-of-bounds with Wall(), i.e. grey
T_EMPTY = 1            # Grid.encode writes 'empty' for a None cell
T_DOOR = 4
EMPTY_CODE = (T_EMPTY, 0, 0)


def atlas_flat_index(hl, d, t, c, s):
    return ((((hl * N_DIR + d) * N_TYPE + t) * N_COLOR + c) * N_STATE + s)


def view_offsets(view_size: int, device) -> torch.Tensor:
    """offs[d, vj, vi] = (dx, dy) of the world cell shown at view cell (vi, vj).

    Closed form taken from ``MiniGridEnv.get_full_render`` and verified against a
    tagged-object trace of ``Grid.slice`` + ``rotate_left`` in calibrate.py.
    """
    DIR_TO_VEC = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=device)
    V = view_size
    vj = torch.arange(V, device=device).view(V, 1)
    vi = torch.arange(V, device=device).view(1, V)
    offs = torch.empty(4, V, V, 2, dtype=torch.long, device=device)
    for d in range(4):
        f = DIR_TO_VEC[d]
        r = torch.tensor([-f[1], f[0]], device=device)
        top = f * (V - 1) - r * (V // 2)
        for k in range(2):
            offs[d, :, :, k] = top[k] - f[k] * vj + r[k] * vi
    return offs


class MiniGridGPU:
    """Holds the atlas and the derived constants; stateless w.r.t. environments."""

    def __init__(self, atlas: np.ndarray, tile_size: int, device="cuda"):
        assert atlas.shape == (N_HL * N_DIR * N_TYPE * N_COLOR * N_STATE,
                               tile_size, tile_size, 3), atlas.shape
        from mg_triton import _WORD_DTYPE, word_size

        self.T = tile_size
        self.device = device
        self.atlas = torch.from_numpy(np.ascontiguousarray(atlas)).to(device)
        w = word_size(tile_size)
        self.atlas_w = self.atlas.reshape(-1).view(_WORD_DTYPE[w]).view(
            self.atlas.shape[0], tile_size * tile_size * 3 // w
        )
        self._offs: dict[int, torch.Tensor] = {}
        self._pow2: dict[int, torch.Tensor] = {}
        # Constants are materialised once. Building them per call from Python lists is
        # a host->device copy on every frame, which also makes the pipeline
        # un-capturable as a CUDA graph.
        self.wall_code = torch.tensor([T_WALL, C_GREY, 0], dtype=torch.uint8, device=device)
        self.empty_code = torch.tensor(EMPTY_CODE, dtype=torch.uint8, device=device)
        self.use_triton_vis = True

    def offs(self, view_size: int) -> torch.Tensor:
        if view_size not in self._offs:
            self._offs[view_size] = view_offsets(view_size, self.device)
        return self._offs[view_size]

    def pow2(self, view_size: int) -> torch.Tensor:
        if view_size not in self._pow2:
            self._pow2[view_size] = (
                torch.ones(view_size * view_size, dtype=torch.int64, device=self.device)
                << torch.arange(view_size * view_size, device=self.device)
            )
        return self._pow2[view_size]

    def vis(self, view, view_size):
        """Visibility mask, (B, V, V) bool indexed [b, i, j]."""
        if not self.use_triton_vis:
            return self.process_vis(view, view_size)
        from mg_triton import process_vis as _pv

        B = view.shape[0]
        t, s = view[..., 0], view[..., 2]
        transp = ~((t == T_WALL) | ((t == T_DOOR) & (s != 0)))
        p2 = self.pow2(view_size)
        bits = (transp.reshape(B, -1).to(torch.int64) * p2).sum(1)
        m = _pv(bits.contiguous(), view_size)
        return ((m.unsqueeze(1) >> torch.arange(view_size * view_size,
                                                device=m.device)) & 1).bool().reshape(
            B, view_size, view_size)

    # ---------------------------------------------------------------- view / vis

    def gather_view(self, code, agent_pos, agent_dir, view_size):
        """Grid.slice + rotate_left**(d+1), batched. Returns (B, V, V, 3) uint8,
        indexed [b, vi, vj] to match minigrid's own [i, j] convention."""
        B, W, H, _ = code.shape
        V = view_size
        offs = self.offs(V)[agent_dir]                       # (B, V, V, 2) [vj, vi]
        wx = agent_pos[:, 0].view(B, 1, 1) + offs[..., 0]
        wy = agent_pos[:, 1].view(B, 1, 1) + offs[..., 1]
        oob = (wx < 0) | (wx >= W) | (wy < 0) | (wy >= H)
        wxc, wyc = wx.clamp(0, W - 1), wy.clamp(0, H - 1)
        flat = (torch.arange(B, device=code.device).view(B, 1, 1) * (W * H)
                + wxc * H + wyc)
        view = code.reshape(B * W * H, 3)[flat.reshape(-1)].reshape(B, V, V, 3)
        view = torch.where(oob.unsqueeze(-1), self.wall_code, view)
        return view.permute(0, 2, 1, 3).contiguous()          # [b, vi, vj]

    @staticmethod
    def process_vis(view, view_size):
        """Batched transcription of ``Grid.process_vis``.

        Sequential in exactly the way the original is: the left-to-right sweep reads
        cells the same sweep just wrote, and the right-to-left sweep reads what the
        left-to-right one wrote, so the two passes cannot be merged or reordered.
        V is 7, so this is ~100 tiny elementwise ops.
        """
        B = view.shape[0]
        V = view_size
        t, s = view[..., 0], view[..., 2]
        transp = ~((t == T_WALL) | ((t == T_DOOR) & (s != 0)))   # see_behind()
        mask = torch.zeros(B, V, V, dtype=torch.bool, device=view.device)
        mask[:, V // 2, V - 1] = True
        for j in reversed(range(V)):
            for i in range(0, V - 1):
                c = mask[:, i, j] & transp[:, i, j]
                mask[:, i + 1, j] |= c
                if j > 0:
                    mask[:, i + 1, j - 1] |= c
                    mask[:, i, j - 1] |= c
            for i in reversed(range(1, V)):
                c = mask[:, i, j] & transp[:, i, j]
                mask[:, i - 1, j] |= c
                if j > 0:
                    mask[:, i - 1, j - 1] |= c
                    mask[:, i, j - 1] |= c
        return mask

    # ------------------------------------------------------------------ indices

    def full_indices(self, code, agent_pos, agent_dir, view_size=7,
                     see_through_walls=False, highlight=True):
        """Tile-atlas index tensor for ``get_full_render``. Returns (B, H, W) int32."""
        B, W, H, _ = code.shape
        dev = code.device
        img = code.permute(0, 2, 1, 3)                       # [b, j, i, 3]
        t, c, s = img[..., 0].long(), img[..., 1].long(), img[..., 2].long()

        hl = torch.zeros(B, H, W, dtype=torch.long, device=dev)
        if highlight:
            view = self.gather_view(code, agent_pos, agent_dir, view_size)
            vis = (torch.ones(B, view_size, view_size, dtype=torch.bool, device=dev)
                   if see_through_walls else self.vis(view, view_size))
            offs = self.offs(view_size)[agent_dir]           # (B, V, V, 2) [vj, vi]
            wx = agent_pos[:, 0].view(B, 1, 1) + offs[..., 0]
            wy = agent_pos[:, 1].view(B, 1, 1) + offs[..., 1]
            keep = (wx >= 0) & (wx < W) & (wy >= 0) & (wy < H) & vis.permute(0, 2, 1)
            flat = (torch.arange(B, device=dev).view(B, 1, 1) * (H * W)
                    + wy.clamp(0, H - 1) * W + wx.clamp(0, W - 1))
            # Off-grid / not-visible view cells are routed to a scratch slot rather
            # than compacted, so the scatter never needs a device->host sync and the
            # whole pipeline stays CUDA-graph capturable.
            scratch = B * H * W
            flat = torch.where(keep, flat, torch.full_like(flat, scratch))
            hl = torch.zeros(scratch + 1, dtype=torch.long, device=dev)
            hl.scatter_(0, flat.reshape(-1), torch.ones_like(flat.reshape(-1)))
            hl = hl[:scratch].reshape(B, H, W)

        d = torch.full((B, H, W), NO_AGENT, dtype=torch.long, device=dev)
        bidx = torch.arange(B, device=dev)
        d[bidx, agent_pos[:, 1], agent_pos[:, 0]] = agent_dir
        return atlas_flat_index(hl, d, t, c, s).to(torch.int32)

    def pov_indices(self, code, agent_pos, agent_dir, carrying, view_size=7,
                    see_through_walls=False):
        """Tile-atlas index tensor for ``get_pov_render``. Returns (B, V, V) int32."""
        B = code.shape[0]
        dev = code.device
        V = view_size
        view = self.gather_view(code, agent_pos, agent_dir, V)     # [b, vi, vj, 3]
        vis = (torch.ones(B, V, V, dtype=torch.bool, device=dev)
               if see_through_walls else self.vis(view, V))
        view = torch.where(vis.unsqueeze(-1), view, self.empty_code)  # process_vis blanks
        view[:, V // 2, V - 1, :] = carrying                       # agent sees its item

        img = view.permute(0, 2, 1, 3)                             # [b, vj, vi, 3]
        t, c, s = img[..., 0].long(), img[..., 1].long(), img[..., 2].long()
        hl = vis.permute(0, 2, 1).long()
        d = torch.full((B, V, V), NO_AGENT, dtype=torch.long, device=dev)
        d[:, V - 1, V // 2] = 3                                    # POV agent faces up
        return atlas_flat_index(hl, d, t, c, s).to(torch.int32)

    # --------------------------------------------------------------------- blit

    def blit_torch(self, idx):
        """Reference blit: advanced-index the atlas, then untile. Two passes."""
        B, H, W = idx.shape
        T = self.T
        tiles = self.atlas[idx.long()]                    # (B, H, W, T, T, 3)
        return tiles.permute(0, 1, 3, 2, 4, 5).reshape(B, H * T, W * T, 3)

    def blit_triton(self, idx, out=None):
        from mg_triton import blit as _blit
        return _blit(self.atlas_w, idx, self.T, out=out)


# ---------------------------------------------------------------- state plumbing

def env_state(env):
    """Extract the minimal state from a live MiniGrid env."""
    u = env.unwrapped
    carrying = u.carrying.encode() if u.carrying is not None else EMPTY_CODE
    return {
        "code": u.grid.encode(),
        "agent_pos": np.asarray(u.agent_pos, dtype=np.int64),
        "agent_dir": int(u.agent_dir),
        "carrying": np.asarray(carrying, dtype=np.uint8),
        "view_size": int(u.agent_view_size),
        "see_through_walls": bool(u.see_through_walls),
    }


def restore_state(env, st):
    """Rebuild a live env from the state alone (used to prove state sufficiency)."""
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import WorldObj

    u = env.unwrapped
    grid, _ = Grid.decode(np.asarray(st["code"]))
    u.grid = grid
    u.width, u.height = grid.width, grid.height
    u.agent_pos = tuple(int(x) for x in st["agent_pos"])
    u.agent_dir = int(st["agent_dir"])
    cc = tuple(int(x) for x in st["carrying"])
    u.carrying = None if cc[0] in (0, 1, 10) else WorldObj.decode(*cc)
    u.agent_view_size = int(st["view_size"])
    u.see_through_walls = bool(st["see_through_walls"])
    return env


def batch_states(states, device="cuda"):
    code = torch.from_numpy(np.stack([s["code"] for s in states])).to(device)
    pos = torch.from_numpy(np.stack([s["agent_pos"] for s in states])).to(device)
    dirs = torch.tensor([s["agent_dir"] for s in states], device=device)
    car = torch.from_numpy(np.stack([s["carrying"] for s in states])).to(device)
    return code, pos, dirs, car
