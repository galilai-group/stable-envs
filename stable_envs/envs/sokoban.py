from __future__ import annotations

import importlib.util
import os
import random

import numpy as np
import torch
from tensordict import TensorDict

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import tilemap as tm
from ..spaces import Dict, RGBBox

# Tile order must match the source room_to_rgb surfaces.
TILE_NAMES = ("wall", "floor", "box_target", "box_on_target", "box", "player", "player_on_target")
N_TILE = len(TILE_NAMES)
TILE_PX = 16

# Tile ID → (occupancy, fixed terrain), including the target beneath a player.
ROOM_STATE_OF = (0, 1, 2, 3, 4, 5, 5)
ROOM_FIXED_OF = (0, 1, 2, 2, 1, 1, 2)

# Distinct source colours need separate labels even when their shades are similar.
PALETTE_NAMES = (
    "background",
    "player",
    "wall_brick",
    "wall_mortar",
    "box_dark",
    "box_mid",
    "box_light",
    "target_dark",
    "target_mid",
    "target_light",
)
OFFICIAL_PALETTE = (
    (0, 0, 0),
    (41, 202, 26),
    (176, 61, 0),
    (176, 154, 142),
    (215, 103, 0),
    (215, 114, 0),
    (215, 196, 0),
    (238, 0, 0),
    (239, 22, 22),
    (242, 26, 26),
)
N_LABEL = len(PALETTE_NAMES)

#: ``CHANGE_COORDINATES``: action-1 mod 4 -> (d_row, d_col). Up, down, left, right.
CHANGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
N_ACTIONS = 9

ROOM_H, ROOM_W = 10, 10
N_CELLS = ROOM_H * ROOM_W
CELL_NAMES = tuple(f"cell[{r},{c}]" for r in range(ROOM_H) for c in range(ROOM_W))


def _tile_dir() -> str:
    """Locate the source PNGs without executing legacy Gym registration."""
    spec = importlib.util.find_spec("gym_sokoban")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a "
            "gather from them and nothing here re-derives them -- so there is no degraded mode to fall "
            "back to. `pip install gym-sokoban`."
        )
    return os.path.join(list(spec.submodule_search_locations)[0], "envs", "surface")


def _build_atlas():
    """Load the seven source tiles and verify their ten-colour label decomposition."""
    import imageio.v2 as imageio

    root = _tile_dir()
    missing = [n for n in TILE_NAMES if not os.path.exists(os.path.join(root, f"{n}.png"))]
    if missing:
        raise ImportError(
            f"gym-sokoban is installed at {root} but {len(missing)} of its seven tile images are "
            f"missing ({', '.join(missing)}). The frame is a gather from those files; a wheel that "
            f"ships the code without the art cannot render this env."
        )
    rgb = np.stack([np.asarray(imageio.imread(os.path.join(root, f"{n}.png")))[..., :3] for n in TILE_NAMES]).astype(
        np.uint8
    )
    pal = np.asarray(OFFICIAL_PALETTE, np.uint8)
    labels = np.full(rgb.shape[:3], 255, np.uint8)
    for k, c in enumerate(pal):
        labels[(rgb == c).all(-1)] = k
    if (labels == 255).any():
        found = np.unique(rgb.reshape(-1, 3), axis=0)
        raise AssertionError(
            f"the shipped tiles use colours OFFICIAL_PALETTE does not name: the atlas has "
            f"{len(found)} distinct colours and the palette names {len(pal)}. Every recoloured "
            f"frame would be wrong in a way no parity test would catch, because parity is measured "
            f"at the default palette where the label map is unused."
        )
    if not np.array_equal(pal[labels], rgb):
        raise AssertionError("palette[labels] does not reproduce the official tiles")
    return rgb, labels


def _check_sokoban_works() -> None:
    """Check assets and source construction before registering the environment.

    Room generation is deferred; missing or incompatible dependencies raise ImportError."""
    _build_atlas()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the legacy gym deprecation banner
        try:
            from gym_sokoban.envs.sokoban_env import SokobanEnv

            SokobanEnv(dim_room=(ROOM_H, ROOM_W), num_boxes=4, reset=False)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                f"gym-sokoban is installed but SokobanEnv does not construct: {type(e).__name__}: "
                f"{e}. It installs against the legacy `gym`, which does not support NumPy 2.0, so a "
                f"skew there presents here."
            ) from e


_check_sokoban_works()


class _Atlas:
    """Cache source RGB tiles and palette labels per device."""

    _mem: dict[str, _Atlas] = {}

    def __init__(self, device="cuda"):
        rgb, labels = _build_atlas()
        self.device = torch.device(device)
        self.rgb = torch.from_numpy(rgb).to(self.device)  # (7,16,16,3) uint8
        self.labels = torch.from_numpy(labels).to(self.device)  # (7,16,16) uint8

    @classmethod
    def get(cls, device="cuda") -> _Atlas:
        key = str(device)
        if key not in cls._mem:
            cls._mem[key] = cls(device)
        return cls._mem[key]


@register
class Sokoban(Env):
    """Render and advance 10×10 Sokoban rooms as seven-valued tile grids.

    Each tile encodes both the occupant and underlying target. This is enough for
    push/move transitions; reward, termination and episode counters are not modelled.
    Source rendering uses 16-pixel tiles; larger sizes replicate those pixels."""

    family = "sokoban"
    domain = "sokoban"

    name = "sokoban"
    reference_id = "Sokoban-v0"
    #: ``SokobanEnv`` defaults; the room shape fixes ``state_dim`` and the frame size
    ROOM_H, ROOM_W = ROOM_H, ROOM_W
    NUM_BOXES = 4
    MAX_STEPS = 120
    default_size = ROOM_H * TILE_PX
    state_dim = N_CELLS
    state_names = CELL_NAMES
    # The tile grid contains all inputs needed by the push/move transition.
    state_layout = ((("cells",), N_CELLS, 0),)

    # 0: no-op; 1–4: push up/down/left/right; 5–8: move in the same order.
    action_dim = 1
    action_low = 0.0
    action_high = float(N_ACTIONS)
    step_reference_checkable = True
    step_reference_exact = True
    step_reference_hold = 1
    step_reference_groups = (("cells", tuple(range(N_CELLS)), "abs"),)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.atlas = _Atlas.get(device=self.device)
        self._pool: np.ndarray | None = None
        self._pool_has: np.ndarray | None = None
        self._default_t: torch.Tensor | None = None
        self._rgb_cache = _RES.BoundedCache(8, name="sokoban.rgb")
        dev = self.device
        self._state_of = torch.tensor(ROOM_STATE_OF, dtype=torch.long, device=dev)
        self._fixed_of = torch.tensor(ROOM_FIXED_OF, dtype=torch.long, device=dev)
        self._change = torch.tensor(CHANGE, dtype=torch.long, device=dev)
        self._act = torch.zeros(self.B, self.action_dim, device=dev, dtype=torch.float32)

    def _scale(self, size: int) -> int:
        """Require an output size that is a positive multiple of the source resolution."""
        t, r = divmod(int(size), self.ROOM_H)
        k, r2 = divmod(t, TILE_PX)
        if r or r2 or k < 1:
            raise ValueError(
                f"Sokoban renders at whole multiples of its 16 px art only: size must be "
                f"{self.ROOM_H * TILE_PX} x k for an integer k >= 1, got {size}"
            )
        return k

    def make_variation_space(self) -> Dict:
        """Expose the ten atlas colours as independent RGB leaves."""
        return Dict({"palette": Dict({n: RGBBox(init_value=c) for n, c in zip(PALETTE_NAMES, OFFICIAL_PALETTE)})})

    def _palette(self, sty) -> torch.Tensor:
        """Return per-row palettes with shape ``(B, 10, 3)``."""
        return torch.stack([torch.trunc(sty[f"palette.{n}"]) for n in PALETTE_NAMES], 1)

    def _is_default(self, sty) -> bool:
        if self._default_t is None:
            self._default_t = torch.as_tensor(self.default_style(), device=self.device, dtype=torch.float32)
        return bool((sty.values[0] == self._default_t).all())

    def _shared_rgb(self, sty, k: int) -> torch.Tensor:
        row = sty.values[0]
        key = np.ascontiguousarray(row.detach().cpu().numpy()).tobytes() + bytes([k])
        hit = self._rgb_cache.get(key)
        if hit is not None:
            return hit
        base = (
            self.atlas.rgb
            if self._is_default(sty)
            else self._palette(sty)[0].to(torch.uint8)[self.atlas.labels.long()]
        )
        out = (base if k == 1 else base.repeat_interleave(k, 1).repeat_interleave(k, 2)).contiguous()
        return self._rgb_cache.put(key, out)

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
        """Gather tile grids into uint8 frames, preserving leading batch dimensions."""
        if supersample not in (None, 1):
            raise ValueError(
                f"Sokoban: supersample={supersample} is not available. The frame is a gather from "
                f"16x16 bitmaps, so there is no coverage to sample twice -- ask for a larger `size`, "
                f"which replicates each art pixel k x k exactly."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        k = self._scale(int(size or self.size))
        T = TILE_PX * k
        sty = self.resolve_style(style, variation, batch=int(s.shape[0]))
        idx = self._tile_indices(s)
        flat = None
        if out is not None:
            flat = out.reshape(-1, self.ROOM_H * T, self.ROOM_W * T, 3)
        if sty.shared:
            img = tm.blit(self._shared_rgb(sty, k), idx, T, out=flat)
        else:
            img = self._blit_styled(idx, sty, k, flat)
        return img.reshape(*lead, *img.shape[1:]) if lead else img

    def _blit_styled(self, idx: torch.Tensor, sty, k: int, out) -> torch.Tensor:
        """Gather labels before recolouring so each row can use its own palette."""
        B, H, W = idx.shape
        lab = self.atlas.labels.long()[idx.long()]  # (B,H,W,16,16)
        lab = lab.permute(0, 1, 3, 2, 4).reshape(B, H * TILE_PX, W * TILE_PX)
        pal = self._palette(sty).to(torch.uint8)  # (B,10,3)
        img = pal[torch.arange(B, device=idx.device).view(B, 1, 1), lab]
        if k > 1:
            img = img.repeat_interleave(k, 1).repeat_interleave(k, 2)
        img = img.contiguous()
        if out is not None:
            out.copy_(img)
            return out
        return img

    def _tile_indices(self, s: torch.Tensor) -> torch.Tensor:
        """Round and wrap cell codes into the seven-tile alphabet."""
        idx = torch.round(s).long() % N_TILE
        return idx.reshape(-1, self.ROOM_H, self.ROOM_W).to(torch.int32)

    def _advance(self) -> None:
        self._advance_into(self._render_state, self._act)

    def _step_state(self) -> TensorDict:
        """Clone live cell rows so later steps cannot mutate the returned snapshot."""
        n = self._live_rows
        return TensorDict(
            {"cells": self._render_state[:n].clone()},
            batch_size=[n],
            device=self.device,
        )

    def _advance_into(self, s, act) -> None:
        """Apply push/move transitions in place to ``s``.

        Out-of-grid cells act as walls. Generated source rooms have a wall border,
        so this extension affects only synthetic boundary states."""
        H, W = self.ROOM_H, self.ROOM_W
        a = act[:, 0].floor().clamp(0, N_ACTIONS - 1).long()
        tile = torch.round(s).long() % N_TILE
        st = self._state_of[tile]  # room_state
        fx = self._fixed_of[tile]  # room_fixed

        # the player is wherever room_state == 5; argmax picks the first, exactly as
        # ``np.argwhere(self.room_state == 5)[0]`` does
        p = (st == 5).to(torch.float32).argmax(dim=1)
        pr, pc = p // W, p % W
        d = self._change[(a - 1).clamp_min(0) % 4]
        nr, nc = pr + d[:, 0], pc + d[:, 1]
        br, bc = nr + d[:, 0], nc + d[:, 1]
        n_in = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        b_in = (br >= 0) & (br < H) & (bc >= 0) & (bc < W)
        ni = (nr.clamp(0, H - 1) * W + nc.clamp(0, W - 1)).unsqueeze(1)
        bi = (br.clamp(0, H - 1) * W + bc.clamp(0, W - 1)).unsqueeze(1)
        z = torch.zeros_like(pr)
        # out of bounds behaves as a wall: state 0, which is in neither [3,4] nor [1,2]
        n_st = torch.where(n_in, st.gather(1, ni).squeeze(1), z)
        n_fx = torch.where(n_in, fx.gather(1, ni).squeeze(1), z)
        b_st = torch.where(b_in, st.gather(1, bi).squeeze(1), z)
        b_fx = torch.where(b_in, fx.gather(1, bi).squeeze(1), z)

        push = (a >= 1) & (a <= 4)
        move = a >= 5
        can_push = push & ((n_st == 3) | (n_st == 4)) & ((b_st == 1) | (b_st == 2))
        # ``_push`` falls back to ``_move`` when there is no box to push; ``_move`` steps onto floor
        # or an empty target only
        walk = (move | (push & ~can_push)) & ((n_st == 1) | (n_st == 2))

        stepped = can_push | walk
        pi = (pr * W + pc).unsqueeze(1)
        # the cell the player leaves reverts to room_fixed, and its tile id IS that value: 0 wall,
        # 1 floor, 2 empty target are the same three numbers in both alphabets
        left = fx.gather(1, pi).squeeze(1)
        enter = torch.where(n_fx == 2, torch.full_like(pr, 6), torch.full_like(pr, 5))
        box = torch.where(b_fx == 2, torch.full_like(pr, 3), torch.full_like(pr, 4))

        out = tile
        out = out.scatter(1, bi, torch.where(can_push, box, out.gather(1, bi).squeeze(1)).unsqueeze(1))
        out = out.scatter(1, ni, torch.where(stepped, enter, out.gather(1, ni).squeeze(1)).unsqueeze(1))
        out = out.scatter(1, pi, torch.where(stepped, left, out.gather(1, pi).squeeze(1)).unsqueeze(1))
        s.copy_(out.to(torch.float32))

    @torch.no_grad()
    def _step(self, action) -> TensorDict:
        """Advance cells using floored, clamped ``(B, 1)`` actions.

        Return State only; reward and termination are neither computed nor reported."""
        self._act.copy_(
            torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, self.action_dim)
        )
        self._advance()
        return self._step_result(self._step_state())


    def _close(self) -> None:
        """Release instance caches while retaining the shared tile atlas."""
        self._pool = None
        self._pool_has = None
        self._default_t = None
        self._rgb_cache.clear()

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw source rollout states, trying to cover every tile when batch ≥ 7.

        Coverage makes colour checks sensitive to palette entries rare in random play."""
        rng = rng or self.rng
        if self._pool is None:
            self._build_pool(512)
        pool, has = self._pool, self._pool_has
        idx = rng.choice(len(pool), batch, replace=batch > len(pool))
        if batch >= N_TILE:
            for _ in range(N_TILE + 1):
                missing = [t for t in range(N_TILE) if not has[idx, t].any()]
                if not missing:
                    break
                for t in missing:
                    cand = np.flatnonzero(has[:, t])
                    idx[rng.integers(0, batch)] = cand[rng.integers(0, len(cand))]
        return pool[idx].astype(np.float32)

    def _build_pool(self, n: int) -> None:
        """Build a deterministic rollout pool and record which tiles each state contains.

        Room generation uses both Python and NumPy global RNGs. Seed both per episode
        and restore their saved states after successful pool construction."""
        u = self._official()
        py_state, np_state = random.getstate(), np.random.get_state()
        out = []
        ep = 0
        while len(out) < n:
            random.seed(ep)
            np.random.seed(ep)
            u.reset()
            r = np.random.default_rng(ep)
            for _ in range(24):
                out.append(self._tiles_of(u).reshape(-1))
                if len(out) >= n:
                    break
                # Source primitives avoid rendering an unused observation.
                self.reference_step(u, [r.integers(0, N_ACTIONS)])
            ep += 1
        random.setstate(py_state)
        np.random.set_state(np_state)
        self._pool = np.stack(out).astype(np.float32)
        self._pool_has = np.stack([(self._pool == t).any(1) for t in range(N_TILE)], 1)

    @classmethod
    def _official(cls):
        """Construct the source directly; its reset signature lacks Gym’s seed argument."""
        from gym_sokoban.envs.sokoban_env import SokobanEnv

        return SokobanEnv(dim_room=(cls.ROOM_H, cls.ROOM_W), num_boxes=cls.NUM_BOXES, max_steps=cls.MAX_STEPS)

    @staticmethod
    def _tiles_of(u) -> np.ndarray:
        """Encode a player on a target as tile 6; preserve other source tile IDs."""
        t = np.array(u.room_state, np.int64).copy()
        t[(np.array(u.room_state) == 5) & (np.array(u.room_fixed) == 2)] = 6
        return t

    @classmethod
    def reference_env(cls, size: int | None = None):
        if size is not None and int(size) != cls.default_size:
            raise ValueError(
                f"the official renderer draws 16 px tiles and nothing else, so there is no reference "
                f"frame at size {size}; parity is measured at {cls.default_size}"
            )
        return cls._official()

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        """Decode tiles into source occupancy, fixed terrain and player position."""
        tile = np.round(np.asarray(st, np.float64)).astype(np.int64).reshape(cls.ROOM_H, cls.ROOM_W)
        tile %= N_TILE
        u.room_state = np.asarray(ROOM_STATE_OF, np.int64)[tile]
        u.room_fixed = np.asarray(ROOM_FIXED_OF, np.int64)[tile]
        where = np.argwhere(u.room_state == 5)
        u.player_position = where[0] if len(where) else np.array([0, 0])
        u.num_env_steps = 0
        u.boxes_on_target = 0

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Render source frames after verifying the requested tile grid was restored."""
        out = []
        for st in np.asarray(states):
            cls.reference_set_state(env, st)
            want = np.round(np.asarray(st, np.float64)).astype(np.int64) % N_TILE
            got = cls._tiles_of(env).reshape(-1)
            if not np.array_equal(got, want):
                bad = int((got != want).sum())
                raise AssertionError(
                    f"Sokoban: the reference did not reach the requested state ({bad} cells differ); "
                    f"parity measured against this frame would compare two scenes"
                )
            out.append(np.asarray(env.render(mode="rgb_array")).copy())
        return np.stack(out)

    @classmethod
    def reference_sim_state(cls, ref) -> np.ndarray:
        return cls._tiles_of(ref).reshape(-1).astype(np.float64)

    @classmethod
    def reference_step(cls, ref, a) -> None:
        """Call source push/move primitives, excluding rendering and episode bookkeeping."""
        act = int(min(max(int(np.floor(float(np.asarray(a).reshape(-1)[0]))), 0), N_ACTIONS - 1))
        if act == 0:
            return
        if act < 5:
            ref._push(act)
        else:
            ref._move(act)

    @classmethod
    def step_reference_reset(cls, ref, seed: int) -> np.ndarray:
        """Seed Python and NumPy global RNGs before generating a source room."""
        random.seed(int(seed))
        np.random.seed(int(seed))
        ref.reset()
        return cls.reference_sim_state(ref)
