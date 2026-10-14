"""A tiny unregistered Env whose frames say which state row drew them."""

from __future__ import annotations

import numpy as np
import torch

from stable_envs import spaces
from stable_envs.env import Env as StableEnv

NAMES = ("agent_x", "agent_y", "block_x", "block_y", "block_angle")


class TinyEnv(StableEnv):
    """5-wide state, a 3x4 frame, one 3-wide appearance knob.

    Pixel (0,0) carries ``state[0]`` and pixel (0,1) the block colour, so "did row 3 get row 3's
    appearance" is an assertion on bytes. Not registered.
    """

    name = "tiny"
    state_dim = 5
    state_names = NAMES
    default_size = 4

    def make_variation_space(self):
        return spaces.Dict({"block": spaces.Dict({"color": spaces.RGBBox(init_value=(10, 20, 30))})})

    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        # Mirrors the unchecked as_tensor + reshape of shipped renders; the base-class hook validates first.
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        sty = self.resolve_style(style, variation, batch=int(s.shape[0]))
        img = torch.zeros(s.shape[0], 3, 4, 3, dtype=torch.uint8, device=self.device)
        img[:, 0, 0, 0] = s[:, 0].to(torch.uint8)
        img[:, 0, 1, :] = sty["block.color"].to(torch.uint8)
        return img.view(*lead, 3, 4, 3) if lead else img


def id_states(n: int, first: int = 0, width: int = 5) -> np.ndarray:
    """``(n, width)`` whose column 0 is the row's identity, so a frame says which state drew it."""
    out = np.zeros((n, width), np.float32)
    out[:, 0] = np.arange(first, first + n, dtype=np.float32)
    return out


def frame_ids(frames: torch.Tensor) -> list[int]:
    return frames[:, 0, 0, 0].tolist()
