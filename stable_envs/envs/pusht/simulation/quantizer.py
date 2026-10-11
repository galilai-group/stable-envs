"""Private source-aligned PushT task support; not a semantic promotion.

Adapted from stable-worldmodel (MIT), by Lucas Maes, Quentin Le Lidec,
Randall Balestriero and contributors, commit
21446f1ede6d5284e981bd7b47f432b994e6d812.
See THIRD_PARTY_NOTICES.md for attribution and the MIT license.
Physics remains official Pymunk 7.2.0 / Munk2D. No stable-worldmodel runtime import.
"""

# Upstream: envs/quantizer.py
# SHA256: 7167a833c4a1603e9912533d19a2fa1c8279eb12f235fd8e27cde9ff0a44bec7

from typing import Protocol

import numpy as np


class Quantizable(Protocol):
    """Protocol for quantizable action spaces."""

    def quantize(self, action: np.ndarray) -> np.ndarray: ...

    def dequantize(self, quantized_action: np.ndarray) -> np.ndarray: ...

    @property
    def action_shape(self) -> tuple[int, ...]: ...


class PolarQuantizer(Quantizable):
    def __init__(
        self,
        num_bins_radial: int,
        num_bins_angular: int,
        max_action_distance: float,
    ):
        self.num_bins_radial = num_bins_radial
        self.num_bins_angular = num_bins_angular
        self.max_action_distance = max_action_distance

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self.num_bins_radial, self.num_bins_angular)

    def quantize(self, action: np.ndarray) -> np.ndarray:
        dx, dy = action
        radius = np.linalg.norm(action)
        theta = np.arctan2(dy, dx) % (2 * np.pi)

        radius_bin = int(
            np.clip(
                radius / self.max_action_distance * self.num_bins_radial,
                0,
                self.num_bins_radial - 1,
            )
        )
        theta_bin = int(
            np.clip(
                theta / (2 * np.pi) * self.num_bins_angular,
                0,
                self.num_bins_angular - 1,
            )
        )
        return np.array([radius_bin, theta_bin], dtype=np.int32)

    def dequantize(self, quantized_action: np.ndarray) -> np.ndarray:
        radius_bin, theta_bin = quantized_action

        new_radius = (radius_bin + 0.5) / self.num_bins_radial * self.max_action_distance
        theta = (theta_bin + 0.5) / self.num_bins_angular * 2 * np.pi

        dx = new_radius * np.cos(theta)
        dy = new_radius * np.sin(theta)

        return np.array([dx, dy], dtype=np.float32)
