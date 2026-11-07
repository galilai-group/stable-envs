"""Wind and rain: the car's airspeed and the road grip, plus the rain look.

Rain is a fixed pool of visual-only thin capsules that live in a box around the car and wrap around as the
car moves, so they stream past at the right relative speed; inactive drops go to geom group 4, which the
renderers do not draw. Rain also brings haze (MuJoCo's fog, scene flag mjRND_FOG, set by
`apply_render_flags`) and dims the sun.

Two things reach the physics: the wind (mean + Ornstein-Uhlenbeck gusts, read by the sim as the car's
airspeed for drag and crosswind yaw) and the tyre grip (a wet road). With no wind and no rain the physics
is that of calm, dry weather.
"""

from __future__ import annotations

import mujoco
import numpy as np

N_RAIN = 600
# box around the car (car frame axes = world axes), biased toward the camera side (-x, -y)
BOX_LO = np.array([-16.0, -16.0, -5.0])
BOX_HI = np.array([12.0, 8.0, 12.0])
RAIN_VEL = np.array([-1.5, 0.8, -14.0])  # m/s, before the wind drift
RAIN_RGBA = (0.75, 0.8, 0.9, 0.45)
RAIN_FOG = (0.55, 0.58, 0.63)  # fog and haze colour
RAIN_SUN = 0.45  # sun brightness in the heaviest rain


def xml() -> str:
    """Rain drop pool, attached to a static body at the origin (positions are world coordinates)."""
    vis = 'contype="0" conaffinity="0" group="4"'
    g = ['<body name="weather" pos="0 0 0">']
    for i in range(N_RAIN):
        g.append(
            f'<geom name="rain{i}" type="capsule" size="0.012 0.3" pos="0 0 -60" rgba="0.75 0.8 0.9 0.45" {vis}/>'
        )
    g.append("</body>")
    return "\n".join(g)


class Weather:
    """Wind (mean + gusts, with its own RNG) and rain (strength 0..1: drops, haze, sun and tyre grip).

    Lanes that share one compiled model each hold a ``Weather``; ``base`` (another lane's) gives the model's
    own friction, sun, haze and fog, and the lane writes its grip into the model with ``apply_grip`` before
    it steps.
    """

    def __init__(self, model: mujoco.MjModel, seed: int = 0, base: Weather | None = None):
        self.m = model
        tires = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tire_{w}") for w in ("fl", "fr", "rl", "rr")]
        self.tires = np.array([t for t in tires if t >= 0], dtype=int)
        self.tire_friction = (base.tire_friction if base else model.geom_friction[self.tires]).copy()
        self.rain_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"rain{i}") for i in range(N_RAIN)]
        )
        self.sun = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
        if base is not None:
            self.sun_diffuse, self.base_haze, self.base_fog = base.sun_diffuse, base.base_haze, base.base_fog
        else:
            self.sun_diffuse = model.light_diffuse[self.sun].copy() if self.sun >= 0 else None
            # the model's own haze and fog (the cinematic look fades to black with it), restored when the rain stops
            self.base_haze = model.vis.rgba.haze.copy()
            self.base_fog = (float(model.vis.map.fogstart), float(model.vis.map.fogend), model.vis.rgba.fog.copy())
        self.set(seed)

    def set(self, seed: int = 0, wind: float | tuple = 0.0, rain: float = 0.0):
        """Configure the weather of an episode.

        Args:
            seed: draws the wind direction, the gusts and the drop pool.
            wind: mean speed in m/s (gusts ~30 % of it) or a (vx, vy) vector.
            rain: strength 0..1 (drops, haze and tyre grip scale with it).
        """
        self.seed = int(seed)
        self.rain = float(np.clip(rain, 0, 1))
        self.gust_std = 0.0
        self.grip = 1.0 - 0.45 * self.rain if self.rain > 0 else 1.0  # wet road: 0.55 x grip in the heaviest rain
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, 2 * np.pi)  # always drawn, so the drop offsets below depend on the seed only
        if np.ndim(wind) == 1:
            self.wind_mean = np.array([wind[0], wind[1], 0.0], float)
        else:
            self.gust_std = 0.3 * float(wind)
            self.wind_mean = float(wind) * np.array([np.cos(ang), np.sin(ang), 0.0])
        self.gust = np.zeros(3)
        self._wind_rng = np.random.default_rng(seed + 1)
        self.apply_grip()
        self.offset_r = rng.uniform(BOX_LO, BOX_HI, (N_RAIN, 3))
        m = self.m
        if self.sun >= 0:
            m.light_diffuse[self.sun] = self.sun_diffuse * (1 - (1 - RAIN_SUN) * self.rain)
        if self.rain > 0:  # haze grows with rain strength
            k = self.rain
            m.vis.map.fogstart, m.vis.map.fogend = 0.25 + (1 - k) * 1.5, 2.2 + (1 - k) * 4.0
            m.vis.rgba.fog[:3] = RAIN_FOG
            m.vis.rgba.haze[:3] = RAIN_FOG
        else:
            m.vis.rgba.haze[:] = self.base_haze
            m.vis.map.fogstart, m.vis.map.fogend, m.vis.rgba.fog[:] = self.base_fog
        m.geom_group[self.rain_ids] = 4
        m.geom_group[self.rain_ids[: int(round(N_RAIN * self.rain))]] = 1  # visible drops scale with the strength
        m.geom_rgba[self.rain_ids] = RAIN_RGBA

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        """Everything the wind model needs to continue: configuration, gust state and its RNG."""
        return {
            "seed": self.seed,
            "rain": self.rain,
            "wind_mean": self.wind_mean.copy(),
            "gust": self.gust.copy(),
            "gust_std": float(self.gust_std),
            "grip": float(self.grip),
            "rng": self._wind_rng.bit_generator.state,
        }

    def set_state(self, st: dict):
        """Restore a `get_state` snapshot.

        The drop pool (from the seed), the mean wind, gusts, tyre grip and the gust RNG.
        """
        same = self.seed == st["seed"] and self.rain == st["rain"] and np.array_equal(self.wind_mean, st["wind_mean"])
        if not same:
            self.set(st["seed"], wind=tuple(st["wind_mean"][:2]), rain=st["rain"])
        self.wind_mean = np.array(st["wind_mean"], float)
        self.gust = np.array(st["gust"], float)
        self.gust_std = float(st["gust_std"])
        self.grip = float(st["grip"])
        self.apply_grip()
        self._wind_rng.bit_generator.state = st["rng"]

    def apply_grip(self):
        """Write this lane's tyre grip into the (possibly shared) model."""
        self.m.geom_friction[self.tires] = self.tire_friction * np.array([self.grip, 1.0, 1.0])

    @property
    def fog(self) -> bool:
        return self.rain > 0

    @property
    def wind(self) -> np.ndarray:
        """Current wind velocity (world frame, m/s)."""
        return self.wind_mean + self.gust

    def step_wind(self, dt: float, tau: float = 1.5):
        if self.gust_std > 0:
            a = np.exp(-dt / tau)
            noise = self._wind_rng.standard_normal(3) * np.array([1, 1, 0.3])
            self.gust = a * self.gust + np.sqrt(1 - a * a) * self.gust_std * noise

    def update(self, t: float, car_pos: np.ndarray):
        """Move the drops for time t (s); they wrap inside the box around the car."""
        if self.rain <= 0:
            return
        vel = RAIN_VEL + 0.8 * self.wind_mean  # drops drift with the wind
        p = self.offset_r + vel * t - car_pos
        self.m.geom_pos[self.rain_ids] = BOX_LO + np.mod(p - BOX_LO, BOX_HI - BOX_LO) + car_pos
        # streaks aligned with the fall direction
        d = (vel + self.gust * 0.5) / np.linalg.norm(vel + self.gust * 0.5)
        z = np.array([0, 0, 1.0])
        ax = np.cross(z, d)
        ang = np.arccos(np.clip(z @ d, -1, 1))
        ax = ax / (np.linalg.norm(ax) + 1e-9)
        self.m.geom_quat[self.rain_ids] = np.r_[np.cos(ang / 2), np.sin(ang / 2) * ax]


def apply_render_flags(scene: mujoco.MjvScene, weather: Weather | None, shadows: bool = False):
    """Shadow maps (off for the world-model camera: a blob shadow under the car stands in) and the rain haze."""
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
    scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = bool(weather is not None and weather.fog)
