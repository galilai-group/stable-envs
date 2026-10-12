from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.render.raster2d import render as render_scene
from stable_envs.render.scene import Scene2D, SceneBuilder
from stable_envs.spaces import Box, Dict, Discrete, RGBBox

GREY = (100, 100, 100)  # WorldSurface.GREY, the road plane and the whole background
WHITE = (255, 255, 255)  # lane markings
BLACK = (60, 60, 60)  # VehicleGraphics.BLACK, the 1 px vehicle outline
BLUE = (100, 200, 255)  # IDMVehicle
GREEN = (50, 200, 0)  # EGO_COLOR (MDPVehicle)
YELLOW = (200, 200, 0)  # DEFAULT_COLOR, and Obstacle
RED = (255, 100, 100)  # crashed


def lighten(c, ratio: float = 0.68):
    """Compute source headlight colour by integer truncation and saturation."""
    return tuple(min(int(v / ratio), 255) for v in c)


STRIPE_SPACING = 4.33
STRIPE_LENGTH = 3.0
STRIPE_WIDTH = 0.3
NONE, STRIPED, CONTINUOUS, CONTINUOUS_LINE = 0, 1, 2, 3


class _Lane:
    def __init__(self, width, line_types):
        self.width = float(width)
        self.line_types = tuple(int(t) for t in line_types)


class Straight(_Lane):
    def __init__(self, start, end, width=4.0, line_types=(STRIPED, STRIPED)):
        super().__init__(width, line_types)
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        d = self.end - self.start
        self.length = float(np.linalg.norm(d))
        self.direction = d / self.length
        self.lateral = np.array([-self.direction[1], self.direction[0]])

    def local_s(self, p: torch.Tensor) -> torch.Tensor:
        d = p - torch.as_tensor(self.start, device=p.device, dtype=p.dtype)
        return d[:, 0] * float(self.direction[0]) + d[:, 1] * float(self.direction[1])

    def position(self, s: torch.Tensor, lat: float | torch.Tensor) -> torch.Tensor:
        st = torch.as_tensor(self.start, device=s.device, dtype=s.dtype)
        dr = torch.as_tensor(self.direction, device=s.device, dtype=s.dtype)
        lt = torch.as_tensor(self.lateral, device=s.device, dtype=s.dtype)
        lat = torch.as_tensor(lat, device=s.device, dtype=s.dtype)
        if lat.ndim:
            lat = lat.unsqueeze(-1)
        return st + s.unsqueeze(-1) * dr + lat * lt


class Sine(Straight):
    """Straight lane with a sinusoidal lateral offset."""

    def __init__(self, start, end, amplitude, pulsation, phase, width=4.0, line_types=(STRIPED, STRIPED)):
        super().__init__(start, end, width, line_types)
        self.amplitude, self.pulsation, self.phase = float(amplitude), float(pulsation), float(phase)

    def position(self, s, lat):
        lat = torch.as_tensor(lat, device=s.device, dtype=s.dtype)
        return super().position(s, lat + self.amplitude * torch.sin(self.pulsation * s + self.phase))


class Circular(_Lane):
    def __init__(
        self, center, radius, start_phase, end_phase, clockwise=True, width=4.0, line_types=(STRIPED, STRIPED)
    ):
        super().__init__(width, line_types)
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.start_phase, self.end_phase = float(start_phase), float(end_phase)
        self.dir = 1.0 if clockwise else -1.0
        self.length = self.radius * (self.end_phase - self.start_phase) * self.dir

    def local_s(self, p):
        c = torch.as_tensor(self.center, device=p.device, dtype=p.dtype)
        d = p - c
        phi = torch.atan2(d[:, 1], d[:, 0])
        # utils.wrap_to_pi, so the branch cut sits at the lane's own start phase
        phi = self.start_phase + torch.remainder(phi - self.start_phase + math.pi, 2 * math.pi) - math.pi
        return self.dir * (phi - self.start_phase) * self.radius

    def position(self, s, lat):
        lat = torch.as_tensor(lat, device=s.device, dtype=s.dtype)
        phi = self.dir * s / self.radius + self.start_phase
        r = self.radius - lat * self.dir
        c = torch.as_tensor(self.center, device=s.device, dtype=s.dtype)
        return c + torch.stack([r * torch.cos(phi), r * torch.sin(phi)], -1)


def straight_road_network(lanes: int, length: float = 10000.0, width: float = 4.0):
    """Build parallel lanes with the source highway boundary markings."""
    out = []
    for i in range(lanes):
        origin = (0.0, i * width)
        end = (length, i * width)
        line = (CONTINUOUS_LINE if i == 0 else STRIPED, CONTINUOUS_LINE if i == lanes - 1 else NONE)
        out.append(Straight(origin, end, width, line))
    return out


def _py_floordiv(a: torch.Tensor, b: float) -> torch.Tensor:
    """Match CPython float floor division when locating dashed stripes.

    The fmod-based correction differs from floor(a / b) near integer quotients;
    a one-unit error selects another stripe."""
    mod = torch.fmod(a, b)
    div = (a - mod) / b
    adj = (mod != 0) & ((b < 0) != (mod < 0))
    div = div - adj.to(div.dtype)
    fl = torch.floor(div)
    fl = fl + ((div - fl) > 0.5).to(div.dtype)
    return torch.where(div == 0, torch.zeros_like(div), fl)


@dataclass(frozen=True)
class _Screen:
    """Immutable geometry for one render size; previews must not mutate env defaults."""

    H: int
    W: int
    cam_off: tuple[float, float]
    stripes_count: int
    line_w: int


class HighwayEnvBase(Env):
    """Render fixed highway-env road networks and a declared vehicle count.

    Rows contain vehicle poses, visible crash flags and optional steering/object
    poses. They omit traffic policy and simulator state; stepping is unsupported.
    Rendering follows source painter order: road, objects, then vehicles."""

    family = "highway"

    #: reference screen, in the same (width, height) order highway-env's config uses
    SCREEN_W, SCREEN_H = 600, 150
    SCALING = 5.5
    CENTERING = (0.3, 0.5)
    #: lanes of the road network, in the order highway-env walks ``network.graph``
    LANES: tuple = ()
    #: number of vehicles, in the reference's draw order (``road.vehicles``)
    N_VEHICLES = 12
    # Ego index determines camera tracking and overlap order; intersection draws it last.
    EGO_INDEX = 0
    #: static road furniture: ``(kind, x, y, heading, length, width)``, kind 'obstacle' or 'landmark'
    OBJECTS: tuple = ()
    # State-driven (kind, length, width) objects precede static OBJECTS.
    # Their (x, y, heading) rows follow vehicles and optional ego steering.
    STATE_OBJECTS: tuple = ()
    VEHICLE_LENGTH, VEHICLE_WIDTH = 5.0, 2.0
    TIRE_LENGTH, TIRE_WIDTH = 1.0, 0.3
    # Only plain Vehicle/BicycleVehicle types draw tyres and expose visible steering.
    EGO_HAS_TIRES = False
    # Plain unpinned vehicles use yellow; Parking explicitly pins its ego green.
    EGO_COLOR_DEFAULT = GREEN
    # A pinned vehicle colour overrides crashed, making the crash flag invisible.
    CRASHED_VISIBLE = True
    #: ``ContinuousAction``'s ``steering_range``, for ``sample_states``
    STEERING_RANGE = math.pi / 4
    #: gymnasium id and the config overrides that make the reference match this class
    reference_id: str | None = None
    reference_config: dict = {}

    def __init__(self, num_envs: int = 1, size: int | None = None, **kw):
        n_obj = len(self.STATE_OBJECTS)
        self.veh_dim = 4 if self.CRASHED_VISIBLE else 3
        self.steer_off = self.veh_dim * self.N_VEHICLES
        self.obj_off = self.steer_off + (1 if self.EGO_HAS_TIRES else 0)
        self.state_dim = self.obj_off + 3 * n_obj
        fields = ("x", "y", "heading", "crashed")[: self.veh_dim]
        names = [f"{'ego' if i == self.EGO_INDEX else f'v{i}'}_{f}" for i in range(self.N_VEHICLES) for f in fields]
        if self.EGO_HAS_TIRES:
            names.append("ego_steering")
        for k, (kind, _, _) in enumerate(self.STATE_OBJECTS):
            names += [f"{kind}{k}_x", f"{kind}{k}_y", f"{kind}{k}_heading"]
        self.state_names = tuple(names)
        # Group vehicle poses, optional steering and object poses in their packed-row order.
        self.state_layout = (
            ((("vehicles",), self.veh_dim * self.N_VEHICLES, 0),)
            + (((("ego", "steering"), 1, self.steer_off),) if self.EGO_HAS_TIRES else ())
            + (((("objects",), 3 * n_obj, self.obj_off),) if n_obj else ())
        )
        # Lane-plan slot counts depend on output geometry; cache separately per size.
        self._plan_cache: dict = {}
        self._obj_runs = None
        super().__init__(num_envs=num_envs, size=size, **kw)

    def _close(self) -> None:
        """Release instance lane plans and object tables."""
        self._plan_cache = {}
        self._obj_runs = None

    def screen(self, size: int | None = None) -> _Screen:
        """Derive output geometry without changing instance defaults.

        Preserve float64 camera arithmetic order: source pixel conversion truncates
        values near integer boundaries."""
        H = int(self.size if size is None else size)
        W = int(round(H * self.SCREEN_W / self.SCREEN_H))
        sc = self.SCALING
        return _Screen(
            H=H,
            W=W,
            cam_off=(float(self.CENTERING[0] * W) / sc, float(self.CENTERING[1] * H) / sc),
            stripes_count=int(2 * (H + W) / (STRIPE_SPACING * sc)),
            line_w=max(int(STRIPE_WIDTH * sc), 1),  # WorldSurface.pix(STRIPE_WIDTH), min 1
        )

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return (height, width), using the family’s fixed aspect ratio."""
        sc = self.screen()
        return int(sc.H), int(sc.W)

    @property
    def H(self) -> int:
        """Frame height at the configured default size."""
        return self.screen().H

    @property
    def W(self) -> int:
        """Frame width at the configured default size."""
        return self.screen().W

    def _pix(self, v: float) -> int:
        return int(v * self.SCALING)

    def _sprite(self, length: float, width: float, with_tires_pad: bool):
        """Build source pixel rectangles for a vehicle or road-object sprite."""
        sc = self.SCALING
        pix = lambda v: int(v * sc)  # WorldSurface.pix
        if with_tires_pad:
            span = length + 2 * self.TIRE_LENGTH
            S = pix(span)
            body = (pix(self.TIRE_LENGTH), pix(span / 2 - width / 2), pix(length), pix(width))
            hx = pix(self.TIRE_LENGTH + length - 0.72)
            hl = (hx, pix(span / 2 - (1.4 * width) / 3), pix(0.72), pix(0.6))
            hr = (hx, pix(span / 2 + (0.6 * width) / 5), pix(0.72), pix(0.6))
            return S, body, hl, hr
        S = pix(length)
        return S, (0, pix(length / 2 - width / 2), pix(length), pix(width)), None, None

    def make_variation_space(self) -> Dict:
        """Expose only appearance leaves used by the declared vehicles and road objects."""
        space: dict = {}
        if self.OBJECTS:
            space["object"] = Dict({"color": RGBBox(init_value=YELLOW)})
        if self.STATE_OBJECTS:
            space["landmark"] = Dict({"color": RGBBox(init_value=BLUE)})
        space["background"] = Dict({"color": RGBBox(init_value=GREY)})
        space["road"] = Dict(
            {
                "line_color": RGBBox(init_value=WHITE),
                "show_lines": Discrete(2, init_value=1),
            }
        )
        space["ego"] = Dict(
            {
                "color": RGBBox(init_value=self.EGO_COLOR_DEFAULT),
                "headlight_color": RGBBox(init_value=lighten(self.EGO_COLOR_DEFAULT)),
            }
        )
        if self.N_VEHICLES > 1:
            space["traffic"] = Dict(
                {"color": RGBBox(init_value=BLUE), "headlight_color": RGBBox(init_value=lighten(BLUE))}
            )
        if self.CRASHED_VISIBLE:
            space["crashed"] = Dict({"color": RGBBox(init_value=RED)})
        space["vehicle"] = Dict(
            {
                "outline_color": RGBBox(init_value=BLACK),
                "scale": Box(low=0.25, high=3.0, shape=(1,), init_value=[1.0]),
            }
        )
        return Dict(space)

    def scene(self, states, variation: dict | None = None, style=None, size: int | None = None) -> Scene2D:
        """Build screen-pixel geometry; ``size`` changes field of view at fixed scale.

        A larger frame reveals more road rather than resampling the same scene."""
        n = int(states.shape[0]) if hasattr(states, "shape") else len(states)
        return self._scene(states, self.resolve_style(style, variation, batch=n), self.screen(size))

    def _scene(self, states, sty, scr: _Screen) -> Scene2D:
        """Build from compact render rows, per-row style and immutable screen geometry."""
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32).reshape(-1, self.state_dim)
        B = s.shape[0]
        sb = SceneBuilder(B, device=self.device, vertices=4)

        # Camera. float64 throughout: pos2pix truncates, and the ego sits exactly on a pixel boundary.
        e0 = self.veh_dim * self.EGO_INDEX
        ego = s[:, e0 : e0 + 2].double()
        origin = torch.stack([ego[:, 0] - scr.cam_off[0], ego[:, 1] - scr.cam_off[1]], -1)

        self._add_lane_markings(sb, origin, sty, B, scr)
        # ``road.objects`` order: the state-driven furniture first (parking-v0 appends its goal
        # landmark before the walls), then the static furniture.
        for k, (kind, length, width) in enumerate(self.STATE_OBJECTS):
            o = self.obj_off + 3 * k
            self._add_boxes(
                sb,
                origin,
                s[:, o : o + 1].double(),
                s[:, o + 1 : o + 2].double(),
                s[:, o + 2 : o + 3],
                length,
                width,
                False,
                sty[f"{kind}.color"],
                None,
                sty,
                B,
                scr,
                cull=False,
            )
        # Keep static headings in float64: quarter-turn classification affects the sprite pivot.
        # Render rows retain float32, matching the source restoration path.
        for (length, width), poses in self._object_runs():
            p = torch.tensor(poses, device=self.device, dtype=torch.float64).unsqueeze(0)
            p = p.expand(B, len(poses), 3)
            self._add_boxes(
                sb,
                origin,
                p[..., 0],
                p[..., 1],
                p[..., 2],
                length,
                width,
                False,
                sty["object.color"],
                None,
                sty,
                B,
                scr,
                cull=False,
            )
        self._add_vehicles(sb, origin, s, sty, B, scr)
        return sb.build(background=sty["background.color"], world=(0.0, 0.0, float(scr.W), float(scr.H)), flip_y=False)

    def _pos2pix(self, origin: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Truncate world-to-screen coordinates toward zero in source arithmetic order."""
        return torch.trunc((p - origin) * self.SCALING)

    def _object_runs(self):
        """Group consecutive equal-sized objects without changing painter order."""
        if self._obj_runs is None:
            runs: list = []
            for _, x, y, heading, length, width in self.OBJECTS:
                if not runs or runs[-1][0] != (length, width):
                    runs.append(((length, width), []))
                runs[-1][1].append((float(x), float(y), float(heading)))
            self._obj_runs = runs
        return self._obj_runs

    def _plan(self, scr: _Screen) -> dict:
        """Cache per-stripe lane parameters and clipping slots for one output size.

        Tables preserve source draw order. Evaluate lane formulas in batches and select
        by lane type instead of regrouping primitives."""
        key = (scr.H, scr.W)
        hit = self._plan_cache.get(key)
        if hit is not None:
            return hit
        lanes = self.LANES
        circ = [isinstance(l, Circular) for l in lanes]
        sine = [isinstance(l, Sine) for l in lanes]

        def straight_of(l):
            """Use finite placeholders for circular lanes, whose straight results are discarded."""
            if isinstance(l, Circular):
                return (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 0.0, 0.0, 0.0
            return (
                tuple(l.start),
                tuple(l.direction),
                tuple(l.lateral),
                getattr(l, "amplitude", 0.0),
                getattr(l, "pulsation", 0.0),
                getattr(l, "phase", 0.0),
            )

        def circular_of(l):
            """Use radius one for straight lanes so the unused circular division stays finite."""
            if isinstance(l, Circular):
                return tuple(l.center), l.radius, l.start_phase, l.dir
            return (0.0, 0.0), 1.0, 0.0, 1.0

        st_p = [straight_of(l) for l in lanes]
        ci_p = [circular_of(l) for l in lanes]
        # groups: one per (lane, side) that draws DASHED stripes, i.e. that needs a slot window
        g_lane, g_len, g_nslots, g_half = [], [], [], []
        rows = []  # (lane, group or -1, j, lat, span, continuous)
        for li, lane in enumerate(lanes):
            n_slots = self._slots_for(lane, scr)
            for side in (0, 1):
                lt = lane.line_types[side]
                if lt == NONE:
                    continue
                lat = (side - 0.5) * lane.width
                if lt == CONTINUOUS_LINE:
                    rows.append((li, -1, 0.0, lat, 0.0, True))
                    continue
                gi = len(g_lane)
                g_lane.append(li)
                g_len.append(lane.length)
                g_nslots.append(float(n_slots))
                g_half.append(float(n_slots // 2))
                span = STRIPE_LENGTH if lt == STRIPED else STRIPE_SPACING
                rows += [(li, gi, float(j), lat, span, False) for j in range(n_slots)]
        lane_of = [r[0] for r in rows]
        t = lambda v, d=torch.float64: torch.tensor(v, device=self.device, dtype=d)  # noqa: E731
        sel = lambda seq, idx: [seq[i] for i in idx]  # noqa: E731
        plan = dict(
            S=len(rows),
            lane_of=torch.tensor(lane_of, device=self.device, dtype=torch.int64),
            grp_of=torch.tensor([max(r[1], 0) for r in rows], device=self.device, dtype=torch.int64),
            j_of=t([r[2] for r in rows]),
            lat=t([r[3] for r in rows]),
            span=t([r[4] for r in rows]),
            cont=t([r[5] for r in rows], torch.bool),
            len_s=t([lanes[i].length for i in lane_of]),
            # per-stripe copies of the lane parametrisation, so the hot path gathers nothing
            p_start=t(sel([p[0] for p in st_p], lane_of)),
            p_dir=t(sel([p[1] for p in st_p], lane_of)),
            p_lateral=t(sel([p[2] for p in st_p], lane_of)),
            p_amp=t(sel([p[3] for p in st_p], lane_of)),
            p_puls=t(sel([p[4] for p in st_p], lane_of)),
            p_phase=t(sel([p[5] for p in st_p], lane_of)),
            p_sine=t(sel(sine, lane_of), torch.bool),
            p_circ=t(sel(circ, lane_of), torch.bool),
            p_c=t(sel([p[0] for p in ci_p], lane_of)),
            p_rad=t(sel([p[1] for p in ci_p], lane_of)),
            p_sph=t(sel([p[2] for p in ci_p], lane_of)),
            p_cdir=t(sel([p[3] for p in ci_p], lane_of)),
            # per-lane, for local_s at the camera origin and centre
            l_start=t([p[0] for p in st_p]),
            l_dir=t([p[1] for p in st_p]),
            l_circ=t(circ, torch.bool),
            l_c=t([p[0] for p in ci_p]),
            l_rad=t([p[1] for p in ci_p]),
            l_sph=t([p[2] for p in ci_p]),
            l_cdir=t([p[3] for p in ci_p]),
            # per-group slot windows
            G=len(g_lane),
            g_lane=torch.tensor(g_lane or [0], device=self.device, dtype=torch.int64),
            g_len=t(g_len or [1.0]),
            g_nslots=t(g_nslots or [1.0]),
            g_half=t(g_half or [0.0]),
        )
        self._plan_cache[key] = plan
        return plan

    def _local_s(self, P: dict, p: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 2)`` points to longitudinal coordinates on every lane: ``(B, L)``."""
        d = p.unsqueeze(1) - P["l_start"]
        s_str = d[..., 0] * P["l_dir"][..., 0] + d[..., 1] * P["l_dir"][..., 1]
        dc = p.unsqueeze(1) - P["l_c"]
        phi = torch.atan2(dc[..., 1], dc[..., 0])
        # utils.wrap_to_pi, so the branch cut sits at the lane's own start phase
        phi = P["l_sph"] + torch.remainder(phi - P["l_sph"] + math.pi, 2 * math.pi) - math.pi
        return torch.where(P["l_circ"], P["l_cdir"] * (phi - P["l_sph"]) * P["l_rad"], s_str)

    def _stripe_pos(self, P: dict, s: torch.Tensor) -> torch.Tensor:
        """Evaluate each stripe’s lane position: ``(B, S)`` → ``(B, S, 2)``."""
        lat = P["lat"]
        lat_e = torch.where(P["p_sine"], lat + P["p_amp"] * torch.sin(P["p_puls"] * s + P["p_phase"]), lat)
        p_str = P["p_start"] + s.unsqueeze(-1) * P["p_dir"] + lat_e.unsqueeze(-1) * P["p_lateral"]
        phi = P["p_cdir"] * s / P["p_rad"] + P["p_sph"]
        r = P["p_rad"] - lat * P["p_cdir"]
        p_cir = P["p_c"] + torch.stack([r * torch.cos(phi), r * torch.sin(phi)], -1)
        return torch.where(P["p_circ"].unsqueeze(-1), p_cir, p_str)

    def _add_lane_markings(self, sb, origin, sty, B, scr: _Screen) -> None:
        P = self._plan(scr)
        if not P["S"]:
            return
        col = sty["road.line_color"]
        show = sty["road.show_lines"] > 0.5
        # window centre, so the slot window follows the camera rather than the reference's s0
        centre = origin + torch.tensor(
            [scr.W / 2.0 / self.SCALING, scr.H / 2.0 / self.SCALING], device=origin.device, dtype=origin.dtype
        )
        s_org = self._local_s(P, origin)  # (B,L)
        s_ctr = self._local_s(P, centre)
        s0 = (_py_floordiv(torch.trunc(s_org), STRIPE_SPACING) - scr.stripes_count // 2) * STRIPE_SPACING
        k0 = self._slot_window_all(P, s0, s_ctr)  # (B,G)

        s0s = s0.index_select(1, P["lane_of"])  # (B,S)
        k = k0.index_select(1, P["grp_of"]) + P["j_of"]
        cont = P["cont"]
        st = torch.where(cont, s0s, s0s + k * STRIPE_SPACING)
        en = torch.where(cont, (s0s + scr.stripes_count * STRIPE_SPACING) + STRIPE_LENGTH, st + P["span"])
        ok = show.unsqueeze(1) & (cont | ((k >= 0) & (k < scr.stripes_count)))

        st = torch.minimum(st.clamp_min(0.0), P["len_s"])
        en = torch.minimum(en.clamp_min(0.0), P["len_s"])
        ok = ok & ((en - st).abs() > 0.5 * STRIPE_LENGTH)  # LaneGraphics.draw_stripes
        o = origin.unsqueeze(1)
        a = (torch.trunc((self._stripe_pos(P, st) - o) * self.SCALING) + 0.5).float()
        b = (torch.trunc((self._stripe_pos(P, en) - o) * self.SCALING) + 0.5).float()
        S = P["S"]
        a, b, hw = self._line_geom(a.reshape(B * S, 2), b.reshape(B * S, 2), scr)
        sb.add_segments(a.reshape(B, S, 2), b.reshape(B, S, 2), hw.reshape(B, S), col, enabled=ok)

    def _slot_window_all(self, P: dict, s0, s_ctr) -> torch.Tensor:
        """Choose the first stripe slot for each lane-side group.

        Clamp short lanes into the window; projecting the camera onto a whole circle
        can otherwise put a visible short arc outside every selected slot."""
        s0g = s0.index_select(1, P["g_lane"])
        sctrg = s_ctr.index_select(1, P["g_lane"])
        k_first = torch.ceil(-s0g / STRIPE_SPACING) - 1
        k_last = torch.floor((P["g_len"] - s0g) / STRIPE_SPACING) + 1
        edge = k_last - P["g_nslots"] + 1
        lo = torch.minimum(k_first, edge)
        hi = torch.maximum(k_first, edge)
        k_cam = torch.round((sctrg - s0g) / STRIPE_SPACING) - P["g_half"]
        return torch.maximum(torch.minimum(k_cam, hi), lo)

    def _slots_for(self, lane, scr: _Screen) -> int:
        """Bound stripe slots by the visible screen span and the source marking window."""
        sc = self.SCALING
        if isinstance(lane, Straight):
            span = (scr.W * abs(lane.direction[0]) + scr.H * abs(lane.direction[1])) / sc
            if isinstance(lane, Sine):  # the ramp's tangent is not the chord's
                span = math.hypot(scr.W, scr.H) / sc
        else:
            span = math.hypot(scr.W, scr.H) / sc
        n = int((span + STRIPE_LENGTH) / STRIPE_SPACING) + 3
        return max(2, min(n, int(lane.length / STRIPE_SPACING) + 3, scr.stripes_count))

    def _line_geom(self, a: torch.Tensor, b: torch.Tensor, scr: _Screen):
        """Approximate pygame line coverage with scaled capsule widths.

        Scale by major-axis extent / segment length to match minor-axis thickness.
        Even widths shift +0.5 along the minor axis regardless of endpoint order."""
        e = b - a
        L = e.norm(dim=-1).clamp_min(1e-9)
        hw = (scr.line_w / 2.0) * e.abs().amax(dim=-1) / L
        if scr.line_w % 2 == 0:
            x_major = e[:, 0].abs() >= e[:, 1].abs()
            shift = torch.stack([torch.where(x_major, 0.0, 0.5), torch.where(x_major, 0.5, 0.0)], -1)
            a, b = a + shift, b + shift
        return a, b, hw

    def _add_vehicles(self, sb, origin, s, sty, B, scr: _Screen) -> None:
        ego_c = sty["ego.color"]
        ego_h = sty["ego.headlight_color"]
        oth_c = sty.get("traffic.color", ego_c)
        oth_h = sty.get("traffic.headlight_color", ego_h)
        crash = sty.get("crashed.color")
        # VehicleGraphics.lighten, in tensor form, so recolouring a crash also recolours its headlights
        crash_h = None if crash is None else torch.clamp(torch.trunc(crash / 0.68), max=255.0)
        v, n = self.veh_dim, self.N_VEHICLES
        cast = s[:, : v * n].reshape(B, n, v)
        ego = torch.zeros(n, dtype=torch.bool, device=self.device)
        ego[self.EGO_INDEX] = True
        ego = ego.view(1, n, 1)
        body = torch.where(ego, ego_c.unsqueeze(1), oth_c.unsqueeze(1)).expand(B, n, 3)
        head = torch.where(ego, ego_h.unsqueeze(1), oth_h.unsqueeze(1)).expand(B, n, 3)
        if crash is not None:
            crashed = (cast[..., 3] > 0.5).unsqueeze(-1)
            body = torch.where(crashed, crash.unsqueeze(1), body)
            head = torch.where(crashed, crash_h.unsqueeze(1), head)
        x, y, hd = cast[..., 0].double(), cast[..., 1].double(), cast[..., 2]
        if not self.EGO_HAS_TIRES:
            self._add_boxes(
                sb, origin, x, y, hd, self.VEHICLE_LENGTH, self.VEHICLE_WIDTH, True, body, head, sty, B, scr
            )
            return
        # Draw the ego’s tyres immediately after its body and before later vehicles.
        e = self.EGO_INDEX
        sl = lambda t, a, b: t[:, a:b]  # noqa: E731
        self._add_boxes(
            sb,
            origin,
            sl(x, 0, e + 1),
            sl(y, 0, e + 1),
            sl(hd, 0, e + 1),
            self.VEHICLE_LENGTH,
            self.VEHICLE_WIDTH,
            True,
            sl(body, 0, e + 1),
            sl(head, 0, e + 1),
            sty,
            B,
            scr,
            steering=s[:, self.steer_off],
            tire_slot=e,
        )
        if e + 1 < n:
            self._add_boxes(
                sb,
                origin,
                sl(x, e + 1, n),
                sl(y, e + 1, n),
                sl(hd, e + 1, n),
                self.VEHICLE_LENGTH,
                self.VEHICLE_WIDTH,
                True,
                sl(body, e + 1, n),
                sl(head, e + 1, n),
                sty,
                B,
                scr,
            )

    def _add_boxes(
        self,
        sb,
        origin,
        x,
        y,
        heading,
        length,
        width,
        is_vehicle,
        body_col,
        head_col,
        sty,
        B,
        scr: _Screen,
        steering=None,
        tire_slot: int = 0,
        cull: bool = True,
    ) -> None:
        """Draw equal-sized sprites with outline, inset body and clipped headlights.

        Positions/headings have shape (B, N). Road objects bypass vehicle visibility
        culling. Clip headlights to the inset because source outlines are painted last."""
        n = int(x.shape[1])
        M = B * n
        S, body, hl, hr = self._sprite(length, width, is_vehicle)
        px = self._pos2pix(origin[:, 0:1], x)
        py = self._pos2pix(origin[:, 1:2], y)
        # WorldSurface.is_visible(margin=50): off-screen VEHICLES are not drawn at all
        vis = (
            (px > -50) & (px < scr.W + 50) & (py > -50) & (py < scr.H + 50)
            if cull
            else torch.ones_like(px, dtype=torch.bool)
        )
        # Snap headings under two degrees in their original dtype before computing pivots.
        h = torch.where(heading.abs() > 2 * math.pi / 180, heading, torch.zeros_like(heading))
        cx, cy = self._pivot(S, h)
        centre = torch.stack([px.float() + cx, py.float() + cy], -1).reshape(M, 2)
        h = h.float()

        # Preserve per-row vehicle scales.
        scale = sty["vehicle.scale"].reshape(B, 1).expand(B, n).reshape(M, 1)
        inset = (body[0] + 1, body[1] + 1, body[2] - 2, body[3] - 2)
        rects = [(body, sty["vehicle.outline_color"]), (inset, body_col)]
        if hl is not None:
            for r in (hl, hr):
                c = _clip_rect(r, inset)
                if c is not None:
                    rects.append((c, head_col))
        ct, st = torch.cos(h), torch.sin(h)
        rot = torch.stack([torch.stack([ct, -st], -1), torch.stack([st, ct], -1)], -2).reshape(M, 2, 2)
        cs, szs, cols = [], [], []
        for r, colour in rects:
            off = (
                torch.tensor(
                    [r[0] + r[2] / 2.0 - S / 2.0, r[1] + r[3] / 2.0 - S / 2.0], device=self.device, dtype=torch.float32
                ).reshape(1, 2)
                * scale
            )
            cs.append((centre + torch.einsum("bij,bj->bi", rot, off)).reshape(B, n, 2))
            szs.append(torch.cat([r[2] * scale, r[3] * scale], dim=1).reshape(B, n, 2))
            cols.append(colour.reshape(B, 1, 3).expand(B, n, 3) if colour.ndim == 2 else colour)
        R = len(rects)
        sb.add_rects(
            torch.stack(cs, 2).reshape(B, n * R, 2),
            torch.stack(szs, 2).reshape(B, n * R, 2),
            torch.stack(cols, 2).reshape(B, n * R, 3),
            angle=h.unsqueeze(2).expand(B, n, R).reshape(B, n * R),
            enabled=vis.unsqueeze(2).expand(B, n, R).reshape(B, n * R),
        )
        if steering is not None:
            k = tire_slot
            self._add_tires(
                sb,
                centre.reshape(B, n, 2)[:, k],
                rot.reshape(B, n, 2, 2)[:, k],
                h[:, k],
                steering,
                S,
                length,
                width,
                scale.reshape(B, n, 1)[:, k],
                sty,
                vis[:, k],
            )

    def _add_tires(self, sb, centre, rot, h, steering, S, length, width, scale, sty, vis) -> None:
        """Apply rotate/blit pivots first within the vehicle sprite, then on screen.

        Only front tyres steer; rear tyres use the identity pivot. Steering does not
        use the heading’s two-degree snap. Polygon coverage approximates the source’s
        nested nearest-neighbour sampling, so thin tyres can differ at their edges."""
        pix = self._pix
        span = length + 2 * self.TIRE_LENGTH
        St = pix(self.TIRE_LENGTH)
        rect = (0, pix(self.TIRE_LENGTH / 2 - self.TIRE_WIDTH / 2), pix(self.TIRE_LENGTH), pix(self.TIRE_WIDTH))
        if rect[2] <= 0 or rect[3] <= 0:  # pix() truncates; nothing to draw below 1 px
            return
        positions = (
            (pix(self.TIRE_LENGTH), pix(span / 2 - width / 2)),
            (pix(self.TIRE_LENGTH), pix(span / 2 + width / 2)),
            (pix(span - self.TIRE_LENGTH), pix(span / 2 - width / 2)),
            (pix(span - self.TIRE_LENGTH), pix(span / 2 + width / 2)),
        )
        black = sty["vehicle.outline_color"]
        off_t = torch.tensor(
            [rect[0] + rect[2] / 2.0 - St / 2.0, rect[1] + rect[3] / 2.0 - St / 2.0],
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 2)
        # The four tyres in one call, with the two rear ones at angle 0 -- ``tire_angles`` is
        # ``[0, 0, steering, steering]`` -- and the sprite offsets stacked alongside them.
        B, T = steering.shape[0], len(positions)
        M = B * T
        a = torch.stack([torch.zeros_like(steering), torch.zeros_like(steering), steering, steering], dim=1).reshape(M)
        pos = torch.tensor(positions, device=self.device, dtype=torch.float32)  # (T,2)
        tcx, tcy = self._pivot(St, a)
        ca, sa = torch.cos(a), torch.sin(a)
        rot_t = torch.stack([torch.stack([ca, -sa], -1), torch.stack([sa, ca], -1)], -2)
        scale_m = scale.reshape(B, 1, 1).expand(B, T, 1).reshape(M, 1)
        local = (torch.stack([tcx, tcy], -1).reshape(B, T, 2) + pos).reshape(M, 2)
        local = (local + torch.einsum("bij,bj->bi", rot_t, off_t.expand(M, 2)) - S / 2.0) * scale_m
        c = centre.reshape(B, 1, 2).expand(B, T, 2).reshape(M, 2) + torch.einsum(
            "bij,bj->bi", rot.reshape(B, 1, 2, 2).expand(B, T, 2, 2).reshape(M, 2, 2), local
        )
        size = torch.cat([rect[2] * scale_m, rect[3] * scale_m], dim=1)
        sb.add_rects(
            c.reshape(B, T, 2),
            size.reshape(B, T, 2),
            black,
            angle=(h.reshape(B, 1) + a.reshape(B, T)),
            enabled=vis.reshape(B, 1).expand(B, T),
        )

    def _pivot(self, S: int, h: torch.Tensor):
        """Return pygame rotate/blit subpixel offsets for sprite width S.

        Quarter-turn detection uses float32 degrees, matching pygame’s tolerance band.
        Otherwise x offset follows rotated-width parity; y offset remains -0.5."""
        a = -h.double()  # VehicleGraphics passes rad2deg(-heading)
        deg = a * (180.0 / math.pi)
        axis = torch.remainder(deg.float().double(), 90.0) == 0
        wr = torch.trunc(S * (a.cos().abs() + a.sin().abs()))
        flat = torch.full_like(wr, -0.5 * (S % 2))
        cx = torch.where(
            axis, flat, torch.where(torch.remainder(wr, 2) == 1, torch.full_like(wr, -0.5), torch.full_like(wr, -1.0))
        )
        cy = torch.where(axis, flat, torch.full_like(wr, -0.5))
        return cx.float(), cy.float()

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
        """Render with ``size`` as height and the class’s aspect ratio as width.

        Derive call-local geometry so preview sizes do not change later renders."""
        scr = self.screen(size)
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        st = s.reshape(-1, self.state_dim)
        sty = self.resolve_style(style, variation, batch=int(st.shape[0]))
        img = render_scene(
            self._scene(st, sty, scr),
            (scr.H, scr.W),
            supersample if supersample is not None else self.ss,
            backend=self.backend,
            out=out,
        )
        return img.view(*lead, *img.shape[1:]) if lead else img

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        out = np.zeros((batch, self.state_dim), dtype=np.float32)
        v = self.veh_dim
        e0 = v * self.EGO_INDEX
        ego = self.ego_spawn(batch, rng)
        out[:, e0 : e0 + 3] = ego
        for i in range(self.N_VEHICLES):
            if i == self.EGO_INDEX:
                continue
            out[:, v * i : v * i + 3] = self.traffic_spawn(batch, rng, ego)
        # Include nearby crashes so sampled batches exercise visible crash colours.
        if self.CRASHED_VISIBLE:
            if self.N_VEHICLES > 1:
                j = v * (1 if self.EGO_INDEX != 1 else 0)
                out[::2, j + 3] = 1.0
                out[::2, j : j + 2] = ego[::2, 0:2] + rng.uniform(-14, 14, (len(out[::2]), 2))
            else:
                # with one vehicle there is no neighbour to crash, so the ego's own flag is the only
                # way the red path is ever drawn
                out[1::2, e0 + 3] = 1.0
        if self.EGO_HAS_TIRES:
            out[:, self.steer_off] = self.ego_steering_spawn(batch, rng)
        for k in range(len(self.STATE_OBJECTS)):
            out[:, self.obj_off + 3 * k : self.obj_off + 3 * k + 3] = self.object_spawn(k, batch, rng)
        return out

    def ego_spawn(self, batch, rng) -> np.ndarray:
        raise NotImplementedError

    def traffic_spawn(self, batch, rng, ego) -> np.ndarray:
        raise NotImplementedError

    def ego_steering_spawn(self, batch, rng) -> np.ndarray:
        """Sample steering uniformly within the source action range."""
        return rng.uniform(-self.STEERING_RANGE, self.STEERING_RANGE, batch).astype(np.float32)

    def object_spawn(self, k: int, batch, rng) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def reference_env(cls, size: int | None = None):
        """Construct and validate the source scene with an enabled offscreen viewer.

        Force the offscreen SDL driver during viewer creation: dummy disables drawing
        in highway-env and would return a blank reference image."""
        import os

        import gymnasium
        import highway_env  # noqa: F401

        h = int(size or cls.default_size)
        w = int(round(h * cls.SCREEN_W / cls.SCREEN_H))
        cfg = {
            "screen_width": w,
            "screen_height": h,
            "scaling": cls.SCALING,
            "centering_position": list(cls.CENTERING),
            "offscreen_rendering": True,
            "show_trajectories": False,
            "render_agent": False,
        }
        cfg.update(cls.reference_config)
        env = gymnasium.make(cls.reference_id, render_mode="rgb_array", config=cfg)
        env.reset(seed=0)
        cls._check_lanes(env.unwrapped)
        prev = os.environ.get("SDL_VIDEODRIVER")
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        try:
            env.unwrapped.render()  # builds the viewer, which reads the variable once
        finally:
            if prev is None:
                os.environ.pop("SDL_VIDEODRIVER", None)
            else:
                os.environ["SDL_VIDEODRIVER"] = prev
        if not env.unwrapped.viewer.enabled:
            raise AssertionError(
                "highway-env's viewer is disabled, so render() returns an untouched surface and every "
                "parity number would be measured against a black image"
            )
        return env

    @staticmethod
    def _lane_signature(l) -> tuple:
        """Encode lane geometry and markings as a tuple rounded to nine decimal places."""
        r = lambda v: round(float(v), 9)
        head = (
            type(l).__name__.replace("Lane", "") or type(l).__name__,
            r(l.width),
            tuple(int(t) for t in l.line_types),
            r(l.length),
        )
        if hasattr(l, "radius"):
            d = getattr(l, "dir", None)
            d = float(l.direction) if d is None else float(d)
            return head + (
                "circular",
                r(l.center[0]),
                r(l.center[1]),
                r(l.radius),
                r(l.start_phase),
                r(l.end_phase),
                r(d),
            )
        tail = ("straight", r(l.start[0]), r(l.start[1]), r(l.end[0]), r(l.end[1]))
        if hasattr(l, "amplitude"):
            tail += (r(l.amplitude), r(l.pulsation), r(l.phase))
        return head + tail

    @classmethod
    def _check_lanes(cls, u) -> None:
        """Compare source and declared lane multisets.

        Order is irrelevant here because all lane markings share one colour;
        vehicle and road-object painter order must still be preserved."""
        got = [l for f in u.road.network.graph for t in u.road.network.graph[f] for l in u.road.network.graph[f][t]]
        mine = sorted(cls._lane_signature(l) for l in cls.LANES)
        theirs = sorted(cls._lane_signature(l) for l in got)
        if mine == theirs:
            return
        only_mine = [s for s in mine if s not in theirs]
        only_theirs = [s for s in theirs if s not in mine]
        raise AssertionError(
            f"{cls.__name__}: the transcribed road is not the one {cls.reference_id} builds "
            f"({len(mine)} lanes transcribed, {len(theirs)} in the package). Transcribed but absent "
            f"from the package: {only_mine[:3]}; in the package but not transcribed: "
            f"{only_theirs[:3]}. Every lane marking would be drawn from a road the reference is not "
            f"using, which presents as a rasterizer error rather than as a wrong constant."
        )

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        """Rebuild the source vehicle list from render rows, retaining the ego object.

        Reset may produce a variable vehicle count. Keeping the ego preserves its
        source class, pinned colour and camera binding. Landmarks are restored unhit."""
        from highway_env.vehicle.behavior import IDMVehicle
        from highway_env.vehicle.kinematics import Vehicle

        ego = u.vehicle
        vd = 4 if cls.CRASHED_VISIBLE else 3
        cast = []
        for i in range(cls.N_VEHICLES):
            pos = np.array([float(st[vd * i]), float(st[vd * i + 1])], dtype=np.float64)
            if i == cls.EGO_INDEX:
                v = ego
                v.position = pos
            else:
                v = IDMVehicle(u.road, pos)
            v.heading = float(st[vd * i + 2])
            v.crashed = bool(st[vd * i + 3] > 0.5) if cls.CRASHED_VISIBLE else False
            cast.append(v)
        u.road.vehicles = cast
        if not cls.CRASHED_VISIBLE and not getattr(ego, "color", None):
            raise AssertionError(
                f"{cls.__name__} declares CRASHED_VISIBLE = False, which is only true because "
                f"{cls.reference_id} pins vehicle.color on its ego and VehicleGraphics.get_color reads "
                f"that before it reads `crashed`. This reference's ego has no pinned colour, so a crash "
                f"WOULD be visible and the state is missing a dimension."
            )
        if cls.EGO_HAS_TIRES:
            # Validate the exact source type; subclasses may suppress tyre rendering.
            if type(ego) is not Vehicle:
                raise AssertionError(
                    f"{cls.__name__}: the reference's ego is a {type(ego).__name__}, but this class "
                    f"declares EGO_HAS_TIRES and VehicleGraphics draws tyres only for a plain Vehicle"
                )
            ego.action = {"steering": float(st[vd * cls.N_VEHICLES]), "acceleration": 0.0}
        n_obj = len(cls.STATE_OBJECTS)
        if n_obj:
            off = vd * cls.N_VEHICLES + (1 if cls.EGO_HAS_TIRES else 0)
            objs = u.road.objects
            if len(objs) < n_obj:
                raise AssertionError(
                    f"{cls.__name__}: the reference holds {len(objs)} road objects and this class "
                    f"drives {n_obj} of them from the state"
                )
            for k in range(n_obj):
                o = objs[k]
                o.position = np.array([float(st[off + 3 * k]), float(st[off + 3 * k + 1])], np.float64)
                o.heading = float(st[off + 3 * k + 2])
                # This render schema has no hit flag; restore landmarks in their unhit colour.
                o.hit = False

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        u = env.unwrapped
        out = []
        vd = 4 if cls.CRASHED_VISIBLE else 3
        n = vd * cls.N_VEHICLES
        for st in np.asarray(states):
            cls.reference_set_state(u, st)
            got = np.array([[*v.position, v.heading] for v in u.road.vehicles], np.float64)
            want = np.asarray(st, np.float64)[:n].reshape(-1, vd)[:, :3]
            if cls.STATE_OBJECTS:
                off = n + (1 if cls.EGO_HAS_TIRES else 0)
                got = np.concatenate(
                    [
                        got,
                        np.array(
                            [[*o.position, o.heading] for o in u.road.objects[: len(cls.STATE_OBJECTS)]], np.float64
                        ),
                    ]
                )
                want = np.concatenate([want, np.asarray(st, np.float64)[off:].reshape(-1, 3)])
            err = np.abs(got - want).max()
            if err > 1e-6:
                raise AssertionError(
                    f"{cls.__name__}: the reference did not reach the requested state (max err "
                    f"{err:.3e}); parity measured against this frame would compare two scenes"
                )
            out.append(u.render().copy())
        return np.stack(out)


def lane_frames(lane, n: int, rng, jitter: float = 1.0) -> np.ndarray:
    """Sample (x, y, heading) along straight or circular lanes with lateral jitter."""
    s = rng.uniform(0.0, lane.length, n)
    lat = rng.uniform(-jitter, jitter, n)
    if isinstance(lane, Circular):
        phi = lane.dir * s / lane.radius + lane.start_phase
        r = lane.radius - lat * lane.dir
        return np.stack(
            [lane.center[0] + r * np.cos(phi), lane.center[1] + r * np.sin(phi), phi + lane.dir * np.pi / 2], 1
        )
    if isinstance(lane, Sine):
        raise NotImplementedError("lane_frames does not cover SineLane; no registered env needs it")
    d, l = lane.direction, lane.lateral
    return np.stack(
        [
            lane.start[0] + s * d[0] + lat * l[0],
            lane.start[1] + s * d[1] + lat * l[1],
            np.full(n, math.atan2(d[1], d[0])),
        ],
        1,
    )


def _on_track(lanes, batch, rng, jitter: float = 1.0, heading_jitter: float = 0.0) -> np.ndarray:
    """Sample one lane per row, then add optional heading noise."""
    pick = rng.integers(0, len(lanes), batch)
    out = np.zeros((batch, 3))
    for i, lane in enumerate(lanes):
        m = pick == i
        if m.any():
            out[m] = lane_frames(lane, int(m.sum()), rng, jitter)
    if heading_jitter:
        out[:, 2] += rng.uniform(-heading_jitter, heading_jitter, batch)
    return out.astype(np.float32)


def _racetrack_lanes():
    """Declare the source racetrack’s nine two-lane sections; phases are in degrees."""
    d = np.deg2rad
    c, s, n = CONTINUOUS, STRIPED, NONE
    return (
        Straight([42, 0], [100, 0], width=5, line_types=(c, s)),
        Straight([42, 5], [100, 5], width=5, line_types=(s, c)),
        Circular([100, -20], 20, d(90), d(-1), clockwise=False, width=5, line_types=(c, n)),
        Circular([100, -20], 25, d(90), d(-1), clockwise=False, width=5, line_types=(s, c)),
        Straight([120, -20], [120, -30], width=5, line_types=(c, n)),
        Straight([125, -20], [125, -30], width=5, line_types=(s, c)),
        Circular([105, -30], 15, d(0), d(-181), clockwise=False, width=5, line_types=(c, n)),
        Circular([105, -30], 20, d(0), d(-181), clockwise=False, width=5, line_types=(s, c)),
        Circular([70, -30], 20, d(0), d(136), clockwise=True, width=5, line_types=(c, s)),
        Circular([70, -30], 15, d(0), d(137), clockwise=True, width=5, line_types=(n, c)),
        Straight([55.7, -15.7], [35.7, -35.7], width=5, line_types=(c, n)),
        Straight([59.3934, -19.2], [39.3934, -39.2], width=5, line_types=(s, c)),
        Circular([18.1, -18.1], 25, d(315), d(170), clockwise=False, width=5, line_types=(c, n)),
        Circular([18.1, -18.1], 30, d(315), d(165), clockwise=False, width=5, line_types=(s, c)),
        Circular([18.1, -18.1], 25, d(170), d(56), clockwise=False, width=5, line_types=(c, n)),
        Circular([18.1, -18.1], 30, d(170), d(58), clockwise=False, width=5, line_types=(s, c)),
        Circular([43.2, 23.4], 23.5, d(240), d(270), clockwise=True, width=5, line_types=(c, s)),
        Circular([43.2, 23.4], 18.5, d(238), d(268), clockwise=True, width=5, line_types=(n, c)),
    )


@register
class Racetrack(HighwayEnvBase):
    """Render racetrack-v0 with one ego and one IDM vehicle.

    The ego’s visible front tyres make steering part of the render row."""

    name = "racetrack"
    default_size = 600
    SCREEN_W, SCREEN_H = 600, 600
    SCALING, CENTERING = 5.5, (0.5, 0.5)
    LANES = _racetrack_lanes()
    N_VEHICLES = 2  # RacetrackEnv._make_vehicles: ego + one front vehicle
    EGO_HAS_TIRES = True
    # Racetrack’s plain, unpinned Vehicle uses the source yellow default.
    EGO_COLOR_DEFAULT = YELLOW
    reference_id = "racetrack-v0"
    reference_config = {"controlled_vehicles": 1, "other_vehicles": 1}

    def ego_spawn(self, batch, rng):
        return _on_track(self.LANES, batch, rng, jitter=1.4, heading_jitter=0.25)

    def traffic_spawn(self, batch, rng, ego):
        return _on_track(self.LANES, batch, rng, jitter=1.4, heading_jitter=0.25)


def _parking_lanes(spots: int = 14, width: float = 4.0, y_offset: float = 10.0, length: float = 8.0):
    """Build both rows of parking bays, upper row first as in source graph iteration."""
    xs = [(k + 1 - spots // 2) * width - width / 2 for k in range(spots)]
    lt = (CONTINUOUS, CONTINUOUS)
    return tuple(
        [Straight([x, y_offset], [x, y_offset + length], width=width, line_types=lt) for x in xs]
        + [Straight([x, -y_offset], [x, -y_offset - length], width=width, line_types=lt) for x in xs]
    )


@register
class Parking(HighwayEnvBase):
    """Render parking-v0 with one ego, a goal landmark and four walls.

    Goal pose and steering are render inputs. The source pins the ego green,
    so crashes do not change its colour and crash flags are omitted."""

    name = "parking"
    default_size = 300
    SCREEN_W, SCREEN_H = 600, 300
    SCALING, CENTERING = 7.0, (0.5, 0.5)
    LANES = _parking_lanes()
    N_VEHICLES = 1
    EGO_HAS_TIRES = True
    CRASHED_VISIBLE = False
    STATE_OBJECTS = (("landmark", 2.0, 2.0),)
    OBJECTS = (
        ("obstacle", 0.0, -21.0, 0.0, 70.0, 1.0),
        ("obstacle", 0.0, 21.0, 0.0, 70.0, 1.0),
        ("obstacle", -35.0, 0.0, math.pi / 2, 42.0, 1.0),
        ("obstacle", 35.0, 0.0, math.pi / 2, 42.0, 1.0),
    )
    #: the 28 bay centres, ``lane.position(lane.length / 2, 0)`` with the lane's own heading
    GOAL_POSES = tuple(
        [
            (float(l.start[0]), float((l.start[1] + l.end[1]) / 2), math.atan2(float(l.end[1] - l.start[1]), 0.0))
            for l in LANES
        ]
    )
    reference_id = "parking-v0"
    reference_config = {"controlled_vehicles": 1, "vehicles_count": 0, "add_walls": True}

    def ego_spawn(self, batch, rng):
        # ``_create_vehicles`` starts the ego at (0, 0) with a heading of 2*pi*U(0,1) and it then
        # drives around the lot, so the spawn is a point and the states are a region
        return np.stack(
            [rng.uniform(-30, 30, batch), rng.uniform(-18, 18, batch), rng.uniform(0, 2 * np.pi, batch)], 1
        ).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        raise AssertionError("parking-v0 has one vehicle; traffic_spawn is unreachable")

    def object_spawn(self, k: int, batch, rng):
        poses = np.asarray(self.GOAL_POSES, np.float32)
        return poses[rng.integers(0, len(poses), batch)]


def _clip_rect(r, box):
    x0, y0 = max(r[0], box[0]), max(r[1], box[1])
    x1, y1 = min(r[0] + r[2], box[0] + box[2]), min(r[1] + r[3], box[1] + box[3])
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1 - x0, y1 - y0)


@register
class Highway(HighwayEnvBase):
    """Render highway-v0’s four straight lanes with an ego-following camera."""

    name = "highway"
    default_size = 150
    SCREEN_W, SCREEN_H = 600, 150
    SCALING, CENTERING = 5.5, (0.3, 0.5)
    LANES = tuple(straight_road_network(4))
    N_VEHICLES = 12
    reference_id = "highway-v0"
    reference_config = {"lanes_count": 4, "controlled_vehicles": 1}

    def ego_spawn(self, batch, rng):
        lane = rng.integers(0, 4, batch) * 4.0
        return np.stack(
            [rng.uniform(100, 900, batch), lane + rng.uniform(-1.2, 1.2, batch), rng.uniform(-0.35, 0.35, batch)], 1
        ).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        lane = rng.integers(0, 4, batch) * 4.0
        return np.stack(
            [
                ego[:, 0] + rng.uniform(-130, 130, batch),
                lane + rng.uniform(-1.2, 1.2, batch),
                rng.uniform(-0.35, 0.35, batch),
            ],
            1,
        ).astype(np.float32)


def _merge_lanes():
    """Build two highway lanes and the source sinusoidal on-ramp."""
    c, s, n = CONTINUOUS_LINE, STRIPED, NONE
    ends = [150, 80, 80, 150]
    y = [0.0, 4.0]
    lt = [(c, s), (n, c)]
    lt_merge = [(c, s), (n, s)]
    out = []
    for i in range(2):
        out.append(Straight([0, y[i]], [sum(ends[:2]), y[i]], line_types=lt[i]))
        out.append(Straight([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=lt_merge[i]))
        out.append(Straight([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=lt[i]))
    amplitude = 3.25
    ljk = Straight([0, 14.5], [ends[0], 14.5], line_types=(c, c))
    lkb = Sine(
        [ends[0], 14.5 - amplitude],
        [sum(ends[:2]), 14.5 - amplitude],
        amplitude,
        2 * np.pi / (2 * ends[1]),
        np.pi / 2,
        line_types=(c, c),
    )
    # lbc starts where the ramp meets the highway: lkb.position(80, 0) = (230, 8)
    out += [ljk, lkb, Straight([sum(ends[:2]), 8.0], [sum(ends[:3]), 8.0], line_types=(n, c))]
    return tuple(out)


@register
class Merge(HighwayEnvBase):
    """Render merge-v0 with a sinusoidal on-ramp and an obstacle at its end."""

    name = "merge"
    default_size = 150
    SCREEN_W, SCREEN_H = 600, 150
    SCALING, CENTERING = 5.5, (0.3, 0.5)
    LANES = _merge_lanes()
    OBJECTS = (("obstacle", 310.0, 8.0, 0.0, 2.0, 2.0),)  # Obstacle(road, lbc.position(80, 0))
    N_VEHICLES = 5  # MergeEnv._make_vehicles: ego + 4
    reference_id = "merge-v0"

    def ego_spawn(self, batch, rng):
        x = rng.uniform(20, 430, batch)
        # Place every third sample near the obstacle to exercise its visible colour.
        x[::3] = rng.uniform(255, 330, len(x[::3]))
        return np.stack(
            [x, rng.integers(0, 2, batch) * 4.0 + rng.uniform(-1.2, 1.2, batch), rng.uniform(-0.3, 0.3, batch)], 1
        ).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        on_ramp = rng.random(batch) < 0.35
        y = np.where(on_ramp, 14.5, rng.integers(0, 2, batch) * 4.0) + rng.uniform(-1.2, 1.2, batch)
        return np.stack([ego[:, 0] + rng.uniform(-110, 110, batch), y, rng.uniform(-0.3, 0.3, batch)], 1).astype(
            np.float32
        )


def _roundabout_lanes():
    """Build two rings of eight arcs and four access branches."""
    radius, alpha, center = 20.0, 24.0, (0.0, 0.0)
    n, c, s = NONE, CONTINUOUS, STRIPED
    radii = [radius, radius + 4]
    line = [(c, s), (n, c)]
    out = []
    seq = [
        (90 - alpha, alpha),
        (alpha, -alpha),
        (-alpha, -90 + alpha),
        (-90 + alpha, -90 - alpha),
        (-90 - alpha, -180 + alpha),
        (-180 + alpha, -180 - alpha),
        (180 - alpha, 90 + alpha),
        (90 + alpha, 90 - alpha),
    ]
    for lane in (0, 1):
        for p0, p1 in seq:
            out.append(
                Circular(center, radii[lane], np.deg2rad(p0), np.deg2rad(p1), clockwise=False, line_types=line[lane])
            )
    access, dev, a = 170.0, 85.0, 5.0
    delta_st = 0.2 * dev
    delta_en = dev - delta_st
    w = 2 * np.pi / dev
    out += [
        Straight([2, access], [2, dev / 2], line_types=(s, c)),
        Sine([2 + a, dev / 2], [2 + a, dev / 2 - delta_st], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([-2 - a, -dev / 2 + delta_en], [-2 - a, dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)),
        Straight([-2, dev / 2], [-2, access], line_types=(n, c)),
        Straight([access, -2], [dev / 2, -2], line_types=(s, c)),
        Sine([dev / 2, -2 - a], [dev / 2 - delta_st, -2 - a], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([-dev / 2 + delta_en, 2 + a], [dev / 2, 2 + a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)),
        Straight([dev / 2, 2], [access, 2], line_types=(n, c)),
        Straight([-2, -access], [-2, -dev / 2], line_types=(s, c)),
        Sine([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)),
        Straight([2, -dev / 2], [2, -access], line_types=(n, c)),
        Straight([-access, 2], [-dev / 2, 2], line_types=(s, c)),
        Sine([-dev / 2, 2 + a], [-dev / 2 + delta_st, 2 + a], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([dev / 2 - delta_en, -2 - a], [-dev / 2, -2 - a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)),
        Straight([-dev / 2, -2], [-access, -2], line_types=(n, c)),
    ]
    return tuple(out)


@register
class Roundabout(HighwayEnvBase):
    """Render roundabout-v0 with two rings, four access branches and five vehicles.

    Its large primitive list requires a runtime kernel loop; fully unrolling it
    can exhaust the CUDA compiler. Thin curved markings retain coverage differences."""

    name = "roundabout"
    default_size = 600
    SCREEN_W, SCREEN_H = 600, 600
    SCALING, CENTERING = 5.5, (0.5, 0.6)
    LANES = _roundabout_lanes()
    N_VEHICLES = 5  # RoundaboutEnv._make_vehicles: ego + 4
    reference_id = "roundabout-v0"

    def ego_spawn(self, batch, rng):
        th = rng.uniform(-np.pi, np.pi, batch)
        r = rng.uniform(20, 26, batch)
        return np.stack([r * np.cos(th), r * np.sin(th), th + np.pi / 2], 1).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        th = rng.uniform(-np.pi, np.pi, batch)
        ring = rng.random(batch) < 0.6
        r = np.where(ring, rng.uniform(20, 26, batch), rng.uniform(30, 70, batch))
        return np.stack([r * np.cos(th), r * np.sin(th), th + np.pi / 2 + rng.uniform(-0.4, 0.4, batch)], 1).astype(
            np.float32
        )


def _intersection_lanes():
    """Build each corner’s approach, turns, through lane and exit."""
    lw = 4.0  # AbstractLane.DEFAULT_WIDTH
    r_turn = lw + 5.0  # 9
    l_turn = r_turn + lw  # 13
    outer = r_turn + lw / 2  # 11
    access = 100.0
    n, c, s = NONE, CONTINUOUS, STRIPED
    out = []
    for corner in range(4):
        a = np.radians(90 * corner)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out.append(Straight(R @ np.array([lw / 2, access + outer]), R @ np.array([lw / 2, outer]), lw, (s, c)))
        out.append(
            Circular(
                R @ np.array([outer, outer]),
                r_turn,
                a + np.radians(180),
                a + np.radians(270),
                clockwise=True,
                width=lw,
                line_types=(n, c),
            )
        )
        # the left turn carries no markings at all, so it contributes no primitives
        out.append(
            Circular(
                R @ np.array([-l_turn + lw / 2, l_turn - lw / 2]),
                l_turn,
                a,
                a - np.radians(90),
                clockwise=False,
                width=lw,
                line_types=(n, n),
            )
        )
        out.append(Straight(R @ np.array([lw / 2, outer]), R @ np.array([lw / 2, -outer]), lw, (s, n)))
        # the exit is built as StraightLane(end, start), i.e. it runs outward from the box
        out.append(Straight(R @ np.array([outer, lw / 2]), R @ np.array([access + outer, lw / 2]), lw, (n, c)))
    return tuple(out)


@register
class Intersection(HighwayEnvBase):
    """Render intersection-v0 with the ego last in vehicle painter order.

    The reference bridge fixes vehicle count explicitly because source reset
    samples traffic probabilistically."""

    name = "intersection"
    default_size = 600
    SCREEN_W, SCREEN_H = 600, 600
    SCALING, CENTERING = 5.5 * 1.3, (0.5, 0.6)
    LANES = _intersection_lanes()
    N_VEHICLES = 10
    EGO_INDEX = N_VEHICLES - 1
    reference_id = "intersection-v0"
    reference_config = {"controlled_vehicles": 1}

    def ego_spawn(self, batch, rng):
        corner = rng.integers(0, 4, batch)
        a = np.radians(90.0 * corner)
        ca, sa = np.cos(a), np.sin(a)
        t = rng.uniform(30, 95, batch)  # metres down the approach lane
        lx, ly = 2.0, 111.0 - t  # lane frame, before the corner rotation
        return np.stack([ca * lx - sa * ly, sa * lx + ca * ly, a - np.pi / 2], 1).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        near_ego = rng.random(batch) < 0.6
        x = np.where(near_ego, ego[:, 0] + rng.uniform(-38, 38, batch), rng.uniform(-45, 45, batch))
        y = np.where(near_ego, ego[:, 1] + rng.uniform(-38, 38, batch), rng.uniform(-45, 45, batch))
        return np.stack([x, y, rng.uniform(-np.pi, np.pi, batch)], 1).astype(np.float32)
