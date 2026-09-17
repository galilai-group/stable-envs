"""highway-env: autonomous-driving scenarios as a batched GPU renderer.

Reference: the ``highway-env`` package (Leurent, 2018), rendered with pygame vector art. A frame is
the grey road plane, the white lane markings of a fixed road network, and a handful of vehicles drawn
as rotated rectangles. That is a pure function of ``(x, y, heading, crashed)`` per vehicle: the camera
follows the ego, so the ego's own coordinates fix the window, and the lane markings follow from the
static network plus that window.

Every constant below was recovered by instrumenting the reference (``scratch/highway/probe_*.py``),
because reading the source is not enough — five things read plausibly and are wrong:

  * **highway-env draws nothing at all under ``SDL_VIDEODRIVER=dummy``.** ``EnvViewer.__init__`` sets
    ``self.enabled = False`` when it sees that driver and ``display()`` returns immediately, so
    ``render()`` hands back the never-touched surface. Measured: 1 unique colour, mean 0.0, against
    7 unique colours and mean 103.8 with ``SDL_VIDEODRIVER=offscreen``. The usual headless recipe
    therefore silently turns every parity number into a comparison against a black image.
  * **The vehicle is not a rotated rectangle** in the reference. It is an axis-aligned rectangle
    rasterized into a 38x38 sprite, passed through ``pygame.transform.rotate`` (nearest neighbour,
    fixed point) and blitted at an origin that ``Rect`` truncates to a whole pixel. It is still
    *exactly* expressible as a polygon: the rotation reads ``src[floor(A·d + t)]``, and ``floor(u) == n``
    is a half-open box test in the rotated frame, so the set of pixels showing a given sprite rectangle
    is a rotated rectangle. Verified on 97 headings: an analytic rectangle reproduces **100% of pixels**
    once the pivot below is applied.
  * **The pivot is not the vehicle's screen position.** With ``Wr = int(S*(|cos a| + |sin a|))`` the
    rotated surface width and ``S`` the sprite width, the rectangle centre sits at
    ``pos + (-0.5, -0.5)`` when ``Wr`` is odd, ``pos + (-1.0, -0.5)`` when it is even, and
    ``pos - 0.5*(S mod 2)`` in both axes when the angle is an exact multiple of 90 degrees. Assuming
    ``pos`` puts the car up to one pixel out. See ``_pivot`` for the measurement.
  * **The ego's screen x is 179, not 180.** ``origin = ego - centering*size/scaling`` and then
    ``int((ego - origin) * scaling)`` reconstructs ``0.3*600`` through two subtractions, and float64
    lands at 179.99999999999997. The quantisation is done here in float64 for that reason; float32
    rounds the other way and shifts the whole frame by a pixel.
  * **The black outline is drawn after the headlights**, so a headlight that reaches the body's front
    edge is painted over. Drawing the outline as a filled rectangle *under* an inset body reproduces
    that only if the headlight rectangles are clipped to the inset body first; without the clip the
    headlights bleed 6 px over the outline per vehicle.

Draw order is the reference's ``EnvViewer.display``: grey fill, lane markings, road objects, then
vehicles in ``road.vehicles`` order (ego first).

Registered: ``Highway`` (highway-v0), ``Merge`` (merge-v0), ``Roundabout`` (roundabout-v0),
``Intersection`` (intersection-v0), ``Racetrack`` (racetrack-v0) and ``Parking`` (parking-v0).

The last two are the ones whose ego is a plain ``Vehicle``, and that costs two extra things.
``VehicleGraphics`` draws tyres only for ``Vehicle``/``BicycleVehicle`` -- an exact ``type()`` check, so
an ``IDMVehicle`` never gets them -- as four small sprites, each its own rotate-and-blit INSIDE the
vehicle sprite, which is then rotated again. **The pivot rule below therefore applies twice**, once
about the tyre sprite's own centre at the steering angle and once about the vehicle sprite's centre at
the heading, and ``steering`` joins the state because the two front tyres turn with it. See
``_add_tires``; measured in ``scratch/highway/calib.py``.

Also not ported: ``show_trajectories`` (the ghost history of past poses) and the lidar observation
overlay, neither of which is on by default. ``ObservationGraphics.display`` draws only for
``LidarObservation``, and no registered env here uses one.

NO ``step``, and the reason is specific rather than a shrug
-----------------------------------------------------------

The classic-control and MiniGrid families in this package ship a batched GPU ``step`` because their
dynamics can be transcribed AND CHECKED — a published closed form or a finite decision table, against a
reference that is installed and can be driven state for state. Neither holds here yet:

  * **The hidden dynamical state is larger than the visible one.** The render state is four numbers per
    vehicle, ``(x, y, heading, crashed)``, because that is all the picture needs. The kinematic bicycle
    model needs ``speed`` on top of it, and the traffic policy needs more per vehicle again:
    ``target_lane_index``, ``target_speed``, the lane-change ``timer``, and the planned route. That is
    fine in itself — classic control hides velocities the same way — but several of those are
    RANDOMISED per vehicle at ``reset`` (``IDMVehicle.randomize_behavior``), so a comparison would have
    to read each vehicle's sampled parameters out of the reference before it could mean anything.
  * **The traffic policy is the real work, not the integrator.** IDM gives an acceleration from the
    vehicle in front; MOBIL decides lane changes from the vehicles behind in two adjacent lanes. Both
    need ``road.network.get_closest_lane_index`` and ``side_lanes`` over a graph of heterogeneous lanes
    (``Straight``, ``Sine``, ``Circular``), batched — and the roundabout adds route following on top.
    A port of the bicycle model alone would run and would not be highway-env.
  * **``highway_env`` is not installed in this environment**, so there is nothing to check a port
    against here. Given how this package works — every constant above was recovered by instrumenting
    the reference, not by reading it — writing the solver first and validating it later is exactly the
    order that produces a plausible env nobody can trust.

The renderer stands on its own and is unaffected: supply states from the official CPU env, or from any
other source, and ``render`` draws them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render.raster2d import render as render_scene
from ..render.scene import Scene2D, SceneBuilder
from ..spaces import Box, Dict, Discrete, RGBBox

# --- palette, straight from highway_env.vehicle.graphics / road.graphics -------------------------
GREY = (100, 100, 100)              # WorldSurface.GREY, the road plane and the whole background
WHITE = (255, 255, 255)             # lane markings
BLACK = (60, 60, 60)                # VehicleGraphics.BLACK, the 1 px vehicle outline
BLUE = (100, 200, 255)              # IDMVehicle
GREEN = (50, 200, 0)                # EGO_COLOR (MDPVehicle)
YELLOW = (200, 200, 0)              # DEFAULT_COLOR, and Obstacle
RED = (255, 100, 100)               # crashed


def lighten(c, ratio: float = 0.68):
    """``VehicleGraphics.lighten``: the headlight colour. Integer truncation, then a clamp at 255."""
    return tuple(min(int(v / ratio), 255) for v in c)


# --- lane geometry, transcribed from highway_env.road.lane ---------------------------------------
STRIPE_SPACING = 4.33
STRIPE_LENGTH = 3.0
STRIPE_WIDTH = 0.3
NONE, STRIPED, CONTINUOUS, CONTINUOUS_LINE = 0, 1, 2, 3


class _Lane:
    """Common part: line types, width, and the arc-length window a marking is clipped to."""

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
    """``SineLane``: a straight lane whose lateral offset oscillates. Used by the merge ramp."""

    def __init__(self, start, end, amplitude, pulsation, phase, width=4.0,
                 line_types=(STRIPED, STRIPED)):
        super().__init__(start, end, width, line_types)
        self.amplitude, self.pulsation, self.phase = float(amplitude), float(pulsation), float(phase)

    def position(self, s, lat):
        lat = torch.as_tensor(lat, device=s.device, dtype=s.dtype)
        return super().position(s, lat + self.amplitude * torch.sin(self.pulsation * s + self.phase))


class Circular(_Lane):
    def __init__(self, center, radius, start_phase, end_phase, clockwise=True, width=4.0,
                 line_types=(STRIPED, STRIPED)):
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
    """``RoadNetwork.straight_road_network``: the highway-v0 layout, N parallel lanes."""
    out = []
    for i in range(lanes):
        origin = (0.0, i * width)
        end = (length, i * width)
        line = (CONTINUOUS_LINE if i == 0 else STRIPED,
                CONTINUOUS_LINE if i == lanes - 1 else NONE)
        out.append(Straight(origin, end, width, line))
    return out


def _py_floordiv(a: torch.Tensor, b: float) -> torch.Tensor:
    """CPython's ``float.__floordiv__``, which is not ``floor(a/b)``.

    ``LaneGraphics.display`` computes ``int(s_origin) // STRIPE_SPACING`` and that value picks which
    stripes exist, so an off-by-one here shifts every dashed marking by 4.33 m. CPython computes
    ``mod = fmod(a, b)``, ``div = (a - mod)/b`` and then rounds ``div`` to the nearest integer when it
    is within half a unit, which differs from ``floor(a/b)`` when ``a/b`` lands just below an integer.
    """
    mod = torch.fmod(a, b)
    div = (a - mod) / b
    adj = (mod != 0) & ((b < 0) != (mod < 0))
    div = div - adj.to(div.dtype)
    fl = torch.floor(div)
    fl = fl + ((div - fl) > 0.5).to(div.dtype)
    return torch.where(div == 0, torch.zeros_like(div), fl)


@dataclass(frozen=True)
class _Screen:
    """Everything about a highway frame that depends on the OUTPUT SIZE, as one immutable value.

    This used to be five attributes on the env, rebound by ``render(size=)`` through a ``_set_screen``
    method — and that made ``size`` a mode rather than an argument: one preview frame at a different
    resolution silently rewrote the resolution of every frame afterwards, while ``env.size`` still
    reported the original. Measured: ``render(st).shape`` (4,150,600,3), then ``render(st, size=300)``,
    then ``render(st).shape`` (4,300,1200,3) with ``env.size == 150``. It cascaded — four unrelated
    dataset round-trip failures whose actual cause was a size leaked three tests earlier.

    A frozen value derived inside the call cannot leak. There is nowhere left to store it.
    """

    H: int
    W: int
    cam_off: tuple[float, float]
    stripes_count: int
    line_w: int


class HighwayEnvBase(Env):
    """Shared renderer for the highway-env family; subclasses supply the road and the cast.

    ``state`` is 4 numbers per vehicle — ``x``, ``y``, ``heading``, ``crashed`` — with the ego first.
    ``crashed`` is in the state because it is visible: ``VehicleGraphics.get_color`` paints a crashed
    vehicle red, so a state without it does not determine the frame in the one situation the env exists
    to study.
    """

    #: reference screen, in the same (width, height) order highway-env's config uses
    SCREEN_W, SCREEN_H = 600, 150
    SCALING = 5.5
    CENTERING = (0.3, 0.5)
    #: lanes of the road network, in the order highway-env walks ``network.graph``
    LANES: tuple = ()
    #: number of vehicles, in the reference's draw order (``road.vehicles``)
    N_VEHICLES = 12
    #: which of them is the ego. It is drawn green and the camera is locked to it; highway-env appends
    #: it first in most envs and last in the intersection family, and the index matters only where two
    #: vehicles overlap, which is exactly the situation these envs are about.
    EGO_INDEX = 0
    #: static road furniture: ``(kind, x, y, heading, length, width)``, kind 'obstacle' or 'landmark'
    OBJECTS: tuple = ()
    #: road furniture whose POSE is part of the state: ``(kind, length, width)``. Drawn BEFORE
    #: ``OBJECTS``, because ``road.objects`` is walked in insertion order and parking-v0 appends its
    #: goal landmark before the four walls. The state carries ``(x, y, heading)`` for each, after the
    #: vehicles and after ``ego_steering``.
    STATE_OBJECTS: tuple = ()
    VEHICLE_LENGTH, VEHICLE_WIDTH = 5.0, 2.0
    TIRE_LENGTH, TIRE_WIDTH = 1.0, 0.3
    #: True where the ego is a plain ``Vehicle`` rather than an ``MDPVehicle``/``IDMVehicle``, which is
    #: the only thing that decides whether ``VehicleGraphics`` draws tyres, and hence whether the
    #: steering angle is visible at all.
    EGO_HAS_TIRES = False
    #: What ``VehicleGraphics.get_color`` returns for THIS env's ego, which is not one colour for the
    #: family and is not the class's own business either. The chain is
    #: ``vehicle.color`` (if set) -> crashed -> LinearVehicle -> IDMVehicle -> MDPVehicle -> DEFAULT,
    #: so an env whose action type produces a plain ``Vehicle`` and does not pin ``color`` falls all
    #: the way through to ``DEFAULT_COLOR``, which is YELLOW -- the obstacle colour. racetrack-v0 is
    #: exactly that case; parking-v0 pins ``vehicle.color = EGO_COLOR`` and is green.
    EGO_COLOR_DEFAULT = GREEN
    #: False where the reference PINS ``vehicle.color`` on the ego. ``get_color`` reads that attribute
    #: BEFORE it looks at ``crashed``, so on such an env a crash changes no pixel and ``crashed`` is
    #: neither a state dimension nor a variation -- carrying it would be a dead dimension and an inert
    #: knob, which is what the conformance suite exists to reject.
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
        fields = ("x", "y", "heading", "crashed")[:self.veh_dim]
        names = [f"{'ego' if i == self.EGO_INDEX else f'v{i}'}_{f}"
                 for i in range(self.N_VEHICLES) for f in fields]
        if self.EGO_HAS_TIRES:
            names.append("ego_steering")
        for k, (kind, _, _) in enumerate(self.STATE_OBJECTS):
            names += [f"{kind}{k}_x", f"{kind}{k}_y", f"{kind}{k}_heading"]
        self.state_names = tuple(names)
        # Step-9 state (contract §3). The packed row is unchanged -- vehicles, then the ego's
        # steering angle if it has tires, then the static objects, exactly the order the offsets
        # above have always fixed -- and the layout names the three runs it already consists of.
        # Assigned per instance because all three widths are functions of the class's own
        # ``N_VEHICLES`` / ``CRASHED_VISIBLE`` / ``EGO_HAS_TIRES`` / ``STATE_OBJECTS`` declarations,
        # which is exactly what ``state_dim`` is computed from two lines up.
        #
        # ``vehicles`` is ONE key of ``N_VEHICLES * veh_dim`` columns rather than a key per vehicle:
        # the vehicles are a homogeneous array whose rows differ only by index, and a state with
        # ``v0``..``v14`` as separate keys would be fifteen names for one fact. Nothing is hidden --
        # every column is covered, so ``pack``/``unpack`` round-trip the frozen layout bit for bit.
        self.state_layout = ((("vehicles",), self.veh_dim * self.N_VEHICLES, 0),) + (
            ((("ego", "steering"), 1, self.steer_off),) if self.EGO_HAS_TIRES else ()) + (
            ((("objects",), 3 * n_obj, self.obj_off),) if n_obj else ())
        #: per output size, the lane-marking tables ``_plan`` builds. Keyed on the frame geometry
        #: because ``stripes_count`` and ``_slots_for`` are functions of it, and NOT stored as
        #: attributes for the reason ``_Screen`` exists: a preview at another size must not change
        #: what the next render draws.
        self._plan_cache: dict = {}
        self._obj_runs = None
        super().__init__(num_envs=num_envs, size=size, **kw)

    def _close(self) -> None:
        """Drop the two per-instance tables. ``Env.close`` calls this once and is idempotent.

        Neither is an external handle -- this family owns no GL context, no simulator and no atlas,
        because it draws every frame itself. What it does own is the lane-marking plan per output
        size and the object run-lengths, both device tensors that can be large for a 600 px
        racetrack, and both rebuildable from the class alone. Releasing them is what a closed env of
        this family has to release; refusing to render afterwards is the rest of the contract.
        """
        self._plan_cache = {}
        self._obj_runs = None

    # --- geometry that depends on the output size -------------------------------------------------

    def screen(self, size: int | None = None) -> _Screen:
        """The frame geometry for one output size. A pure function of ``size`` and the class.

        Called per render and never stored, which is what makes ``size`` an argument again. The
        arithmetic is unchanged and evaluated in the reference's order, so the float64 results stay
        bit-identical; see the module docstring on the ego landing at x=179 rather than 180.
        """
        H = int(self.size if size is None else size)
        W = int(round(H * self.SCREEN_W / self.SCREEN_H))
        sc = self.SCALING
        return _Screen(
            H=H, W=W,
            cam_off=(float(self.CENTERING[0] * W) / sc, float(self.CENTERING[1] * H) / sc),
            stripes_count=int(2 * (H + W) / (STRIPE_SPACING * sc)),
            line_w=max(int(STRIPE_WIDTH * sc), 1),            # WorldSurface.pix(STRIPE_WIDTH), min 1
        )

    @property
    def H(self) -> int:
        """Frame height at the env's OWN size. Read-only: a render cannot move it any more."""
        return self.screen().H

    @property
    def W(self) -> int:
        """Frame width at the env's own size, from the class's fixed aspect ratio."""
        return self.screen().W

    def _pix(self, v: float) -> int:
        """``WorldSurface.pix``: ``int(length * scaling)``."""
        return int(v * self.SCALING)

    def _sprite(self, length: float, width: float, with_tires_pad: bool):
        """Sprite-pixel rectangles for one vehicle or road object, exactly as the reference builds them."""
        sc = self.SCALING
        pix = lambda v: int(v * sc)                                    # WorldSurface.pix
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

    # --- variation --------------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Only the knobs that can move a pixel in THIS env, in a fixed order.

        Three of the six leaves are conditional, and each condition is a measurement rather than a
        preference: ``object`` needs road furniture; ``landmark`` needs a ``Landmark`` among it (an
        Obstacle is yellow and a Landmark is BLUE, so one 'object' colour cannot serve both);
        ``traffic`` needs a second vehicle (parking-v0 has one); and ``crashed`` needs a crash to be
        able to reach a pixel, which it cannot where the reference pins ``vehicle.color``. A knob that
        cannot move the picture is what ``check_variations_do_something`` exists to reject.

        Built as an ordered list rather than by merging dicts so that adding a conditional leaf cannot
        renumber the flat style layout of the envs that do not have it -- a stored ``(state, style)``
        pair is only reproducible while the offsets hold still.
        """
        space: dict = {}
        if self.OBJECTS:
            space["object"] = Dict({"color": RGBBox(init_value=YELLOW)})
        if self.STATE_OBJECTS:
            space["landmark"] = Dict({"color": RGBBox(init_value=BLUE)})
        space["background"] = Dict({"color": RGBBox(init_value=GREY)})
        space["road"] = Dict({
            "line_color": RGBBox(init_value=WHITE),
            "show_lines": Discrete(2, init_value=1),
        })
        space["ego"] = Dict({
            "color": RGBBox(init_value=self.EGO_COLOR_DEFAULT),
            "headlight_color": RGBBox(init_value=lighten(self.EGO_COLOR_DEFAULT)),
        })
        if self.N_VEHICLES > 1:
            space["traffic"] = Dict({"color": RGBBox(init_value=BLUE),
                                     "headlight_color": RGBBox(init_value=lighten(BLUE))})
        if self.CRASHED_VISIBLE:
            space["crashed"] = Dict({"color": RGBBox(init_value=RED)})
        space["vehicle"] = Dict({
            "outline_color": RGBBox(init_value=BLACK),
            "scale": Box(low=0.25, high=3.0, shape=(1,), init_value=[1.0]),
        })
        return Dict(space)

    # --- the scene ---------------------------------------------------------------------------------

    def scene(self, states, variation: dict | None = None, style=None, size: int | None = None
              ) -> Scene2D:
        """``size`` is declared here, and its presence is a statement about this family.

        The world rectangle below is ``(0, 0, W, H)`` in SCREEN PIXELS at a fixed ``scaling``, so a
        bigger frame shows more road rather than the same road more finely — ``size`` is a field of
        view here, not a resolution. Callers that need to know which of the two they are dealing with
        (the downsample-consistency check is one) build the scene at two sizes and compare ``world``,
        which is why the argument exists and why envs whose world is in world units do not declare it.
        """
        n = int(states.shape[0]) if hasattr(states, "shape") else len(states)
        return self._scene(states, self.resolve_style(style, variation, batch=n), self.screen(size))

    def _scene(self, states, sty, scr: _Screen) -> Scene2D:
        """Build the scene from the compact state, a resolved per-env style, and one output geometry."""
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32).reshape(-1, self.state_dim)
        B = s.shape[0]
        sb = SceneBuilder(B, device=self.device, vertices=4)

        # Camera. float64 throughout: pos2pix truncates, and the ego sits exactly on a pixel boundary.
        e0 = self.veh_dim * self.EGO_INDEX
        ego = s[:, e0:e0 + 2].double()
        origin = torch.stack([ego[:, 0] - scr.cam_off[0], ego[:, 1] - scr.cam_off[1]], -1)

        self._add_lane_markings(sb, origin, sty, B, scr)
        # ``road.objects`` order: the state-driven furniture first (parking-v0 appends its goal
        # landmark before the walls), then the static furniture.
        for k, (kind, length, width) in enumerate(self.STATE_OBJECTS):
            o = self.obj_off + 3 * k
            self._add_boxes(sb, origin, s[:, o:o + 1].double(), s[:, o + 1:o + 2].double(),
                            s[:, o + 2:o + 3], length, width, False, sty[f"{kind}.color"], None,
                            sty, B, scr, cull=False)
        # Static furniture, in class order, batched over the runs that share a sprite. float64, and
        # NOT because more precision is nicer: ``_pivot``'s first question is whether the angle is an
        # exact multiple of 90 degrees, and parking-v0's walls are built at exactly ``np.pi/2``.
        # Narrowed to float32 that becomes 90.0000025 degrees, the exact-multiple branch is missed,
        # and the four walls land a pixel out -- 1,140 px a frame, measured, all of it on their 1 px
        # outline. A heading that comes from the STATE is float32 in both renderers and agrees; one
        # that is a class constant must not be narrowed.
        for (length, width), poses in self._object_runs():
            p = torch.tensor(poses, device=self.device, dtype=torch.float64).unsqueeze(0)
            p = p.expand(B, len(poses), 3)
            self._add_boxes(sb, origin, p[..., 0], p[..., 1], p[..., 2], length, width, False,
                            sty["object.color"], None, sty, B, scr, cull=False)
        self._add_vehicles(sb, origin, s, sty, B, scr)
        return sb.build(background=sty["background.color"],
                        world=(0.0, 0.0, float(scr.W), float(scr.H)), flip_y=False)

    def _pos2pix(self, origin: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """``WorldSurface.pos2pix``: ``int((p - origin) * scaling)``, truncating toward zero."""
        return torch.trunc((p - origin) * self.SCALING)

    def _object_runs(self):
        """``OBJECTS`` split into CONSECUTIVE runs that share a sprite: ``((length, width), poses)``.

        Consecutive, not grouped by size: ``road.objects`` is walked in insertion order and that
        order is the z-buffer, so two walls of one length either side of a wall of another must stay
        three draws. parking-v0's four walls happen to be written as two runs of two.
        """
        if self._obj_runs is None:
            runs: list = []
            for _, x, y, heading, length, width in self.OBJECTS:
                if not runs or runs[-1][0] != (length, width):
                    runs.append(((length, width), []))
                runs[-1][1].append((float(x), float(y), float(heading)))
            self._obj_runs = runs
        return self._obj_runs

    def _plan(self, scr: _Screen) -> dict:
        """Every lane marking this env can draw at this frame size, as flat per-stripe tables.

        The loop this replaces issued one ``add_segment`` per stripe and roundabout-v0 has 1,248 of
        them: 53,481 torch calls and 315 ms to build 436 primitives, the SAME 315 ms at batch 16 as
        at batch 1024. That is not arithmetic, it is dispatch, and the fix is to do the arithmetic
        once on ``(B, S)`` rather than S times on ``(B,)``.

        What the plan holds is exactly what the loop's control flow decided and nothing that depends
        on the state: which lane a stripe belongs to, its lateral offset, its span, its slot index
        within its own (lane, side) window, and whether it is the single long stripe a continuous
        line draws. Stripes are laid out in the reference's draw order, so no reordering happens
        afterwards — order is the z-buffer here (``render/scene.py``), and a marking that painted in
        a different order would be a different picture wherever two markings overlap.

        The two lane parametrisations are both EVALUATED for every stripe and selected between,
        rather than gathered into groups and stitched back together. A circular lane is ~10 flops
        that a straight lane throws away, against an index_select and a scatter per group; and a
        select cannot get the order wrong.
        """
        key = (scr.H, scr.W)
        hit = self._plan_cache.get(key)
        if hit is not None:
            return hit
        lanes = self.LANES
        f64 = dict(device=self.device, dtype=torch.float64)
        circ = [isinstance(l, Circular) for l in lanes]
        sine = [isinstance(l, Sine) for l in lanes]

        def straight_of(l):
            """``(start, direction, lateral, amplitude, pulsation, phase)`` — a circular lane
            supplies harmless stand-ins, because its rows are selected away."""
            if isinstance(l, Circular):
                return (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 0.0, 0.0, 0.0
            return (tuple(l.start), tuple(l.direction), tuple(l.lateral),
                    getattr(l, "amplitude", 0.0), getattr(l, "pulsation", 0.0),
                    getattr(l, "phase", 0.0))

        def circular_of(l):
            """``(centre, radius, start_phase, dir)``; radius 1 for a straight lane so the division
            in the unused branch is finite rather than an inf that ``where`` merely discards."""
            if isinstance(l, Circular):
                return tuple(l.center), l.radius, l.start_phase, l.dir
            return (0.0, 0.0), 1.0, 0.0, 1.0

        st_p = [straight_of(l) for l in lanes]
        ci_p = [circular_of(l) for l in lanes]
        # groups: one per (lane, side) that draws DASHED stripes, i.e. that needs a slot window
        g_lane, g_len, g_nslots, g_half = [], [], [], []
        rows = []                                   # (lane, group or -1, j, lat, span, continuous)
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
        sel = lambda seq, idx: [seq[i] for i in idx]                                 # noqa: E731
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
        """``AbstractLane.local_coordinates`` for every lane at once: ``(B,2)`` -> ``(B,L)``."""
        d = p.unsqueeze(1) - P["l_start"]
        s_str = d[..., 0] * P["l_dir"][..., 0] + d[..., 1] * P["l_dir"][..., 1]
        dc = p.unsqueeze(1) - P["l_c"]
        phi = torch.atan2(dc[..., 1], dc[..., 0])
        # utils.wrap_to_pi, so the branch cut sits at the lane's own start phase
        phi = P["l_sph"] + torch.remainder(phi - P["l_sph"] + math.pi, 2 * math.pi) - math.pi
        return torch.where(P["l_circ"], P["l_cdir"] * (phi - P["l_sph"]) * P["l_rad"], s_str)

    def _stripe_pos(self, P: dict, s: torch.Tensor) -> torch.Tensor:
        """``lane.position(s, lat)`` for every stripe at once: ``(B,S)`` -> ``(B,S,2)``."""
        lat = P["lat"]
        lat_e = torch.where(P["p_sine"],
                            lat + P["p_amp"] * torch.sin(P["p_puls"] * s + P["p_phase"]),
                            lat)
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
        centre = origin + torch.tensor([scr.W / 2.0 / self.SCALING, scr.H / 2.0 / self.SCALING],
                                       device=origin.device, dtype=origin.dtype)
        s_org = self._local_s(P, origin)                                        # (B,L)
        s_ctr = self._local_s(P, centre)
        s0 = (_py_floordiv(torch.trunc(s_org), STRIPE_SPACING)
              - scr.stripes_count // 2) * STRIPE_SPACING
        k0 = self._slot_window_all(P, s0, s_ctr)                                # (B,G)

        s0s = s0.index_select(1, P["lane_of"])                                  # (B,S)
        k = k0.index_select(1, P["grp_of"]) + P["j_of"]
        cont = P["cont"]
        st = torch.where(cont, s0s, s0s + k * STRIPE_SPACING)
        en = torch.where(cont, (s0s + scr.stripes_count * STRIPE_SPACING) + STRIPE_LENGTH,
                         st + P["span"])
        ok = show.unsqueeze(1) & (cont | ((k >= 0) & (k < scr.stripes_count)))

        st = torch.minimum(st.clamp_min(0.0), P["len_s"])
        en = torch.minimum(en.clamp_min(0.0), P["len_s"])
        ok = ok & ((en - st).abs() > 0.5 * STRIPE_LENGTH)                # LaneGraphics.draw_stripes
        o = origin.unsqueeze(1)
        a = (torch.trunc((self._stripe_pos(P, st) - o) * self.SCALING) + 0.5).float()
        b = (torch.trunc((self._stripe_pos(P, en) - o) * self.SCALING) + 0.5).float()
        S = P["S"]
        a, b, hw = self._line_geom(a.reshape(B * S, 2), b.reshape(B * S, 2), scr)
        sb.add_segments(a.reshape(B, S, 2), b.reshape(B, S, 2), hw.reshape(B, S), col, enabled=ok)

    def _slot_window_all(self, P: dict, s0, s_ctr) -> torch.Tensor:
        """First stripe index of the window each (lane, side) group gets: ``(B,L)`` -> ``(B,G)``.

        Following the camera alone is wrong for short lanes. A circular lane's ``local_coordinates``
        projects the camera onto the whole circle, so for an arc of 14.66 m the projection is routinely
        tens of metres outside ``[0, length]``, the whole window then clips to a degenerate segment and
        the arc disappears. Measured before this fix: **66% of the roundabout's white pixels missing**.
        The window is therefore clamped to cover the lane whenever it is short enough to fit, and only
        follows the camera when it is not.
        """
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
        """How many dashed-stripe slots a lane needs to cover everything that can reach the screen.

        The reference emits ``stripes_count`` stripes per marking regardless of the lane — measured on
        highway-v0, from x = -749 px to x = +743 px for a 600 px window. A stripe further along the lane
        than the screen's extent in that direction cannot paint a pixel, so only that many slots are
        built; the rest are still evaluated against the reference's window so nothing extra is drawn.
        """
        sc = self.SCALING
        if isinstance(lane, Straight):
            span = (scr.W * abs(lane.direction[0]) + scr.H * abs(lane.direction[1])) / sc
            if isinstance(lane, Sine):                 # the ramp's tangent is not the chord's
                span = math.hypot(scr.W, scr.H) / sc
        else:
            span = math.hypot(scr.W, scr.H) / sc
        n = int((span + STRIPE_LENGTH) / STRIPE_SPACING) + 3
        return max(2, min(n, int(lane.length / STRIPE_SPACING) + 3, scr.stripes_count))

    def _line_geom(self, a: torch.Tensor, b: torch.Tensor, scr: _Screen):
        """Capsule that reproduces ``pygame.draw.line`` of width ``scr.line_w``.

        pygame lights ``w`` consecutive pixels per column of the segment's MAJOR axis, i.e. it
        thresholds the distance measured along the MINOR axis at ``w/2``. A capsule thresholds the
        PERPENDICULAR distance, and the two coincide only for axis-aligned lines; scaling the
        half-width by ``|major| / length`` converts one threshold into the other.

        For EVEN ``w`` the run is also shifted one pixel toward +minor, because an even number of
        pixels cannot straddle a pixel centre symmetrically. Measured directly: a width-2 horizontal
        line at y=30 lights rows [30, 31], width 3 lights [29, 30, 31], width 4 lights [29..32], and
        reversing the endpoints does not move any of them, so the shift is absolute rather than
        direction-dependent.

        Measured agreement, 300 random segments each: at width 1, 12.48% of lit pixels wrong with a
        flat 0.5 half-width against **1.79%** with this rule. At width 2 every symmetric candidate —
        capsule, scaled capsule, or square-capped quad — is ~50% wrong, which is what an unshifted band
        looks like when the truth is shifted by one pixel; this rule gives **0.25%** on axis-aligned
        segments and 1.35% on arbitrary ones. Width 2 is what intersection-v0 needs, since its
        ``scaling`` of 7.15 makes ``pix(0.3) = 2``.
        """
        e = b - a
        L = e.norm(dim=-1).clamp_min(1e-9)
        hw = (scr.line_w / 2.0) * e.abs().amax(dim=-1) / L
        if scr.line_w % 2 == 0:
            x_major = e[:, 0].abs() >= e[:, 1].abs()
            shift = torch.stack([torch.where(x_major, 0.0, 0.5),
                                 torch.where(x_major, 0.5, 0.0)], -1)
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
        cast = s[:, :v * n].reshape(B, n, v)
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
            self._add_boxes(sb, origin, x, y, hd, self.VEHICLE_LENGTH, self.VEHICLE_WIDTH, True,
                            body, head, sty, B, scr)
            return
        # ``road.vehicles`` order with the ego's four tyres in the middle of it: everything up to
        # and including the ego, then its tyres, then the rest. Draw order is the z-buffer, so the
        # tyres cannot be moved to the end of the cast even though only the ego has any.
        e = self.EGO_INDEX
        sl = lambda t, a, b: t[:, a:b]                                            # noqa: E731
        self._add_boxes(sb, origin, sl(x, 0, e + 1), sl(y, 0, e + 1), sl(hd, 0, e + 1),
                        self.VEHICLE_LENGTH, self.VEHICLE_WIDTH, True, sl(body, 0, e + 1),
                        sl(head, 0, e + 1), sty, B, scr,
                        steering=s[:, self.steer_off], tire_slot=e)
        if e + 1 < n:
            self._add_boxes(sb, origin, sl(x, e + 1, n), sl(y, e + 1, n), sl(hd, e + 1, n),
                            self.VEHICLE_LENGTH, self.VEHICLE_WIDTH, True, sl(body, e + 1, n),
                            sl(head, e + 1, n), sty, B, scr)

    def _add_boxes(self, sb, origin, x, y, heading, length, width, is_vehicle, body_col, head_col,
                   sty, B, scr: _Screen, steering=None, tire_slot: int = 0,
                   cull: bool = True) -> None:
        """``n`` rotated sprites of ONE size: black footprint, inset body, two clipped headlights.

        ``x``/``y``/``heading`` are ``(B,n)`` and every box in the call shares the sprite, so its
        pixel rectangles and its pivot rule are computed once. Boxes of different sizes are separate
        calls -- ``_sprite`` truncates to whole pixels, so two lengths are two different sprites and
        not one sprite scaled.

        ``cull`` is False for road furniture and True for vehicles, and that asymmetry is the
        reference's: ``VehicleGraphics.display`` returns early on ``surface.is_visible(position)``
        while ``RoadObjectGraphics.display`` has no such test. Measured on parking-v0, the env with the
        largest road objects in the family, it is worth 7 px a frame.
        """
        n = int(x.shape[1])
        M = B * n
        S, body, hl, hr = self._sprite(length, width, is_vehicle)
        px = self._pos2pix(origin[:, 0:1], x)
        py = self._pos2pix(origin[:, 1:2], y)
        # WorldSurface.is_visible(margin=50): off-screen VEHICLES are not drawn at all
        vis = ((px > -50) & (px < scr.W + 50) & (py > -50) & (py < scr.H + 50) if cull
               else torch.ones_like(px, dtype=torch.bool))
        # VehicleGraphics.display: headings under 2 degrees are snapped to zero. The snap and the pivot
        # are evaluated in the heading's OWN dtype so a float64 class constant stays float64; the
        # geometry handed to the builder is float32 like every other primitive.
        h = torch.where(heading.abs() > 2 * math.pi / 180, heading, torch.zeros_like(heading))
        cx, cy = self._pivot(S, h)
        centre = torch.stack([px.float() + cx, py.float() + cy], -1).reshape(M, 2)
        h = h.float()

        # One value PER ENVIRONMENT, not a shared scalar: a batch is allowed to carry 4096 different
        # vehicle sizes, and collapsing to row 0 is the failure the per-env conformance check exists for.
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
            off = torch.tensor([r[0] + r[2] / 2.0 - S / 2.0, r[1] + r[3] / 2.0 - S / 2.0],
                               device=self.device, dtype=torch.float32).reshape(1, 2) * scale
            cs.append((centre + torch.einsum("bij,bj->bi", rot, off)).reshape(B, n, 2))
            szs.append(torch.cat([r[2] * scale, r[3] * scale], dim=1).reshape(B, n, 2))
            cols.append(colour.reshape(B, 1, 3).expand(B, n, 3) if colour.ndim == 2 else colour)
        R = len(rects)
        sb.add_rects(torch.stack(cs, 2).reshape(B, n * R, 2), torch.stack(szs, 2).reshape(B, n * R, 2),
                     torch.stack(cols, 2).reshape(B, n * R, 3),
                     angle=h.unsqueeze(2).expand(B, n, R).reshape(B, n * R),
                     enabled=vis.unsqueeze(2).expand(B, n, R).reshape(B, n * R))
        if steering is not None:
            k = tire_slot
            self._add_tires(sb, centre.reshape(B, n, 2)[:, k], rot.reshape(B, n, 2, 2)[:, k],
                            h[:, k], steering, S, length, width,
                            scale.reshape(B, n, 1)[:, k], sty, vis[:, k])

    def _add_tires(self, sb, centre, rot, h, steering, S, length, width, scale, sty, vis) -> None:
        """Four tyres, which is THE PIVOT RULE APPLIED TWICE.

        ``VehicleGraphics.display`` rasterises each tyre into its own ``pix(1) x pix(1)`` surface,
        ``blit_rotate``\\ s it into the VEHICLE surface at the steering angle, and only then rotates the
        vehicle surface by the heading and blits that onto the screen. Both blits truncate their origin
        to a whole pixel, so ``_pivot`` -- the sub-pixel offset measured for the outer blit -- applies
        to the inner one as well, with ``S`` the tyre sprite's own size:

            local centre = tire_position + pivot(S_tire, steering) + R(steering) @ tyre-rect offset
            screen centre = vehicle centre + R(heading) @ (local centre - S_vehicle/2)
            screen angle  = heading + steering

        The rear pair is blitted at angle 0, which takes ``pygame.transform.rotate``'s identity path and
        ``_pivot``'s exact-multiple-of-90 branch; only the FRONT pair turns (``tire_angles`` is
        ``[0, 0, steering, steering]``). The heading is snapped to zero below 2 degrees and the steering
        angle is NOT -- the snap happens once, on ``v.heading``, after the tyres are already drawn.

        What this cannot be is exact by construction. The outer rotation resamples the vehicle surface
        with ``src[floor(A.d + t)]``, so a rectangle of vehicle-surface PIXELS maps to a rotated
        rectangle tested at the floored coordinate rather than at the pixel itself; on a body 27 px long
        that is invisible, and on a tyre ``pix(0.3) = 1`` px thick it is up to a pixel of the tyre. The
        residual is bounded by the tyres' own area, 4 x pix(1) x pix(0.3) px per ego, and it is reported
        with the rest of the boundary disagreement in ``docs/research/highway.md``.
        """
        pix = self._pix
        span = length + 2 * self.TIRE_LENGTH
        St = pix(self.TIRE_LENGTH)
        rect = (0, pix(self.TIRE_LENGTH / 2 - self.TIRE_WIDTH / 2),
                pix(self.TIRE_LENGTH), pix(self.TIRE_WIDTH))
        if rect[2] <= 0 or rect[3] <= 0:                 # pix() truncates; nothing to draw below 1 px
            return
        positions = ((pix(self.TIRE_LENGTH), pix(span / 2 - width / 2)),
                     (pix(self.TIRE_LENGTH), pix(span / 2 + width / 2)),
                     (pix(span - self.TIRE_LENGTH), pix(span / 2 - width / 2)),
                     (pix(span - self.TIRE_LENGTH), pix(span / 2 + width / 2)))
        black = sty["vehicle.outline_color"]
        off_t = torch.tensor([rect[0] + rect[2] / 2.0 - St / 2.0, rect[1] + rect[3] / 2.0 - St / 2.0],
                             device=self.device, dtype=torch.float32).reshape(1, 2)
        # The four tyres in one call, with the two rear ones at angle 0 -- ``tire_angles`` is
        # ``[0, 0, steering, steering]`` -- and the sprite offsets stacked alongside them.
        B, T = steering.shape[0], len(positions)
        M = B * T
        a = torch.stack([torch.zeros_like(steering), torch.zeros_like(steering),
                         steering, steering], dim=1).reshape(M)
        pos = torch.tensor(positions, device=self.device, dtype=torch.float32)     # (T,2)
        tcx, tcy = self._pivot(St, a)
        ca, sa = torch.cos(a), torch.sin(a)
        rot_t = torch.stack([torch.stack([ca, -sa], -1), torch.stack([sa, ca], -1)], -2)
        scale_m = scale.reshape(B, 1, 1).expand(B, T, 1).reshape(M, 1)
        local = (torch.stack([tcx, tcy], -1).reshape(B, T, 2) + pos).reshape(M, 2)
        local = (local + torch.einsum("bij,bj->bi", rot_t, off_t.expand(M, 2)) - S / 2.0) * scale_m
        c = (centre.reshape(B, 1, 2).expand(B, T, 2).reshape(M, 2)
             + torch.einsum("bij,bj->bi", rot.reshape(B, 1, 2, 2).expand(B, T, 2, 2).reshape(M, 2, 2),
                            local))
        size = torch.cat([rect[2] * scale_m, rect[3] * scale_m], dim=1)
        sb.add_rects(c.reshape(B, T, 2), size.reshape(B, T, 2), black,
                     angle=(h.reshape(B, 1) + a.reshape(B, T)),
                     enabled=vis.reshape(B, 1).expand(B, T))

    def _pivot(self, S: int, h: torch.Tensor):
        """Sub-pixel pivot of pygame's rotate + truncated blit.

        Measured by searching, for 181 headings and sprite sizes S in {11, 14, 21, 38}, every offset
        that makes an analytic rectangle reproduce 100% of the pixels:

          * a rotation by a multiple of 90 degrees takes pygame's axis-aligned fast path, where the
            only sub-pixel effect is the blit truncating ``pos - S/2``: offset ``-0.5*(S mod 2)``;
          * otherwise the offset is ``(-0.5, -0.5)`` when the rotated surface width
            ``Wr = int(S*(|cos a| + |sin a|))`` is odd and ``(-1.0, -0.5)`` when it is even.

        The parity that matters is ``Wr``'s, not ``S``'s and not ``Wr - S``'s: S=11 with Wr=13 gives
        -0.5 while S=38 with Wr=46 gives -1.0, and both have an even ``Wr - S``. The x and y rules
        differ because ``blit_rotate`` builds the two coordinates from ``min_box`` and ``max_box``
        respectively, which are not symmetric.

        **"A multiple of 90 degrees" is not an exact test, and taking it as one costs a pixel.**
        ``pygame.transform.rotate`` takes its fast path on a band around each quarter turn, and the
        band is exactly half a FLOAT32 ulp wide: bisected on 490x490, 294x294 and 38x38 sprites, a
        rotation stops being bit-identical to the quarter turn at ``3.815e-06`` degrees from 90,
        ``7.629e-06`` from 180 and ``1.526e-05`` from 270 -- which is ``2**-18``, ``2**-17``, ``2**-16``,
        i.e. half an ulp of 90, 180 and 270 in float32. So the angle is rounded to float32 and then
        tested. That is not academic here: a heading of ``pi/2`` carried through a float32 STATE comes
        back as 90.0000025 degrees, inside the band, and an exact-equality test sends it down the
        ``Wr`` branch instead -- worth 23 px a frame on parking-v0's 14x14 goal landmark alone.
        Measured in ``scratch/highway/probe_rot_tolerance.py``.
        """
        a = -h.double()                       # VehicleGraphics passes rad2deg(-heading)
        deg = a * (180.0 / math.pi)
        axis = torch.remainder(deg.float().double(), 90.0) == 0
        wr = torch.trunc(S * (a.cos().abs() + a.sin().abs()))
        flat = torch.full_like(wr, -0.5 * (S % 2))
        cx = torch.where(axis, flat, torch.where(torch.remainder(wr, 2) == 1,
                                                 torch.full_like(wr, -0.5),
                                                 torch.full_like(wr, -1.0)))
        cy = torch.where(axis, flat, torch.full_like(wr, -0.5))
        return cx.float(), cy.float()

    # --- rendering ---------------------------------------------------------------------------------

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``size`` is the frame HEIGHT; the width follows the env's own aspect ratio.

        highway-v0 is 600x150 and the intersection envs are square. Forcing a square frame would either
        crop the road or change ``scaling``, and ``scaling`` is inside every ``int()`` in the reference,
        so changing it changes the picture rather than its resolution.

        The geometry for the requested size is derived here and thrown away here. Nothing about the
        call survives it: ``render`` is a read, and a preview frame at another resolution must not
        change what the next ``render()`` returns.
        """
        scr = self.screen(size)
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        st = s.reshape(-1, self.state_dim)
        sty = self.resolve_style(style, variation, batch=int(st.shape[0]))
        img = render_scene(self._scene(st, sty, scr), (scr.H, scr.W),
                           supersample if supersample is not None else self.ss,
                           backend=self.backend, out=out)
        return img.view(*lead, *img.shape[1:]) if lead else img

    # --- states ------------------------------------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        out = np.zeros((batch, self.state_dim), dtype=np.float32)
        v = self.veh_dim
        e0 = v * self.EGO_INDEX
        ego = self.ego_spawn(batch, rng)
        out[:, e0:e0 + 3] = ego
        for i in range(self.N_VEHICLES):
            if i == self.EGO_INDEX:
                continue
            out[:, v * i:v * i + 3] = self.traffic_spawn(batch, rng, ego)
        # Half the batch carries a crashed vehicle next to the ego, so the red-vehicle path is
        # exercised; without it the 'crashed.color' knob would look inert and the crashed state dims
        # would look dead, and both would be a test passing for the wrong reason.
        if self.CRASHED_VISIBLE:
            if self.N_VEHICLES > 1:
                j = v * (1 if self.EGO_INDEX != 1 else 0)
                out[::2, j + 3] = 1.0
                out[::2, j:j + 2] = ego[::2, 0:2] + rng.uniform(-14, 14, (len(out[::2]), 2))
            else:
                # with one vehicle there is no neighbour to crash, so the ego's own flag is the only
                # way the red path is ever drawn
                out[1::2, e0 + 3] = 1.0
        if self.EGO_HAS_TIRES:
            out[:, self.steer_off] = self.ego_steering_spawn(batch, rng)
        for k in range(len(self.STATE_OBJECTS)):
            out[:, self.obj_off + 3 * k:self.obj_off + 3 * k + 3] = self.object_spawn(k, batch, rng)
        return out

    def ego_spawn(self, batch, rng) -> np.ndarray:
        raise NotImplementedError

    def traffic_spawn(self, batch, rng, ego) -> np.ndarray:
        raise NotImplementedError

    def ego_steering_spawn(self, batch, rng) -> np.ndarray:
        """``ContinuousAction`` clips the steering to ``+-steering_range``; measured on 240 real steps
        of racetrack-v0 and parking-v0, it reaches +-0.785 rad and averages 0.41 in magnitude."""
        return rng.uniform(-self.STEERING_RANGE, self.STEERING_RANGE, batch).astype(np.float32)

    def object_spawn(self, k: int, batch, rng) -> np.ndarray:
        raise NotImplementedError

    # --- reference bridge ---------------------------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int | None = None):
        """The official env, sized and populated to match this class.

        ``SDL_VIDEODRIVER`` is forced to ``offscreen``: under ``dummy`` highway-env disables its viewer
        and returns an untouched surface, so a parity test would compare against a black image and pass
        or fail for reasons that have nothing to do with rendering.
        """
        import os
        import gymnasium
        import highway_env  # noqa: F401
        h = int(size or cls.default_size)
        w = int(round(h * cls.SCREEN_W / cls.SCREEN_H))
        cfg = {"screen_width": w, "screen_height": h, "scaling": cls.SCALING,
               "centering_position": list(cls.CENTERING), "offscreen_rendering": True,
               "show_trajectories": False, "render_agent": False}
        cfg.update(cls.reference_config)
        env = gymnasium.make(cls.reference_id, render_mode="rgb_array", config=cfg)
        env.reset(seed=0)
        cls._check_lanes(env.unwrapped)
        prev = os.environ.get("SDL_VIDEODRIVER")
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        try:
            env.unwrapped.render()          # builds the viewer, which reads the variable once
        finally:
            if prev is None:
                os.environ.pop("SDL_VIDEODRIVER", None)
            else:
                os.environ["SDL_VIDEODRIVER"] = prev
        if not env.unwrapped.viewer.enabled:
            raise AssertionError(
                "highway-env's viewer is disabled, so render() returns an untouched surface and every "
                "parity number would be measured against a black image")
        return env

    @staticmethod
    def _lane_signature(l) -> tuple:
        """Everything about a lane that reaches a pixel, rounded to 1e-9, as one comparable tuple."""
        r = lambda v: round(float(v), 9)
        head = (type(l).__name__.replace("Lane", "") or type(l).__name__,
                r(l.width), tuple(int(t) for t in l.line_types), r(l.length))
        if hasattr(l, "radius"):
            d = getattr(l, "dir", None)
            d = float(l.direction) if d is None else float(d)
            return head + ("circular", r(l.center[0]), r(l.center[1]), r(l.radius),
                           r(l.start_phase), r(l.end_phase), r(d))
        tail = ("straight", r(l.start[0]), r(l.start[1]), r(l.end[0]), r(l.end[1]))
        if hasattr(l, "amplitude"):
            tail += (r(l.amplitude), r(l.pulsation), r(l.phase))
        return head + tail

    @classmethod
    def _check_lanes(cls, u) -> None:
        """``LANES`` must still be the road the package builds. Re-derived, never trusted.

        The four envs ported first have small hand-written networks; racetrack-v0 has 18 lanes with
        radii and phases in degrees and parking-v0 has 28 generated from a loop, and a transcription
        that size has a defect rate. A wrong radius is not a crash -- it is a lane marking a few pixels
        off, which looks exactly like a rasterizer bug -- so the transcription is checked against the
        official network every time a reference env is built.

        Compared as a MULTISET rather than in order, and that is a statement about the renderer rather
        than a convenience. ``RoadNetwork.graph`` is a dict of dicts, so it groups every ``a->b`` lane
        before every ``b->c`` one while ``MergeEnv._make_road`` writes them section by section; the two
        orders differ for three of the six envs here. Order would matter if lanes painted over each
        other in different colours, and they do not -- every marking is ``road.line_color`` -- so what
        has to agree is which lanes exist, not the sequence they are walked in.
        """
        got = [l for f in u.road.network.graph for t in u.road.network.graph[f]
               for l in u.road.network.graph[f][t]]
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
            f"using, which presents as a rasterizer error rather than as a wrong constant.")

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        """Drive the reference to ``st`` by rebuilding ``road.vehicles`` outright.

        Rebuilding rather than repositioning what happens to be there: several of these envs decide
        their cast at random (``IntersectionEnv._spawn_vehicle`` draws against a probability and
        ``_clear_vehicles`` deletes vehicles that leave), so the number of vehicles after ``reset`` is
        not a property of the env id. The state is the definition of the scene, and the reference is
        made to hold exactly that scene. The ego object is kept, because it must stay the class the
        env's action type produced or it renders in the wrong colour and ``env.vehicle`` — which fixes
        the camera — stops pointing at it.
        """
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
                f"WOULD be visible and the state is missing a dimension.")
        if cls.EGO_HAS_TIRES:
            # ``VehicleGraphics`` draws tyres for ``type(v) in [Vehicle, BicycleVehicle]`` -- an exact
            # type check -- so a reference whose ego has drifted to another class would silently draw
            # no tyres and the parity number would describe a different picture.
            if type(ego) is not Vehicle:
                raise AssertionError(
                    f"{cls.__name__}: the reference's ego is a {type(ego).__name__}, but this class "
                    f"declares EGO_HAS_TIRES and VehicleGraphics draws tyres only for a plain Vehicle")
            ego.action = {"steering": float(st[vd * cls.N_VEHICLES]), "acceleration": 0.0}
        n_obj = len(cls.STATE_OBJECTS)
        if n_obj:
            off = vd * cls.N_VEHICLES + (1 if cls.EGO_HAS_TIRES else 0)
            objs = u.road.objects
            if len(objs) < n_obj:
                raise AssertionError(
                    f"{cls.__name__}: the reference holds {len(objs)} road objects and this class "
                    f"drives {n_obj} of them from the state")
            for k in range(n_obj):
                o = objs[k]
                o.position = np.array([float(st[off + 3 * k]), float(st[off + 3 * k + 1])], np.float64)
                o.heading = float(st[off + 3 * k + 2])
                # a Landmark turns GREEN once hit, which is a flag the state does not carry because
                # nothing in the picture distinguishes 'reached' from 'about to be reached'
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
                got = np.concatenate([got, np.array(
                    [[*o.position, o.heading] for o in u.road.objects[:len(cls.STATE_OBJECTS)]],
                    np.float64)])
                want = np.concatenate([want, np.asarray(st, np.float64)[off:].reshape(-1, 3)])
            err = np.abs(got - want).max()
            if err > 1e-6:
                raise AssertionError(
                    f"{cls.__name__}: the reference did not reach the requested state (max err "
                    f"{err:.3e}); parity measured against this frame would compare two scenes")
            out.append(u.render().copy())
        return np.stack(out)


def lane_frames(lane, n: int, rng, jitter: float = 1.0) -> np.ndarray:
    """``(n, 3)`` of ``(x, y, heading)`` at random points along ``lane``, with a lateral jitter.

    The reference spawns every vehicle with ``make_on_lane`` and then drives it, so a spawn
    distribution that is not on the road exercises a picture the env never shows -- and on a racetrack
    where the whole frame is lane markings that is most of the frame. ``heading_at`` is transcribed
    per lane class, which is the same arithmetic ``AbstractLane.heading_at`` does.
    """
    s = rng.uniform(0.0, lane.length, n)
    lat = rng.uniform(-jitter, jitter, n)
    if isinstance(lane, Circular):
        phi = lane.dir * s / lane.radius + lane.start_phase
        r = lane.radius - lat * lane.dir
        return np.stack([lane.center[0] + r * np.cos(phi), lane.center[1] + r * np.sin(phi),
                         phi + lane.dir * np.pi / 2], 1)
    if isinstance(lane, Sine):
        raise NotImplementedError("lane_frames does not cover SineLane; no registered env needs it")
    d, l = lane.direction, lane.lateral
    return np.stack([lane.start[0] + s * d[0] + lat * l[0],
                     lane.start[1] + s * d[1] + lat * l[1],
                     np.full(n, math.atan2(d[1], d[0]))], 1)


def _on_track(lanes, batch, rng, jitter: float = 1.0, heading_jitter: float = 0.0) -> np.ndarray:
    """``(batch, 3)`` poses spread over a whole lane network, one lane drawn per environment."""
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
    """``RacetrackEnv._make_road``: nine sections of two 5 m lanes, read off the official network.

    Emitted by ``scratch/highway/probe_racetrack.py`` from ``road.network`` rather than transcribed
    from ``_make_road`` by eye, and re-checked against the package by ``_check_lanes`` every time a
    reference env is built. Phases are written in degrees because that is the unit the source uses and
    the unit a typo is visible in.
    """
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
    """racetrack-v0: a closed circuit of straights and arcs, ego plus one IDM vehicle.

    The first of the two envs whose ego is a plain ``Vehicle``, so its four tyres are drawn and
    ``ego_steering`` is a state dimension. See ``_add_tires`` for the nested pivot; nothing else here
    differs from the rest of the family.
    """

    name = "racetrack"
    default_size = 600
    SCREEN_W, SCREEN_H = 600, 600
    SCALING, CENTERING = 5.5, (0.5, 0.5)
    LANES = _racetrack_lanes()
    N_VEHICLES = 2                          # RacetrackEnv._make_vehicles: ego + one front vehicle
    EGO_HAS_TIRES = True
    #: measured, not assumed: racetrack-v0 does not pin ``vehicle.color`` and its ego is a plain
    #: ``Vehicle``, so ``get_color`` falls past LinearVehicle/IDMVehicle/MDPVehicle to DEFAULT_COLOR.
    #: The ego is YELLOW here, the same colour an obstacle is elsewhere in the family; assuming the
    #: family's green cost 1,717 interior pixels a frame before the diff named it.
    EGO_COLOR_DEFAULT = YELLOW
    reference_id = "racetrack-v0"
    reference_config = {"controlled_vehicles": 1, "other_vehicles": 1}

    def ego_spawn(self, batch, rng):
        return _on_track(self.LANES, batch, rng, jitter=1.4, heading_jitter=0.25)

    def traffic_spawn(self, batch, rng, ego):
        return _on_track(self.LANES, batch, rng, jitter=1.4, heading_jitter=0.25)


def _parking_lanes(spots: int = 14, width: float = 4.0, y_offset: float = 10.0,
                   length: float = 8.0):
    """``ParkingEnv._create_road``: two rows of ``spots`` bays, all four sides continuous.

    Built by the same loop the reference uses, and in the same ORDER -- ``RoadNetwork.graph`` groups
    every ``a->b`` lane before every ``b->c`` one, so the upper row comes first, not the two rows
    interleaved as the loop is written.
    """
    xs = [(k + 1 - spots // 2) * width - width / 2 for k in range(spots)]
    lt = (CONTINUOUS, CONTINUOUS)
    return tuple([Straight([x, y_offset], [x, y_offset + length], width=width, line_types=lt)
                  for x in xs]
                 + [Straight([x, -y_offset], [x, -y_offset - length], width=width, line_types=lt)
                    for x in xs])


@register
class Parking(HighwayEnvBase):
    """parking-v0: a 14-bay lot, one ego, a goal landmark and four walls.

    Three things are specific to this env and each is measured rather than assumed:

      * its ``scaling`` is 7, so ``pix(STRIPE_WIDTH) = 2`` and the markings take the even-width branch
        of ``_line_geom`` -- the same branch intersection-v0 needed;
      * the GOAL is a ``Landmark`` whose bay is drawn at random each episode, so its pose is part of
        the state rather than a class constant, and it is drawn BLUE (``RoadObjectGraphics.get_color``)
        rather than the obstacles' yellow. The 28 poses it takes are the bay centres, which is what
        ``object_spawn`` draws from;
      * the four walls are ``Obstacle``\\ s 70 m and 42 m long -- 490 px and 294 px at this scaling --
        and ``RoadObjectGraphics.display``, unlike ``VehicleGraphics.display``, does NOT test
        ``is_visible``. ``_add_box(cull=False)`` is that difference. Measured on 64 states with the ego
        driven to the edges of the lot, culling road objects the way vehicles are culled costs 7 px a
        frame -- small, because a wall leaves the frame along its 7 px THICKNESS and the visibility
        margin is 50 px, so it is already invisible by the time its centre is culled. It is followed
        because it is the reference's rule, not because the difference is large.

    **``crashed`` is not in this env's state, and that is a measurement rather than an omission.**
    ``ParkingEnv._create_vehicles`` sets ``vehicle.color = VehicleGraphics.EGO_COLOR``, and
    ``get_color`` returns ``vehicle.color`` BEFORE it ever tests ``crashed`` -- so the ego is green
    whether or not it has crashed, there is no second vehicle to paint red, and both the state
    dimension and the ``crashed.color`` knob would be dead weight. ``reference_set_state`` asserts the
    pinned colour is still there, so a package that stopped pinning it fails loudly instead of
    rendering a green car where the reference draws a red one. The same reasoning removes ``traffic``:
    ``vehicles_count`` is 0, so there is no traffic to colour.
    """

    name = "parking"
    default_size = 300
    SCREEN_W, SCREEN_H = 600, 300
    SCALING, CENTERING = 7.0, (0.5, 0.5)
    LANES = _parking_lanes()
    N_VEHICLES = 1
    EGO_HAS_TIRES = True
    CRASHED_VISIBLE = False
    STATE_OBJECTS = (("landmark", 2.0, 2.0),)
    OBJECTS = (("obstacle", 0.0, -21.0, 0.0, 70.0, 1.0),
               ("obstacle", 0.0, 21.0, 0.0, 70.0, 1.0),
               ("obstacle", -35.0, 0.0, math.pi / 2, 42.0, 1.0),
               ("obstacle", 35.0, 0.0, math.pi / 2, 42.0, 1.0))
    #: the 28 bay centres, ``lane.position(lane.length / 2, 0)`` with the lane's own heading
    GOAL_POSES = tuple([(float(l.start[0]), float((l.start[1] + l.end[1]) / 2),
                         math.atan2(float(l.end[1] - l.start[1]), 0.0)) for l in LANES])
    reference_id = "parking-v0"
    reference_config = {"controlled_vehicles": 1, "vehicles_count": 0, "add_walls": True}

    def ego_spawn(self, batch, rng):
        # ``_create_vehicles`` starts the ego at (0, 0) with a heading of 2*pi*U(0,1) and it then
        # drives around the lot, so the spawn is a point and the states are a region
        return np.stack([rng.uniform(-30, 30, batch), rng.uniform(-18, 18, batch),
                         rng.uniform(0, 2 * np.pi, batch)], 1).astype(np.float32)

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
    """highway-v0: four straight lanes, ego plus traffic, camera locked to the ego."""

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
        return np.stack([rng.uniform(100, 900, batch), lane + rng.uniform(-1.2, 1.2, batch),
                         rng.uniform(-0.35, 0.35, batch)], 1).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        lane = rng.integers(0, 4, batch) * 4.0
        return np.stack([ego[:, 0] + rng.uniform(-130, 130, batch),
                         lane + rng.uniform(-1.2, 1.2, batch),
                         rng.uniform(-0.35, 0.35, batch)], 1).astype(np.float32)


def _merge_lanes():
    """``MergeEnv._make_road``: two highway lanes in three sections plus a sinusoidal on-ramp."""
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
    lkb = Sine([ends[0], 14.5 - amplitude], [sum(ends[:2]), 14.5 - amplitude],
               amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=(c, c))
    # lbc starts where the ramp meets the highway: lkb.position(80, 0) = (230, 8)
    out += [ljk, lkb, Straight([sum(ends[:2]), 8.0], [sum(ends[:3]), 8.0], line_types=(n, c))]
    return tuple(out)


@register
class Merge(HighwayEnvBase):
    """merge-v0: a two-lane highway with a sinusoidal on-ramp and a yellow obstacle at its end."""

    name = "merge"
    default_size = 150
    SCREEN_W, SCREEN_H = 600, 150
    SCALING, CENTERING = 5.5, (0.3, 0.5)
    LANES = _merge_lanes()
    OBJECTS = (("obstacle", 310.0, 8.0, 0.0, 2.0, 2.0),)      # Obstacle(road, lbc.position(80, 0))
    N_VEHICLES = 5                                            # MergeEnv._make_vehicles: ego + 4
    reference_id = "merge-v0"

    def ego_spawn(self, batch, rng):
        x = rng.uniform(20, 430, batch)
        # Every third environment is parked next to the obstacle at x=310. The camera window is 109 m
        # wide, so a uniform spawn puts the obstacle on screen only ~27% of the time and the
        # 'object.color' knob would then be checked against batches that sometimes do not contain it —
        # a test whose verdict depends on the draw is not a test.
        x[::3] = rng.uniform(255, 330, len(x[::3]))
        return np.stack([x, rng.integers(0, 2, batch) * 4.0 + rng.uniform(-1.2, 1.2, batch),
                         rng.uniform(-0.3, 0.3, batch)], 1).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        on_ramp = rng.random(batch) < 0.35
        y = np.where(on_ramp, 14.5, rng.integers(0, 2, batch) * 4.0) + rng.uniform(-1.2, 1.2, batch)
        return np.stack([ego[:, 0] + rng.uniform(-110, 110, batch), y,
                         rng.uniform(-0.3, 0.3, batch)], 1).astype(np.float32)


def _roundabout_lanes():
    """``RoundaboutEnv._make_road``: two concentric rings of eight arcs, plus four access branches."""
    radius, alpha, center = 20.0, 24.0, (0.0, 0.0)
    n, c, s = NONE, CONTINUOUS, STRIPED
    radii = [radius, radius + 4]
    line = [(c, s), (n, c)]
    out = []
    seq = [(90 - alpha, alpha), (alpha, -alpha), (-alpha, -90 + alpha), (-90 + alpha, -90 - alpha),
           (-90 - alpha, -180 + alpha), (-180 + alpha, -180 - alpha), (180 - alpha, 90 + alpha),
           (90 + alpha, 90 - alpha)]
    for lane in (0, 1):
        for p0, p1 in seq:
            out.append(Circular(center, radii[lane], np.deg2rad(p0), np.deg2rad(p1),
                                clockwise=False, line_types=line[lane]))
    access, dev, a = 170.0, 85.0, 5.0
    delta_st = 0.2 * dev
    delta_en = dev - delta_st
    w = 2 * np.pi / dev
    out += [
        Straight([2, access], [2, dev / 2], line_types=(s, c)),
        Sine([2 + a, dev / 2], [2 + a, dev / 2 - delta_st], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([-2 - a, -dev / 2 + delta_en], [-2 - a, dev / 2], a, w, -np.pi / 2 + w * delta_en,
             line_types=(c, c)),
        Straight([-2, dev / 2], [-2, access], line_types=(n, c)),
        Straight([access, -2], [dev / 2, -2], line_types=(s, c)),
        Sine([dev / 2, -2 - a], [dev / 2 - delta_st, -2 - a], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([-dev / 2 + delta_en, 2 + a], [dev / 2, 2 + a], a, w, -np.pi / 2 + w * delta_en,
             line_types=(c, c)),
        Straight([dev / 2, 2], [access, 2], line_types=(n, c)),
        Straight([-2, -access], [-2, -dev / 2], line_types=(s, c)),
        Sine([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en,
             line_types=(c, c)),
        Straight([2, -dev / 2], [2, -access], line_types=(n, c)),
        Straight([-access, 2], [-dev / 2, 2], line_types=(s, c)),
        Sine([-dev / 2, 2 + a], [-dev / 2 + delta_st, 2 + a], a, w, -np.pi / 2, line_types=(c, c)),
        Sine([dev / 2 - delta_en, -2 - a], [-dev / 2, -2 - a], a, w, -np.pi / 2 + w * delta_en,
             line_types=(c, c)),
        Straight([-dev / 2, -2], [-access, -2], line_types=(n, c)),
    ]
    return tuple(out)


@register
class Roundabout(HighwayEnvBase):
    """roundabout-v0: a two-ring roundabout with four access branches, ego plus four vehicles.

    At 604 primitives this is the largest scene in the repo, and it was unrenderable until the
    rasterizer stopped unrolling its primitive list: ``_scene_kernel`` used ``tl.static_range``, which
    made the primitive count a compile-time constant, and ptxas aborted the process above ~340. The
    unroll is gone (``raster2d: runtime primitive loop``) and this env is the reason it went; the
    measured ceiling is kept in ``docs/research/highway.md`` because it will come back if anyone
    reintroduces unrolling.

    Its markings are the one place a capsule and a Bresenham line genuinely differ: the arcs are drawn
    by the reference as chains of short 1 px lines at arbitrary angles, and those account for the
    remaining ~0.06% of pixels. All of them are 1 px wide, so none is an interior pixel.
    """

    name = "roundabout"
    default_size = 600
    SCREEN_W, SCREEN_H = 600, 600
    SCALING, CENTERING = 5.5, (0.5, 0.6)
    LANES = _roundabout_lanes()
    N_VEHICLES = 5                       # RoundaboutEnv._make_vehicles: ego + 4
    reference_id = "roundabout-v0"

    def ego_spawn(self, batch, rng):
        th = rng.uniform(-np.pi, np.pi, batch)
        r = rng.uniform(20, 26, batch)
        return np.stack([r * np.cos(th), r * np.sin(th), th + np.pi / 2], 1).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        th = rng.uniform(-np.pi, np.pi, batch)
        ring = rng.random(batch) < 0.6
        r = np.where(ring, rng.uniform(20, 26, batch), rng.uniform(30, 70, batch))
        return np.stack([r * np.cos(th), r * np.sin(th),
                         th + np.pi / 2 + rng.uniform(-0.4, 0.4, batch)], 1).astype(np.float32)


def _intersection_lanes():
    """``IntersectionEnv._make_road``: four corners, each an approach, two turns, a through and an exit."""
    lw = 4.0                                   # AbstractLane.DEFAULT_WIDTH
    r_turn = lw + 5.0                          # 9
    l_turn = r_turn + lw                       # 13
    outer = r_turn + lw / 2                    # 11
    access = 100.0
    n, c, s = NONE, CONTINUOUS, STRIPED
    out = []
    for corner in range(4):
        a = np.radians(90 * corner)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out.append(Straight(R @ np.array([lw / 2, access + outer]),
                            R @ np.array([lw / 2, outer]), lw, (s, c)))
        out.append(Circular(R @ np.array([outer, outer]), r_turn,
                            a + np.radians(180), a + np.radians(270),
                            clockwise=True, width=lw, line_types=(n, c)))
        # the left turn carries no markings at all, so it contributes no primitives
        out.append(Circular(R @ np.array([-l_turn + lw / 2, l_turn - lw / 2]), l_turn,
                            a, a - np.radians(90), clockwise=False, width=lw, line_types=(n, n)))
        out.append(Straight(R @ np.array([lw / 2, outer]),
                            R @ np.array([lw / 2, -outer]), lw, (s, n)))
        # the exit is built as StraightLane(end, start), i.e. it runs outward from the box
        out.append(Straight(R @ np.array([outer, lw / 2]),
                            R @ np.array([access + outer, lw / 2]), lw, (n, c)))
    return tuple(out)


@register
class Intersection(HighwayEnvBase):
    """intersection-v0: a four-way junction, ego approaching from the south, traffic crossing.

    Two things make this one different from the rest of the family. Its ``scaling`` is ``5.5 * 1.3``,
    so ``pix(STRIPE_WIDTH) = 2`` and the markings leave pygame's Bresenham path — see ``_line_geom``
    for the width-2 rule and what it cost to get wrong. And the ego is appended to ``road.vehicles``
    LAST rather than first, so it paints on top of the traffic; ``EGO_INDEX`` says so, because at a
    junction vehicles overlap by design and the order is the difference between the ego being visible
    and being buried.

    The reference decides its cast at random — ``_spawn_vehicle`` draws against a probability and
    ``_clear_vehicles`` deletes vehicles that leave — so the number of vehicles after ``reset`` is not
    a property of the env id. ``reference_set_state`` rebuilds ``road.vehicles`` outright, which is
    what makes a fixed-width state well defined here at all.
    """

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
        t = rng.uniform(30, 95, batch)                      # metres down the approach lane
        lx, ly = 2.0, 111.0 - t                             # lane frame, before the corner rotation
        return np.stack([ca * lx - sa * ly, sa * lx + ca * ly,
                         a - np.pi / 2], 1).astype(np.float32)

    def traffic_spawn(self, batch, rng, ego):
        near_ego = rng.random(batch) < 0.6
        x = np.where(near_ego, ego[:, 0] + rng.uniform(-38, 38, batch), rng.uniform(-45, 45, batch))
        y = np.where(near_ego, ego[:, 1] + rng.uniform(-38, 38, batch), rng.uniform(-45, 45, batch))
        return np.stack([x, y, rng.uniform(-np.pi, np.pi, batch)], 1).astype(np.float32)
