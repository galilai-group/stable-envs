"""Procedural MJCF of the world: the jet Roadster, the loop / ring / landing-pad course and the scenarios.

The car is a sprung, steerable four-wheel chassis carrying nine cold-gas jets that are
modelled as site-attached force actuators:

    lift_fl, lift_fr, lift_rl, lift_rr   downward nozzles -> +z body force (hover, roll, pitch)
    boost_l, boost_r                     rear nozzles     -> +x body force (forward thrust)
    retro                                nose nozzle      -> -x body force (air brake)
    lat_front, lat_rear                  side nozzle pairs -> +/-y body force (sway, yaw)

The textures are generated in memory (textures.assets()); the XML names them by file name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import body_mesh, decor, fx, scenarios, weather
from .body_mesh import mesh_xml
from .course import Course
from .util import vec


@dataclass(frozen=True)
class Jet:
    name: str
    pos: tuple[float, float, float]  # nozzle position in chassis frame
    direction: tuple[float, float, float]  # unit thrust direction in chassis frame
    max_force: float  # [N]
    bidirectional: bool = False  # ctrl in [-1, 1] (paired nozzles) instead of [0, 1]


JETS: tuple[Jet, ...] = (
    Jet("lift_fl", (0.95, 0.55, -0.18), (0, 0, 1), 9000.0),
    Jet("lift_fr", (0.95, -0.55, -0.18), (0, 0, 1), 9000.0),
    Jet("lift_rl", (-0.95, 0.55, -0.18), (0, 0, 1), 9000.0),
    Jet("lift_rr", (-0.95, -0.55, -0.18), (0, 0, 1), 9000.0),
    Jet("boost_l", (-2.12, 0.45, 0.05), (1, 0, 0), 6500.0),
    Jet("boost_r", (-2.12, -0.45, 0.05), (1, 0, 0), 6500.0),
    Jet("retro", (2.12, 0.0, 0.0), (-1, 0, 0), 8000.0),
    Jet("lat_front", (1.6, 0.0, 0.12), (0, 1, 0), 3500.0, bidirectional=True),
    Jet("lat_rear", (-1.6, 0.0, 0.12), (0, 1, 0), 3500.0, bidirectional=True),
)

WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_RADIUS = 0.35
WHEEL_POS = {"fl": (1.4, 0.86), "fr": (1.4, -0.86), "rl": (-1.4, 0.86), "rr": (-1.4, -0.86)}
SPAWN_Z = 0.56
WM_CAM_POS = np.array([-8.0, -9.0, 3.0])
# Silver paint, dark bronze glazing, white light bar, red tail lights, white-blue cold-gas jets
PAINT = "0.63 0.65 0.68 1"
GLASS = "0.11 0.085 0.065 1"
LED = "0.85 0.95 1 1"
TAIL = "1 0.05 0.05 1"
CALIPER = "0.85 0.08 0.08 1"
PLUME_CORE = (0.92, 0.97, 1.0)
PLUME_HALO = (0.45, 0.72, 1.0)
GLOW = (0.95, 0.95, 1.0)  # nozzle colour at full throttle


def _cam_xyaxes(pos, target) -> str:
    """Camera xyaxes for a camera at `pos` looking at `target` with world-up roughly +z."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return vec(np.r_[x, y])


def flame_geoms(jet: Jet) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(geom name, nozzle position, exhaust direction) for every visual flame of a jet."""
    p, d = np.asarray(jet.pos, float), np.asarray(jet.direction, float)
    if not jet.bidirectional:
        return [(f"flame_{jet.name}", p, -d)]
    # paired side nozzles at the body edge: '+' fires for positive ctrl (exhausts toward -d)
    return [
        (f"flame_{jet.name}_pos", p - 0.95 * d, -d),
        (f"flame_{jet.name}_neg", p + 0.95 * d, d),
    ]


def _car_xml(c: Course) -> str:
    chase = np.array([-14.0, -8.0, 5.0])
    out = [
        f'<body name="chassis" pos="{c.start_x} 0 {SPAWN_Z}">',
        '<freejoint name="root"/>',
        f'<camera name="chase" mode="track" pos="{vec(chase)}" xyaxes="{_cam_xyaxes(chase, [0, 0, 0])}"/>',
        # world-model camera: follows the car's position with a fixed world orientation (no swinging)
        f'<camera name="wm" mode="track" pos="{vec(WM_CAM_POS)}" xyaxes="{_cam_xyaxes(WM_CAM_POS, [3, 0, 0])}" fovy="60"/>',
        '<inertial pos="0 0 0" mass="1700" diaginertia="650 2600 3000"/>',
        # hidden collision hull (kept between the axles so the nose clears the ramp)
        '<geom name="chassis_col" type="box" size="1.75 0.85 0.22" pos="0 0 0.12" group="3" rgba="0 0 0 0" friction="0.6"/>',
        '<geom name="roof_col" type="box" size="0.8 0.6 0.1" pos="-0.15 0 0.55" group="3" rgba="0 0 0 0"/>',
    ]
    vis = 'contype="0" conaffinity="0" density="0"'
    out += _details(vis)

    for jet in JETS:  # engine housings, glowing nozzles, two-layer exhaust plumes
        lift = jet.name.startswith("lift")
        boost = jet.name.startswith("boost")
        for name, p, ex in flame_geoms(jet):
            p, ex = _port(jet, p, ex)  # nozzle housings and glow discs on the skin, not poking through it
            if lift:  # ducted lift-fan pod under the floor
                out.append(
                    f'<geom type="cylinder" size="0.21 0.07" pos="{vec(p - ex * 0.06)}" zaxis="{vec(ex)}" '
                    f'rgba="0.16 0.17 0.19 1" material="metal" {vis}/>'
                )
                out.append(
                    f'<geom type="cylinder" size="0.23 0.015" pos="{vec(p - ex * 0.0)}" zaxis="{vec(ex)}" '
                    f'rgba="0.7 0.72 0.75 1" material="metal" {vis}/>'
                )
                glow_r = 0.16
            elif boost:  # rocket nozzle recessed fully behind the leaning diffuser face
                out.append(
                    f'<geom type="cylinder" size="0.085 0.08" pos="{vec(p - ex * 0.12)}" '
                    f'zaxis="{vec(ex)}" rgba="0.06 0.06 0.07 1" material="metal" {vis}/>'
                )
                glow_r = 0.025  # inside the small nozzle bores
            else:  # small reaction-control port, flush with the body
                out.append(
                    f'<geom type="cylinder" size="0.045 0.03" pos="{vec(p - ex * 0.03)}" zaxis="{vec(ex)}" '
                    f'rgba="0.05 0.05 0.06 1" {vis}/>'
                )
                glow_r = 0.035
            glow_off = 0.025 if lift else (0.012 if boost else 0.004)  # just proud of the skin
            glow_a = 0 if jet.bidirectional else 1  # side ports: invisible until they fire (RoadsterSim.draw_effects)
            out.append(
                f'<geom name="glow_{name[6:]}" type="cylinder" size="{glow_r} 0.006" '
                f'pos="{vec(p + ex * glow_off)}" zaxis="{vec(ex)}" rgba="0.1 0.25 0.35 {glow_a}" material="flame" {vis}/>'
            )
            p, ex = next((q, e) for n_, q, e in flame_geoms(jet) if n_ == name)  # flames stay at the jet
            out.append(
                f'<geom name="{name}" type="capsule" size="0.06 0.05" pos="{vec(p + ex * 0.2)}" '
                f'zaxis="{vec(ex)}" rgba="0.9 0.97 1 0" material="flame" group="1" {vis}/>'
            )
            out.append(
                f'<geom name="{name}_halo" type="capsule" size="0.14 0.05" pos="{vec(p + ex * 0.2)}" '
                f'zaxis="{vec(ex)}" rgba="0.45 0.75 1 0" material="flame" group="1" {vis}/>'
            )
        out.append(f'<site name="{jet.name}" pos="{vec(jet.pos)}" size="0.05" rgba="1 0.5 0 0" zaxis="0 0 1"/>')
    out += [  # lights
        '<light name="headlight_l" mode="track" pos="1.95 0.66 0.25" dir="1 0 -0.12" directional="false" '
        'cutoff="28" exponent="12" diffuse="0.9 0.9 0.85" specular="0.2 0.2 0.2" attenuation="1 0.02 0.002" castshadow="false"/>',
        '<light name="headlight_r" mode="track" pos="1.95 -0.66 0.25" dir="1 0 -0.12" directional="false" '
        'cutoff="28" exponent="12" diffuse="0.9 0.9 0.85" specular="0.2 0.2 0.2" attenuation="1 0.02 0.002" castshadow="false"/>',
        '<light name="engine_glow" mode="track" pos="0 0 -0.3" dir="0 0 -1" directional="false" cutoff="70" '
        'exponent="2" diffuse="0 0 0" specular="0 0 0" attenuation="1 0.1 0.05" castshadow="false"/>',
    ]
    out.append('<site name="imu" pos="0 0 0" size="0.02" rgba="0 0 0 0"/>')
    for w in WHEELS:
        x, y = WHEEL_POS[w]
        steer = w.startswith("f")
        out.append(f'<body name="knuckle_{w}" pos="{x} {y} -0.15">')
        out.append('<inertial pos="0 0 0" mass="10" diaginertia="0.1 0.1 0.1"/>')
        out.append(
            f'<joint name="susp_{w}" type="slide" axis="0 0 1" range="-0.14 0.12" '
            'stiffness="90000" springref="-0.06" damping="7000" armature="1"/>'
        )
        if steer:
            out.append(
                f'<joint name="steer_{w}" type="hinge" axis="0 0 1" range="-0.55 0.55" damping="80" armature="2"/>'
            )
        cs = 1 if y > 0 else -1
        out.append(
            f'<geom type="box" size="0.09 0.03 0.06" pos="0.0 {0.1 * cs} 0.17" rgba="{CALIPER}" '
            f'material="paint" {vis}/>'
        )
        out.append(f'<body name="wheel_{w}">')
        out.append('<inertial pos="0 0 0" mass="30" diaginertia="1.1 1.84 1.1"/>')
        out.append(f'<joint name="spin_{w}" type="hinge" axis="0 1 0" damping="0.4"/>')
        out.append(
            f'<geom name="tire_{w}" type="cylinder" size="{WHEEL_RADIUS} 0.14" euler="90 0 0" '
            'rgba="0.06 0.06 0.06 1" friction="1.3 0.02 0.002" condim="4" priority="1"/>'
        )
        side = 1 if y > 0 else -1
        out += _wheel_face(side, vis)
        out.append("</body></body>")
    out.append("</body>")
    return "\n".join(out)


def _wheel_face(side: int, vis: str) -> list[str]:
    # light aero disc behind a black split six-spoke star, dark lip and centre cap
    # faces just proud of the tyre's outer face (y = 0.14), up to y = 1.027: the body side over the
    # arches is flared to 1.03-1.08 so no part of the wheel reaches it
    out = [
        f'<geom type="cylinder" size="0.325 0.006" pos="0 {0.138 * side} 0" euler="90 0 0" '
        f'rgba="0.08 0.08 0.09 1" {vis}/>',
        f'<geom type="cylinder" size="0.31 0.01" pos="0 {0.142 * side} 0" euler="90 0 0" '
        f'rgba="0.6 0.62 0.65 1" material="metal" {vis}/>',
    ]  # low-profile tyre: 2.5 cm black ring
    for a in range(0, 360, 60):
        for da in (-5, 5):
            ang = np.deg2rad(a + da)
            out.append(
                f'<geom type="box" size="0.15 0.006 0.012" pos="{0.155 * np.cos(ang):.4f} {0.153 * side} '
                f'{0.155 * np.sin(ang):.4f}" euler="0 {-(a + da)} 0" rgba="0.05 0.05 0.06 1" material="metal" {vis}/>'
            )
    out.append(
        f'<geom type="cylinder" size="0.05 0.01" pos="0 {0.157 * side} 0" euler="90 0 0" '
        f'rgba="0.05 0.05 0.06 1" material="metal" {vis}/>'
    )
    return out


def _details(vis: str) -> list[str]:
    """Lights, aero and trim, after photos of the reveal car.

    Every part sits on (or a few mm proud of) the faceted shell, inside its outline; the shell's own black
    band mesh is the lower fascia, rocker band and underside.
    """
    dark = "0.03 0.03 0.035 1"
    keys = body_mesh.REVEAL_KEYS[np.argsort(body_mesh.REVEAL_KEYS[:, 0])]
    nose = keys[-1]  # the blunt nose face (x = nose[0]); its crease nose[5] is the blade line
    vent_y, lean = body_mesh.side_point(0.86, 0.0)  # side plane behind the front wheel
    door_y, door_lean = body_mesh.side_point(0.95, 0.0)
    door_lo, door_hi = float(np.interp(0.95, keys[:, 0], keys[:, 3])), float(np.interp(0.95, keys[:, 0], keys[:, 5]))
    tail, foot = keys[0], keys[1]  # tail-edge station and the station at the foot of the undercut diffuser face
    dif_x, dif_z = 0.5 * (tail[0] + foot[0]), 0.5 * (tail[2] + foot[2])  # centre of the diffuser face
    dif_tilt = float(np.degrees(np.arctan2(tail[0] - foot[0], tail[2] - foot[2])))  # its lean (deg about y)
    out = [
        f'<geom type="mesh" mesh="car_body" rgba="{PAINT}" material="car_paint" {vis}/>',
        f'<geom type="mesh" mesh="car_glass" rgba="{GLASS}" material="visor" {vis}/>',
        f'<geom type="mesh" mesh="car_band" rgba="0.05 0.05 0.055 1" material="car_paint" {vis}/>',
        # one thin white light bar along the blade line across the blunt nose face, over the black fascia
        f'<geom type="box" size="0.005 {nose[4] - 0.02} 0.006" pos="{nose[0] + 0.005} 0 {nose[5] - 0.014}" rgba="{LED}" material="lamp" {vis}/>',
        # thin splitter lip at the foot of the fascia
        f'<geom type="box" size="0.04 {0.9 * nose[1] - 0.03} 0.005" pos="{nose[0] - 0.03} 0 {nose[2] - 0.006}" rgba="{dark}" {vis}/>',
        # rear: two thin red light lines just under the sharp tail edge (the upper one wraps the corners),
        # plate centred in the silver band below; the black diffuser is the shell's undercut lower face
        f'<geom type="box" size="0.006 {tail[4] - 0.015} 0.005" pos="{tail[0] - 0.006} 0 {tail[5] - 0.006}" rgba="{TAIL}" material="lamp" {vis}/>',
        f'<geom type="box" size="0.005 {tail[4] - 0.09} 0.0035" pos="{tail[0] - 0.005} 0 {tail[5] - 0.03}" rgba="{TAIL}" material="lamp" {vis}/>',
        f'<geom type="box" size="0.005 0.17 0.04" pos="{tail[0] - 0.005} 0 {0.5 * (tail[2] + tail[5])}" rgba="0.94 0.94 0.92 1" {vis}/>',
        # vertical fins standing proud of the leaning diffuser face
        *[
            f'<geom type="box" size="0.025 0.008 0.16" pos="{dif_x - 0.016} {y} {dif_z - 0.004}" euler="0 {dif_tilt} 0" rgba="{dark}" {vis}/>'
            for y in (-0.7, -0.35, 0.0, 0.35, 0.7)
        ],
        # small diagonal vent slash behind the front wheel with an amber marker; faint door cut line behind it
        *[
            f'<geom type="box" size="0.012 0.005 0.07" pos="0.84 {sy * (vent_y + 0.005)} 0.05" euler="{sy * lean} -38 0" '
            f'rgba="{dark}" {vis}/>'
            for sy in (1, -1)
        ],
        *[
            f'<geom type="box" size="0.008 0.004 0.03" pos="0.80 {sy * (vent_y + 0.012)} 0.1" euler="{sy * lean} -38 0" '
            f'rgba="1 0.55 0.1 1" material="lamp" {vis}/>'
            for sy in (1, -1)
        ],
        *[
            f'<geom type="box" size="0.002 0.003 {(door_hi - door_lo) / 2 - 0.02}" pos="0.95 {sy * (door_y + 0.002)} '
            f'{(door_hi + door_lo) / 2}" euler="{sy * door_lean} 0 0" rgba="{dark}" {vis}/>'
            for sy in (1, -1)
        ],
    ]
    for y in (-0.45, -0.15, 0.15, 0.45):  # row of small round thruster nozzles recessed in the diffuser face
        out.append(
            f'<geom type="cylinder" size="0.04 0.012" pos="-2.108 {y} 0.05" zaxis="1 0 0" '
            f'rgba="0.5 0.52 0.55 1" material="metal" {vis}/>'
        )
        out.append(
            f'<geom type="cylinder" size="0.028 0.015" pos="-2.108 {y} 0.05" zaxis="1 0 0" rgba="0.02 0.02 0.02 1" {vis}/>'
        )
        if abs(y) < 0.3:  # the inner pair has no jet: same teal disc as the boost nozzles' glow at idle
            out.append(
                f'<geom type="cylinder" size="0.025 0.006" pos="-2.132 {y} 0.05" zaxis="1 0 0" rgba="0.1 0.25 0.35 1" '
                f'material="flame" {vis}/>'
            )
    # no wing and no lip: the reveal car's rear deck ends in a clean cut-off edge
    return out


def _port(jet: Jet, p: np.ndarray, ex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where a jet's visible nozzle sits on the shell: (port position, outward normal).

    The thrust site and the flame stay at the Jet definition; only the housing / glow geoms move onto the skin.
    """
    if jet.name == "retro":  # in the black band on the nose blade face
        nose = body_mesh.REVEAL_KEYS[np.argmax(body_mesh.REVEAL_KEYS[:, 0])]
        return np.array([nose[0], 0.0, 0.5 * (nose[2] + nose[3])]), ex
    if jet.name.startswith("lat"):  # flush with the side plane, just clear of the wheel arch
        x = np.sign(p[0]) * 1.92
        y, lean = body_mesh.side_point(x, p[2])
        n = np.array([0.0, np.cos(np.radians(lean)), np.sin(np.radians(lean))]) * np.sign(ex[1])
        return np.array([x, np.sign(ex[1]) * y, p[2]]), n
    return p, ex


def _car_meshes() -> str:
    """The faceted shell split into paint / glass / black band meshes (body_mesh.shell_faceted)."""
    shell = body_mesh.shell_faceted()
    return "\n    ".join(mesh_xml(f"car_{part}", *shell[part]) for part in ("body", "glass", "band"))


def _course_xml(c: Course) -> str:
    out = []
    road_len = -c.start_x + 25.0
    out.append(
        f'<geom name="road" type="box" size="{road_len / 2} {c.road_width / 2} 0.005" '
        f'pos="{-road_len / 2 + 2} 0 0.005" material="asphalt" contype="0" conaffinity="0"/>'
    )
    for x in np.arange(c.start_x - 20, 0, 8.0):
        out.append(
            f'<geom type="box" size="1.6 0.08 0.006" pos="{x} 0 0.008" rgba="0.95 0.85 0.3 1" '
            'contype="0" conaffinity="0"/>'
        )
    for y in (c.road_width / 2 - 0.2, -c.road_width / 2 + 0.2):
        out.append(
            f'<geom type="box" size="{road_len / 2} 0.08 0.006" pos="{-road_len / 2 + 2} {y} 0.008" '
            'rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>'
        )
    # ramp: one collision box per planar segment
    poly = c.ramp_polyline()
    t = 0.3
    for i, ((x0, z0), (x1, z1)) in enumerate(zip(poly[:-1], poly[1:])):
        mid = np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])
        L = float(np.hypot(x1 - x0, z1 - z0))
        a = np.arctan2(z1 - z0, x1 - x0)
        n = np.array([-np.sin(a), 0.0, np.cos(a)])
        ctr = mid - n * t / 2
        out.append(
            f'<geom name="ramp_{i}" type="box" size="{L / 2 + 0.02} {c.ramp_width / 2} {t / 2}" '
            f'pos="{vec(ctr)}" euler="0 {-np.rad2deg(a):.4f} 0" material="ramp" friction="1.1"/>'
        )
    # flight rings (visual only) on poles
    for k, ring in enumerate(c.rings):
        cx, cy, cz = ring.center
        n_seg = 28
        for j in range(n_seg):
            a0, a1 = 2 * np.pi * j / n_seg, 2 * np.pi * (j + 1) / n_seg
            p0 = np.array([cx, cy + ring.radius * np.cos(a0), cz + ring.radius * np.sin(a0)])
            p1 = np.array([cx, cy + ring.radius * np.cos(a1), cz + ring.radius * np.sin(a1)])
            out.append(
                f'<geom name="ring{k}_{j}" type="capsule" size="0.22" fromto="{vec(p0)} {vec(p1)}" '
                'material="ring" contype="0" conaffinity="0"/>'
            )
        bottom = cz - ring.radius
        out.append(
            f'<geom type="cylinder" size="0.18 {bottom / 2}" pos="{cx} {cy} {bottom / 2}" '
            'material="metal" rgba="0.35 0.35 0.4 1" contype="0" conaffinity="0"/>'
        )
    # elevated landing pad
    px, py = c.pad_center
    sx, sy = c.pad_size
    h = c.pad_height
    out.append(
        f'<geom name="pad" type="box" size="{sx} {sy} {h / 2}" pos="{px} {py} {h / 2}" material="pad" friction="1.2"/>'
    )
    for dx, dy, hx, hy in ((0, -2.2, 3.0, 0.35), (0, 2.2, 3.0, 0.35), (0, 0, 0.35, 2.2)):  # "H"
        out.append(
            f'<geom type="box" size="{hx} {hy} 0.01" pos="{px + dx} {py + dy} {h + 0.005}" '
            'rgba="1 1 1 1" contype="0" conaffinity="0"/>'
        )
    for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        out.append(
            f'<geom type="box" size="0.4 0.12 0.01" pos="{px + 6 * np.cos(ang)} {py + 6 * np.sin(ang)} {h + 0.004}" '
            f'euler="0 0 {np.rad2deg(ang) + 90:.2f}" rgba="1 0.8 0.1 1" contype="0" conaffinity="0"/>'
        )
    for sxs in (-1, 1):
        for sys_ in (-1, 1):
            out.append(
                f'<geom type="sphere" size="0.25" pos="{px + sxs * sx} {py + sys_ * sy} {h + 0.25}" '
                'material="lamp" rgba="0.2 1 0.3 1" contype="0" conaffinity="0"/>'
            )
    return "\n".join(out)


def _actuators_xml() -> str:
    out = ["<actuator>"]
    for w in WHEELS:
        out.append(f'<motor name="drive_{w}" joint="spin_{w}" gear="1100" ctrlrange="-1 1" ctrllimited="true"/>')
    for w in ("fl", "fr"):
        out.append(f'<position name="steer_{w}" joint="steer_{w}" kp="9000" ctrlrange="-0.5 0.5" ctrllimited="true"/>')
    for jet in JETS:
        g = np.asarray(jet.direction, float) * jet.max_force
        lo = -1 if jet.bidirectional else 0
        out.append(
            f'<motor name="{jet.name}" site="{jet.name}" gear="{vec(g)} 0 0 0" ctrlrange="{lo} 1" ctrllimited="true"/>'
        )
    out.append("</actuator>")
    return "\n".join(out)


def _exclude_xml() -> str:
    # the hull would otherwise collide with its own (grand-child) tires
    return "\n".join(f'    <exclude body1="chassis" body2="wheel_{w}"/>' for w in WHEELS) + "\n"


def build_xml(course: Course | None = None) -> str:
    """The world model XML.

    The car, the main course, the traffic-jam highway, the rooftop city and the car-hop
    plaza, the rain pool and the smoke pool. The ground is a non-repeating texture so that pixel
    observations identify where the car is.
    """
    c = course or Course()
    side_pos = np.array([10.0, -45.0, 12.0])
    return f"""<mujoco model="roadster_jets">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" density="0"/>
  <size memory="64M"/>
  <statistic extent="100" center="50 0 5"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="50"/>
    <quality shadowsize="8192" offsamples="4"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <map znear="0.002" zfar="12" haze="0.3" shadowclip="0.35"/>
    <rgba haze="0.75 0.82 0.92 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.35 0.55 0.85" rgb2="0.85 0.9 0.97" width="512" height="512"/>
    <texture name="ground" type="2d" file="terrain.png"/>
    <texture name="pad" type="2d" builtin="checker" rgb1="0.22 0.23 0.26" rgb2="0.2 0.21 0.24" width="256" height="256"/>
    <material name="ground" texture="ground" texrepeat="1 1"/>
    <material name="asphalt" rgba="0.16 0.16 0.18 1"/>
    <material name="ramp" rgba="0.32 0.33 0.36 0.45"/>
    <material name="pad" texture="pad" texrepeat="6 4"/>
    <material name="paint" specular="1" shininess="0.95" reflectance="0.0"/>
    <material name="glass" specular="1" shininess="1"/>
    <material name="visor" specular="1" shininess="1" reflectance="0"/>
    <material name="car_paint" specular="1.0" shininess="0.95"/>
    {_car_meshes()}
    <material name="metal" specular="0.8" shininess="0.7"/>
    <material name="lamp" emission="1"/>
    <material name="flame" emission="1"/>
    {fx.ASSET}
    <material name="ring" rgba="0.1 0.85 1 1" emission="0.8"/>
    {scenarios.assets_xml()}
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 100" dir="-0.4 0.3 -1" diffuse="0.8 0.78 0.72" specular="0.3 0.3 0.3" castshadow="true"/>
    <geom name="ground" type="plane" size="500 300 1" pos="100 0 0" material="ground" friction="1.0"/>
    <camera name="side" pos="{vec(side_pos)}" xyaxes="{_cam_xyaxes(side_pos, [10, 0, 4])}"/>
{_course_xml(c)}
    <geom name="car_shadow" type="ellipsoid" size="2.3 1.2 0.01" pos="0 0 -50" rgba="0 0 0 0.45" contype="0" conaffinity="0" group="2"/>
{scenarios.world_xml()}
{decor.loop_decor(c)}
{weather.xml()}
{fx.xml()}
{_car_xml(c)}
  </worldbody>
  <contact>
{_exclude_xml()}  </contact>
{_actuators_xml()}
</mujoco>
"""
