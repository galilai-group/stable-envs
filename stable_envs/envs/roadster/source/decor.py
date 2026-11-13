"""Visual-only scenery for the loop course, the traffic-jam highway, the rooftop city and the car-hop plaza.

Everything here is contype=0 / conaffinity=0 (no physics) and sits outside the flight corridors, so it
does not affect tasks, collisions or the scripted expert.
"""

from __future__ import annotations

import numpy as np

from .util import vec

VIS = 'contype="0" conaffinity="0"'


# ---------------------------------------------------------------------------------------------- materials
def assets_xml() -> str:
    """Scenery textures (by their ``textures.assets()`` names) and materials."""
    return "\n".join(
        [
            '<texture name="facade_glass" type="2d" file="facade_glass.png"/>',
            '<material name="facade_glass" texture="facade_glass" texrepeat="2 3" specular="1" shininess="0.95"/>',
            '<texture name="facade_stone" type="2d" file="facade_stone.png"/>',
            '<material name="facade_stone" texture="facade_stone" texrepeat="3 4" specular="0.2" shininess="0.3"/>',
            '<texture name="banner" type="2d" file="banner.png"/>',
            '<material name="banner" texture="banner" texrepeat="1 1" emission="0.2"/>',
            '<texture name="crowd" type="2d" file="crowd.png"/>',
            '<material name="crowd" texture="crowd" texrepeat="4 1"/>',
            '<texture name="sign" type="2d" file="sign.png"/>',
            '<material name="sign" texture="sign" texrepeat="1 1" emission="0.15"/>',
            '<texture name="grass" type="2d" file="grass.png"/>',
            '<material name="grass" texture="grass" texrepeat="12 1"/>',
            '<material name="steel" rgba="0.7 0.72 0.76 1" specular="0.9" shininess="0.8"/>',
            '<material name="kerb_red" rgba="0.85 0.1 0.12 1" specular="0.3"/>',
            '<material name="kerb_white" rgba="0.95 0.95 0.95 1" specular="0.3"/>',
            '<material name="leaf" rgba="0.2 0.45 0.2 1" specular="0.1"/>',
            '<material name="leaf2" rgba="0.15 0.38 0.18 1" specular="0.1"/>',
            '<material name="bark" rgba="0.38 0.26 0.16 1"/>',
            '<material name="hill" rgba="0.42 0.55 0.32 1"/>',
        ]
    )


# ---------------------------------------------------------------------------------------------- props
def tree(x: float, y: float, s: float = 1.0, pine: bool = False) -> str:
    out = [f'<geom type="cylinder" size="{0.18 * s} {1.3 * s}" pos="{x} {y} {1.3 * s}" material="bark" {VIS}/>']
    if pine:
        for k, (r, z) in enumerate(((1.6, 2.6), (1.25, 3.6), (0.85, 4.5), (0.45, 5.2))):
            out.append(
                f'<geom type="ellipsoid" size="{r * s} {r * s} {0.65 * s}" pos="{x} {y} {z * s}" '
                f'material="{"leaf2" if k % 2 else "leaf"}" {VIS}/>'
            )
    else:
        for dx, dy, dz, r in ((0, 0, 3.2, 1.6), (0.7, 0.3, 2.7, 1.1), (-0.6, -0.4, 2.8, 1.2), (0.1, 0.6, 3.8, 1.0)):
            out.append(
                f'<geom type="ellipsoid" size="{r * s} {r * s} {r * 0.9 * s}" '
                f'pos="{x + dx * s} {y + dy * s} {dz * s}" material="leaf" {VIS}/>'
            )
    return "\n".join(out)


def lamp(x: float, y: float, toward_y: float, h: float = 8.0) -> str:
    arm = np.sign(toward_y - y) * 1.6
    return "\n".join(
        [
            f'<geom type="cylinder" size="0.1 {h / 2}" pos="{x} {y} {h / 2}" rgba="0.35 0.37 0.4 1" material="steel" {VIS}/>',
            f'<geom type="capsule" size="0.06" fromto="{x} {y} {h} {x} {y + arm} {h + 0.2}" rgba="0.35 0.37 0.4 1" {VIS}/>',
            f'<geom type="ellipsoid" size="0.35 0.22 0.08" pos="{x} {y + arm} {h + 0.12}" rgba="0.25 0.25 0.27 1" {VIS}/>',
            f'<geom type="ellipsoid" size="0.28 0.17 0.03" pos="{x} {y + arm} {h + 0.05}" rgba="1 0.92 0.7 1" material="lamp" {VIS}/>',
        ]
    )


def parked_car(i: int, color: str, size) -> list[str]:
    """Visual body for traffic car slot i (sedan / SUV / hatchback by index) inside the collision box."""
    hx, hy, _ = size
    kind = ("sedan", "suv", "hatch")[i % 3]
    roof_h = {"sedan": 1.15, "suv": 1.38, "hatch": 1.25}[kind]
    cab_len = {"sedan": 0.5, "suv": 0.68, "hatch": 0.6}[kind]
    cab_x = {"sedan": -0.15, "suv": -0.35, "hatch": -0.45}[kind]
    g = [
        f'<geom type="box" size="{hx - 0.05} {hy} 0.28" pos="0 0 0.55" rgba="{color}" material="paint" {VIS}/>',
        f'<geom type="ellipsoid" size="{hx * 0.42} {hy * 0.98} 0.22" pos="{hx * 0.55} 0 0.72" rgba="{color}" material="paint" {VIS}/>',
        f'<geom type="ellipsoid" size="{hx * 0.35} {hy * 0.98} 0.22" pos="{-hx * 0.62} 0 0.74" rgba="{color}" material="paint" {VIS}/>',
        f'<geom type="ellipsoid" size="{hx * cab_len} {hy * 0.9} {roof_h - 0.78}" pos="{cab_x} 0 0.8" '
        f'rgba="0.07 0.09 0.13 1" material="glass" {VIS}/>',
        f'<geom type="box" size="{hx * cab_len * 0.75} {hy * 0.82} 0.03" pos="{cab_x} 0 {roof_h}" rgba="{color}" '
        f'material="paint" {VIS}/>',
        f'<geom type="box" size="{hx * 0.98} {hy * 1.01} 0.09" pos="0 0 0.32" rgba="0.08 0.08 0.09 1" {VIS}/>',
        # grille, number plates
        f'<geom type="box" size="0.02 {hy * 0.45} 0.06" pos="{hx - 0.02} 0 0.5" rgba="0.05 0.05 0.06 1" material="metal" {VIS}/>',
        f'<geom type="box" size="0.01 0.24 0.06" pos="{hx + 0.005} 0 0.38" rgba="0.92 0.92 0.88 1" {VIS}/>',
        f'<geom type="box" size="0.01 0.24 0.06" pos="{-hx - 0.005} 0 0.5" rgba="0.92 0.92 0.88 1" {VIS}/>',
    ]
    for sy in (-1, 1):  # side windows with a chrome sill, door cut, mirror, dark wheel wells
        g.append(
            f'<geom type="box" size="{hx * cab_len * 0.85} 0.012 {(roof_h - 0.86) / 2}" '
            f'pos="{cab_x} {sy * hy * 0.86} {(roof_h + 0.86) / 2}" rgba="0.06 0.08 0.11 1" material="glass" {VIS}/>'
        )
        g.append(
            f'<geom type="box" size="{hx * cab_len * 0.85} 0.014 0.008" pos="{cab_x} {sy * hy * 0.87} 0.86" '
            f'rgba="0.8 0.82 0.85 1" material="metal" {VIS}/>'
        )
        g.append(
            f'<geom type="box" size="0.006 0.006 0.22" pos="{cab_x + 0.05} {sy * (hy + 0.003)} 0.58" '
            f'rgba="0.05 0.05 0.05 1" {VIS}/>'
        )
        g.append(
            f'<geom type="ellipsoid" size="0.08 0.07 0.045" pos="{cab_x + hx * cab_len * 0.8} {sy * (hy + 0.06)} 0.88" '
            f'rgba="{color}" material="paint" {VIS}/>'
        )
        for sx in (-1, 1):
            g.append(
                f'<geom type="cylinder" size="0.4 0.005" pos="{sx * (hx - 0.68)} {sy * (hy + 0.002)} 0.36" '
                f'euler="90 0 0" rgba="0.03 0.03 0.03 1" {VIS}/>'
            )
    for sy in (-1, 1):
        g.append(
            f'<geom type="ellipsoid" size="0.04 0.2 0.06" pos="{hx - 0.03} {sy * hy * 0.68} 0.66" '
            f'rgba="1 0.97 0.85 1" material="lamp" {VIS}/>'
        )
        g.append(
            f'<geom type="ellipsoid" size="0.04 0.2 0.06" pos="{-hx + 0.03} {sy * hy * 0.68} 0.7" '
            f'rgba="1 0.08 0.06 1" material="lamp" {VIS}/>'
        )
        for sx in (-1, 1):
            x, y = sx * (hx - 0.68), sy * (hy - 0.02)
            g.append(
                f'<geom type="cylinder" size="0.34 0.12" pos="{x} {y} 0.34" euler="90 0 0" rgba="0.05 0.05 0.05 1" {VIS}/>'
            )
            g.append(
                f'<geom type="cylinder" size="0.2 0.01" pos="{x} {y + sy * 0.12} 0.34" euler="90 0 0" '
                f'rgba="0.75 0.76 0.8 1" material="metal" {VIS}/>'
            )
    return g


# ---------------------------------------------------------------------------------------------- loop course
def loop_decor(course) -> str:
    c = course
    out = []
    poly = c.ramp_polyline()
    t = 0.3
    for i, ((x0, z0), (x1, z1)) in enumerate(zip(poly[:-1], poly[1:])):
        mid = np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])
        L = float(np.hypot(x1 - x0, z1 - z0))
        a = np.arctan2(z1 - z0, x1 - x0)
        n = np.array([-np.sin(a), 0.0, np.cos(a)])  # riding-surface normal (toward the loop center)
        e = f'euler="0 {-np.rad2deg(a):.4f} 0"'
        mat = "kerb_red" if i % 2 == 0 else "kerb_white"
        for y in (c.ramp_width / 2 - 0.25, -c.ramp_width / 2 + 0.25):  # curbs on the riding surface
            out.append(
                f'<geom type="box" size="{L / 2 + 0.01} 0.25 0.03" pos="{vec(mid + n * 0.03 + np.array([0, y, 0]))}" '
                f'{e} material="{mat}" {VIS}/>'
            )
        if i % 2 == 0:  # dashed center line
            out.append(
                f'<geom type="box" size="{L / 4} 0.08 0.02" pos="{vec(mid + n * 0.025)}" {e} '
                f'rgba="1 0.85 0.2 0.9" {VIS}/>'
            )
        for y in (c.ramp_width / 2 + 0.12, -c.ramp_width / 2 - 0.12):  # side walls with a steel cap
            out.append(
                f'<geom type="box" size="{L / 2} 0.06 0.45" pos="{vec(mid - n * (t / 2) + np.array([0, y, 0]) + n * 0.45)}" '
                f'{e} rgba="0.75 0.8 0.88 0.35" material="glass" {VIS}/>'
            )
            out.append(
                f'<geom type="box" size="{L / 2} 0.09 0.04" pos="{vec(mid + n * 0.95 + np.array([0, y, 0]))}" '
                f'{e} material="steel" {VIS}/>'
            )
        # steel truss frames on the outside of the loop every 6 segments
        if i % 6 == 3:
            outer = mid - n * (t + 0.6)
            for y in (c.ramp_width / 2 + 0.6, -c.ramp_width / 2 - 0.6):
                foot = np.array([outer[0], y, 0.0])
                top = np.array([outer[0], y, outer[2]])
                out.append(
                    f'<geom type="capsule" size="0.16" fromto="{vec(foot)} {vec(top)}" material="steel" {VIS}/>'
                )
                for zb in np.arange(4.0, outer[2] - 1.0, 4.0):  # horizontal bracing rings
                    out.append(
                        f'<geom type="cylinder" size="0.22 0.06" pos="{foot[0]} {y} {zb}" material="steel" {VIS}/>'
                    )
            out.append(
                f'<geom type="capsule" size="0.1" fromto="{outer[0]} {c.ramp_width / 2 + 0.6} {outer[2]} '
                f'{outer[0]} {-c.ramp_width / 2 - 0.6} {outer[2]}" material="steel" {VIS}/>'
            )
    # launch arch over the road
    ax = -28.0
    for y in (-c.road_width / 2 - 1.0, c.road_width / 2 + 1.0):
        out.append(f'<geom type="box" size="0.35 0.35 3.2" pos="{ax} {y} 3.2" rgba="0.12 0.12 0.14 1" {VIS}/>')
    out.append(
        f'<geom type="box" size="0.3 {c.road_width / 2 + 1.35} 0.75" pos="{ax} 0 7.0" material="banner" {VIS}/>'
    )
    # grandstand on the camera side of the run-up
    for k in range(5):
        out.append(
            f'<geom type="box" size="22 {1.0} {0.4 + 0.45 * k}" pos="-55 {-12.5 - 1.6 * k} {0.4 + 0.45 * k}" '
            f'material="crowd" {VIS}/>'
        )
    out.append(f'<geom type="box" size="22.5 4.5 0.12" pos="-55 -16.5 5.2" rgba="0.85 0.15 0.15 1" {VIS}/>')
    for x in (-76, -55, -34):
        out.append(f'<geom type="cylinder" size="0.12 2.6" pos="{x} -20.5 2.6" material="steel" {VIS}/>')
    # flags and cones along the run-up
    cols = ("0.85 0.1 0.12 1", "0.1 0.3 0.8 1", "0.95 0.8 0.1 1", "0.95 0.95 0.95 1")
    for k, x in enumerate(np.arange(-120, -5, 10.0)):
        for y in (-c.road_width / 2 - 1.5, c.road_width / 2 + 1.5):
            out.append(f'<geom type="cylinder" size="0.04 2.5" pos="{x} {y} 2.5" material="steel" {VIS}/>')
            out.append(f'<geom type="box" size="0.6 0.02 0.4" pos="{x - 0.6} {y} 4.55" rgba="{cols[k % 4]}" {VIS}/>')
        out.append(
            f'<geom type="ellipsoid" size="0.2 0.2 0.4" pos="{x + 5} {c.road_width / 2 + 0.5} 0.4" '
            f'rgba="1 0.45 0.05 1" {VIS}/>'
        )
    for x, y, s, pine in (
        (-90, 26, 1.2, False),
        (-70, 30, 1.0, True),
        (-40, 28, 1.3, False),
        (-110, -32, 1.1, True),
        (-20, -34, 1.2, False),
        (-95, -36, 1.0, False),
        (-130, 22, 1.2, True),
    ):
        out.append(tree(x, y, s, pine))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- highway
def traffic_decor(y0: float) -> str:
    out = []
    rng = np.random.default_rng(21)
    for side in (-1, 1):  # grass verges, guardrail, trees, distant hills
        out.append(f'<geom type="box" size="140 9 0.004" pos="30 {y0 + side * 15.2} 0.004" material="grass" {VIS}/>')
        out.append(f'<geom type="box" size="140 0.04 0.12" pos="30 {y0 + side * 7.6} 0.85" material="steel" {VIS}/>')
        for x in np.arange(-105, 170, 4.0):
            out.append(
                f'<geom type="cylinder" size="0.05 0.45" pos="{x} {y0 + side * 7.6} 0.45" material="steel" {VIS}/>'
            )
        for x in np.arange(-100, 165, 11.0):
            yy = y0 + side * rng.uniform(13, 24)
            out.append(tree(x + rng.uniform(-3, 3), yy, rng.uniform(0.8, 1.3), pine=rng.random() < 0.4))
        for x in np.arange(-150, 260, 70.0):
            out.append(
                f'<geom type="ellipsoid" size="{rng.uniform(40, 70)} 30 {rng.uniform(12, 25)}" '
                f'pos="{x} {y0 + side * rng.uniform(85, 110)} -2" material="hill" {VIS}/>'
            )
    for x in np.arange(-50, 95, 18.0):  # proper lamp heads (poles already there)
        for y in (y0 - 7.5, y0 + 7.5):
            out.append(
                f'<geom type="ellipsoid" size="0.32 0.2 0.07" pos="{x} {y - (1.5 if y > y0 else -1.5)} 7.95" '
                f'rgba="1 0.92 0.7 1" material="lamp" {VIS}/>'
            )
    gx = 92.0  # sign gantry downstream of the jam (outside the hop's flight path)
    for y in (y0 - 7.0, y0 + 7.0):
        out.append(f'<geom type="box" size="0.25 0.25 3.9" pos="{gx} {y} 3.9" material="steel" {VIS}/>')
    out.append(f'<geom type="box" size="0.3 7.3 0.25" pos="{gx} {y0} 7.8" material="steel" {VIS}/>')
    out.append(
        f'<geom type="box" size="0.08 3.6 0.9" pos="{gx - 0.3} {y0 - 1.5} 7.0" material="sign" euler="0 0 0" {VIS}/>'
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- city
def roof_props(cx: float, cy: float, hx: float, hy: float, h: float) -> list[str]:
    """AC units, a water tank, railings and a blinking-style antenna light on a roof at height h."""
    out = []
    for sx in (-1, 1):  # railings along the roof edge
        out.append(
            f'<geom type="box" size="{hx} 0.04 0.5" pos="{cx} {cy + sx * (hy - 0.1)} {h + 0.6}" '
            f'rgba="0.75 0.78 0.82 0.6" material="glass" {VIS}/>'
        )
        out.append(
            f'<geom type="box" size="0.04 {hy} 0.5" pos="{cx + sx * (hx - 0.1)} {cy} {h + 0.6}" '
            f'rgba="0.75 0.78 0.82 0.6" material="glass" {VIS}/>'
        )
    corners = [(hx - 3, hy - 3), (hx - 3, -(hy - 3)), (-(hx - 3), hy - 3), (-(hx - 3), -(hy - 3))]
    for k, (dx, dy) in enumerate(corners):
        if k == 0:
            out.append(
                f'<geom type="cylinder" size="1.2 1.4" pos="{cx + dx} {cy + dy} {h + 2.0}" rgba="0.55 0.4 0.3 1" {VIS}/>'
            )
            out.append(
                f'<geom type="ellipsoid" size="1.25 1.25 0.5" pos="{cx + dx} {cy + dy} {h + 3.4}" rgba="0.45 0.32 0.25 1" {VIS}/>'
            )
            for ox in (-0.8, 0.8):
                out.append(
                    f'<geom type="cylinder" size="0.08 0.6" pos="{cx + dx + ox} {cy + dy} {h + 0.6}" material="steel" {VIS}/>'
                )
        else:
            out.append(
                f'<geom type="box" size="1.0 0.8 0.6" pos="{cx + dx} {cy + dy} {h + 0.7}" rgba="0.82 0.83 0.85 1" '
                f'material="metal" {VIS}/>'
            )
            out.append(
                f'<geom type="cylinder" size="0.5 0.05" pos="{cx + dx} {cy + dy} {h + 1.33}" rgba="0.3 0.3 0.32 1" {VIS}/>'
            )
    return out


def city_decor(yc: float, towers, scenery) -> str:
    out = []
    rng = np.random.default_rng(31)
    # streets with lane lines and sidewalks between the blocks
    for x in (-50, 20, 60, 100):
        out.append(f'<geom type="box" size="0.12 90 0.012" pos="{x} {yc} 0.02" rgba="0.95 0.85 0.3 1" {VIS}/>')
        out.append(f'<geom type="box" size="2.5 90 0.03" pos="{x - 6} {yc} 0.03" rgba="0.6 0.6 0.58 1" {VIS}/>')
    for y in (-15, 22, -45):
        out.append(f'<geom type="box" size="110 0.12 0.012" pos="40 {yc + y} 0.02" rgba="0.95 0.95 0.95 1" {VIS}/>')
    for k in range(18):  # street trees and lamps
        x, y = rng.uniform(-60, 140), yc + rng.choice([-15, 22, -45]) + rng.choice([-5, 5])
        out.append(tree(x, y, rng.uniform(0.7, 1.0)))
        out.append(lamp(x + 8, y, y, 6.0))
    for name, cx, cy, hx, hy, h in towers:
        if name == "obs_tower_block":  # setback crown on the tall blocker tower
            out.append(
                f'<geom type="box" size="{hx - 2} {hy - 2} 3" pos="{cx} {yc + cy} {h + 3}" material="facade_glass" {VIS}/>'
            )
        else:  # the props sit in the roof corners, clear of the parking spot and the helipad
            out += roof_props(cx, yc + cy, hx, hy, h)
    gx, gy, _, _, gh = towers[1][1:]
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):  # helipad edge lights
        out.append(
            f'<geom type="sphere" size="0.18" pos="{gx + 8.5 * np.cos(ang)} {yc + gy + 8.5 * np.sin(ang)} {gh + 0.25}" '
            f'rgba="0.3 1 0.4 1" material="lamp" {VIS}/>'
        )
    for cx, cy, hx, hy, h in scenery:  # roof crowns and antennas on the background towers
        out.append(
            f'<geom type="box" size="{hx * 0.6} {hy * 0.6} 1.5" pos="{cx} {yc + cy} {h + 1.5}" rgba="0.55 0.57 0.6 1" {VIS}/>'
        )
        if h > 35:
            out.append(f'<geom type="cylinder" size="0.1 4" pos="{cx} {yc + cy} {h + 7}" material="steel" {VIS}/>')
            out.append(
                f'<geom type="sphere" size="0.3" pos="{cx} {yc + cy} {h + 11}" rgba="1 0.1 0.1 1" material="lamp" {VIS}/>'
            )
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- car-hop plaza
def hop_plaza(x0: float, y0: float) -> str:
    """Paved strip for the car-hop scenario: asphalt, parking bay lines, curbs, lamps and a few trees."""
    out = [f'<geom type="box" size="72 11 0.006" pos="{x0 + 66} {y0} 0.006" material="asphalt" {VIS}/>']
    for x in np.arange(x0 - 4, x0 + 136, 6.0):  # bay lines on both sides of the hop lane
        for side in (-1, 1):
            out.append(
                f'<geom type="box" size="0.06 2.2 0.004" pos="{x} {y0 + side * 7.5} 0.012" '
                f'rgba="0.95 0.95 0.95 1" {VIS}/>'
            )
    for side in (-1, 1):
        out.append(
            f'<geom type="box" size="72 0.06 0.004" pos="{x0 + 66} {y0 + side * 4.8} 0.012" '
            f'rgba="0.95 0.8 0.2 1" {VIS}/>'
        )
        out.append(
            f'<geom type="box" size="72 0.2 0.08" pos="{x0 + 66} {y0 + side * 11.2} 0.08" material="concrete" {VIS}/>'
        )
        for x in np.arange(x0, x0 + 135, 22.0):
            out.append(lamp(x, y0 + side * 12.5, y0, 7.0))
        for x in np.arange(x0 + 11, x0 + 135, 22.0):
            out.append(tree(x, y0 + side * 15.0, 0.9))
    return "\n".join(out)
