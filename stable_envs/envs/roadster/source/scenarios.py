"""Extra scenarios placed in their own corners of the world (no recompiling to switch tasks).

traffic (y = +200): a straight highway with jersey barriers. A jam of parked cars (mocap bodies, moved
    each episode) blocks both lanes. The Roadster has to lift itself on its jets, fly over the jam and
    land on the road beyond.
rooftop (y = -200): a city block. The car starts parked on a tower roof, must fly around a taller tower
    that blocks the direct line and land on the helipad of another, higher roof.
carhop (y = 0, past the landing pad): a lane packed nose to tail with parked cars (mocap bodies). The car
    starts on the right of the lane, jumps sideways over one car with a barrel roll and parks on the left.
"""

from __future__ import annotations

import mujoco
import numpy as np

from . import decor

TRAFFIC_Y = 200.0
CITY_Y = -200.0
HOP_X0, HOP_Y = 262.0, 0.0  # car-hop plaza: start line x and centre line (past the landing pad)
N_HOP = 14  # mocap slots for the lane of parked cars
N_TRAFFIC = 10  # mocap car slots
CAR_SIZE = (2.2, 0.9, 0.7)  # half extents of a parked car's collision box
TRUCK_SIZE = (5.0, 1.25, 1.7)
CAR_COLORS = [
    "0.75 0.1 0.1 1",
    "0.1 0.3 0.7 1",
    "0.92 0.92 0.9 1",
    "0.12 0.12 0.13 1",
    "0.9 0.6 0.1 1",
    "0.2 0.55 0.3 1",
    "0.55 0.55 0.6 1",
    "0.45 0.15 0.5 1",
    "0.95 0.85 0.2 1",
    "0.3 0.3 0.35 1",
]

# rooftop city: (name, centre x, centre y offset, half size x, half size y, height)
TOWERS = (
    ("roof_start", 0.0, 0.0, 12.0, 12.0, 24.0),
    ("roof_goal", 78.0, 14.0, 12.0, 12.0, 34.0),
    ("obs_tower_block", 40.0, 6.0, 9.0, 9.0, 62.0),
)
SCENERY_TOWERS = (
    (-30, 30, 10, 10, 40),
    (-28, -28, 9, 12, 30),
    (30, -35, 11, 9, 48),
    (80, -30, 10, 10, 26),
    (115, 20, 12, 10, 52),
    (40, 45, 10, 12, 30),
    (115, -25, 9, 9, 38),
    (-35, 2, 8, 9, 18),
)


def assets_xml() -> str:
    return (
        '<texture name="facade" type="2d" file="facade.png"/>\n'
        '<material name="facade" texture="facade" texrepeat="3 4" specular="0.4" shininess="0.6"/>\n'
        '<material name="concrete" rgba="0.62 0.62 0.6 1"/>\n'
        '<material name="barrier" rgba="0.85 0.85 0.82 1"/>\n' + decor.assets_xml() + "\n"
    )


def _parked_car(i: int, color: str, prefix: str = "tcar", y: float = TRAFFIC_Y) -> str:
    """Mocap parked car: collision box + a car-shaped body (sedan / SUV / hatchback)."""
    hx, hy, hz = CAR_SIZE
    g = [
        f'<body name="{prefix}{i}" mocap="true" pos="0 {y} -30">',
        f'<geom name="obs_{prefix}{i}" type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" rgba="0 0 0 0" group="3"/>',
    ]
    g += decor.parked_car(i, color, CAR_SIZE)
    g.append("</body>")
    return "\n".join(g)


def _truck() -> str:
    hx, hy, hz = TRUCK_SIZE
    vis = 'contype="0" conaffinity="0"'
    g = [
        f'<body name="truck" mocap="true" pos="0 {TRAFFIC_Y} -30">',
        f'<geom name="obs_truck" type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" rgba="0 0 0 0" group="3"/>',
        f'<geom type="box" size="{hx - 1.0} {hy} {hz - 0.2}" pos="-1.0 0 {hz + 0.2}" rgba="0.92 0.92 0.94 1" {vis}/>',
        f'<geom type="box" size="0.9 {hy} 1.1" pos="{hx - 0.9} 0 1.4" rgba="0.15 0.35 0.7 1" material="paint" {vis}/>',
        f'<geom type="box" size="0.3 {hy * 0.85} 0.35" pos="{hx - 0.6} 0 1.95" rgba="0.08 0.1 0.14 1" material="glass" {vis}/>',
    ]
    for x in (hx - 1.0, -hx + 1.5, -hx + 3.0):
        for sy in (-1, 1):
            g.append(
                f'<geom type="cylinder" size="0.5 0.2" pos="{x} {sy * (hy - 0.1)} 0.5" euler="90 0 0" '
                f'rgba="0.05 0.05 0.05 1" {vis}/>'
            )
    g.append("</body>")
    return "\n".join(g)


def world_xml() -> str:
    out = []
    y0 = TRAFFIC_Y
    # highway: asphalt, lane markings, jersey barriers (collision) along the jam section
    out.append(
        f'<geom name="hwy" type="box" size="140 6 0.005" pos="30 {y0} 0.005" material="asphalt" '
        'contype="0" conaffinity="0"/>'
    )
    for x in np.arange(-105, 170, 9.0):
        out.append(
            f'<geom type="box" size="1.8 0.08 0.006" pos="{x} {y0} 0.008" rgba="0.95 0.95 0.95 1" '
            'contype="0" conaffinity="0"/>'
        )
    for y in (y0 - 5.7, y0 + 5.7):
        out.append(
            f'<geom type="box" size="140 0.08 0.006" pos="30 {y} 0.008" rgba="0.95 0.85 0.3 1" '
            'contype="0" conaffinity="0"/>'
        )
    for k, y in enumerate((y0 - 6.4, y0 + 6.4)):
        out.append(f'<geom name="obs_barrier{k}" type="box" size="75 0.3 0.55" pos="20 {y} 0.55" material="barrier"/>')
    for x in np.arange(-50, 95, 18.0):  # light poles
        for y in (y0 - 7.5, y0 + 7.5):
            arm = 1.5 if y > y0 else -1.5
            out.append(
                f'<geom type="cylinder" size="0.1 4" pos="{x} {y} 4" rgba="0.4 0.42 0.45 1" '
                'contype="0" conaffinity="0"/>'
            )
            out.append(
                f'<geom type="box" size="0.12 {abs(arm) / 2} 0.06" pos="{x} {y - arm / 2} 8" '
                'rgba="0.4 0.42 0.45 1" contype="0" conaffinity="0"/>'
            )
    for i in range(N_TRAFFIC):
        out.append(_parked_car(i, CAR_COLORS[i % len(CAR_COLORS)]))
    out.append(_truck())

    # rooftop city
    yc = CITY_Y
    out.append(
        f'<geom type="box" size="110 90 0.01" pos="40 {yc} 0.01" material="asphalt" contype="0" conaffinity="0"/>'
    )
    tower_mats = {"roof_start": "facade_stone", "roof_goal": "facade_glass", "obs_tower_block": "facade"}
    for name, cx, cy, hx, hy, h in TOWERS:
        out.append(
            f'<geom name="{name}" type="box" size="{hx} {hy} {h / 2}" pos="{cx} {yc + cy} {h / 2}" '
            f'material="{tower_mats[name]}" friction="1.2"/>'
        )
        out.append(
            f'<geom type="box" size="{hx + 0.3} {hy + 0.3} 0.4" pos="{cx} {yc + cy} {h - 0.3}" '
            'material="concrete" contype="0" conaffinity="0"/>'
        )
    sx, sy, _, _, gh = TOWERS[1][1:]
    gx, gy = sx, yc + sy
    for dx, dy, hx, hy in ((0, -2.2, 3.0, 0.35), (0, 2.2, 3.0, 0.35), (0, 0, 0.35, 2.2)):  # helipad "H"
        out.append(
            f'<geom type="box" size="{hx} {hy} 0.01" pos="{gx + dx} {gy + dy} {gh + 0.11}" '
            'rgba="1 1 1 1" contype="0" conaffinity="0"/>'
        )
    for ang in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        out.append(
            f'<geom type="box" size="0.4 0.12 0.01" pos="{gx + 6 * np.cos(ang)} {gy + 6 * np.sin(ang)} '
            f'{gh + 0.11}" euler="0 0 {np.rad2deg(ang) + 90:.2f}" rgba="1 0.8 0.1 1" '
            'contype="0" conaffinity="0"/>'
        )
    bx, by, _, bhy, bh = TOWERS[2][1:]
    out.append(
        f'<geom type="cylinder" size="0.15 5" pos="{bx} {yc + by} {bh + 5}" rgba="0.6 0.6 0.62 1" '
        'contype="0" conaffinity="0"/>'
    )  # antenna
    out.append(
        f'<geom type="sphere" size="0.4" pos="{bx} {yc + by} {bh + 10}" rgba="1 0.1 0.1 1" material="lamp" '
        'contype="0" conaffinity="0"/>'
    )
    for k, (cx, cy, hx, hy, h) in enumerate(SCENERY_TOWERS):
        mat = ("facade_glass", "facade_stone", "facade")[k % 3]
        out.append(
            f'<geom name="obs_city{k}" type="box" size="{hx} {hy} {h / 2}" pos="{cx} {yc + cy} {h / 2}" '
            f'material="{mat}"/>'
        )
    s0x, s0y, _, _, s0h = TOWERS[0][1:]
    for dy in (-3, 3):  # parking lines on the start roof
        out.append(
            f'<geom type="box" size="3 0.06 0.01" pos="{s0x - 4} {yc + s0y + dy} {s0h + 0.11}" '
            'rgba="1 1 1 1" contype="0" conaffinity="0"/>'
        )
    out.append(decor.traffic_decor(TRAFFIC_Y))
    out.append(decor.city_decor(CITY_Y, TOWERS, SCENERY_TOWERS))
    # car-hop plaza: a wide paved strip with a row of parked cars (moved each episode)
    out.append(decor.hop_plaza(HOP_X0, HOP_Y))
    for i in range(N_HOP):
        out.append(_parked_car(i, CAR_COLORS[(i * 3 + 1) % len(CAR_COLORS)], prefix="hcar", y=HOP_Y))
    return "\n".join(out)


def mocap_id(model, name: str) -> int:
    """Index into data.mocap_pos / mocap_quat of the mocap body called `name`."""
    return int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])


def place_hop(model, data, rng: np.random.Generator) -> list[tuple[float, float]]:
    """A long lane packed with 10-14 parked cars nose to tail (0.4-1.2 m apart) along the plaza's centre line.

    Returns each car's (x_rear, x_front).
    """
    mid = lambda name: mocap_id(model, name)  # noqa: E731
    n = int(rng.integers(10, N_HOP + 1))
    x = HOP_X0 + rng.uniform(2, 6)
    cars = []
    for i in range(N_HOP):
        if i < n:
            yaw = rng.uniform(-0.04, 0.04)
            data.mocap_pos[mid(f"hcar{i}")] = (x + CAR_SIZE[0], HOP_Y + rng.uniform(-0.2, 0.2), 0.0)
            data.mocap_quat[mid(f"hcar{i}")] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
            cars.append((x, x + 2 * CAR_SIZE[0]))
            x += 2 * CAR_SIZE[0] + rng.uniform(0.4, 1.2)
        else:
            data.mocap_pos[mid(f"hcar{i}")] = (0.0, HOP_Y, -30.0)
    return cars


def place_traffic(model, data, rng: np.random.Generator) -> dict:
    """Random jam: 2-4 cars per lane bumper to bumper, sometimes a truck. Returns jam extents."""
    mid = lambda name: mocap_id(model, name)  # noqa: E731
    x_start = rng.uniform(0.0, 8.0)
    ends = []
    slot = 0
    use_truck = rng.random() < 0.35
    truck_lane = rng.integers(2)
    for lane, ly in enumerate((-2.3, 2.3)):
        x = x_start + rng.uniform(-2, 2)
        if use_truck and lane == truck_lane:
            data.mocap_pos[mid("truck")] = (x + TRUCK_SIZE[0], TRAFFIC_Y + ly, 0.0)
            x += 2 * TRUCK_SIZE[0] + rng.uniform(0.8, 2.0)
        n = int(rng.integers(2, 5))
        for _ in range(n):
            if slot >= N_TRAFFIC:
                break
            data.mocap_pos[mid(f"tcar{slot}")] = (x + CAR_SIZE[0], TRAFFIC_Y + ly + rng.uniform(-0.3, 0.3), 0.0)
            yaw = rng.uniform(-0.06, 0.06)
            data.mocap_quat[mid(f"tcar{slot}")] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
            x += 2 * CAR_SIZE[0] + rng.uniform(0.8, 2.0)
            slot += 1
        ends.append(x)
    for i in range(slot, N_TRAFFIC):
        data.mocap_pos[mid(f"tcar{i}")] = (0.0, TRAFFIC_Y, -30.0)
    if not use_truck:
        data.mocap_pos[mid("truck")] = (0.0, TRAFFIC_Y, -30.0)
    top = 2 * TRUCK_SIZE[2] if use_truck else 2 * CAR_SIZE[2]
    return {"x_start": x_start - 1.0, "x_end": max(ends), "top": top}
