"""Faceted body shell of the car (visual only).

One straight-sided section per key station along the car's x axis (chassis frame: ground at z = -0.51,
wheel centers at x = +-1.4, z = -0.16, radius 0.35), with the wheel arches cut in. Quads between
neighboring stations are split along the less-twisted diagonal and every triangle gets its own
vertices, so each panel is shaded flat and every crease stays sharp. Overall 4.34 m long, 2.16 m wide.
"""

from __future__ import annotations

import numpy as np

# Key stations. Explicit section
# vertices per station (all y are half widths):
#   w, zb, zc      body side at the top of the black sill band (y, z of the band top), floor z
#   wb, zbelt      shoulder crease: front fender peak -> door top -> rear haunch edge -> tail light
#   ys, zs         window sill (cabin) / inner edge of the haunch deck (rear)
#   yh, zh         window header (cabin) / outer foot of the rear buttress
#   wt, ze         top edge: fender blade crest (front), roof edge (cabin), buttress crest (rear)
#   wc, zcn        inner foot of the fender blade on the hood / rear channel edge (= top edge in the cabin)
#   zt             center line: hood, canopy ridge, recessed channel floor
# Plan view: raised fender blades frame a lower trapezoid hood; the canopy is a teardrop, widest at the
# cowl just ahead of the front wheels (1.3), running as a fastback down to -1.8 between two shallow
# buttresses; the side windows are triangles pointing at the C-pillar over the rear wheel (-1.2). The
# tail is narrower than the haunches and ends in a sharp edge over an undercut face: the last station
# only holds the top 20 cm, so the strips to the station before it form the leaning black diffuser face.
# Wheel arches are not stations here: shell_faceted() cuts them in.
REVEAL_KEYS = np.array(
    [
        # x, w, zb, zc, wb, zbelt, ys, zs, yh, zh, wt, ze, wc, zcn, zt
        (
            -2.14,
            0.88,
            0.14,
            0.17,
            0.85,
            0.30,
            0.54,
            0.315,
            0.40,
            0.315,
            0.25,
            0.33,
            0.23,
            0.325,
            0.33,
        ),  # sharp tail edge; the face below z = 0.14 is undercut
        (
            -2.06,
            0.94,
            -0.24,
            0.02,
            0.90,
            0.31,
            0.56,
            0.345,
            0.42,
            0.345,
            0.26,
            0.36,
            0.24,
            0.35,
            0.355,
        ),  # foot of the diffuser; tail narrower than the haunches
        (
            -1.80,
            1.06,
            -0.33,
            0.05,
            1.02,
            0.35,
            0.60,
            0.385,
            0.46,
            0.385,
            0.30,
            0.40,
            0.28,
            0.39,
            0.395,
        ),  # end of the rear glass, short deck behind
        (
            -1.50,
            1.08,
            -0.36,
            0.08,
            1.04,
            0.345,
            0.62,
            0.40,
            0.50,
            0.40,
            0.36,
            0.445,
            0.34,
            0.435,
            0.44,
        ),  # rear haunch (widest), shallow buttresses
        (
            -1.20,
            1.07,
            -0.37,
            0.10,
            1.03,
            0.335,
            1.025,
            0.336,
            1.025,
            0.336,
            0.40,
            0.51,
            0.39,
            0.50,
            0.515,
        ),  # C-pillar over the rear wheel: side window point
        (
            -0.60,
            1.02,
            -0.38,
            -0.06,
            0.98,
            0.32,
            0.975,
            0.33,
            0.55,
            0.555,
            0.50,
            0.565,
            0.50,
            0.565,
            0.57,
        ),  # fastback roof; sill band kicks up behind
        (
            0.0,
            1.0,
            -0.38,
            -0.07,
            0.97,
            0.31,
            0.965,
            0.32,
            0.66,
            0.615,
            0.62,
            0.63,
            0.62,
            0.63,
            0.64,
        ),  # roof peak (1.15 m above ground)
        (
            0.50,
            1.0,
            -0.38,
            -0.08,
            0.975,
            0.30,
            0.97,
            0.31,
            0.74,
            0.46,
            0.70,
            0.475,
            0.70,
            0.475,
            0.488,
        ),  # on the windscreen plane
        (
            1.10,
            1.045,
            -0.36,
            -0.10,
            1.03,
            0.295,
            1.015,
            0.296,
            1.015,
            0.296,
            0.84,
            0.30,
            0.84,
            0.30,
            0.306,
        ),  # A-pillar foot: window point
        (
            1.30,
            1.05,
            -0.35,
            -0.12,
            1.035,
            0.30,
            1.025,
            0.301,
            1.025,
            0.301,
            0.93,
            0.31,
            0.85,
            0.24,
            0.245,
        ),  # cowl: glass spans between the fender blades
        (
            1.70,
            1.045,
            -0.335,
            -0.15,
            1.03,
            0.265,
            1.015,
            0.266,
            1.015,
            0.266,
            0.93,
            0.275,
            0.87,
            0.22,
            0.215,
        ),  # fender blade over the wheel, flat hood
        (
            2.00,
            0.98,
            -0.33,
            -0.16,
            0.94,
            0.17,
            0.935,
            0.171,
            0.935,
            0.171,
            0.86,
            0.18,
            0.81,
            0.155,
            0.15,
        ),  # transverse hood crease
        (
            2.20,
            0.80,
            -0.30,
            -0.15,
            0.78,
            -0.12,
            0.775,
            -0.119,
            0.775,
            -0.119,
            0.74,
            0.02,
            0.70,
            0.03,
            0.035,
        ),  # blunt nose: blade line (light bar) at mid height, black fascia below
    ]
)


# x ranges of the glazing: canopy (roof strip) and side windows
GLASS_X = (-1.8, 1.3)
SIDE_GLASS_X = (-1.2, 1.1)

# wheel arches: (x half length at the floor, at the top), opening height by axle
ARCH_AXLES = (1.4, -1.4)  # wheel center x (model.WHEEL_POS); front first
ARCH_X = (0.44, 0.30)
ARCH_Z = (0.215, 0.215)  # 3 cm over the tire tops (wheel center settles at z = -0.164, r = 0.35)
ARCH_WALL_Y = 0.56  # inner arch wall (tire inner face at 0.72; 0.57 with the wheels steered 0.5 rad)


def mesh_xml(name: str, V: np.ndarray, F: np.ndarray) -> str:
    v = " ".join(f"{x:.4f}" for x in V.ravel())
    f = " ".join(str(int(i)) for i in F.ravel())
    return f'<mesh name="{name}" vertex="{v}" face="{f}" smoothnormal="false" inertia="shell"/>'


# ---------------------------------------------------------------------------------------------- faceted
# Panel-style shell: one straight-sided section per station, quads between
# neighboring stations split along the less-twisted diagonal, and unshared vertices per triangle so
# every panel is shaded flat and every crease stays sharp.
# Paint, glass and the black band are *one* shell: every triangle belongs to exactly one of the three
# meshes, because overlaid slabs z-fight with the body and, on twisted quads, wind inside out and get culled.

# section vertices (+y side, then mirrored): 0 floor center, 1 keel, 2 arch wall top, 3 floor edge / arch
# ceiling edge, 4 sill band top, 5 shoulder crease, 6 sill, 7 header, 8 top edge (fender / roof /
# buttress crest), 9 blade foot / channel edge, 10 ridge, then 11..19 mirrored (channel edge L .. keel L);
# strip j runs from vertex j to j + 1
_STRIPS = {
    "floor": (0, 19),
    "wall": (1, 18),
    "ceiling": (2, 17),
    "band": (3, 16),
    "side": (4, 15),
    "flank": (5, 14),
    "window": (6, 13),
    "header": (7, 12),
    "fender": (8, 11),
    "roof": (9, 10),
}
_NV = 20


def _polygon(w, zb, zc, wb, zbelt, ys, zs, yh, zh, wt, ze, wc, zcn, zt, za=None):
    """Section (y, z) polygon (20 points) from one REVEAL_KEYS row.

    za > zb opens a wheel arch: the outer path (rocker, sill band, side plane) is cut at height za and every
    vertex below it collapses onto the cut point; a vertical inner wall at ARCH_WALL_Y closes the tunnel, so
    every strip keeps its direction (planar quads) and the arch stays a side opening only.
    """
    zc = min(max(zc, zb + 0.03), zbelt - 0.03)
    ye = 0.9 * w
    yr = min(ARCH_WALL_Y, ye - 0.02)
    za = zb if za is None else min(max(za, zb), zbelt - 0.03)
    outer = [(ye, zb), (w, zc), (wb, zbelt), (ys, zs), (yh, zh), (wt, ze)]
    pa = (ye, zb)
    for (y0, z0), (y1, z1) in zip(outer[:-1], outer[1:]):
        if z0 <= za <= z1 and z1 > z0:
            pa = (y0 + (y1 - y0) * (za - z0) / (z1 - z0), za)
            break
    outer = [pt if pt[1] > za else pa for pt in outer]
    right = [(0.0, zb), (yr, zb), (yr, za)] + outer + [(wc, zcn)]
    return np.asarray(right + [(0.0, zt)] + [(-y, z) for y, z in reversed(right[1:])])


def _arch_height(x: float) -> float:
    """Wheel-arch opening height (below the floor = closed) at station x, a trapezoid per axle."""
    z = -9.0
    for wx, top in zip(ARCH_AXLES, ARCH_Z):
        z = max(z, float(np.interp(abs(x - wx), [ARCH_X[1], ARCH_X[0]], [top, -9.0])))
    return z


def _faceted_stations() -> np.ndarray:
    """Key stations plus the wheel-arch stations (vertex values interpolated): (x, REVEAL_KEYS cols, za)."""
    keys = REVEAL_KEYS[np.argsort(REVEAL_KEYS[:, 0])]
    ax = [wx + s * h for wx in ARCH_AXLES for s in (-1, 1) for h in ARCH_X]
    xs = np.unique(np.round(np.concatenate([keys[:, 0], ax]), 6))
    cols = [np.interp(xs, keys[:, 0], keys[:, k]) for k in range(1, keys.shape[1])]
    return np.column_stack([xs] + cols + [[_arch_height(x) for x in xs]])


def _ear_clip(P: np.ndarray, idx: list[int]) -> list[tuple[int, int, int]]:
    """Triangles (as entries of idx) of the simple 2D polygon P (n,2) by ear clipping.

    Duplicate vertices (closed arches, collapsed strips) are skipped.
    """
    keep = [i for i in range(len(P)) if np.linalg.norm(P[i] - P[i - 1]) > 1e-9]
    pts, ids = P[keep], [idx[i] for i in keep]
    sgn = np.sign(sum(pts[i - 1, 0] * pts[i, 1] - pts[i, 0] * pts[i - 1, 1] for i in range(len(pts))))
    order, tris = list(range(len(pts))), []
    while len(order) > 2:
        for n_ in range(len(order)):
            a, b, c = order[n_ - 1], order[n_], order[(n_ + 1) % len(order)]
            ab, bc = pts[b] - pts[a], pts[c] - pts[b]
            cross = ab[0] * bc[1] - ab[1] * bc[0]
            if abs(cross) < 1e-12 or cross * sgn < 0:  # flat or reflex corner: not an ear (a flat one
                continue  # is used by a neighbor's ear, so the ring's collinear edges stay in the cap)
            tri = pts[[a, b, c]]
            inside = False
            for o in order:
                if o in (a, b, c):
                    continue
                d = [
                    (
                        (tri[j][0] - tri[j - 1][0]) * (pts[o][1] - tri[j - 1][1])
                        - (tri[j][1] - tri[j - 1][1]) * (pts[o][0] - tri[j - 1][0])
                    )
                    * sgn
                    for j in range(3)
                ]
                if min(d) > -1e-12:
                    inside = True
                    break
            if not inside:
                tris.append((ids[a], ids[b], ids[c]))
                order.pop(n_)
                break
        else:
            break  # no ear found (should not happen for a simple polygon)
    return tris


def _flat(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unshare vertices so each triangle gets its own flat normal."""
    return V[F].reshape(-1, 3), np.arange(F.size).reshape(-1, 3)


def _tri_normal(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def shell_faceted() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The whole faceted shell as {"body", "glass", "band": (V, F)}; the three share one surface."""
    st = _faceted_stations()
    rings = [np.column_stack([np.full(_NV, s[0]), _polygon(*s[1:])]) for s in st]
    k = _NV
    V = np.concatenate(rings)
    F, tag = [], []
    strip_of = {j: name for name, (a, b) in _STRIPS.items() for j in (a, b)}
    for i in range(len(rings) - 1):
        a, b = i * k, (i + 1) * k
        x0, x1 = st[i, 0], st[i + 1, 0]
        for j in range(k):
            j2 = (j + 1) % k
            q = [a + j, a + j2, b + j2, b + j]  # quad corners, consistent orientation
            # both diagonals; take the one whose two triangles bend least (the crease a twisted quad
            # must have then follows the smoother line)
            cands = [((q[0], q[1], q[3]), (q[3], q[1], q[2])), ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
            best = None
            for tris in cands:
                t = np.asarray(tris)
                n = _tri_normal(V, t)
                ok = np.linalg.norm(np.cross(V[t[:, 1]] - V[t[:, 0]], V[t[:, 2]] - V[t[:, 0]]), axis=1) > 1e-9
                bend = float(np.arccos(np.clip(n[0] @ n[1], -1, 1))) if ok.all() else 0.0
                if best is None or bend < best[0] - 1e-6:
                    best = (bend, t[ok])
            strip = strip_of[j]
            part = "body"
            if strip in ("floor", "wall", "ceiling", "band"):
                part = "band"
            elif strip == "roof" and GLASS_X[0] - 1e-6 <= x0 and x1 <= GLASS_X[1] + 1e-6:
                part = "glass"
            elif strip == "window" and SIDE_GLASS_X[0] - 1e-6 <= x0 and x1 <= SIDE_GLASS_X[1] + 1e-6:
                part = "glass"  # (zero-height window quads, e.g. at the pillars, are dropped)
            for t in best[1]:
                F.append(tuple(t))
                tag.append(part)
    # end caps (tail panel, blunt nose face), split along the crease chord so the nose's lower half is
    # the black fascia; each half is ear-clipped (the fan from a center point fails on the two-lobed
    # buttress / channel outline of the tail)
    cr = (_STRIPS["flank"][0], _STRIPS["flank"][1] + 1)  # crease vertices R, L
    for base, lower in ((0, "body"), ((len(rings) - 1) * k, "band")):
        lo = list(range(base + 1, base + cr[0] + 1)) + list(range(base + cr[1], base + k)) + [base]
        up = list(range(base + cr[0], base + cr[1] + 1))
        for idx, part in ((lo, lower), (up, "body")):
            tris = _ear_clip(V[idx][:, 1:], idx)
            if base == 0:  # the tail ring is wound the other way round seen from outside
                tris = [t[::-1] for t in tris]
            F += tris
            tag += [part] * len(tris)
    F = np.asarray(F)
    tag = np.asarray(tag)
    area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    F, tag = F[area > 1e-9], tag[area > 1e-9]
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() < 0:  # wind outward
        F = F[:, ::-1]
    return {part: _flat(V, F[tag == part]) for part in ("body", "glass", "band")}


def side_point(x: float, z: float) -> tuple[float, float]:
    """(y, lean) of the faceted side at (x, z): y of the +y surface and the plane's inward lean in degrees.

    For mounting flush parts (vents, door line, side thruster ports). Uses the side plane (band top ->
    crease) or the flank above the crease (crease -> sill), whichever holds z.
    """
    keys = REVEAL_KEYS[np.argsort(REVEAL_KEYS[:, 0])]
    w, zb, zc, wb, zbelt, ys, zs = [float(np.interp(x, keys[:, 0], keys[:, k])) for k in range(1, 8)]
    zc = min(max(zc, zb + 0.03), zbelt - 0.03)
    (y0, z0), (y1, z1) = ((w, zc), (wb, zbelt)) if z <= zbelt else ((wb, zbelt), (ys, zs))
    y = y0 + (y1 - y0) * (z - z0) / (z1 - z0)
    return y, float(np.degrees(np.arctan2(y0 - y1, z1 - z0)))
