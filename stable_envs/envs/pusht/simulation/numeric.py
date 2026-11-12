"""Numeric replay values preserve Python/NumPy representation and dtype tags.

Scalar-versus-array distinctions affect source arithmetic. PCG64 words and
seed metadata remain integers; replay recipes contain no engine handles.
"""

from __future__ import annotations

import copy
import ctypes
import struct

import numpy as np
from pymunk._chipmunk_cffi import ffi, lib

DTYPES = tuple(np.dtype(v) for v in ("float32", "float64", "uint8", "int32", "int64", "bool"))


# Dtypes whose float64 payload is always exact; integers can exceed 2**53.
_EXACT = frozenset(DTYPES) - {np.dtype("int64")}


def put_value(result, key, value):
    array = np.asarray(value)
    kind = type(value)
    if kind is np.ndarray or kind is float or kind is bool or isinstance(value, np.generic):
        if array.dtype in _EXACT and (array.dtype.kind != "f" or np.isfinite(array).all()):
            representation = 4 if kind is np.ndarray else 0 if kind is float else 2 if kind is bool else 3
            result[key + "/value"] = array.astype(np.float64)
            result[key + "/type"] = np.array([representation, DTYPES.index(array.dtype)], np.uint8)
            return
    if type(value) is float:
        representation = 0
    elif type(value) is int:
        representation = 1
    elif type(value) is bool:
        representation = 2
    elif isinstance(value, np.generic):
        representation = 3
    elif isinstance(value, np.ndarray):
        representation = 4
    else:
        raise ValueError(f"Unsupported PushT numeric representation: {type(value)}")
    if array.dtype not in DTYPES or not np.isfinite(array).all():
        raise ValueError("Unsupported PushT numeric dtype/value")
    payload = array.astype(np.float64)
    if not np.array_equal(payload.astype(array.dtype), array):
        raise ValueError("PushT numeric value cannot be represented losslessly")
    result[key + "/value"] = payload
    result[key + "/type"] = np.array([representation, DTYPES.index(array.dtype)], np.uint8)


def get_value(state, key):
    value, tag = state[key + "/value"], state[key + "/type"]
    if tag.dtype != np.uint8 or tag.shape != (2,) or tag[0] > 4 or tag[1] >= len(DTYPES):
        raise ValueError("Invalid PushT numeric type tag")
    representation, dtype = int(tag[0]), DTYPES[int(tag[1])]
    if value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError("Invalid PushT numeric payload")
    if representation < 4 and value.shape:
        raise ValueError("Scalar representation requires scalar payload")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = value.astype(dtype)
    if not np.array_equal(converted, value):
        raise ValueError("Lossy PushT numeric payload")
    if representation < 3:
        expected = (np.dtype("float64"), np.dtype("int64"), np.dtype("bool"))[representation]
        if dtype != expected:
            raise ValueError("Invalid Python scalar dtype tag")
        return (float, int, bool)[representation](converted)
    return converted[()] if representation == 3 else converted.copy()


def encode_rng(state):
    if state is None:
        return np.zeros(11, np.int64)
    if state["bit_generator"] != "PCG64":
        raise ValueError("PushT requires PCG64")
    words, mask = state["state"], 0xFFFFFFFF
    value, inc = words["state"], words["inc"]
    return np.array(
        [
            1,
            value & mask,
            value >> 32 & mask,
            value >> 64 & mask,
            value >> 96 & mask,
            inc & mask,
            inc >> 32 & mask,
            inc >> 64 & mask,
            inc >> 96 & mask,
            state["has_uint32"],
            state["uinteger"],
        ],
        np.int64,
    )


def decode_rng(row):
    if row.dtype != np.int64 or row.shape != (11,) or row[0] not in (0, 1):
        raise ValueError("Invalid PushT PCG64 presence/shape/dtype")
    if not row[0]:
        if np.any(row):
            raise ValueError("Absent PushT PCG64 must have zero padding")
        return None
    if np.any(row[1:] < 0) or np.any(row[1:] > 0xFFFFFFFF) or row[9] not in (0, 1) or not row[5] % 2:
        raise ValueError("Invalid PushT PCG64 words/increment/cache")
    return dict(
        bit_generator="PCG64",
        state={k: sum(int(row[1 + j * 4 + i]) << (32 * i) for i in range(4)) for j, k in enumerate(("state", "inc"))},
        has_uint32=int(row[9]),
        uinteger=int(row[10]),
    )


def rng_of(node):
    return None if node._np_random is None else copy.deepcopy(node._np_random.bit_generator.state)


def rng_view(node):
    """Return the generator state for reading; ``bit_generator.state`` already builds a new dict."""
    return None if node._np_random is None else node._np_random.bit_generator.state


def install_rng(node, state):
    node._np_random = None if state is None else np.random.default_rng(0)
    if state is not None:
        node._np_random.bit_generator.state = copy.deepcopy(state)


def encode_seed(seed):
    # Gym's fresh entropy is commonly 128 bits, rather than a signed int64.
    if seed is None:
        return np.array([2, 0, 0, 0, 0], np.int64)
    if not isinstance(seed, (int, np.integer)) or not -1 <= seed < 2**128:
        raise ValueError("Unsupported Gym seed metadata")
    return np.array([int(seed == -1), *[(max(0, int(seed)) >> (32 * i)) & 0xFFFFFFFF for i in range(4)]], np.int64)


def decode_seed(row):
    if row.shape != (5,) or row.dtype != np.int64 or row[0] not in (0, 1, 2):
        raise ValueError("Invalid Gym seed metadata")
    if np.any(row[1:] < 0) or np.any(row[1:] > 0xFFFFFFFF) or row[0] and np.any(row[1:]):
        raise ValueError("Invalid Gym seed words")
    return (None if row[0] == 2 else -1) if row[0] else sum(int(row[i + 1]) << (32 * i) for i in range(4))


BODY_FIELDS = (
    ("position", 2),
    ("velocity", 2),
    ("angle", 1),
    ("angular_velocity", 1),
    ("force", 2),
    ("torque", 1),
    ("center_of_gravity", 2),
    ("mass", 1),
    ("moment", 1),
)
#: Width of a ``body_row``: both bodies' fields, then the seven observation values.
BODY_ROW = 2 * sum(width for _, width in BODY_FIELDS) + 7
#: ``current`` State fields in export order.
CURRENT_KEYS = (
    *(
        f"body/{name}/{key}"
        for name in ("agent", "block")
        for field, _ in BODY_FIELDS
        for key in ((f"{field}_infinite", field) if field in ("mass", "moment") else (field,))
    ),
    "observation",
    "contacts",
    "render_segments",
    "render_segment_count",
    "goal_image",
    "latest_present",
    "latest/value",
    "latest/type",
    "goal/value",
    "goal/type",
)


def body_row(env):
    """Return both bodies' fields and the observation as one list of floats."""
    # The Chipmunk calls that the pymunk Body properties make.
    row = []
    for body in (env.agent._body, env.block._body):
        position, velocity = lib.cpBodyGetPosition(body), lib.cpBodyGetVelocity(body)
        force, center = lib.cpBodyGetForce(body), lib.cpBodyGetCenterOfGravity(body)
        row.extend(
            (
                position.x,
                position.y,
                velocity.x,
                velocity.y,
                lib.cpBodyGetAngle(body),
                lib.cpBodyGetAngularVelocity(body),
                force.x,
                force.y,
                lib.cpBodyGetTorque(body),
                center.x,
                center.y,
                lib.cpBodyGetMass(body),
                lib.cpBodyGetMoment(body),
            )
        )
    # Same arithmetic as ``env._get_obs``.
    row.extend((row[0], row[1], row[13], row[14], row[17] % (2 * np.pi), row[2], row[3]))
    return row


def body_fields(rows):
    """Split ``[N, BODY_ROW]`` float64 rows into owned ``body/...`` and ``observation`` columns."""
    result = {}
    offset = 0
    for name in ("agent", "block"):
        for field, width in BODY_FIELDS:
            value = rows[:, offset : offset + width].copy() if width > 1 else rows[:, offset].copy()
            offset += width
            if field in ("mass", "moment"):
                # Encode infinite kinematic mass/inertia as a flag plus finite zero padding.
                infinite = np.isposinf(value)
                result[f"body/{name}/{field}_infinite"] = infinite
                value = np.where(infinite, 0.0, value)
            result[f"body/{name}/{field}"] = value
    result["observation"] = rows[:, offset:].copy()
    return result


def episode_fields(env):
    """Return the ``current`` fields that stay fixed from one reset to the next."""
    result = {"goal_image": env._goal.copy()}
    put_value(result, "goal", env.goal_state)
    return result


def latest_fields(env):
    result = {"latest_present": np.array(env.latest_action is not None, np.bool_)}
    put_value(result, "latest", np.zeros(2, np.float64) if env.latest_action is None else env.latest_action)
    return result


def install_latest(owners, values, tags, present):
    """Set each owner's latest action from the ``current/latest`` columns of the CUDA physics."""
    for owner, value, tag, flag in zip(owners, values, tags, present, strict=True):
        owner.env.latest_action = value.astype(np.float32 if tag[1] == 0 else np.float64) if flag else None


#: Most convex pieces of one body; the plus has three.
MAX_SHAPES = 3
_UNIT_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


#: Byte offsets of private Munk2D 2.0.1 fields (pymunk 7.2.0, 64-bit): cpBody ``p`` (the center of gravity)
#: and ``v_bias``; cpArbiter
#: ``body_b``, ``count`` and ``contacts``; cpContact size, ``r2`` and ``jnAcc``.
_BODY_P, _BODY_BIAS = 64, 192
_ARBITER_BODY_B, _ARBITER_COUNT, _ARBITER_CONTACTS = 64, 104, 112
_CONTACT_SIZE, _CONTACT_R2, _CONTACT_IMPULSE = 96, 16, 56
#: Most contacts a supported scene holds, as ``contacts.contact_segments`` bounds them.
MAX_CONTACTS = 90


def _address(handle):
    return int(ffi.cast("uintptr_t", handle))


def _doubles(address, count):
    return tuple((ctypes.c_double * count).from_address(address))


def _near(a, b):
    return all(abs(x - y) <= 1e-9 for x, y in zip(a, b, strict=True))


_BROKEN = "PushT cannot read the Pymunk solver state of this pymunk build"


def solver_rows(env):
    """Return the bias velocities and cached contact impulses that the last Pymunk step left.

    The next Pymunk step moves each body by its bias velocity and starts its solver from the cached impulses.
    Pymunk does not expose either, so they are read from Munk2D's structs at the pinned version; the reads are
    checked against the public accessors. ``contacts`` rows are the contact point on body ``b`` and its normal
    impulse, NaN past the last contact.
    """
    bias = np.zeros((2, 3))
    for index, body in enumerate((env.agent, env.block)):
        address = _address(body._body)
        center = lib.cpBodyLocalToWorld(body._body, lib.cpBodyGetCenterOfGravity(body._body))
        if not _near(_doubles(address + _BODY_P, 2), (center.x, center.y)):
            raise RuntimeError(_BROKEN)
        bias[index] = _doubles(address + _BODY_BIAS, 3)
    contacts = np.full((MAX_CONTACTS, 3), np.nan)
    row, seen = 0, set()
    arbiters = []
    for body in (env.agent, env.block):
        body.each_arbiter(arbiters.append)
    for arbiter in arbiters:
        address = _address(arbiter._arbiter)
        if address in seen:
            continue
        seen.add(address)
        count = ctypes.c_int.from_address(address + _ARBITER_COUNT).value
        points = arbiter.contact_point_set.points
        if count != len(points):
            raise RuntimeError(_BROKEN)
        base = _pointer(address + _ARBITER_CONTACTS)
        bx, by = _doubles(_pointer(address + _ARBITER_BODY_B) + _BODY_P, 2)
        for index in range(count):
            contact = base + index * _CONTACT_SIZE
            rx, ry = _doubles(contact + _CONTACT_R2, 2)
            point = (bx + rx, by + ry)
            if not any(_near(point, p) for q in points for p in (q.point_a, q.point_b)):
                raise RuntimeError(_BROKEN)
            if row == MAX_CONTACTS:
                raise RuntimeError("PushT contact cache exceeds the supported geometry bound")
            contacts[row] = (*point, _doubles(contact + _CONTACT_IMPULSE, 1)[0])
            row += 1
    return bias, contacts


#: More private Munk2D 2.0.1 layouts: cpSpace ``shapeIDCounter`` and ``dynamicShapes``; cpBBTree ``root``;
#: cpShape ``bb`` and ``hashid``; a tree Node (``obj``, ``bb``, ``parent``, two children or a leaf's stamp and
#: pair list) and a Pair (two threads of ``prev``, ``leaf``, ``next``).
_SPACE_IDS, _SPACE_DYNAMIC = 128, 144
_TREE_ROOT = 48
_SHAPE_BB, _SHAPE_HASH = 64, 184
_PRIMES = (5, 13, 23, 47)


def _pointer(address):
    return ctypes.c_void_p.from_address(address).value or 0


def _leaf_order(hashes):
    """The order ``cpHashSetEach`` visits leaves inserted in ``hashes`` order (``cpHashSet`` with its resizes)."""
    size, table = _PRIMES[0], [[] for _ in range(_PRIMES[0])]
    for count, (leaf, value) in enumerate(hashes, 1):
        table[value % size].insert(0, (leaf, value))
        if count >= size:
            size = next(p for p in _PRIMES if p >= size + 1)
            grown = [[] for _ in range(size)]
            for chain in table:
                for item in chain:
                    grown[item[1] % size].insert(0, item)
            table = grown
    return [leaf for chain in table for leaf, _ in chain]


_NODE = struct.Struct("<Q4dQQQ")
_PAIR = struct.Struct("<6Q")


def broadphase_rows(env, leaves):
    """Return Munk2D's dynamic bounding-box tree of ``env`` in ``numba_broadphase``'s layout.

    ``leaves`` maps each dynamic shape to its leaf index. Also returns the leaves in hash-set order. The
    walls are the scene's first four shapes, so a wall's shape id is its index.
    """
    from . import numba_broadphase as bp

    space = _address(env.space._space)
    if ctypes.c_size_t.from_address(space + _SPACE_IDS).value != len(env.space.shapes):
        raise RuntimeError(_BROKEN)
    shapes, hashes = {}, []
    for shape, leaf in leaves.items():
        address = _address(shape._shape)
        shapes[address] = leaf
        if _doubles(address + _SHAPE_BB, 4) != tuple(shape.bb):
            raise RuntimeError(_BROKEN)
        hashes.append((leaf, ctypes.c_size_t.from_address(address + _SHAPE_HASH).value))

    def other(node):
        obj = _pointer(node)
        if obj in shapes:
            return index[node]
        wall = ctypes.c_size_t.from_address(obj + _SHAPE_HASH).value
        if wall >= 4:
            raise RuntimeError(_BROKEN)
        return bp.LEAVES + wall

    tree = np.full(bp.WIDTH, -1, np.int64)
    tree[1 : bp.LIST : 3] = -2
    bounds = np.zeros((bp.NODES, 4))
    root = _pointer(_pointer(space + _SPACE_DYNAMIC) + _TREE_ROOT)
    index, rows, inner = {}, {}, bp.LEAVES
    stack = [(root, -1)] if root else []
    while stack:
        node, parent = stack.pop()
        obj, left, bottom, right, top, _, a, b = _NODE.unpack(ctypes.string_at(node, _NODE.size))
        if obj:
            this = shapes[obj]
        else:
            this, inner = inner, inner + 1
            stack.extend(((b, this), (a, this)))
        index[node], rows[node] = this, (obj, a, b)
        tree[1 + 3 * this] = parent
        bounds[this] = left, bottom, right, top
        if parent < 0:
            tree[0] = this
    for node, this in index.items():
        obj, a, b = rows[node]
        if not obj:
            tree[2 + 3 * this], tree[3 + 3 * this] = index[a], index[b]
            continue
        # A leaf's union field ``b`` is its pair list.
        pair, j = b, 0
        while pair:
            _, first, after_first, _, second, after_second = _PAIR.unpack(ctypes.string_at(pair, _PAIR.size))
            if first == node:
                tree[bp.LIST + this * bp.PAIRS + j] = 2 * other(second)
                pair = after_first
            else:
                tree[bp.LIST + this * bp.PAIRS + j] = 2 * other(first) + 1
                pair = after_second
            j += 1
    order = np.full(bp.LEAVES, -1, np.int64)
    visit = _leaf_order(sorted(hashes, key=lambda item: item[1]))
    order[: len(visit)] = visit
    return tree, bounds, order


def _pieces(body):
    """A body's shapes in a fixed order: by vertices, a circle first."""
    return sorted(
        body.shapes,
        key=lambda shape: tuple(np.array(shape.get_vertices()).ravel()) if hasattr(shape, "get_vertices") else (),
    )


def geometry_columns(owners, shapes=None):
    """Return the body rows and convex pieces that the CUDA physics reads, as batch columns.

    Pieces are padded to ``shapes``, by default the most pieces of any body in the batch.
    """
    bodies = [(owner.env.agent, owner.env.block) for owner in owners]
    counts = [max(len(body.shapes) for body in pair) for pair in bodies]
    shapes = max(counts) if shapes is None else shapes
    rows, vertices, centers, inverse, radii, enabled, biases, impulses = [], [], [], [], [], [], [], []
    trees, bounds, orders = [], [], []
    for owner, pair in zip(owners, bodies, strict=True):
        bias, contacts = solver_rows(owner.env)
        biases.append(bias)
        impulses.append(contacts)
        pieces = [_pieces(body) for body in pair]
        leaves = {
            shape: MAX_SHAPES * index + rank for index, body in enumerate(pieces) for rank, shape in enumerate(body)
        }
        tree, bound, order = broadphase_rows(owner.env, leaves)
        trees.append(tree)
        bounds.append(bound)
        orders.append(order)
        radius = 0.0
        lane_vertices, lane_enabled = [], []
        for name, body, body_pieces in zip(("agent", "block"), pair, pieces, strict=True):
            polys = np.zeros((shapes, 4, 2), np.float64)
            valid = np.zeros(shapes, np.bool_)
            for index, shape in enumerate(body_pieces):
                if hasattr(shape, "get_vertices"):
                    polys[index] = shape.get_vertices()
                else:
                    radius = shape.radius
                valid[index] = True
            # Nonzero padded edges avoid NaNs in inactive polygon contacts.
            polys[~valid] = _UNIT_SQUARE
            if name == "agent" and not np.isfinite(body.mass):
                polys[:] = _UNIT_SQUARE
            lane_vertices.append(polys)
            lane_enabled.append(valid)
        rows.append([[*body.position, *body.velocity, body.angle, body.angular_velocity] for body in pair])
        vertices.append(lane_vertices)
        centers.append([body.center_of_gravity for body in pair])
        inverse.append([[1 / body.mass, 1 / body.moment] for body in pair])
        radii.append(radius)
        enabled.append(lane_enabled)
    return {
        "body": np.array(rows),
        "vertices": np.array(vertices),
        "centers": np.array(centers),
        "inverse": np.array(inverse),
        "radius": np.array(radii),
        "enabled": np.array(enabled),
        "circle": np.array([int(owner.env.variation_space["agent"]["shape"].value) == 0 for owner in owners]),
        "shapes": np.array(counts, np.int64),
        "bias": np.array(biases),
        "cached": np.array(impulses),
        "tree": np.array(trees),
        "tree_bounds": np.array(bounds),
        "tree_order": np.array(orders),
    }


def capture(env, collector=None, episode=None):
    """Return the ``current`` State fields of ``env`` in export order.

    ``episode`` is a cached ``episode_fields(env)``; it is copied.
    """
    from .contacts import capture_contacts

    result = {k: np.array(v[0]) for k, v in body_fields(np.array([body_row(env)], np.float64)).items()}
    result["contacts"] = np.array(env.n_contact_points, np.int64)
    result["render_segments"], result["render_segment_count"] = capture_contacts(env.space, collector)
    result.update(latest_fields(env))
    for key, value in (episode_fields(env) if episode is None else episode).items():
        result[key] = value.copy()
    return {key: result[key] for key in CURRENT_KEYS}
