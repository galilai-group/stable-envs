from __future__ import annotations

import functools
import hashlib

import numpy as np
import torch
from tensordict import TensorDict

from .compatibility import CompatibleCube, CompatibleScene

OBJECT_QPOS_ADR = 14
TARGET_ALPHA = 0.2

STATIC = ("cube.size", "camera.angle_delta", "light.intensity")
# agent.color is these materials' RGB, written per world by the renderers.
AGENT_MATERIALS = ("ur5e/robotiq/black", "ur5e/robotiq/pad_gray")


class _RenderScene(CompatibleScene):
    """Give each button its own render material while preserving source material parameters.

    MuJoCo honours geom RGB overrides; MJWarp reads material RGB. Independent
    materials let the two button-state bits select different colours.
    """

    def modify_mjcf_model(self, model):
        model = super().modify_mjcf_model(model)
        for i in range(2):
            geom = model.find("geom", f"btngeom_{i}")
            name = f"se_button_{i}"
            material = model.find("material", name)
            if material is None:
                attrs = {k: v for k, v in geom.material.get_attributes().items() if k != "name"}
                material = model.asset.add("material", name=name, **attrs)
            geom.material = material
        return model

    def _apply_button_states(self):
        super()._apply_button_states()
        for i, geoms in enumerate(self._button_geom_ids_list):
            rgba = self.model.geom_rgba[geoms[0]]
            # MuJoCo's exact compiler-default RGBA means "use material".
            # Preserve that source behavior for a lock color of exactly 0.5.
            if np.array_equal(rgba, [0.5, 0.5, 0.5, 1.0]):
                rgba = self.model.material("btn_red").rgba
            self.model.material(f"se_button_{i}").rgba[:] = rgba


def signature(family):
    return np.frombuffer(
        hashlib.sha256(f"ogb_pixels_v1|{family}|ogbench=1.2.1|camera=front_pixels".encode()).digest(), np.uint8
    ).copy()


def fields(family):
    nq = 21 if family == "Cube" else 25
    result = {
        ("renderer", "signature"): ((32,), torch.uint8),
        ("renderer", "size"): ((1,), torch.int64),
        ("renderer", "supersample"): ((1,), torch.int64),
        ("renderer", "camera"): ((1,), torch.int64),
        ("pose", "qpos"): ((nq,), torch.float64),
        ("pose", "mocap_pos"): ((3,), torch.float64),
        ("pose", "mocap_quat"): ((4,), torch.float64),
        ("appearance", "cube.size"): ((1,), torch.float64),
        ("appearance", "agent.color"): ((3,), torch.float64),
        ("appearance", "camera.angle_delta"): ((1, 2), torch.float64),
        ("appearance", "light.intensity"): ((1,), torch.float64),
        ("appearance", "floor.color"): ((2, 3), torch.float64),
        ("appearance", "cube.color"): ((1, 3), torch.float64),
        ("appearance", "target_alpha"): ((1,), torch.float32),
    }
    if family == "Scene":
        result[("appearance", "lock_color")] = ((2, 3), torch.float64)
        result[("pose", "buttons")] = ((2,), torch.int64)
    return result


def header(family, b, size, supersample, device):
    """Renderer fields shared by every row of a RenderState batch."""
    if type(size) is not int or type(supersample) is not int:
        raise ValueError("Render size and supersample must be integers")
    return {
        ("renderer", "signature"): torch.as_tensor(signature(family), device=device).expand(b, 32).clone(),
        ("renderer", "size"): torch.full((b, 1), size, dtype=torch.int64, device=device),
        ("renderer", "supersample"): torch.full((b, 1), supersample, dtype=torch.int64, device=device),
        ("renderer", "camera"): torch.zeros((b, 1), dtype=torch.int64, device=device),
        ("appearance", "target_alpha"): torch.full((b, 1), TARGET_ALPHA, dtype=torch.float32, device=device),
    }


def projected_keys(family):
    """The State leaves ``project`` reads."""
    keys = [("physics", name) for name in ("qpos", "mocap_pos", "mocap_quat")]
    keys += [("model", "compiled", name) for name in (*STATIC, "agent.color", "floor.color")]
    keys.append(("variation", "current", "cube.color"))
    if family == "Scene":
        keys += [("variation", "current", "lock_color"), ("scene", "cur_button_states")]
    return keys


def state_columns(family, columns, size, supersample):
    """NumPy RenderState columns from trusted stacked State leaves, as ``project`` would fill them."""
    b = len(columns["physics", "qpos"])
    values = {
        ("renderer", "size"): np.full((b, 1), size, np.int64),
        ("renderer", "supersample"): np.full((b, 1), supersample, np.int64),
    }
    for name in ("qpos", "mocap_pos", "mocap_quat"):
        values["pose", name] = columns["physics", name]
    for name in (*STATIC, "agent.color", "floor.color"):
        values["appearance", name] = columns["model", "compiled", name]
    values["appearance", "cube.color"] = columns["variation", "current", "cube.color"]
    if family == "Scene":
        values["appearance", "lock_color"] = columns["variation", "current", "lock_color"]
        values["pose", "buttons"] = columns["scene", "cur_button_states"]
    return values


def project(family, state, size, supersample, check=True):
    b = len(state)
    device = state.device or state["physics", "qpos"].device
    values = header(family, b, size, supersample, device)
    for name in ("qpos", "mocap_pos", "mocap_quat"):
        values["pose", name] = state["physics", name].clone()
    for name in (*STATIC, "agent.color", "floor.color"):
        values["appearance", name] = state["model", "compiled", name].clone()
    values["appearance", "cube.color"] = state["variation", "current", "cube.color"].clone()
    if family == "Scene":
        values["appearance", "lock_color"] = state["variation", "current", "lock_color"].clone()
        values["pose", "buttons"] = state["scene", "cur_button_states"].clone()
    result = TensorDict(values, [b], device=device)
    if check:
        validate(family, result)
    return result


def validate(family, state):
    if not isinstance(state, TensorDict) or len(state.batch_size) != 1 or len(state) < 1:
        raise TypeError("Named Cube/Scene RenderState with a nonempty batch is required")
    schema = fields(family)
    expected = set(schema) | {key[0] for key in schema}
    if set(state.keys(True, False)) != expected:
        raise ValueError("RenderState has missing or extra fields")
    if state.device is not None and state.device.type == "cpu":
        return _validate_host(family, state, schema)
    for key, (shape, dtype) in schema.items():
        value = state[key]
        if value.shape != (len(state), *shape) or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError(f"Invalid RenderState dtype/shape/value: {key}")
    sig = torch.as_tensor(signature(family), device=state["renderer", "signature"].device)
    if not torch.all(state["renderer", "signature"] == sig):
        raise ValueError("Incompatible renderer family/asset/schema profile")
    for name, maximum in [("size", 4096), ("supersample", 4)]:
        value = state["renderer", name]
        if torch.any((value < 1) | (value > maximum)) or not torch.all(value == value[0]):
            raise ValueError("RenderState requires one valid square resolution/supersampling per batch")
    if torch.any(state["renderer", "camera"] != 0):
        raise ValueError("Only registered front_pixels camera is supported")
    if not torch.all(state["appearance", "target_alpha"] == TARGET_ALPHA):
        raise ValueError("Only registered visible target alpha 0.2 is supported")
    qadr = OBJECT_QPOS_ADR
    for q in (state["pose", "qpos"][:, qadr + 3 : qadr + 7], state["pose", "mocap_quat"]):
        if not torch.allclose(
            torch.linalg.vector_norm(q, dim=-1),
            torch.ones(len(state), dtype=q.dtype, device=q.device),
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("RenderState quaternion must be unit length")
    for name in (
        "cube.color",
        "agent.color",
        "floor.color",
        "light.intensity",
        *(("lock_color",) if family == "Scene" else ()),
    ):
        v = state["appearance", name]
        if torch.any((v < 0) | (v > 1)):
            raise ValueError("RenderState color/intensity range")
    angle = state["appearance", "camera.angle_delta"]
    if torch.any(angle.abs() > (10 if family == "Cube" else 5)):
        raise ValueError("RenderState camera-angle range")
    sz = state["appearance", "cube.size"]
    if torch.any((sz < min(0.01, float(np.float32(0.01)))) | (sz > 0.03)):
        raise ValueError("RenderState cube-size range")
    if family == "Scene" and torch.any((state["pose", "buttons"] < 0) | (state["pose", "buttons"] > 1)):
        raise ValueError("RenderState button bits")
    return state


@functools.lru_cache
def _signature(family):
    return signature(family)


def _validate_host(family, state, schema):
    """The checks of ``validate`` on NumPy views of a CPU RenderState."""
    n = len(state)
    leaves = dict(state.items(True, True))
    values = {}
    for key, (shape, dtype) in schema.items():
        value = leaves[key]
        if value.shape != (n, *shape) or value.dtype != dtype:
            raise ValueError(f"Invalid RenderState dtype/shape/value: {key}")
        array = values[key] = value.detach().numpy()
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"Invalid RenderState dtype/shape/value: {key}")
    if not np.all(values["renderer", "signature"] == _signature(family)):
        raise ValueError("Incompatible renderer family/asset/schema profile")
    for name, maximum in [("size", 4096), ("supersample", 4)]:
        value = values["renderer", name]
        if np.any((value < 1) | (value > maximum)) or not np.all(value == value[0]):
            raise ValueError("RenderState requires one valid square resolution/supersampling per batch")
    if np.any(values["renderer", "camera"] != 0):
        raise ValueError("Only registered front_pixels camera is supported")
    if not np.all(values["appearance", "target_alpha"] == np.float32(TARGET_ALPHA)):
        raise ValueError("Only registered visible target alpha 0.2 is supported")
    qadr = OBJECT_QPOS_ADR
    for q in (values["pose", "qpos"][:, qadr + 3 : qadr + 7], values["pose", "mocap_quat"]):
        if not np.all(np.abs(np.sqrt(np.sum(q * q, axis=-1)) - 1.0) <= 1e-6):
            raise ValueError("RenderState quaternion must be unit length")
    for name in (
        "cube.color",
        "agent.color",
        "floor.color",
        "light.intensity",
        *(("lock_color",) if family == "Scene" else ()),
    ):
        v = values["appearance", name]
        if np.any((v < 0) | (v > 1)):
            raise ValueError("RenderState color/intensity range")
    angle = values["appearance", "camera.angle_delta"]
    if np.any(np.abs(angle) > (10 if family == "Cube" else 5)):
        raise ValueError("RenderState camera-angle range")
    sz = values["appearance", "cube.size"]
    if np.any((sz < min(0.01, float(np.float32(0.01)))) | (sz > 0.03)):
        raise ValueError("RenderState cube-size range")
    if family == "Scene" and np.any((values["pose", "buttons"] < 0) | (values["pose", "buttons"] > 1)):
        raise ValueError("RenderState button bits")
    return state


def build_model(family, row=None, names=None):
    """Compile renderer assets and configuration without resetting or stepping physics.

    ``names`` selects the row's leaves to compile; the others keep their defaults.
    """
    cls = CompatibleCube if family == "Cube" else _RenderScene
    env = cls(seed=0)
    if names is None:
        names = () if row is None else (*STATIC, *(("lock_color",) if family == "Scene" else ()))
    try:
        for name in names:
            value = row["appearance", name].detach().cpu().numpy().copy()
            if name == "lock_color":
                env.variation_space._walk(name).set_value(value)
            else:
                env.variation_space._walk(name)._value = value
        # Floor, cube and agent RGB use the renderer’s per-world colour paths.
        env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
        env.compile_model_and_data()
        env._cube_colors_from_variation()
        for ids in env._cube_target_geom_ids_list:
            env.model.geom_rgba[ids, 3] = TARGET_ALPHA
        if family == "Scene":
            env._cur_button_states = np.zeros(2, np.int64)
            env._apply_button_states()
        return env
    except BaseException:
        env.close()
        raise
