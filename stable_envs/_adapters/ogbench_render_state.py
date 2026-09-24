"""Numeric pixel inputs for the registered Cube/Scene renderer boundary."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from tensordict import TensorDict

from .ogbench_manipulation_compat import CompatibleCube, CompatibleScene

STATIC = ("cube.size", "agent.color", "camera.angle_delta", "light.intensity")


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


def project(family, state, size, supersample):
    if type(size) is not int or type(supersample) is not int:
        raise ValueError("Render size and supersample must be integers")
    b = len(state)
    device = state.device or state["physics", "qpos"].device
    values = {
        ("renderer", "signature"): torch.as_tensor(signature(family), device=device).expand(b, 32).clone(),
        ("renderer", "size"): torch.full((b, 1), size, dtype=torch.int64, device=device),
        ("renderer", "supersample"): torch.full((b, 1), supersample, dtype=torch.int64, device=device),
        ("renderer", "camera"): torch.zeros((b, 1), dtype=torch.int64, device=device),
        ("appearance", "target_alpha"): torch.full((b, 1), 0.2, dtype=torch.float32, device=device),
    }
    for name in ("qpos", "mocap_pos", "mocap_quat"):
        values["pose", name] = state["physics", name].clone()
    for name in (*STATIC, "floor.color"):
        values["appearance", name] = state["model", "compiled", name].clone()
    values["appearance", "cube.color"] = state["variation", "current", "cube.color"].clone()
    if family == "Scene":
        values["appearance", "lock_color"] = state["variation", "current", "lock_color"].clone()
        values["pose", "buttons"] = state["scene", "cur_button_states"].clone()
    result = TensorDict(values, [b], device=device)
    validate(family, result)
    return result


def validate(family, state):
    if not isinstance(state, TensorDict) or len(state.batch_size) != 1 or len(state) < 1:
        raise TypeError("Named Cube/Scene RenderState with a nonempty batch is required")
    schema = fields(family)
    expected = set(schema) | {key[0] for key in schema}
    if set(state.keys(True, False)) != expected:
        raise ValueError("RenderState has missing or extra fields")
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
    if not torch.all(state["appearance", "target_alpha"] == 0.2):
        raise ValueError("Only registered visible target alpha 0.2 is supported")
    qadr = 14
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


def build_model(family, row):
    """Compile render-only assets/configuration. Never reset or step physics."""
    cls = CompatibleCube if family == "Cube" else CompatibleScene
    env = cls(seed=0)
    try:
        for name in STATIC:
            env.variation_space._walk(name)._value = row["appearance", name].detach().cpu().numpy().copy()
        if family == "Scene":
            env.variation_space._walk("lock_color").set_value(row["appearance", "lock_color"].cpu().numpy().copy())
        # Floor/cube RGB use the frozen renderer's existing per-world color paths.
        env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
        env.compile_model_and_data()
        env._cube_colors_from_variation()
        for ids in env._cube_target_geom_ids_list:
            env.model.geom_rgba[ids, 3] = 0.2
        if family == "Scene":
            env._cur_button_states = np.zeros(2, np.int64)
            env._apply_button_states()
        return env
    except BaseException:
        env.close()
        raise
