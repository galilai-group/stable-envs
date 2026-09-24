import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np

from stable_envs._adapters.ogbench_render_state import build_model

root = Path.cwd()
p = root / "scripts/recovery/ogbench_renderer_gate.py"
s = importlib.util.spec_from_file_location("gate", p)
g = importlib.util.module_from_spec(s)
s.loader.exec_module(g)
f = root / "docs/research/semantic_recovery_evidence/ogb_renderer_closeout/fixtures"
report = []
for c in json.loads((f / "cases.json").read_text()):
    rs = g.load_state(f / c["render"])
    family = c["family"]
    a = build_model(family, rs[0])
    o = g.owned_oracle()
    b = (o.OwnedCube if family == "Cube" else o.OwnedScene)(entropy_seed=17)
    vals = {
        k: rs["appearance", k][0].numpy().copy()
        for k in ["cube.size", "agent.color", "camera.angle_delta", "light.intensity", "cube.color", "floor.color"]
        + (["lock_color"] if family == "Scene" else [])
    }
    b.reset(seed=17, options={"task_id": 1, "variation": [], "variation_values": vals})
    if family == "Scene":
        for e in [a, b]:
            e._cur_button_states = rs["pose", "buttons"][0].numpy().copy()
            e._apply_button_states()
    a.variation_space._walk("cube.color").set_value(vals["cube.color"])
    a._cube_colors_from_variation()
    fields = [
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_rgba",
        "mat_rgba",
        "mat_emission",
        "mat_specular",
        "mat_shininess",
        "mat_reflectance",
        "light_pos",
        "light_dir",
        "light_diffuse",
        "light_ambient",
        "light_specular",
        "body_pos",
        "body_quat",
        "mesh_vert",
        "mesh_normal",
        "mesh_texcoord",
        "mesh_face",
    ]
    bad = [k for k in fields if not np.array_equal(getattr(a.model, k), getattr(b.model, k))]
    for k in ["cam_pos", "cam_quat", "cam_fovy", "cam_intrinsic"]:
        ai = mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
        bi = mujoco.mj_name2id(b.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
        if not np.array_equal(getattr(a.model, k)[ai], getattr(b.model, k)[bi]):
            bad.append(k)
    report.append(
        dict(
            family=family,
            case=c["label"],
            different_render_model_fields=bad,
            checked_fields=fields + ["cam_pos", "cam_quat", "cam_fovy", "cam_intrinsic"],
        )
    )
    a.close()
    b.close()
print(json.dumps(report, indent=2))
assert not any(x["different_render_model_fields"] for x in report)
