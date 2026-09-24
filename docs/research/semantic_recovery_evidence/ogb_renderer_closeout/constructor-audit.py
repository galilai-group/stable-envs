import json

from stable_envs._adapters.ogbench_manipulation_compat import CompatibleCube, CompatibleScene
from stable_envs.envs.ogbench import manipulation as frozen

out = {}
for family, cls in [("Cube", CompatibleCube), ("Scene", CompatibleScene)]:
    e = cls(seed=0)
    e.reset(seed=0)
    try:
        status = "constructed"
        ref = frozen._ogbench_make(family, {})
        ref.close()
    except Exception as ex:
        status = f"{type(ex).__name__}: {ex}"
    out[family] = {
        "frozen_constructor": status,
        "registered_render_size": frozen.SPEC[family]["size"],
        "cpu_engine_render_size": e._render_height,
        "camera_count": e.model.ncam,
        "target_geom_ids": [x.tolist() if hasattr(x, "tolist") else x for x in e._cube_target_geom_ids_list],
        "target_rgba": [e.model.geom_rgba[x].tolist() for x in e._cube_target_geom_ids_list],
        "appearance_digest": frozen._appearance_digest(e.model, e._scene_option.sitegroup),
        "required_static_variations": {},
    }
    for name, value in [
        ("agent.color", [0.2, 0.3, 0.4]),
        ("camera.angle_delta", [[2.0, 1.0]]),
        ("light.intensity", [0.9]),
        ("cube.size", [0.03]),
    ]:
        e.reset(seed=0, options={"variation": [], "variation_values": {name: value}})
        out[family]["required_static_variations"][name] = frozen._appearance_digest(e.model, e._scene_option.sitegroup)
    e.close()
print(json.dumps(out, indent=2))
