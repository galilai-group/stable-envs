"""Direct GPU lifetime checks for the five Workstream-I backend objects.

The environment oracle checks public pixels.  This probe checks the lower-level ownership contract
directly so a cache or environment wrapper cannot make every ``_release`` body unreachable while the
job still reports success.  Run one backend per compatible Oscar venv.
"""
from __future__ import annotations

import argparse
import gc
import json


def _delta(after: dict, before: dict, section: str) -> dict[str, int]:
    a, b = after.get(section, {}), before.get(section, {})
    return {key: a.get(key, 0) - b.get(key, 0) for key in sorted(set(a) | set(b))
            if a.get(key, 0) != b.get(key, 0)}


def _nvdiff() -> tuple[list[str], dict]:
    import numpy as np
    import torch

    from stable_envs.render import nvdiff as ND

    vert = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
    norm = np.array([[0.0, 0.0, 1.0]] * 3)
    scene = ND.Scene.from_triangles(
        vert, norm, [[0, 1, 2]], [0, 0, 0], rgba=[[1.0, 1.0, 1.0, 1.0]], device="cuda")
    renderer = ND.BatchRenderer(
        None, res=16, scene=scene, device="cuda", textures=False, mipmap=False, skybox=False)
    renderer.owns_scene = True

    identity = np.eye(3)
    cam0 = ND.Camera(np.array([0.0, 0.0, 3.0]), identity, 60.0, 0.1, 10.0)
    first = renderer._ray_dirs(cam0, 16, 1).clone()
    for i in range(ND.RAY_CACHE_MAX + 1):
        angle = 0.1 * (i + 1)
        c, s = np.cos(angle), np.sin(angle)
        mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        renderer._ray_dirs(ND.Camera(cam0.pos, mat, 60.0, 0.1, 10.0), 16, 1)
    again = renderer._ray_dirs(cam0, 16, 1)
    assert torch.equal(first, again), "LRU overflow changed the rebuilt ray grid"
    assert renderer._ray_cache.evictions >= 2

    renderer.close()
    assert renderer.closed and scene.closed
    assert renderer.ctx is None and renderer.scene is None
    for name in ("vert", "norm", "uv", "vgeom", "face", "fgeom", "mjgeom", "S3", "S4", "S3p",
                 "mat_rgb", "alpha", "spec", "shininess", "emission", "texid"):
        assert getattr(scene, name) is None, f"Scene.close left {name} alive"
    return ["nvdiff.Scene", "nvdiff.BatchRenderer"], {
        "ray_cache_evictions": renderer._ray_cache.evictions,
        "ray_cache_rebuild_identical": True,
    }


def _mjwarp() -> tuple[list[str], dict]:
    import mujoco

    from stable_envs.render import _resources as RES
    from stable_envs.render import mjwarp as MW

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><camera name='camera' pos='0 -3 1'/>"
        "<geom type='box' size='.5 .5 .5'/></worldbody></mujoco>")
    renderer = MW.BatchRenderer(model, batch=2, res=16, capture=False)
    renderer.close()
    assert renderer.closed
    assert renderer.graph is None and renderer._bgra is None
    assert renderer.rc is None and renderer.d is None and renderer.m is None
    try:
        renderer.render(None)
    except RES.UseAfterCloseError:
        pass
    else:
        raise AssertionError("closed MJWarp renderer did not raise UseAfterCloseError")
    return ["mjwarp.BatchRenderer"], {}


def _rendernet() -> tuple[list[str], dict]:
    import torch

    from stable_envs.render import _resources as RES
    from stable_envs.render import rendernet as RN

    # ``load`` / ``unsealed`` move weights onto ``device``. Constructing ``LoadedCorrector``
    # around a CPU module is not a supported path (job 6509711: cuda input, CPU weights).
    corrector = RN.unsealed(
        RN.ESPCN(in_size=16, out_size=16, width=8, depth=2),
        cell={"env_class": "ResourceProbe"},
        render={"render_size": 16, "out_size": 16, "tex_lod": 1, "backface_cull": False},
        net="resource_probe",
        device="cuda",
        mode="eager",
        net_batch=2,
    )
    runner = corrector.runner()
    runner(torch.zeros((1, 16, 16, 3), dtype=torch.uint8, device="cuda"))
    corrector.close()
    assert corrector.closed and runner.closed and corrector.module is None
    try:
        corrector.runner()
    except RES.UseAfterCloseError:
        pass
    else:
        raise AssertionError("closed corrector did not raise UseAfterCloseError")
    return ["rendernet.LoadedCorrector", "rendernet.Runner"], {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("mjwarp", "nvdiff", "rendernet"))
    args = parser.parse_args()

    import torch

    from stable_envs.render import _resources as RES

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.synchronize()
    before = RES.counts()
    live_before = {kind: RES.live(kind) for kind in (
        "mjwarp.BatchRenderer", "nvdiff.Scene", "nvdiff.BatchRenderer",
        "rendernet.LoadedCorrector", "rendernet.Runner")}
    memory_before = int(torch.cuda.memory_allocated())

    kinds, detail = {"mjwarp": _mjwarp, "nvdiff": _nvdiff, "rendernet": _rendernet}[args.backend]()

    torch.cuda.synchronize()
    memory_open_or_closed = int(torch.cuda.memory_allocated())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory_after = int(torch.cuda.memory_allocated())
    after = RES.counts()

    created = _delta(after, before, "created")
    closed = _delta(after, before, "closed")
    failed = _delta(after, before, "release_failed")
    for kind in kinds:
        assert created.get(kind) == 1, (kind, created)
        assert closed.get(kind) == 1, (kind, closed)
        assert RES.live(kind) == live_before[kind], (kind, live_before[kind], RES.live(kind))
    assert not failed, failed
    assert memory_after <= memory_open_or_closed, (memory_after, memory_open_or_closed)

    record = {
        "backend": args.backend,
        "kinds": kinds,
        "created": created,
        "closed": closed,
        "release_failed": failed,
        "memory_before": memory_before,
        "memory_after_close": memory_after,
        "memory_before_empty_cache": memory_open_or_closed,
        **detail,
    }
    print("@@JSON@@" + json.dumps(record, sort_keys=True))
    print("RESOURCE_PROBE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
