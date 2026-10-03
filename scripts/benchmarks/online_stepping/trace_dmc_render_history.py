"""Trace a frozen failing gate without changing its rendering or assertions.

Run beside diagnose_dmc_export_frames.py with the original gate arguments.
Groups byte-identical compact renderer inputs and compares scene fields and
images across their occurrences. This is diagnostic evidence, not acceptance.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        h.update(key.encode() + b"\0" + array.dtype.str.encode())
        h.update(str(array.shape).encode() + b"\0" + array.tobytes())
    return h.hexdigest()


def fields(obj, prefix):
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if isinstance(value, (np.ndarray, int, float, bool)):
            result[prefix + key] = np.array(value, copy=True)
    return result


def scene_fields(scene):
    result = fields(scene, "scene/")
    # Only active geometry and lights are inputs; unused slots may be uninitialized.
    result["scene/geomorder"] = np.array(scene.geomorder[:scene.ngeom], copy=True)
    for name, objects in (("geom", scene.geoms[:scene.ngeom]),
                          ("light", scene.lights[:scene.nlight]), ("camera", scene.camera)):
        for i, obj in enumerate(objects):
            result.update(fields(obj, f"{name}/{i}/"))
    return result


def differences(left, right):
    return [key for key in sorted(left.keys() | right.keys())
            if key not in left or key not in right
            or digest({key: left[key]}) != digest({key: right[key]})]


def main():
    import diagnose_dmc_export_frames as diagnostic
    import mujoco

    from stable_envs.envs.dmcontrol import trackcom_opengl

    output = Path(sys.argv[sys.argv.index("--output") + 1]).parent
    target = output / "render_trace.json"
    if target.exists() or (output / "render_trace.npz").exists():
        raise FileExistsError(target)
    original_batch = trackcom_opengl.render
    original_scene = mujoco.Renderer.update_scene
    original_render = mujoco.Renderer.render
    pending, records = [], []

    def batch(env, states, style, size, out):
        assert not pending
        rows = states.detach().cpu().numpy()
        common = {"pristine/" + key: value for key, value in env._pristine().items()}
        common["size"] = np.array(size)
        common["slots"] = np.array(repr(env._slots()))
        for i, row in enumerate(rows):
            inputs = dict(common, state=row)
            inputs.update({"style/" + key: value[i].detach().cpu().numpy()
                           for key, value in style.items()})
            pending.append(dict(input_sha256=digest(inputs), owner=id(env), lane=i))
        try:
            return original_batch(env, states, style, size, out)
        finally:
            pending.clear()

    def update(renderer, data, *args, **kwargs):
        result = original_scene(renderer, data, *args, **kwargs)
        if pending:
            record = pending[0]
            record["scene_before"] = scene_fields(renderer.scene)
            record["kinematics"] = {key: np.array(getattr(data, key), copy=True) for key in
                                    ("qpos", "qvel", "xpos", "xmat", "geom_xpos", "geom_xmat",
                                     "cam_xpos", "cam_xmat", "light_xpos", "light_xdir", "subtree_com")}
        return result

    def render(renderer, *args, **kwargs):
        result = original_render(renderer, *args, **kwargs)
        if pending:
            record = pending.pop(0)
            record["scene_after"] = scene_fields(renderer.scene)
            record["image"] = result.copy()
            records.append(record)
        return result

    trackcom_opengl.render = batch
    mujoco.Renderer.update_scene = update
    mujoco.Renderer.render = render
    report = dict(status="RUNNING", diagnostic_only=True,
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        diagnostic.main()
        report["status"] = "GATE_DID_NOT_REPRODUCE"
    except BaseException as exc:
        report.update(status="GATE_RAISED", error=repr(exc))
        raise
    finally:
        trackcom_opengl.render = original_batch
        mujoco.Renderer.update_scene = original_scene
        mujoco.Renderer.render = original_render
        groups, mismatch, arrays = {}, [], {}
        for index, record in enumerate(records):
            key = record["input_sha256"]
            for previous in groups.get(key, []):
                other = records[previous]
                if np.array_equal(other["image"], record["image"]):
                    continue
                delta = other["image"].astype(float) - record["image"].astype(float)
                mismatch.append(dict(first=previous, second=index, input_sha256=key,
                    differing_channels=int(np.count_nonzero(delta)),
                    normalized_mse=float(np.mean(delta ** 2) / 255 ** 2),
                    differing_fields={section: differences(other[section], record[section])
                                      for section in ("scene_before", "scene_after", "kinematics")}))
                for n in (previous, index):
                    arrays[f"{n}/image"] = records[n]["image"]
                    for section in ("scene_before", "scene_after", "kinematics"):
                        arrays.update({f"{n}/{section}/{k}": v for k, v in records[n][section].items()})
            groups.setdefault(key, []).append(index)
        report.update(render_count=len(records), input_groups=len(groups),
                      repeated_input_groups=sum(len(v) > 1 for v in groups.values()),
                      mismatches=mismatch,
                      records=[{k: r[k] for k in ("input_sha256", "owner", "lane")} for r in records])
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "render_trace.npz", **arrays)
        target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
