"""One shared multi-object renderer must match independent Blender oracles."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("recording rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.testing.geometry import depth_mse, mask_iou
from tests.helpers.reference.object_permanence_recordings import (
    DEPTH_MSE_MAX,
    MASK_IOU_MIN,
    RECORDED_TASKS,
    RECORDINGS,
    load_geometry,
)


@pytest.mark.parametrize("task", RECORDED_TASKS)
@pytest.mark.parametrize("cache_static", [True, False])
def test_recorded_scene_passes_source_depth_and_all_object_masks(task, cache_static, record_property):
    manifest, oracle = load_geometry(task)
    frames = oracle["geometry_frames"] - 1
    with (
        RigidTrajectory(RECORDINGS, f"{task}-nominal") as clip,
        RigidSceneRenderer(clip, cache_static=cache_static) as renderer,
    ):
        output = renderer.render_many(frames, modes=("depth", "ids"))
        depths, ids = output["depth"].cpu().numpy(), output["ids"].cpu().numpy()
    worst, minimum = 0.0, 1.0
    hidden = {name: 0 for name in manifest.get("targets", ["small_orange_ball"])}
    for row, frame in enumerate(frames + 1):
        expected = oracle["object_ids"][row]
        roi = (expected > 0) | (ids[row] > 0)
        mse = float(depth_mse(depths[row], oracle["depth"][row], roi))
        assert mse <= DEPTH_MSE_MAX, (task, int(frame), mse)
        worst = max(worst, mse)
        for name, object_id in manifest["object_ids"].items():
            actual_mask, expected_mask = ids[row] == object_id, expected == object_id
            iou = float(mask_iou(actual_mask, expected_mask))
            if np.isnan(iou):
                assert not actual_mask.any() and not expected_mask.any()
                if name in hidden:
                    hidden[name] += 1
            else:
                assert iou >= MASK_IOU_MIN, (task, int(frame), name, iou)
                minimum = min(minimum, iou)
    expected_hidden = {
        name: int(np.all(oracle["object_ids"] != manifest["object_ids"][name], axis=(1, 2)).sum()) for name in hidden
    }
    assert hidden == expected_hidden
    if task != "g64":
        assert all(count > 0 for count in hidden.values())
    else:
        # Source G64 has a real window: partial visibility, despite prose tags
        # claiming complete hiding. Preserve the actual source mesh semantics.
        assert all(count == 0 for count in hidden.values())
    record_property("depth_mse_max", worst)
    record_property("mask_iou_min", minimum)
    record_property("hidden_target_frames", hidden)
    print(task, {"depth_mse_max": worst, "mask_iou_min": minimum, "hidden": hidden})


def test_multi_outputs_batches_semantic_ids_and_caller_buffers(tmp_path):
    clip = RigidTrajectory(RECORDINGS, "g62-nominal")
    with RigidSceneRenderer(clip, size=90) as renderer:
        frames = np.array([119, 0, 59, 44, 0])
        images = renderer.render_many(frames)
        for mode in images:
            assert torch.equal(images[mode], renderer.render(frames, mode=mode))
            for index in (0, 2):
                assert torch.equal(
                    images[mode][index : index + 1], renderer.render(frames[index : index + 1], mode=mode)
                )
        out = {mode: torch.empty_like(value) for mode, value in images.items()}
        assert renderer.render_many(frames, out=out) is out
        assert all(torch.equal(out[m], images[m]) for m in images)
        bad = {m: torch.full_like(v, 17) for m, v in out.items()}
        bad["depth"] = bad["depth"].cpu()
        with pytest.raises(ValueError, match="out"):
            renderer.render_many(frames, out=bad)
        assert all(torch.all(value == 17) for value in bad.values())
    assert not clip.closed  # Renderer borrows, it never closes caller data.
    clip.close()

    import json
    import shutil

    changed_root = tmp_path / "changed-ids"
    shutil.copytree(RECORDINGS, changed_root)
    path = changed_root / "clips/g62-nominal/manifest.json"
    manifest = json.loads(path.read_text())
    for obj in manifest["objects"]:
        obj["id"] += 20
    path.write_text(json.dumps(manifest))
    with RigidTrajectory(changed_root, "g62-nominal") as other, RigidSceneRenderer(other, size=90) as renderer:
        changed = renderer.render(frames, mode="ids")
        assert torch.equal(changed, torch.where(images["ids"] > 0, images["ids"] + 20, 0))


def test_source_gate_rejects_shifted_camera(tmp_path):
    import json
    import shutil

    root = tmp_path / "changed-camera"
    shutil.copytree(RECORDINGS, root)
    path = root / "clips/g62-nominal/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["camera"]["world"][0][3] += 0.2
    path.write_text(json.dumps(manifest))
    _, oracle = load_geometry("g62")
    with RigidTrajectory(root, "g62-nominal") as clip, RigidSceneRenderer(clip) as renderer:
        output = renderer.render_many([0], modes=("depth", "ids"))
        depth, ids = output["depth"][0].cpu().numpy(), output["ids"][0].cpu().numpy()
    assert depth_mse(depth, oracle["depth"][0], oracle["object_ids"][0] > 0) > DEPTH_MSE_MAX
    assert mask_iou(ids == 1, oracle["object_ids"][0] == 1) < MASK_IOU_MIN


@pytest.mark.parametrize("task", RECORDED_TASKS)
def test_cached_and_full_scene_outputs_are_bit_identical_for_all_frames(task):
    frames = np.arange(120)[::-1].copy()
    with RigidTrajectory(RECORDINGS, f"{task}-nominal") as clip:
        with (
            RigidSceneRenderer(clip, size=90) as cached,
            RigidSceneRenderer(clip, size=90, cache_static=False) as full,
        ):
            expected = full.render_many(frames)
            actual = cached.render_many(frames)
            assert all(torch.equal(actual[m], expected[m]) for m in actual)
            assert cached.renderer._static_coverage.builds == 1
            # Repeated/shuffled sampling must retain the seed and output contract.
            selected = np.array([44, 0, 119, 44, 62])
            again = cached.render_many(selected)
            assert all(torch.equal(again[m], expected[m][119 - selected]) for m in again)
            assert cached.renderer._static_coverage.builds == 1


@pytest.mark.parametrize("all_static", [True, False])
def test_static_only_and_dynamic_only_recordings_and_shared_dynamic_nodes(tmp_path, all_static):
    import hashlib
    import json
    import shutil

    root = tmp_path / "partition"
    shutil.copytree(RECORDINGS, root)
    folder = root / "clips/g62-nominal"
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    if all_static:
        manifest["dynamic_nodes"] = []
        poses = np.empty((120, 0, 7), np.float32)
        np.save(folder / "poses.npy", poses)
        manifest["arrays"]["poses"].update(
            shape=list(poses.shape), sha256=hashlib.sha256((folder / "poses.npy").read_bytes()).hexdigest()
        )
    for i, obj in enumerate(manifest["objects"]):
        obj["node"] = -1 if all_static else i % 2
    path.write_text(json.dumps(manifest))
    with RigidTrajectory(root, "g62-nominal") as clip:
        with (
            RigidSceneRenderer(clip, size=90) as cached,
            RigidSceneRenderer(clip, size=90, cache_static=False) as full,
        ):
            expected, actual = full.render_many([0, 44, 119]), cached.render_many([0, 44, 119])
            assert all(torch.equal(actual[m], expected[m]) for m in actual)
            assert cached.renderer._static_coverage.builds == int(all_static)
