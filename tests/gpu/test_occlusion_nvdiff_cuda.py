"""Engine-free rasterization must pass the unmodified source geometry gates."""

import subprocess
import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("nvdiff needs CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence.rendering.g34_nvdiff import G34Renderer
from stable_envs.envs.object_permanence.replay import G34Replay
from stable_envs.testing.geometry import depth_mse, mask_iou
from tests.helpers.reference.occlusion_ball import DEPTH_MSE_MAX, MASK_IOU_MIN, load_oracle


def test_replay_poses_match_all_120_independently_evaluated_source_frames():
    _, oracle = load_oracle()
    with G34Replay() as replay:
        rows = replay.poses(replay.state_at(torch.arange(120))).numpy()
    np.testing.assert_allclose(rows[:, :3], oracle["positions"], rtol=0, atol=3e-7)
    q = oracle["quaternions"]
    sign = np.where((q * rows[:, 3:]).sum(-1, keepdims=True) < 0, -1, 1)
    np.testing.assert_allclose(rows[:, 3:], q * sign, rtol=0, atol=1e-6)


@pytest.mark.parametrize("coverage,background", [("scaled", "refined"), ("scaled", "native"), ("refined", "refined")])
def test_nvdiff_frozen_blender_depth_and_masks(record_property, coverage, background):
    _, oracle = load_oracle()
    with G34Replay(size=180, coverage_mode=coverage, background_coverage=background) as replay:
        state = replay.state_at(torch.as_tensor(oracle["geometry_frames"].astype(np.int64) - 1))
        live = replay.get_state()
        ids = replay.render(state, mode="ids").cpu().numpy()
        depth = replay.render(state, mode="depth").cpu().numpy()
        worst, minimum, hidden = 0.0, 1.0, 0
        for i, frame in enumerate(oracle["geometry_frames"]):
            reference = oracle["object_ids"][i]
            roi = ((reference >= 4) | (ids[i] >= 4)) & (oracle["depth"][i] > 0)
            mse = float(depth_mse(depth[i], oracle["depth"][i], roi))
            assert mse <= DEPTH_MSE_MAX, (frame, mse)
            worst = max(worst, mse)
            for obj in (4, 5):
                iou = float(mask_iou(ids[i] == obj, reference == obj))
                if np.isnan(iou):
                    assert obj == 5 and not (ids[i] == obj).any()
                    hidden += 1
                else:
                    assert iou >= MASK_IOU_MIN, (frame, obj, iou)
                    minimum = min(minimum, iou)
        assert hidden == 5
        assert (live == replay.get_state()).all()
        record_property("depth_mse_max", worst)
        record_property("visible_mask_iou_min", minimum)
        record_property("both_hidden_frames", hidden)


def test_multi_outputs_share_dynamic_and_background_coverage(monkeypatch):
    with G34Replay(size=180) as replay:
        state = replay.state_at(torch.arange(129).remainder(120))
        # Construct without baking so both static and dynamic counts are visible.
        replay._renderer = G34Renderer(size=180)
        renderer = replay._renderer.renderer
        calls = []
        raster = renderer._rasterize

        def counted(*args):
            calls.append(1)
            return raster(*args)

        monkeypatch.setattr(renderer, "_rasterize", counted)
        outputs = replay.render_many(state)
        assert len(calls) == 3  # one static pass and two dynamic chunks
        assert set(replay._renderer._plate) == {"rgb", "depth", "ids"}
        for mode, image in outputs.items():
            assert torch.equal(image, replay.render(state, mode=mode))
        buffers = {m: torch.empty_like(v) for m, v in outputs.items()}
        assert replay.render_many(state, out=buffers) is buffers
        assert all(torch.equal(buffers[m], outputs[m]) for m in outputs)


@pytest.mark.parametrize("size", [90, 180])
def test_nvdiff_all_frames_batch_invariance_buffers_and_cursor(size):
    with G34Replay(size=size) as replay:
        states = replay.state_at(torch.arange(120))
        before = states.clone()
        live = replay.advance(44)
        for mode in ("rgb", "depth", "ids"):
            batch = replay.render(states, mode=mode)
            output = torch.empty_like(batch)
            assert replay.render(states, mode=mode, out=output) is output
            assert torch.equal(batch, output)
            for index in (0, 44, 59, 75, 119):
                assert torch.equal(replay.render(states[index : index + 1], mode=mode)[0], batch[index]), (mode, index)
            with pytest.raises(ValueError, match="out"):
                replay.render(states, mode=mode, out=output.cpu())
        assert (states == before).all() and (replay.get_state() == live).all()


def test_nvdiff_geometry_negative_control():
    _, oracle = load_oracle()
    with G34Replay() as replay:
        rows = replay.poses(replay.get_state())
    rows[0, 0] += 0.1
    with G34Renderer(size=180) as renderer:
        ids = renderer.render(rows, mode="ids")[0].cpu().numpy()
        assert mask_iou(ids == 5, oracle["object_ids"][0] == 5) < MASK_IOU_MIN
    depth = oracle["depth"][0].copy()
    roi = oracle["object_ids"][0] == 4
    depth[roi] += 0.01
    assert depth_mse(depth, oracle["depth"][0], roi) > DEPTH_MSE_MAX


@pytest.mark.parametrize("missing_triton", [False, True])
def test_cold_nvdiff_render_with_mujoco_and_blender_blocked(missing_triton):
    code = """
import sys
sys.modules['mujoco'] = None
sys.modules['bpy'] = None
from stable_envs.envs.object_permanence import G34Replay
with G34Replay(size=90) as replay:
    assert replay.render(replay.advance(44)).shape == (1, 90, 160, 3)
    assert replay.render(mode='ids').max() >= 4
assert sys.modules['mujoco'] is None and sys.modules['bpy'] is None
"""
    if missing_triton:
        code = "import sys\nsys.modules['triton'] = None\n" + code
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90)


def test_cached_modes_and_multi_chunk_partition_do_not_change_images_or_inputs():
    with G34Replay(size=180) as replay:
        state = replay.state_at(torch.arange(257).remainder(120))
        before = state.clone()
        first = replay.render(state)
        renderer = replay._renderer
        assert set(renderer._plate) == {"rgb"}
        assert renderer.patch == 12
        for mode in ("rgb", "depth", "ids"):
            batch = first if mode == "rgb" else replay.render(state, mode=mode)
            for start, stop in ((0, 1), (127, 130), (255, 257)):
                assert torch.equal(batch[start:stop], replay.render(state[start:stop], mode=mode))
        assert set(renderer._plate) == {"rgb", "depth", "ids"}
        assert (state == before).all()
        plate = renderer._plate["rgb"]
        output = torch.full_like(first[:1], 19)
        invalid = replay.poses(state[:1])
        invalid[:, 2] = 20
        with pytest.raises(ValueError, match="bounds"):
            renderer.render(invalid, out=output)
        assert (output == 19).all() and renderer._plate["rgb"] is plate


def test_supplied_trajectory_can_choose_a_larger_fixed_window_without_changing_state():
    with G34Replay() as nominal:
        rows = nominal.poses(nominal.state_at(torch.arange(120)))
    rows[:, 1] -= 2  # Nearer to the camera; nominal 12px window cannot fit.
    with G34Replay(rows, size=180) as small, G34Replay(rows, size=180, window_size=24) as large:
        state = small.state_at(torch.tensor([44]))
        with pytest.raises(ValueError, match="bounds"):
            small.render(state)
        image = large.render(state)
        assert large._renderer.patch == 24 and image.shape == (1, 180, 320, 3)
        assert (state == large.state_at(torch.tensor([44]))).all()
