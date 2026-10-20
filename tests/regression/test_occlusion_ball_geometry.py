"""Geometry fidelity to Blender, independently of RGB materials and lighting."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

import stable_envs as se
from stable_envs.testing.geometry import depth_mse, mask_iou
from tests.helpers.reference.occlusion_ball import DEPTH_MSE_MAX, MASK_IOU_MIN, load_oracle
from tests.helpers.reference.semantics import assert_same_state


@pytest.mark.parametrize("backend", ["opengl", "raycast"])
def test_source_depth_masks_and_occlusion_boundaries(record_property, backend):
    _, oracle = load_oracle()
    with se.MuJoCoOcclusionBall(size=180, backend=backend) as env:
        initial = env.reset()
        states = env.step(np.zeros((1, 119, 1)), frames=None)["states"]
        live = env.get_state()
        worst_mse, minimum_iou, hidden = 0.0, 1.0, 0
        for index, frame in enumerate(oracle["geometry_frames"]):
            state = initial if frame == 1 else states[:, int(frame) - 2]
            held = state.clone()
            full = env.render(state)
            assert torch.equal(full, env._render(env._pack_render_state(state)))
            geometry = env.geometry(state)
            assert_same_state(env.get_state(), live)
            assert_same_state(state, held)
            ids = geometry["object_ids"][0].numpy()
            source_ids, source_depth = oracle["object_ids"][index], oracle["depth"][index]
            roi = ((source_ids >= 4) | (ids >= 4)) & (source_depth > 0)
            mse = float(depth_mse(geometry["depth"][0].numpy(), source_depth, roi))
            assert mse <= DEPTH_MSE_MAX, (int(frame), mse)
            worst_mse = max(worst_mse, mse)
            for object_id in (4, 5):
                iou = float(mask_iou(ids == object_id, source_ids == object_id))
                if np.isnan(iou):
                    assert object_id == 5 and not (ids == 5).any()
                    hidden += 1
                else:
                    assert iou >= MASK_IOU_MIN, (int(frame), object_id, iou)
                    minimum_iou = min(minimum_iou, iou)
        assert hidden == 5
        record_property("depth_mse_max", worst_mse)
        record_property("visible_mask_iou_min", minimum_iou)
        record_property("both_hidden_frames", hidden)


@pytest.mark.parametrize("backend", ["opengl", "raycast"])
def test_geometry_gate_detects_shifted_motion_and_wrong_depth(backend):
    _, oracle = load_oracle()
    with se.MuJoCoOcclusionBall(size=180, backend=backend) as env:
        state = env.reset()
        offset = env._simulator.pose_offset
        state["physics", "integration"][:, offset] += 0.1
        shifted = env.geometry(state)["object_ids"][0].numpy()
        assert mask_iou(shifted == 5, oracle["object_ids"][0] == 5) < MASK_IOU_MIN
        depth = oracle["depth"][0].copy()
        roi = oracle["object_ids"][0] == 4
        depth[roi] += 0.01
        assert depth_mse(depth, oracle["depth"][0], roi) > DEPTH_MSE_MAX


@pytest.mark.parametrize("backend", ["opengl", "raycast"])
def test_render_buffers_time_axes_modes_and_later_transitions(backend):
    with se.MuJoCoOcclusionBall(num_envs=2, backend=backend) as env, se.MuJoCoOcclusionBall(num_envs=2) as control:
        state = env.reset()
        expected = env.render(state)
        output = torch.empty_like(expected)
        assert env.render(state, out=output, variation={}) is output
        assert torch.equal(output, expected)
        rows = env._pack_render_state(state).reshape(1, 2, 7)
        sequence_output = torch.empty(1, 2, 90, 160, 3, dtype=torch.uint8)
        assert env._render(rows, out=sequence_output) is sequence_output
        assert torch.equal(sequence_output[0], expected)
        geometry = env.geometry(state)
        assert geometry["depth"].dtype == torch.float32 and geometry["object_ids"].dtype == torch.int16
        assert torch.equal(env.render(state), expected)
        assert_same_state(env.step(np.zeros((2, 1)), frames=None), control.step(np.zeros((2, 1)), frames=None))
        for options in ({"variation": {"unknown": 1}}, {"size": 84}, {"supersample": 2}):
            with pytest.raises(ValueError):
                env.render(state, **options)


@pytest.mark.parametrize("size", [90, 180])
def test_all_frames_mixed_batches_and_warm_cache_do_not_call_native_draw(size, monkeypatch):
    from stable_envs.envs.object_permanence.rendering.g34_native import _bundle

    with se.MuJoCoOcclusionBall(size=size, backend="raycast") as fast:
        initial = fast.reset()
        rollout = fast.step(np.zeros((1, 119, 1)), frames=None)["states"]
        states = torch.cat([initial, rollout.squeeze(0)], dim=0)
        saved = states.clone()
        live = fast.get_state()
        actual_geometry = fast.geometry(states)
        image = fast.render(states)
        for frame in (0, 22, 44, 59, 97, 98, 104, 119):
            assert torch.equal(fast.render(states[frame : frame + 1])[0], image[frame])
            single = fast.geometry(states[frame : frame + 1])
            for key in ("depth", "object_ids"):
                assert torch.equal(single[key][0], actual_geometry[key][frame])
        renderer = _bundle(size, size * 16 // 9)[2]

        def forbidden_draw(*args, **kwargs):
            raise AssertionError("A warm raycast render must not call native OpenGL")

        monkeypatch.setattr(renderer, "render", forbidden_draw)
        assert torch.equal(fast.render(states), image)
        assert torch.equal(fast.geometry(states)["object_ids"], actual_geometry["object_ids"])
        image.zero_()  # outputs must not alias the shared background plate
        assert fast.render(states).any()
        assert_same_state(states, saved)
        assert_same_state(fast.get_state(), live)
