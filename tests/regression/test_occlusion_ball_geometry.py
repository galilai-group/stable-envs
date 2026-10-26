"""Geometry fidelity to Blender, independently of RGB materials and lighting."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

import stable_envs as se
from stable_envs.testing.geometry import depth_mse, mask_iou
from tests.helpers.reference.occlusion_ball import (
    DEPTH_MSE_MAX,
    MASK_IOU_MIN,
    load_oracle,
)
from tests.helpers.reference.semantics import assert_same_state


@pytest.mark.parametrize("backend", ["opengl", "raycast"])
def test_source_depth_masks_and_occlusion_boundaries(record_property, backend, device="cpu"):
    # Preserve the original native/analytic certification scope; expanded source
    # coverage belongs to the shared recording renderer, not this legacy model.
    _, oracle = load_oracle(legacy=backend == "opengl")
    with se.MuJoCoOcclusionBall(size=180, backend=backend, device=device) as env:
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
            ids = geometry["object_ids"][0].cpu().numpy()
            source_ids, source_depth = oracle["object_ids"][index], oracle["depth"][index]
            roi = ((source_ids >= 4) | (ids >= 4)) & (source_depth > 0)
            mse = float(depth_mse(geometry["depth"][0].cpu().numpy(), source_depth, roi))
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
        assert hidden == (5 if backend == "opengl" else 26)
        record_property("depth_mse_max", worst_mse)
        record_property("visible_mask_iou_min", minimum_iou)
        record_property("both_hidden_frames", hidden)
