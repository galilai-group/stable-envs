"""Contract for the public stable-worldmodel interoperability facade."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import stable_envs as snv
from stable_envs import interop, registry

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
FROZEN_MATRIX = ROOT / "tests" / "fixtures" / "regression" / "gate_f_matrix.json"


def _frozen_rows() -> list[dict]:
    rows = json.loads(FROZEN_MATRIX.read_text())["rows"]
    assert len(rows) == 42
    return rows


def test_public_facade_keeps_optional_adapter_implementation_lazy():
    code = """
import sys
import stable_envs.interop as interop
assert len(interop.SWM_IDS) == 42
assert "stable_envs._interop.worldmodel" not in sys.modules
assert "stable_worldmodel" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_all_42_frozen_swm_ids_resolve_to_the_same_canonical_classes():
    expected = {row["swm_id"]: tuple(row["stable_envs_cells"]) for row in _frozen_rows()}
    assert interop.SWM_IDS is registry.SWM_IDS
    assert interop.SWM_IDS == expected
    assert len(expected) == 42

    for swm_id, class_names in expected.items():
        assert interop.canonical_names(swm_id) == class_names
        for class_name in class_names:
            cls = registry._class_for_frozen_cell(class_name)
            if cls is not None:
                assert getattr(snv, cls.__name__) is cls
                if class_name != cls.__name__:
                    with pytest.raises(KeyError):
                        registry._resolve(class_name, "test")
        if len(class_names) == 1:
            live = registry._class_for_frozen_cell(class_names[0])
            if live is not None:
                assert snv.info(swm_id)["class"] is live

    assert interop.SYMBOLIC_IDS == (
        "swm/CraftaxClassicSymbolic-v1",
        "swm/CraftaxSymbolic-v1",
    )


def test_variation_map_bridge_is_public():
    env = snv.PushT(num_envs=1, device="cpu", size=32)
    try:
        mapping = interop.variation_map(env)
        assert isinstance(mapping, interop.VariationMap)
        assert mapping.swm_id == "swm/PushT-v1"
        style = env.default_variation()
        variation = interop.variation_from_style(env, style)
        restored = interop.style_from_variation(env, variation)
        np.testing.assert_array_equal(restored, style)
    finally:
        env.close()


def test_swap_renderer_keeps_the_live_swm_pusht_and_replaces_only_pixels():
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_worldmodel")

    official = gymnasium.make("swm/PushT-v1", render_mode="rgb_array")
    renderer = None
    try:
        official.reset(seed=0)
        official_type = type(official.unwrapped)
        original_render = official.unwrapped.render
        returned = interop.swap_renderer(
            official,
            "PushT",
            interop.pusht_state,
            size=32,
            device="cpu",
        )
        renderer = official.unwrapped._stable_envs_renderer

        assert returned is official
        assert type(official.unwrapped) is official_type
        assert callable(official.unwrapped._stable_envs_original_render)
        assert official.unwrapped._stable_envs_original_render.__func__ is original_render.__func__
        frame = official.render()
        assert frame.shape == (32, 32, 3)
        assert frame.dtype == np.uint8

        action = official.action_space.sample()
        _, reward, terminated, truncated, _ = official.step(action)
        assert np.isscalar(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    finally:
        if renderer is not None:
            renderer.close()
        official.close()


def test_registered_gym_id_runs_through_an_actual_swm_world():
    swm = pytest.importorskip("stable_worldmodel")

    ids = interop.register_gym_ids(
        ["PushT"],
        prefix="stable_envs_step9",
        device="cpu",
        size=32,
    )
    env_id = "stable_envs_step9/PushT-v0"
    assert ids == {env_id: "PushT"}

    world = swm.World(
        env_id,
        num_envs=1,
        image_shape=(32, 32),
        max_episode_steps=4,
    )
    try:
        world.reset(seed=0)
        assert isinstance(world.envs.envs[0].unwrapped, interop.StableEnvGym)
        assert world.infos["pixels"].shape == (1, 1, 32, 32, 3)
        assert world.infos["pixels"].dtype == np.uint8
    finally:
        world.close()


def test_native_swm_world_accepts_stable_envs_when_upstream_exposes_env():
    swm = pytest.importorskip("stable_worldmodel")
    signature = inspect.signature(swm.World)
    if "env" not in signature.parameters:
        pytest.skip(
            "installed stable_worldmodel.World requires positional `env_name: str`; "
            "World.__init__ builds functools.partial(_make_env, env_name, ...) and "
            "_make_env always calls gym.make(env_name), so this version has no `env=` path"
        )

    env = snv.PushT(num_envs=1, device="cpu", size=32)
    world = None
    try:
        world = swm.World(
            env=env,
            num_envs=1,
            image_shape=(32, 32),
            max_episode_steps=4,
        )
        world.reset(seed=0)
        assert world.infos["pixels"].shape[-3:] == (32, 32, 3)
    finally:
        if world is not None:
            world.close()
        env.close()
