"""Craftax CPU lanes split over XLA host devices give the same results as lanes on one device."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("craftax")

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

# Seven lanes over four devices: two lanes per device and one padding row.
ROLLOUT = """
import sys
import numpy as np
import torch
import stable_envs as se
import jax

name, out = sys.argv[1], sys.argv[2]
saved = {"devices": np.array(len(jax.devices("cpu")))}
with se.make(name, num_envs=7, device="cpu") as env:
    first = env.reset(seed=11)
    g = torch.Generator().manual_seed(3)
    acts = torch.randint(0, env.action_high + 1, (7, 12, 1), generator=g)
    result = env.step(acts, states=slice(None), frames=slice(None))
    for key in ("reward", "terminated", "valid", "frames"):
        saved[key] = result[key].numpy()
    for key, value in result["states"].items(include_nested=True, leaves_only=True):
        saved["states/" + "/".join(key)] = value.numpy()
    first["engine", "player_health"][2] = 0
    env.set_state(first)
    masked = env.step(acts[:, :4], states=None, frames=slice(None))
    saved["masked_frames"] = masked["frames"].numpy()
    saved["masked_valid"] = masked["valid"].numpy()
    saved["render"] = env.render(result["states"][:, 5]).numpy()
    for key, value in env.get_state().items(include_nested=True, leaves_only=True):
        saved["live/" + "/".join(key)] = value.numpy()
np.savez(out, **saved)
"""


def _rollout(tmp_path, name, devices: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "JAX_NUM_CPU_DEVICES")}
    env["JAX_PLATFORMS"] = "cpu"
    if devices is not None:
        env["JAX_NUM_CPU_DEVICES"] = devices
    out = tmp_path / f"{name}-{devices}.npz"
    subprocess.run([sys.executable, "-c", ROLLOUT, name, str(out)], check=True, cwd=ROOT, env=env, timeout=900)
    return dict(np.load(out))


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxPixels"])
def test_lanes_split_over_host_devices_equal_lanes_on_one_device(tmp_path, name):
    one = _rollout(tmp_path, name, "1")
    split = _rollout(tmp_path, name, "4")
    assert int(one.pop("devices")) == 1 and int(split.pop("devices")) == 4
    assert one.keys() == split.keys()
    for key in one:
        np.testing.assert_array_equal(split[key], one[key], err_msg=key)


CHECK = """
import json
import os
import jax
from stable_envs._core.cpus import cpu_limit
if os.environ.get("START_JAX"):
    jax.numpy.zeros(1).block_until_ready()
import stable_envs as se
with se.make("CraftaxClassicSymbolic", num_envs=2, device="cpu") as env:
    env.reset(seed=0)
print(json.dumps({"devices": len(jax.devices("cpu")), "limit": cpu_limit()}))
"""


def _devices(env_overrides: dict) -> tuple[dict, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "JAX_NUM_CPU_DEVICES")}
    env.update(JAX_PLATFORMS="cpu", STABLE_ENVS_VERBOSE="INFO", **env_overrides)
    run = subprocess.run(
        [sys.executable, "-c", CHECK], check=True, cwd=ROOT, env=env, timeout=600, capture_output=True, text=True
    )
    return json.loads(run.stdout.strip().splitlines()[-1]), run.stderr


def test_craftax_asks_for_one_host_device_per_usable_cpu():
    seen, _ = _devices({})
    assert seen["devices"] == seen["limit"]


def test_a_device_count_set_by_the_user_is_kept():
    seen, _ = _devices({"XLA_FLAGS": "--xla_force_host_platform_device_count=3"})
    assert seen["devices"] == 3


def test_jax_started_before_craftax_keeps_one_device_and_says_so():
    seen, err = _devices({"START_JAX": "1"})
    assert seen["devices"] == 1
    if seen["limit"] > 1:
        assert "JAX started with one CPU device" in err
