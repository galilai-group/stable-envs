"""Fetch smoke tests: the twelve registered cells on CPU, and the renderer base without a GPU.

Pixels from the mover pass need CUDA and are checked by the source-pixel contracts; here the CPU
half is exercised: simulation and restorable State through every cell, and the MuJoCo-baked static
plate through ``FetchBase`` with ``warp``/``mujoco_warp`` stubbed in a child interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium_robotics")

import stable_envs as snv

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCENES = {"Reach": 15, "Push": 22, "Slide": 22, "PickAndPlace": 22}
CELLS = [f"Fetch{scene}{suffix}" for scene in SCENES for suffix in ("", "Dense", "Dict")]


@pytest.mark.parametrize("cell", CELLS)
def test_cell_builds_resets_steps_and_restores_state(cell):
    scene = next(s for s in SCENES if cell.startswith(f"Fetch{s}"))
    nq = SCENES[scene]
    with snv.make(cell, num_envs=2, device="cpu", seed=3) as env:
        state = env.reset(seed=3)
        assert state["qpos"].shape == (2, nq) and state["goal"].shape == (2, 3)
        assert env.state_dim == nq + 3
        actions = np.random.default_rng(0).uniform(-1, 1, (2, 3, 4)).astype(np.float32)
        out = env.step(actions, frames=None)
        assert out["valid"].all()
        assert torch.isfinite(out["state"]["qpos"]).all()
        assert not torch.equal(out["state"]["qpos"], state["qpos"]), "the arm moved"
        env.set_state(state)
        again = env.step(actions, frames=None)
        assert torch.equal(again["state"]["qpos"], out["state"]["qpos"])


def test_dense_reward_differs_from_sparse_reward():
    actions = np.zeros((1, 2, 4), np.float32)
    rewards = {}
    for cell in ("FetchReach", "FetchReachDense"):
        with snv.make(cell, num_envs=1, device="cpu", seed=0) as env:
            env.reset(seed=0)
            rewards[cell] = env.step(actions, frames=None)["reward"]
    assert torch.isin(rewards["FetchReach"], torch.tensor([-1.0, 0.0])).all()
    assert not torch.isin(rewards["FetchReachDense"], torch.tensor([-1.0, 0.0])).all()


PROBE = r"""
import importlib.machinery as im
import json, sys, types
import numpy as np, torch
for nm in ("warp", "mujoco_warp"):
    m = types.ModuleType(nm); m.__spec__ = im.ModuleSpec(nm, None); sys.modules[nm] = m
import stable_envs as snv
import stable_envs.envs.fetch.envs as F

states_dir = sys.argv[1]
F._state_pool = lambda scene, ref_id, state_dim, nq, n=512: np.load(
    f"{states_dir}/Fetch{scene}.npz")["states"].astype(np.float32)

out = {}
for name in ("FetchReach", "FetchPush"):
    public = getattr(F, name)
    cls = type(name, (F.FetchBase,), dict(SCENE=public.SCENE, REF_ID=public.REF_ID))
    corpus = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)[:6]
    env = cls(num_envs=6, device="cpu")
    env._set_render_state(corpus)
    packed = env._get_render_state()
    plate = env._plate(64, None).cpu().numpy()
    varied = env._plate(64, {"floor.color": np.array([0.9, 0.1, 0.1])}).cpu().numpy()
    out[name] = {
        "roundtrip_exact": bool(np.array_equal(packed.cpu().numpy(), corpus)),
        "layout": {".".join(k): [w, o] for k, w, o in env._layout()},
        "state_names": list(env.state_names),
        "plate_shape": list(plate.shape),
        "plate_repeatable": bool(np.array_equal(plate, env._plate(64, None).cpu().numpy())),
        "floor_color_changes_plate": bool(not np.array_equal(plate, varied)),
        "movers": int(env._mover_lut().sum()),
        "robot_geoms": len(env._b.info["robot"]),
    }
    env.close()
    out[name]["closed_refuses_render"] = False
    try:
        env._render()
    except RuntimeError:
        out[name]["closed_refuses_render"] = True
print("@@JSON@@" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def renderer_base() -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("MUJOCO_GL", None)
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PROBE), str(ROOT / "tests/fixtures/states/fidelity")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )
    if r.returncode != 0:
        pytest.fail(f"renderer probe failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    line = next(ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("@@JSON@@"))
    return json.loads(line[len("@@JSON@@") :])


@pytest.mark.parametrize("name,nq", [("FetchReach", 15), ("FetchPush", 22)])
def test_renderer_state_is_qpos_plus_goal_and_roundtrips(renderer_base, name, nq):
    r = renderer_base[name]
    assert r["layout"] == {"qpos": [nq, 0], "goal": [3, nq]}
    assert r["state_names"][:3] == ["slide0", "slide1", "slide2"]
    assert r["state_names"][-3:] == ["goal_x", "goal_y", "goal_z"]
    assert r["roundtrip_exact"]


@pytest.mark.parametrize("name", ["FetchReach", "FetchPush"])
def test_static_plate_is_repeatable_follows_appearance_and_robot_is_a_mover(renderer_base, name):
    r = renderer_base[name]
    assert r["plate_shape"] == [64, 64, 3]
    assert r["plate_repeatable"] and r["floor_color_changes_plate"]
    assert r["movers"] >= r["robot_geoms"] > 0


@pytest.mark.parametrize("name", ["FetchReach", "FetchPush"])
def test_closed_renderer_refuses_to_render(renderer_base, name):
    assert renderer_base[name]["closed_refuses_render"]
