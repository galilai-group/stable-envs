"""FrameLoader frames match ``env.render`` of each batch, for full States and compact rows.

The frames are byte-equal, except for renderers whose bytes depend on the batch size: their largest per-frame MSE
(0-255 scale) stays under ``MSE_BOUND``, far below the MSE between any renderer and its source. Each family runs
where its extra is installed; the CUDA cases need a CUDA device.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as snv

NAMES = ("PushT", "TwoRoom", "DMCReacher", "OGBCube")
DEVICES = ("cpu", pytest.param("cuda", marks=pytest.mark.gpu))
ROWS = 37
BATCHES = (1, 8, 16)
#: Largest per-frame MSE between a chunk and a per-batch render. MuJoCo Warp draws OGBCube with a few pixels
#: 1 level apart at another world count (measured on an RTX A5500: 94 pixels of 1 of 37 frames, MSE 0.0019).
MSE_BOUND = {("OGBCube", "cuda"): 0.005}


def _make(name: str, **kwargs):
    try:
        return snv.make(name, **kwargs)
    except ModuleNotFoundError as e:
        pytest.skip(str(e))


def _states(name: str) -> snv.State:
    """``ROWS`` full States along seeded random-action rollouts, on the CPU."""
    with _make(name, num_envs=ROWS, device="cpu") as env:
        env.reset(seed=3)
        rng = np.random.default_rng(3)
        for _ in range(6):
            out = env.step(env.action_spec.sample(ROWS, rng), frames=None)
        return out["state"].cpu()


def _rows(name: str):
    with _make(name, num_envs=1, device="cpu") as env:
        try:
            return env.sample_states(ROWS, rng=np.random.default_rng(4))
        except NotImplementedError:
            return None


def _check(name, device, data):
    data = data.to(device) if isinstance(data, snv.State) else torch.as_tensor(data, device=device)
    for b in BATCHES:
        with snv.make(name, num_envs=b, device=device) as env:
            loader = snv.FrameLoader(env, data, b)
            assert loader.chunk_size == ROWS
            bound = MSE_BOUND.get((name, device), 0.0)
            for got, start in zip(loader, range(0, ROWS, b), strict=True):
                want = env.render(data[start : start + b])
                assert got.device == want.device and got.dtype == torch.uint8 and got.shape == want.shape
                mse = (got.double() - want.double()).pow(2).flatten(1).mean(1).max().item()
                assert mse <= bound, f"{name} {device} batch {b} rows {start}: frame MSE {mse:.5f} > {bound}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_state_frames_equal_render_per_batch(name, device):
    if name not in snv.list_envs():
        pytest.skip(f"{name} is not installed")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    _check(name, device, _states(name))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_row_frames_equal_render_per_batch(name, device):
    if name not in snv.list_envs():
        pytest.skip(f"{name} is not installed")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    rows = _rows(name)
    if rows is None:
        pytest.skip(f"{name} has no sample_states")
    try:
        with snv.make(name, num_envs=1, device=device) as env:
            env.render(torch.as_tensor(rows[:1], device=device))
    except TypeError:
        pytest.skip(f"{name} does not draw rows on {device}")
    _check(name, device, rows)
