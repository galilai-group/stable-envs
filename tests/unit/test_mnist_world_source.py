"""MNISTWorld against episodes of the published Flow Equivariant World Models dataset.

The paper's generator is not published, so the reference is its output: two validation episodes per
configuration of ``flowm123/mnist-world``, frozen by ``scripts/testing/freeze_mnist_world_corpus.py``
into ``tests/fixtures/semantics/mnist_world``. Each episode is replayed with its recorded digits and
actions; every step must reproduce the recorded digit positions and camera exactly, and the frames
must match the decoded MP4 frames up to H.264 noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as snv

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "semantics" / "mnist_world"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
#: The dataset's "dynamic_fo" files are a world of 80 with 8 digits, not the preset's 32 and 3.
OPTIONS = {
    "dynamic_po": {"preset": "dynamic_po"},
    "static_po": {"preset": "static_po"},
    "dynamic_fo_no_sm": {"preset": "dynamic_fo_no_sm"},
    "dynamic_fo_ws80": {"preset": "dynamic_fo", "world_size": 80, "num_digits": 8},
}
#: Measured on these episodes: mean |difference| 0.27-0.53 / 255 and at least 99.95% of pixels within 8;
#: the rest are H.264 ringing at digit edges.
MEAN_ABS_MAX = 1.0
WITHIN8_MIN = 0.99


def _episodes(config):
    with np.load(ROOT / f"{config}.npz") as f:
        values = {key: f[key] for key in f.files}
    count = len({key.split("/")[0] for key in values})
    return [{key.split("/")[1]: v for key, v in values.items() if key.startswith(f"{e}/")} for e in range(count)]


def _start(env, episodes):
    """A State at the first frame of each episode, one episode per lane."""
    n = len(episodes)

    def stack(name, fn):
        return torch.as_tensor(np.stack([fn(e[name]) for e in episodes]), dtype=torch.int64)

    return TensorDict(
        {
            "digits": {
                "image": stack("mnist_index", lambda x: x),
                "label": stack("labels", lambda x: x),
                "position": stack("positions", lambda x: x[0].reshape(-1)),
                "velocity": stack("velocity", lambda x: x.reshape(-1)),
            },
            "camera": stack("camera", lambda x: x[0]),
            "step": torch.zeros(n, 1, dtype=torch.int64),
            "rng": {"seed": torch.zeros(n, 1, dtype=torch.int64), "episode": torch.zeros(n, 1, dtype=torch.int64)},
        },
        batch_size=[n],
    )


@pytest.mark.parametrize("config", sorted(OPTIONS))
def test_lanes_match_the_published_dataset(config):
    episodes = _episodes(config)
    steps = MANIFEST["frame_steps"]
    with snv.MNISTWorld(num_envs=len(episodes), device="cpu", **OPTIONS[config]) as env:
        env.set_state(_start(env, episodes))
        frames = {0: env.render()}
        actions = np.stack([e["actions"] for e in episodes], 1)  # T, N, 2
        assert (actions[0] == 0).all()  # the first frame has no action
        for t in range(1, len(actions)):
            state = env.step(actions[t], frames=None)["state"]
            for lane, e in enumerate(episodes):
                assert state["camera"][lane].tolist() == e["camera"][t].tolist(), (config, lane, t)
                assert state["digits", "position"][lane].tolist() == e["positions"][t].reshape(-1).tolist()
            if t in steps:
                frames[t] = env.render()
    ours = torch.stack([frames[t] for t in steps], 1)[..., 0].numpy().astype(float)  # N, S, V, V
    recorded = np.stack([e["frames"] for e in episodes]).astype(float)
    diff = np.abs(ours - recorded)
    assert diff.mean() <= MEAN_ABS_MAX, (config, diff.mean())
    assert (diff <= 8).mean() >= WITHIN8_MIN, (config, (diff <= 8).mean())


def test_the_corpus_is_the_one_the_manifest_names():
    import hashlib

    for config in OPTIONS:
        digest = hashlib.sha256((ROOT / f"{config}.npz").read_bytes()).hexdigest()
        assert digest == MANIFEST["splits"][config]["sha256"], config
