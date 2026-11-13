"""Rendering under DDP and on two devices in one process.

Under DDP, the union of what the ranks render must be bit-identical to one process rendering the same
states. On two devices in one process (``DataParallel``, manual multi-GPU), frames must match across
devices; module-level caches are keyed by scene content, not device.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import assert_frames_identical, make_or_skip, sampled_states

import stable_envs as se
from stable_envs.testing._data import render_states

#: One env per render backend (2-D rasterizer, 3-D ray caster), where a device-blind cache could live.
ENVS = ["PushT", "PointMazeMediumNavigate"]

#: Seed for the states both sides of each comparison draw, so they draw the same states.
STATE_SEED = 20260812


def _registered_or_skip(name: str) -> None:
    """Skip when ``name`` is not registered in this interpreter.

    ``ENVS`` is a fixed pick of backends, not a family sweep, so ``resolve_family`` does not apply.
    """
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter, so neither backend is reachable")


@pytest.mark.parametrize("name", ENVS)
def test_two_devices_in_one_process(name):
    """The same states on cuda:0 and cuda:1 give the same pixels.

    The ``render/raster2d.py`` static-prefix cache is not keyed by device, and the tile atlas is keyed
    by ``str(device)``, which is ``"cuda"`` whichever device is current.
    """
    _registered_or_skip(name)
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")

    env0 = make_or_skip(name, num_envs=8, device="cuda:0")
    states = sampled_states(env0, 8, STATE_SEED)
    f0 = env0.render(states).cpu()

    env1 = make_or_skip(name, num_envs=8, device="cuda:1")
    f1 = env1.render(states).cpu()

    assert_frames_identical(
        f0,
        f1,
        what=f"{name}: cuda:0 against cuda:1",
        why="check device-blind caches: the static-prefix cache is keyed by scene content and the "
        "tile atlas by the string 'cuda'",
    )


def _ddp_child(rank: int, world: int, name: str, n: int, q):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    from stable_envs.testing._data import StateDataset, collate_batched

    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=rank, world_size=world)

    proto = se.make(name, num_envs=1, device=f"cuda:{rank}")
    states = proto.sample_states(n, rng=np.random.default_rng(STATE_SEED))
    del proto

    ds = StateDataset(name, states, device=f"cuda:{rank}")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    # The dataset renders in ``__getitems__`` (hence ``collate_batched``) onto this rank's GPU, so
    # workers would only duplicate the CUDA context.
    loader = DataLoader(ds, batch_size=16, shuffle=False, sampler=sampler, num_workers=0, collate_fn=collate_batched)

    idxs, frames = list(iter(sampler)), []
    for batch in loader:
        frames.append(batch.cpu())
    q.put((rank, idxs, torch.cat(frames)))
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("name", ENVS)
def test_ddp_shards_and_matches_single_process(name):
    """Two ranks, disjoint shards, and the union is bit-identical to one process."""
    _registered_or_skip(name)
    import multiprocessing as mp

    n, world = 64, 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ddp_child, args=(r, world, name, n, q)) for r in range(world)]
    for p in procs:
        p.start()
    got = {}
    for _ in range(world):
        rank, idxs, frames = q.get(timeout=600)
        got[rank] = (idxs, frames)
    for p in procs:
        p.join(timeout=600)
        assert p.exitcode == 0, f"rank died with exit code {p.exitcode}"

    # 1. the shards partition the dataset
    all_idx = [i for r in range(world) for i in got[r][0]]
    assert sorted(all_idx) == list(range(n)), "DistributedSampler did not partition the states"
    assert len(set(got[0][0]) & set(got[1][0])) == 0, "ranks share states"

    # 2. the union is byte-identical to one process rendering the same seeded states
    proto = se.make(name, num_envs=1, device="cuda:0")
    states = proto.sample_states(n, rng=np.random.default_rng(STATE_SEED))
    del proto
    ref = render_states(name, states, device="cuda:0").cpu()

    for r in range(world):
        idxs, frames = got[r]
        assert frames.shape[0] == len(idxs)
        assert_frames_identical(
            frames,
            ref[idxs],
            what=f"{name}: rank {r} against a single process",
            why="module-level caches are per-rank, so a difference means a rank rendered from a "
            "different device, atlas or cached plate",
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs"]))
