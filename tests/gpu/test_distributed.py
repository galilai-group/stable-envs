"""Does the state->frame path survive DDP, and does it survive two devices in ONE process?

Two different questions, and only the first one is the one people ask.

**DDP** is one process per GPU, so every module-level cache in this repository is per-rank by
construction and the dataset is an ordinary map-style dataset that ``DistributedSampler`` shards like
any image folder. The test asserts the property that actually matters and is easy to lose: the union of
what the ranks render is BIT-IDENTICAL to what one process renders from the same states. A renderer
that silently used a different device, a different atlas or a stale cached plate would still train, and
would still produce a loss curve, and would be wrong.

**Two devices in one process** is the case DDP does not cover and that ``DataParallel`` and any
manual multi-GPU script hit immediately. It is tested separately because the caches are keyed by scene
content, not by device.
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
from stable_envs._data import render_states

#: One env per RENDER BACKEND rather than per family: the 2-D rasterizer, the tile atlas and the 3-D
#: ray caster are the three places a device-blind cache can live, and every other registered env
#: renders through one of them.
ENVS = ["PushT", "MiniGridEmpty", "PointmazeMediumNavigate"]

#: The states both halves of this file draw. A CONSTANT, and not the env's own generator: the
#: cross-device and cross-process comparisons are only meaningful if both sides drew the same numbers,
#: and ``sample_states(n)`` with no generator consumes from ``env.rng``, so it depended on how many
#: draws each side had already made. The single-process reference below made none and the DDP child
#: made one, which is how a difference between two RANKS and a difference between two DRAWS became
#: indistinguishable.
STATE_SEED = 20260812


def _registered_or_skip(name: str) -> None:
    """A name this interpreter does not have is a fact about the interpreter, not a failure.

    Kept as a skip rather than a family-emptiness failure (see ``_shared.resolve_family``) because
    ``ENVS`` is three deliberately-chosen backends rather than a family sweep: if all three were
    missing the registry itself would be empty, which ``test_registry_is_not_empty`` already fails on.
    """
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter, so neither backend is reachable")


@pytest.mark.parametrize("name", ENVS)
def test_two_devices_in_one_process(name):
    """The same states on cuda:0 and cuda:1 must give the same pixels.

    The static-prefix cache in ``render/raster2d.py`` is keyed by (resolution, world, geometry blob)
    and NOT by device, and the tile atlas is keyed by ``str(device)`` -- which is the literal string
    ``"cuda"`` on every rank, while ``torch.device("cuda")`` resolves to whichever device is current.
    Either one can hand back a tensor that lives on the wrong GPU.
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
        why="a device-blind cache is the first place to look — the static-prefix cache is keyed by "
        "scene content and the tile atlas by the STRING 'cuda', which is the same on every rank "
        "while the device it resolves to is not",
    )


def _ddp_child(rank: int, world: int, name: str, n: int, q):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    from stable_envs._data import StateDataset, collate_batched

    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=rank, world_size=world)

    proto = se.make(name, num_envs=1, device=f"cuda:{rank}")
    states = proto.sample_states(n, rng=np.random.default_rng(STATE_SEED))
    del proto

    ds = StateDataset(name, states, device=f"cuda:{rank}")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    # The caller's own DataLoader (``ds.loader()`` is gone). ``collate_batched`` because this dataset
    # renders in ``__getitems__``; ``num_workers=0`` because those frames are already on this rank's
    # GPU and a worker would only duplicate the CUDA context to ship them back.
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

    # 2. the union is what one process would have rendered, to the byte. Drawn from the same explicit
    #    seed the children used, so "the ranks disagree with one process" cannot be "the two sides
    #    drew different states".
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
            why="every module-level cache in this repository is per-rank by construction, so a "
            "difference means a rank rendered from a different device, atlas or cached plate — "
            "which would still train and would still produce a loss curve",
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs"]))
