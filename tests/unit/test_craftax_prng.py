"""Bit-exact checks against frozen JAX partitionable-Threefry vectors."""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest
import torch

from stable_envs.envs.craftax import prng

GOLDEN = (
    pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "regression" / "fidelity" / "prng_threefry_golden.json"
)


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.is_file():
        pytest.skip(f"{GOLDEN} is not present in this checkout")
    return json.loads(GOLDEN.read_text())


def _keys(rows):
    return torch.tensor([r["key"] for r in rows], dtype=torch.int64).to(torch.int32)


def test_the_golden_file_says_which_jax_path_it_froze(golden):
    """Require the JAX version and PRNG configuration in the reference data."""
    assert golden["jax_threefry_partitionable"] is True, golden
    assert golden["jax_default_prng_impl"] == "threefry2x32", golden
    assert golden["jax"], golden


def test_uniform_matches_jax_bit_for_bit_on_small_shapes(golden):
    rows = golden["full"]
    assert len(rows) >= 50, "the golden file has shrunk; regenerate it"
    for r in rows:
        keys = _keys([r])
        got = prng.uniform(keys, tuple(r["shape"]))[0].numpy()
        want = np.asarray(r["uniform_bits"], np.uint32).reshape(tuple(r["shape"]))
        assert np.array_equal(np.ascontiguousarray(got).view(np.uint32), want), f"key={r['key']} shape={r['shape']}"


def test_uniform_matches_jax_on_the_render_sized_fields(golden):
    """Check the field sizes used by Classic and full Craftax."""
    for r in golden["digest"]:
        got = prng.uniform(_keys([r]), tuple(r["shape"]))[0].numpy()
        b = np.ascontiguousarray(got).view(np.uint32).ravel()
        assert hashlib.sha256(b.tobytes()).hexdigest() == r["sha256"], f"key={r['key']} shape={r['shape']}"
        assert [int(v) for v in b[:8]] == r["first8"]
        assert [int(v) for v in b[-8:]] == r["last8"]


def test_the_scaled_night_field_matches(golden):
    """Check the complete night-noise expression, including float32 scaling."""
    for r in golden["scaled"]:
        got = prng.uniform(_keys([r]), tuple(r["shape"]))[0] * 95 + 32
        b = np.ascontiguousarray(got.numpy()).view(np.uint32).ravel()
        assert hashlib.sha256(b.tobytes()).hexdigest() == r["sha256"], f"key={r['key']} shape={r['shape']}"


def test_a_batch_of_keys_is_the_same_as_the_keys_one_at_a_time(golden):
    """Each key produces the same field alone or in a batch."""
    rows = [r for r in golden["digest"] if tuple(r["shape"]) == (49, 63)]
    keys = _keys(rows)
    batched = prng.uniform(keys, (49, 63))
    for i, r in enumerate(rows):
        one = prng.uniform(_keys([r]), (49, 63))[0]
        assert torch.equal(batched[i].view(torch.int32), one.view(torch.int32)), r["key"]


def test_permuting_the_batch_permutes_the_output_and_nothing_else(golden):
    rows = golden["digest"][:12]
    keys = _keys(rows)
    a = prng.uniform(keys, (8, 8))
    perm = torch.tensor([7, 0, 3, 11, 2, 9, 1, 5, 4, 10, 6, 8])[: keys.shape[0]]
    b = prng.uniform(keys[perm], (8, 8))
    assert torch.equal(a[perm].view(torch.int32), b.view(torch.int32))


def test_counter_offset_chunks_a_field_without_changing_it(golden):
    """Chunks must reconstruct the same field as a single draw."""
    rows = [r for r in golden["digest"] if tuple(r["shape"]) == (49, 63)][:4]
    keys = _keys(rows)
    whole = prng.uniform(keys, (49 * 63,))
    parts = []
    off = 0
    for n in (1000, 2000, 49 * 63 - 3000):
        parts.append(prng.uniform(keys, (n,), offset=off))
        off += n
    assert torch.equal(torch.cat(parts, dim=1).view(torch.int32), whole.view(torch.int32))


def test_key_halves_round_trip_exactly_through_float32():
    """Key halves must survive float32 render projections without losing bits."""
    rng = np.random.default_rng(0)
    raw = np.concatenate(
        [
            np.array(
                [[0, 0], [0xFFFFFFFF, 0xFFFFFFFF], [0x80000000, 0x7FFFFFFF], [0x00010000, 0x0000FFFF]], np.uint32
            ),
            rng.integers(0, 2**32, size=(20000, 2), dtype=np.uint64).astype(np.uint32),
        ]
    )
    keys = torch.tensor(raw.astype(np.int64), dtype=torch.int64).to(torch.int32)
    halves = prng.key_to_halves(keys)
    assert int(halves.max()) <= 0xFFFF and int(halves.min()) >= 0

    # Render projections store key halves as float32.
    through_f32 = torch.as_tensor(halves.numpy().astype(np.float32).astype(np.int64))
    back = prng.key_from_halves(through_f32)
    assert torch.equal(back, keys)

    # Full uint32 words lose precision in float32.
    bad = raw.astype(np.float32).astype(np.uint64).astype(np.uint32)
    assert (bad == raw).mean() < 0.1, "if this passes, the four-slot key encoding is unnecessary"


def test_key_from_halves_tolerates_a_conformance_probe(golden):
    """Perturbing a key half must produce a valid, different key."""
    keys = _keys(golden["digest"][:4])
    halves = prng.key_to_halves(keys).to(torch.float32)
    probed = (halves + 25.0).round().to(torch.int64) % 65536
    out = prng.key_from_halves(probed)
    assert out.dtype == torch.int32 and out.shape == keys.shape
    assert not torch.equal(out, keys), "a +25 probe on a key half must reach the output"


def test_rotation_mask_is_the_one_that_is_easy_to_get_wrong():
    """Compare signed-tensor rotations with unsigned Python-integer arithmetic."""
    rng = np.random.default_rng(7)
    vals = np.concatenate(
        [
            np.array([0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF], np.uint32),
            rng.integers(0, 2**32, size=512, dtype=np.uint64).astype(np.uint32),
        ]
    )
    x = torch.tensor(vals.astype(np.int64), dtype=torch.int64).to(torch.int32)
    for d in (13, 15, 26, 6, 17, 29, 16, 24):
        want = np.array([((int(v) << d) | (int(v) >> (32 - d))) & 0xFFFFFFFF for v in vals], np.uint32)
        got = prng._rotl32(x, d).numpy().astype(np.int64) & 0xFFFFFFFF
        assert np.array_equal(got.astype(np.uint32), want), d


def test_uniform_is_in_range_and_is_not_degenerate(golden):
    keys = _keys(golden["digest"][:8])
    u = prng.uniform(keys, (64, 64))
    assert float(u.min()) >= 0.0 and float(u.max()) < 1.0
    assert 0.45 < float(u.mean()) < 0.55
    assert u.reshape(8, -1).std(dim=1).min() > 0.25


def test_uniform_takes_no_range_arguments():
    """Range scaling stays at the call site to preserve its float32 rounding."""
    import inspect

    p = inspect.signature(prng.uniform).parameters
    assert "minval" not in p and "maxval" not in p, "prng.uniform must not take a range argument"
    assert [n for n in p] == ["keys", "shape", "offset"]
