"""``stable_envs/render/prng.py`` against golden vectors frozen from a real jax.

The repo ships no jax and no Oscar interpreter has one, so a test that imported jax would skip
everywhere and assert nothing. ``tests/data/prng_threefry_golden.json`` is therefore frozen by
``scratch/step7/craftax-classic-full/gen_prng_golden.py`` against jax 0.11.1 with
``jax_threefry_partitionable = True``, and this module asserts against the file.

What is being asserted is **bit** agreement, not closeness. ``jax.random.uniform`` builds its
floats by randomising the mantissa of an exponent-1 float, so every value is exactly representable
and a one-bit difference in the hash shows up as a completely different number rather than as a
small error. There is no tolerance to choose.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest
import torch

from stable_envs.render import prng

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "prng_threefry_golden.json"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.is_file():
        pytest.skip(f"{GOLDEN} is not present in this checkout")
    return json.loads(GOLDEN.read_text())


def _keys(rows):
    return torch.tensor([[np.int64(k).astype(np.int64) for k in r["key"]] for r in rows],
                        dtype=torch.int64).to(torch.int32)


def test_the_golden_file_says_which_jax_path_it_froze(golden):
    """A golden file that does not say which of jax's two code paths it came from is worthless.

    The partitionable and pre-0.4.36 paths give different bits for the same key, so a vector frozen
    without recording the flag cannot be checked against anything later.
    """
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
        assert np.array_equal(np.ascontiguousarray(got).view(np.uint32), want), (
            f"key={r['key']} shape={r['shape']}")


def test_uniform_matches_jax_on_the_render_sized_fields(golden):
    """The shapes Craftax Classic (49x63, 112x144) and full Craftax (63x77, 90x110) actually draw."""
    for r in golden["digest"]:
        got = prng.uniform(_keys([r]), tuple(r["shape"]))[0].numpy()
        b = np.ascontiguousarray(got).view(np.uint32).ravel()
        assert hashlib.sha256(b.tobytes()).hexdigest() == r["sha256"], (
            f"key={r['key']} shape={r['shape']}")
        assert [int(v) for v in b[:8]] == r["first8"]
        assert [int(v) for v in b[-8:]] == r["last8"]


def test_the_scaled_night_field_matches(golden):
    """``uniform(key, shape) * 95 + 32`` — the expression the renderer evaluates, not a component."""
    for r in golden["scaled"]:
        got = prng.uniform(_keys([r]), tuple(r["shape"]))[0] * 95 + 32
        b = np.ascontiguousarray(got.numpy()).view(np.uint32).ravel()
        assert hashlib.sha256(b.tobytes()).hexdigest() == r["sha256"], (
            f"key={r['key']} shape={r['shape']}")


def test_a_batch_of_keys_is_the_same_as_the_keys_one_at_a_time(golden):
    """Batching must not change a row. This is the batch-invariance rule applied to the PRNG."""
    rows = [r for r in golden["digest"] if tuple(r["shape"]) == (49, 63)]
    keys = _keys(rows)
    batched = prng.uniform(keys, (49, 63))
    for i, r in enumerate(rows):
        one = prng.uniform(_keys([r]), (49, 63))[0]
        assert torch.equal(batched[i].view(torch.int32), one.view(torch.int32)), r["key"]


def test_permuting_the_batch_permutes_the_output_and_nothing_else(golden):
    rows = [r for r in golden["digest"] if tuple(r["shape"]) == (3, 5) or True][:12]
    keys = _keys(rows)
    a = prng.uniform(keys, (8, 8))
    perm = torch.tensor([7, 0, 3, 11, 2, 9, 1, 5, 4, 10, 6, 8])[: keys.shape[0]]
    b = prng.uniform(keys[perm], (8, 8))
    assert torch.equal(a[perm].view(torch.int32), b.view(torch.int32))


def test_counter_offset_chunks_a_field_without_changing_it(golden):
    """Drawing (49,63) in three chunks must rebuild the field jax drew in one call."""
    rows = [r for r in golden["digest"] if tuple(r["shape"]) == (49, 63)][:4]
    keys = _keys(rows)
    whole = prng.uniform(keys, (49 * 63,))
    parts = []
    off = 0
    for n in (1000, 2000, 49 * 63 - 3000):
        parts.append(prng.uniform(keys, (n,), offset=off))
        off += n
    assert torch.equal(torch.cat(parts, dim=1).view(torch.int32), whole.view(torch.int32))


def test_random_bits_is_what_uniform_is_built_from(golden):
    keys = _keys(golden["digest"][:4])
    bits = prng.random_bits(keys, 64)
    u = prng.uniform(keys, (64,))
    rebuilt = (((bits >> 9) & 0x7FFFFF) | 0x3F800000).view(torch.float32) - 1.0
    assert torch.equal(u.view(torch.int32), rebuilt.view(torch.int32))


def test_key_halves_round_trip_exactly_through_float32():
    """The reason an env carries FOUR numbers for a key and not two.

    A uint32 does not survive a float32 state row; two 16-bit halves do. This asserts the property
    the state encoding depends on, on the worst cases and on a large random sample.
    """
    rng = np.random.default_rng(0)
    raw = np.concatenate([
        np.array([[0, 0], [0xFFFFFFFF, 0xFFFFFFFF], [0x80000000, 0x7FFFFFFF],
                  [0x00010000, 0x0000FFFF]], np.uint32),
        rng.integers(0, 2 ** 32, size=(20000, 2), dtype=np.uint64).astype(np.uint32)])
    keys = torch.tensor(raw.astype(np.int64), dtype=torch.int64).to(torch.int32)
    halves = prng.key_to_halves(keys)
    assert int(halves.max()) <= 0xFFFF and int(halves.min()) >= 0

    # the round trip that matters is through float32, which is what a state row is
    through_f32 = torch.as_tensor(halves.numpy().astype(np.float32).astype(np.int64))
    back = prng.key_from_halves(through_f32)
    assert torch.equal(back, keys)

    # and a bare uint32 through float32 does NOT round-trip, which is why the halves exist
    bad = raw.astype(np.float32).astype(np.uint64).astype(np.uint32)
    assert (bad == raw).mean() < 0.1, "if this passes, the four-slot key encoding is unnecessary"


def test_key_from_halves_tolerates_a_conformance_probe(golden):
    """``check_state_dims_matter`` perturbs a state dimension by +25; the decode must stay total."""
    keys = _keys(golden["digest"][:4])
    halves = prng.key_to_halves(keys).to(torch.float32)
    probed = (halves + 25.0).round().to(torch.int64) % 65536
    out = prng.key_from_halves(probed)
    assert out.dtype == torch.int32 and out.shape == keys.shape
    assert not torch.equal(out, keys), "a +25 probe on a key half must reach the output"


def test_rotation_mask_is_the_one_that_is_easy_to_get_wrong():
    """``_rotl32`` keeps ``d`` bits after a shift right by ``32-d``, not ``32-d`` bits.

    The wrong mask produces a hash that looks like noise and is wrong everywhere, which is exactly
    the failure that is invisible without a golden vector. Asserted directly against a reference
    rotation computed in python ints.
    """
    rng = np.random.default_rng(7)
    vals = np.concatenate([np.array([0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF], np.uint32),
                           rng.integers(0, 2 ** 32, size=512, dtype=np.uint64).astype(np.uint32)])
    x = torch.tensor(vals.astype(np.int64), dtype=torch.int64).to(torch.int32)
    for d in (13, 15, 26, 6, 17, 29, 16, 24):
        want = np.array([((int(v) << d) | (int(v) >> (32 - d))) & 0xFFFFFFFF for v in vals],
                        np.uint32)
        got = prng._rotl32(x, d).numpy().astype(np.int64) & 0xFFFFFFFF
        assert np.array_equal(got.astype(np.uint32), want), d


def test_uniform_is_in_range_and_is_not_degenerate(golden):
    keys = _keys(golden["digest"][:8])
    u = prng.uniform(keys, (64, 64))
    assert float(u.min()) >= 0.0 and float(u.max()) < 1.0
    assert 0.45 < float(u.mean()) < 0.55
    assert u.reshape(8, -1).std(dim=1).min() > 0.25


def test_uniform_takes_no_range_arguments(golden):
    """Scaling into a range is the CALLER's expression, on purpose.

    ``jax.random.uniform(key, shape, dtype, minval, maxval)`` computes
    ``floats*(maxval-minval)+minval``, and under ``jax.jit`` XLA contracts that into a fused
    multiply-add, which rounds once where a separate multiply and add round twice. Independent
    review measured **508 of 4096 values differing** for the span ``[32, 127)`` at shape
    ``(32, 127)`` — while a power-of-two span agrees, which is precisely the coincidence that lets
    such a bug survive a test. A range argument here would be bit-exact against one jax call and not
    another, depending on a fusion decision the caller cannot see, so there is none.

    What pins the real expression is ``test_the_scaled_night_field_matches``: both craftax renderers
    write ``uniform(key, shape) * 95 + 32`` themselves and un-jitted, and that is checked against
    jax-generated values.
    """
    import inspect
    p = inspect.signature(prng.uniform).parameters
    assert "minval" not in p and "maxval" not in p, (
        "prng.uniform grew a range argument; see this test's docstring for why it must not have one")
    assert [n for n in p] == ["keys", "shape", "offset"]


def test_render_module_exports_are_the_minimal_interface():
    """W2 consumes this module verbatim; keep the surface small and named."""
    assert set(n for n in dir(prng) if not n.startswith("_")) >= {
        "threefry2x32", "random_bits", "uniform", "key_from_halves", "key_to_halves"}


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (49, 63)])
def test_odd_element_counts_work(golden, shape):
    """The partitionable path has no pairing, so an odd element count is not a special case.

    Asserted because the pre-0.4.36 path DID pad to an even count, and a port that carried that
    padding over would be wrong only on odd shapes.
    """
    rows = [r for r in golden["full"] if tuple(r["shape"]) == shape]
    if not rows:
        pytest.skip(f"no golden vector at {shape}")
    for r in rows:
        got = prng.uniform(_keys([r]), shape)[0].numpy()
        want = np.asarray(r["uniform_bits"], np.uint32).reshape(shape)
        assert np.array_equal(np.ascontiguousarray(got).view(np.uint32), want)
