"""Replay from recorded checkpoints after validated numeric RNG roundtrips."""

import copy
import json
from pathlib import Path

import numpy as np
from gates import compare, digest, plain, run_reset
from ownership import OwnedCube, OwnedScene, restore
from rng_codec import decode_mt, decode_pcg, decode_seed, encode_seed, numeric_roundtrip

OUT = Path(__file__).resolve().parent


def decode(value):
    if isinstance(value, dict) and set(value) == {"dtype", "shape", "data"}:
        return np.array(value["data"], dtype=value["dtype"]).reshape(value["shape"])
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def main():
    assert (OUT / "gates.log").read_text().splitlines()[-1] == "OGB_RNG_UNPATCHED_REPLAY_AND_THREADED_ISOLATION_PASS"
    recorded = decode(json.loads((OUT / "replay_results.json").read_text()))
    results = {}
    for family, cls in [("Cube", OwnedCube), ("Scene", OwnedScene)]:
        records = []
        for row in recorded["families"][family]["replay"]:
            if row["options"] is not None:
                continue
            original = row["checkpoint"]
            converted, numeric = numeric_roundtrip(original)
            assert not compare(original, converted)
            env = cls(entropy_seed=991)
            try:
                env.reset(seed=991, options={"variation": [], "task_id": 4})
                restore(env, converted)
                actual = run_reset(env, row["options"])
                diff = compare(actual, row["result"])
                assert not diff, diff
                restore(env, converted)
                seeded_first = run_reset(env, row["options"], seed=row["seed"])
                restore(env, converted)
                seeded_second = run_reset(env, row["options"], seed=row["seed"])
                assert not compare(seeded_first, seeded_second)
                records.append(
                    {
                        "seed": row["seed"],
                        "numeric_rng_leaves": numeric,
                        "replayed_sha256": digest(actual),
                        "expected_sha256": digest(row["result"]),
                        "diff": diff,
                        "seeded_replay_sha256": digest(seeded_first),
                    }
                )
            finally:
                env.close()
        results[family] = records
    bad_cases = []
    valid = results["Cube"][0]["numeric_rng_leaves"]
    for label, leaves, decoder, key, value in [
        ("pcg wrong dtype", valid["environment"], decode_pcg, "state", np.zeros(16, np.int64)),
        ("pcg wrong shape", valid["environment"], decode_pcg, "state", np.zeros(15, np.uint8)),
        ("pcg flag", valid["environment"], decode_pcg, "has_uint32", np.array(2, np.uint8)),
        ("pcg increment", valid["environment"], decode_pcg, "increment", np.zeros(16, np.uint8)),
        ("mt wrong length", valid["task"], decode_mt, "key", np.zeros(623, np.uint32)),
        ("mt position", valid["task"], decode_mt, "position", np.array(625, np.int64)),
        ("mt flag", valid["task"], decode_mt, "has_gauss", np.array(2, np.uint8)),
        ("mt nonfinite cache", valid["task"], decode_mt, "cached_gaussian", np.array(np.nan, np.float64)),
    ]:
        candidate = copy.deepcopy(leaves)
        candidate[key] = value
        try:
            decoder(candidate)
        except ValueError:
            bad_cases.append(label)
        else:
            raise AssertionError(label)
    for seed_value in [None, -1, 0, 17, 2**128 - 1, 2**256 + 17]:
        assert decode_seed(encode_seed(seed_value)) == seed_value
    results["seed_metadata_roundtrips"] = [None, -1, 0, 17, str(2**128 - 1), str(2**256 + 17)]
    results["malformed_rejections"] = bad_cases
    (OUT / "numeric_rng_results.json").write_text(json.dumps(plain(results), indent=2) + "\n")
    print("OGB_NUMERIC_RNG_REPLAY_PASS six_resets=exact malformed_rejections=8")


if __name__ == "__main__":
    main()
