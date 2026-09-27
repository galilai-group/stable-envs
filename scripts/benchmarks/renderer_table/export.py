"""Independently recompute retained raw frames and export compact, immutable evidence.

Run on a compute node. Raw NPY images stay in scratch; retain State/action recipes,
all metrics/timings, hashes, and one worst witness per selected registration.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .common import METRICS, array_hash, load_state, quality, state_hash, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("selection", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--witnesses",
        nargs="*",
        default=[
            "PyFlytRocketLanding",
            "PointMazeLargeNavigate",
            "DMCFingerTurnHard",
            "OGBScene",
            "FetchPush",
            "CraftaxPixels",
        ],
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = json.loads(args.selection.read_text())
    manifest = {"files": {}, "raw_scratch_files": {}, "independent_frames_recomputed": 0, "registrations": {}}
    new_selection = {}

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest["files"][str(target.relative_to(args.output))] = digest(target)
        return str(target)

    for registration, chosen in selected.items():
        stem = registration.replace("/", "_")
        folder = args.output / stem
        folder.mkdir()
        new_selection[registration] = {}
        for kind, path in chosen.items():
            if path:
                new_selection[registration][kind] = copy(Path(path), folder / (kind + ".json"))
        if "probe" in chosen:
            continue
        cp, gp = Path(chosen["cpu"]), Path(chosen["gpu"])
        cpu, gpu = json.loads(cp.read_text()), json.loads(gp.read_text())
        assert cpu["status"] == gpu["status"] == "COMPLETE"
        assert gpu["corpus_result_sha256"] == digest(cp)
        recipes = {r["case"]: r for r in cpu.get("recipes", [])}
        for recipe in recipes.values():
            path = cp.parent / recipe["actions_file"]
            a = np.load(path, allow_pickle=False)
            assert array_hash(a) == recipe["actions_hash"]
            copy(path, folder / "corpus" / path.name)
        witness = None
        for f, g in zip(cpu["frames"], gpu["frames"], strict=True):
            assert f["index"] == g["index"] and f["state_hash"] == g["state_hash"]
            path = cp.parent / f["state_file"]
            assert state_hash(load_state(path)) == f["state_hash"]
            copy(path, folder / "corpus" / path.name)
            if "device_replay_proof" in g:
                proof = g["device_replay_proof"]
                path = gp.parent / proof["actual_gpu_state_file"]
                assert state_hash(load_state(path)) == proof["actual_gpu_state_hash"]
                copy(path, folder / "corpus" / path.name)
            actions = np.load(cp.parent / recipes[f["case"]]["actions_file"], allow_pickle=False)
            assert array_hash(actions[: f["step"]]) == f["prefix_hash"]
            op = cp.parent / f["oracle_file"]
            original = np.load(op, allow_pickle=False)
            assert hashlib.sha256(original.tobytes()).hexdigest() == f["oracle_sha256"]
            manifest["raw_scratch_files"][str(op)] = digest(op)
            for backend, result, directory in [("cpu", f, cp.parent), ("gpu", g, gp.parent)]:
                if backend + "_quality" not in result:
                    continue
                path = directory / f"frame-{f['index']:04d}-{backend}.npy"
                candidate = np.load(path, allow_pickle=False)
                got = quality(candidate, original)
                expected = result[backend + "_quality"]
                for key in METRICS:
                    assert np.isclose(got[key], expected[key], rtol=1e-12, atol=1e-12), (
                        registration,
                        f["index"],
                        backend,
                        key,
                    )
                for key in ["candidate_sha256", "oracle_sha256", "shape"]:
                    assert got[key] == expected[key]
                manifest["raw_scratch_files"][str(path)] = digest(path)
                manifest["independent_frames_recomputed"] += 1
                if cpu["name"] == "PyFlytRocketLanding" and f["index"] == 19 and backend == "gpu":
                    assert f["seed"] == 0 and f["step"] == 600 and f["action_dtype"] == "float32"
                    for suffix, image in [
                        ("candidate", candidate),
                        ("original", original),
                        (
                            "absolute-difference",
                            np.abs(candidate.astype(np.int16) - original.astype(np.int16)).astype(np.uint8),
                        ),
                    ]:
                        path = folder / ("retained-horizon600-" + suffix + ".png")
                        Image.fromarray(image).save(path)
                        manifest["files"][str(path.relative_to(args.output))] = digest(path)
                if witness is None or got["mse_normalized"] > witness[0]:
                    witness = (got["mse_normalized"], candidate, original, f["index"], backend)
        if witness and cpu["name"] in args.witnesses:
            error, candidate, original, index, backend = witness
            for suffix, image in [
                ("candidate", candidate),
                ("original", original),
                (
                    "absolute-difference",
                    np.abs(candidate.astype(np.int16) - original.astype(np.int16)).astype(np.uint8),
                ),
            ]:
                path = folder / ("worst-" + suffix + ".png")
                Image.fromarray(image).save(path)
                manifest["files"][str(path.relative_to(args.output))] = digest(path)
            write(
                folder / "witness.json",
                dict(
                    index=index,
                    backend=backend,
                    mse_normalized=error,
                    difference="absolute RGB difference, uint8, no amplification",
                ),
            )
            path = folder / "witness.json"
            manifest["files"][str(path.relative_to(args.output))] = digest(path)
        manifest["registrations"][registration] = dict(
            frames=len(cpu["frames"]), status="RAW_FRAMES_AND_STATE_RECIPES_VERIFIED"
        )
    write(args.output / "selection.json", new_selection)
    write(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
