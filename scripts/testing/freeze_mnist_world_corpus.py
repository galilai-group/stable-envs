"""Freeze a MNIST World reference corpus from the Flow Equivariant World Models dataset.

The generator of the paper is not published; its output is, as ``flowm123/mnist-world`` on the
Hugging Face Hub. Each episode there is an MP4 of the agent's 32 x 32 view and a ``.pt`` file with the
actions, digit labels, initial positions, velocities, the digit positions at every step and the camera
centres. This script downloads shard ``0.tar`` of a validation split per configuration at a pinned
revision, and keeps, for the first ``EPISODES`` episodes:

- the recorded trajectory: actions, digit positions over time, camera centres;
- the MNIST training image of each digit, identified by matching the frames against MNIST
  (the dataset stores only labels);
- the decoded frames at ``FRAME_STEPS``.

The ``.pt`` files are read as raw int64 storages from their zip archive, without unpickling. Frames are
decoded with PyAV. Run it from the repository root with numpy and av installed:

    python -m scripts.testing.freeze_mnist_world_corpus [--source DIR]

``--source`` names a directory that already holds the shards as ``<config>/0.tar``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pickletools
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO = "flowm123/mnist-world"
REVISION = "436521f8930d63658553e593d70d8fdf61b26104"
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}
#: config -> (shard path in the dataset, world size, digits, self-motion bound, speed bound)
SPLITS = {
    "dynamic_po": ("dynamic_po/dynamic_validation_200/0.tar", 50, 5, 10, 1),
    "static_po": ("static_po/static_validation_200/0.tar", 50, 5, 10, 0),
    "dynamic_fo_no_sm": ("dynamic_fo_no_sm/dynamic_validation_smallworld_no_em_200/0.tar", 32, 3, 0, 1),
    # The files published as dynamic_fo are a world of 80 with 8 digits ("ws80"), not the paper's 32 and 3.
    "dynamic_fo_ws80": ("dynamic_fo/dynamic_validation_ws80_200/0.tar", 80, 8, 10, 1),
}
EPISODES = 2
FRAME_STEPS = (0, 1, 2, 49, 50, 51, 120, 199)
WINDOW = 32
DIGIT = 28
OUT = Path("tests/fixtures/semantics/mnist_world")


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path


def _mnist_train(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    def idx(name):
        path = _download(MNIST_URL + name, cache / name)
        if hashlib.md5(path.read_bytes()).hexdigest() != MNIST_MD5[name]:
            raise RuntimeError(f"{path} does not match its md5")
        b = gzip.decompress(path.read_bytes())
        dims = [int.from_bytes(b[4 + 4 * i : 8 + 4 * i], "big") for i in range(b[3])]
        return np.frombuffer(b, np.uint8, offset=4 + 4 * b[3]).reshape(dims)

    return idx("train-images-idx3-ubyte.gz"), idx("train-labels-idx1-ubyte.gz")


def _read_pt(data: bytes) -> dict[str, np.ndarray]:
    """The tensors of a flat ``{name: LongTensor}`` torch save, from the archive's raw storages."""
    z = zipfile.ZipFile(io.BytesIO(data))
    root = z.namelist()[0].split("/")[0]
    listing = io.StringIO()
    pickletools.dis(z.read(f"{root}/data.pkl"), out=listing)
    text = listing.getvalue()
    if set(re.findall(r"GLOBAL\s+'torch (\w+Storage)'", text)) != {"LongStorage"}:
        raise ValueError("expected only int64 tensors")
    names = [s for s in re.findall(r"BINUNICODE '([^']*)'", text) if not s.isdigit() and s not in ("cpu", "storage")]
    keys = sorted((n for n in z.namelist() if "/data/" in n), key=lambda s: int(s.rsplit("/", 1)[1]))
    return {name: np.frombuffer(z.read(key), "<i8") for name, key in zip(names, keys, strict=True)}


def _frames(data: bytes) -> np.ndarray:
    import av

    with av.open(io.BytesIO(data)) as container:
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    if not (frames[..., 0] == frames[..., 2]).all():
        raise ValueError("expected grey frames")
    return frames[..., 0]


def _local(pos, cam, world):
    """Per frame and digit, the digit-image row and column under each window pixel, or -1."""
    t, k = pos.shape[1], pos.shape[0]
    rows = (cam[:, 1, None] - WINDOW // 2 + np.arange(WINDOW)) % world  # T, V
    cols = (cam[:, 0, None] - WINDOW // 2 + np.arange(WINDOW)) % world
    u = (rows[None] - pos[..., 1, None]) % world  # K, T, V
    v = (cols[None] - pos[..., 0, None]) % world
    inside = (u[..., :, None] < DIGIT) & (v[..., None, :] < DIGIT)  # K, T, V, V
    uu = np.broadcast_to(u[..., :, None], inside.shape)
    vv = np.broadcast_to(v[..., None, :], inside.shape)
    assert inside.shape == (k, t, WINDOW, WINDOW)
    return inside, np.where(inside, uu, 0), np.where(inside, vv, 0)


def _identify(frames, pos, cam, labels, world, images, image_labels, rounds=4, sample=2048):
    """The MNIST training index of each digit, by coordinate descent on the error of the frames.

    A digit starts as the image of its label closest on the pixels where it is the only digit in view.
    Each round then picks, digit by digit, the image of its label that makes the frames closest with the
    other digits held, on up to ``sample`` pixels where the digit is in view. A digit that never reaches
    the view keeps the first image of its label; it does not change a frame.
    """
    inside, u, v = _local(pos, cam, world)
    alone = inside & (inside.sum(0) == 1)[None]
    flat = images.reshape(len(images), DIGIT * DIGIT)
    frames = frames.astype(np.int32)
    rng = np.random.default_rng(0)
    candidates = [np.nonzero(image_labels == label)[0] for label in labels]
    found = np.asarray([c[0] for c in candidates])
    for k in range(len(labels)):
        where = np.argwhere(alone[k])
        if len(where):
            t, i, j = where[rng.permutation(len(where))[:sample]].T
            pixels = flat[candidates[k][:, None], (u[k, t, i, j] * DIGIT + v[k, t, i, j])[None]]
            found[k] = candidates[k][np.argmin(np.abs(pixels.astype(np.int32) - frames[t, i, j]).sum(1))]
    # Each digit keeps one pixel sample, so its candidates' pixels are gathered once.
    samples = []
    for k in range(len(labels)):
        where = np.argwhere(inside[k])
        if not len(where):
            samples.append(None)
            continue
        t, i, j = where[rng.permutation(len(where))[:sample]].T
        uu, vv = u[:, t, i, j], v[:, t, i, j]  # K, P
        pixels = flat[candidates[k][:, None], (uu[k] * DIGIT + vv[k])[None]].astype(np.int32)  # C, P
        samples.append((inside[:, t, i, j], uu * DIGIT + vv, pixels, frames[t, i, j]))
    errors = [float("nan")] * len(labels)
    for _ in range(rounds):
        for k, drawn in enumerate(samples):
            if drawn is None:
                continue
            seen, index, pixels, target = drawn
            others = np.where(seen, flat[found[:, None], index], 0).astype(np.int32)
            others = others.sum(0) - others[k]
            err = np.abs(np.minimum(others[None] + pixels, 255) - target[None]).mean(1)
            found[k] = candidates[k][np.argmin(err)]
            errors[k] = float(err.min())
    return found, errors


def _render(pos, cam, sprites, world):
    inside, u, v = _local(pos, cam, world)
    layers = np.where(inside, sprites[np.arange(len(sprites))[:, None, None, None], u, v], 0)
    return np.minimum(layers.astype(np.int32).sum(0), 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="directory holding the shards as <config>/0.tar")
    parser.add_argument("--cache", type=Path, default=Path(".venvs/mnist-world-corpus"))
    args = parser.parse_args()
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"{OUT} exists; move it aside first")
    images, image_labels = _mnist_train(args.cache / "mnist")
    manifest = {
        "generator": "scripts/testing/freeze_mnist_world_corpus.py",
        "dataset": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "frame_steps": list(FRAME_STEPS),
        "splits": {},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for config, (shard, world, digits, motion, speed) in SPLITS.items():
        tar_path = (
            args.source / config / "0.tar"
            if args.source
            else _download(
                f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{shard}", args.cache / config / "0.tar"
            )
        )
        tar = tarfile.open(tar_path)
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        episodes = sorted(n[:-3] for n in members if n.endswith(".pt"))[:EPISODES]
        out: dict[str, np.ndarray] = {}
        report = []
        for e, name in enumerate(episodes):
            meta = _read_pt(tar.extractfile(members[name + ".pt"]).read())
            frames = _frames(tar.extractfile(members[name + ".mp4"]).read()).astype(float)
            steps = len(frames)
            actions = meta["actions"].reshape(steps, 2)
            labels = meta["digit_labels"]
            if "digit_positions_over_time" in meta:
                pos = meta["digit_positions_over_time"].reshape(digits, steps, 2)
                camera = np.concatenate([meta["camera_centers_input"], meta["camera_centers_target"]]).reshape(
                    steps, 2
                )
                velocity = meta["digit_velocities"].reshape(digits, 2)
            else:  # the static split stores where the digits stay, and no camera
                pos = np.repeat(meta["digit_positions"].reshape(digits, 1, 2), steps, 1)
                camera = np.cumsum(actions, 0) % world
                velocity = np.zeros((digits, 2), np.int64)
            assert len(labels) == digits and pos.max() < world and np.abs(actions).max() <= motion
            assert np.abs(velocity).max() <= speed
            assert (camera == np.cumsum(actions, 0) % world).all(), "camera is not the summed actions"
            assert ((pos[:, 1:] - pos[:, :-1] - velocity[:, None]) % world == 0).all(), (
                "digits do not move by velocity"
            )
            index, errors = _identify(frames, pos, camera, labels, world, images, image_labels)
            rendered = _render(pos, camera, images[index], world).astype(float)
            diff = np.abs(rendered - frames)
            report.append(
                {
                    "episode": name,
                    "match_mean_abs": [round(x, 3) for x in errors],
                    "frames_mean_abs": round(float(diff.mean()), 4),
                    "frames_max_abs": int(diff.max()),
                }
            )
            out[f"{e}/mnist_index"] = index
            out[f"{e}/labels"] = labels
            out[f"{e}/velocity"] = velocity
            out[f"{e}/actions"] = actions
            out[f"{e}/positions"] = pos.transpose(1, 0, 2)  # T, K, 2
            out[f"{e}/camera"] = camera
            out[f"{e}/frames"] = frames[list(FRAME_STEPS)].astype(np.uint8)
        path = OUT / f"{config}.npz"
        np.savez_compressed(path, **out)
        manifest["splits"][config] = {
            "shard": shard,
            "shard_sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
            "world_size": world,
            "num_digits": digits,
            "max_self_motion": motion,
            "max_speed": speed,
            "episodes": report,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(config, report, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
