"""Reject damaged or misleading disk trajectories before simulator restoration."""

import json
import zipfile

import numpy as np
import pytest

from stable_envs.testing.corpus import load_trajectory, write_trajectory
from stable_envs.testing.evidence import sha256


def values():
    return {
        "actions": np.zeros((2, 3, 1), np.float32),
        "state/position": np.zeros((3, 3, 2), np.float64),
        "state/rng/state": np.full((3, 3, 4), 2**63 - 1, np.int64),
        "reward": np.zeros((2, 3), np.float64),
        "terminated": np.zeros((2, 3), bool),
        "truncated": np.zeros((2, 3), bool),
    }


def write(path, arrays=None):
    identity = dict(commit="test-fixture", files={"fixture": "test-hash"}, versions={"fixture": "1"})
    return write_trajectory(
        path,
        registration="fixture/v1",
        seed=17,
        source=identity,
        producer=identity,
        arrays=values() if arrays is None else arrays,
    )


def reseal(path):
    # A trusted seal cannot make structurally invalid content into a valid State.
    files = ("manifest.json", "trajectory.npz")
    (path / "SHA256SUMS").write_text("".join(f"{sha256(path / f)}  {f}\n" for f in files))


def test_owned_typed_state_and_write_once(tmp_path):
    path = tmp_path / "run"
    saved = write(path)
    state = saved.state(1)
    assert state.batch_size == (3,)
    assert state["rng", "state"][0, 0].item() == 2**63 - 1
    state["rng", "state"].zero_()
    assert saved.state(1)["rng", "state"][0, 0].item() == 2**63 - 1
    with pytest.raises(ValueError):
        saved.arrays["actions"][0] = 1
    with pytest.raises(FileExistsError):
        write(path)
    with pytest.raises(IndexError):
        saved.state(-1)
    with pytest.raises(ValueError, match="registration"):
        load_trajectory(path, registration="sibling/v1")


@pytest.mark.parametrize("defect", ["truncated", "object", "nonfinite", "overlap", "flags", "missing"])
def test_invalid_arrays_never_create_corpus(tmp_path, defect):
    arrays = values()
    if defect == "truncated":
        arrays["state/position"] = arrays["state/position"][:-1]
    elif defect == "object":
        arrays["state/position"] = arrays["state/position"].astype(object)
    elif defect == "nonfinite":
        arrays["state/position"][0, 0, 0] = np.nan
    elif defect == "overlap":
        arrays["state/rng"] = np.zeros((3, 3), np.int64)
    elif defect == "flags":
        arrays["terminated"] = arrays["terminated"].astype(np.int64)
    else:
        del arrays["truncated"]
    with pytest.raises(ValueError):
        write(tmp_path / "run", arrays)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("defect", ["hash", "schema", "duplicate", "identity", "missing"])
def test_loader_rejects_bad_artifacts(tmp_path, defect):
    path = tmp_path / "run"
    write(path)
    manifest = path / "manifest.json"
    if defect in {"hash", "schema", "identity"}:
        metadata = json.loads(manifest.read_text())
        if defect == "identity":
            metadata["source"] = {}
        else:
            metadata["arrays"]["state/rng/state"]["dtype"] = "float64"
        manifest.write_text(json.dumps(metadata))
        if defect != "hash":
            reseal(path)
    elif defect == "duplicate":
        with zipfile.ZipFile(path / "trajectory.npz", "a") as archive:
            data = archive.read("actions.npy")
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("actions.npy", data)
        reseal(path)
    else:
        (path / "trajectory.npz").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_trajectory(path, registration="fixture/v1")
