"""Verify downloaded Oscar bytes and independently recompute every frame comparison."""

import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REMOTE = HERE / "remote"
EVIDENCE = REMOTE / "evidence"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = REMOTE / "evidence.tar.gz"
    expected = (REMOTE / "evidence.tar.gz.sha256").read_text().split()[0]
    assert sha(archive) == expected == "6f584dc2f29f592b75c77eaf2237e4879d7b07eb5ec5add517fd8a5147581cd4"
    remote_log = (HERE / "remote-status.log").read_text()
    assert expected in remote_log and sha(REMOTE / "slurm-6669909.log") in remote_log
    with tarfile.open(archive) as tar:
        tar.extractall(REMOTE, filter="data")
    hashes = {}
    for line in (EVIDENCE / "SHA256SUMS.evidence").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(EVIDENCE / name) == digest, name
        hashes[name] = digest
    assert (EVIDENCE / "terminal.txt").read_text().splitlines() == ["OGB_CUBE_SCENE_GPU_GATE_PASS"]
    families = {}
    for family, case_count, cpu_count in [("Cube", 13, 45), ("Scene", 22, 48)]:
        folder = EVIDENCE / family
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "PASS" and len(result["cases"]) == case_count
        assert json.loads((folder / "exits.json").read_text()) == {"cpu_exit": 0, "gpu_exit": 0}
        assert f"OGB_{family.upper()}_GPU_GATE_PASS" in (folder / "gpu.log").read_text().splitlines()
        tests = list(ET.parse(folder / "cpu.xml").getroot().iter("testcase"))
        assert len(tests) == cpu_count
        assert not any(t.find(n) is not None for t in tests for n in ["skipped", "failure", "error"])
        rows = []
        for i, entry in enumerate(result["cases"]):
            serial = f"{i:03d}-{entry['label']}"
            frames = {
                kind: np.load(folder / f"{serial}-{kind}.npy", allow_pickle=False)
                for kind in ["full", "render", "baseline", "source"]
            }
            assert all(frame.dtype == np.uint8 for frame in frames.values())
            assert frames["full"].shape == ((224, 224, 3) if family == "Cube" else (200, 200, 3))
            assert np.array_equal(frames["full"], frames["render"])
            assert np.array_equal(frames["full"], frames["baseline"])
            assert entry["comparison"] == {"exact": True, "different_channels": 0, "max_abs": 0}
            assert entry["nonmutation"] and entry["full_private_exact"]
            if entry["kind"] == "variation" and not entry["label"].startswith("size-"):
                assert entry["physics_unchanged"]
            residual = dict(
                source_exact_fraction=float(np.mean(np.all(frames["source"] == frames["baseline"], axis=-1))),
                source_max_abs=int(np.abs(frames["source"].astype(int) - frames["baseline"].astype(int)).max()),
            )
            assert residual == entry["source_visual_residual"]
            rows.append(dict(label=entry["label"], full_render_baseline_exact=True, source_visual_residual=residual))
        for key in ["batch_order_exact", "chunk_size_exact", "independent_environment_exact", "no_context_released"]:
            assert result[key]
        assert all(result["visual_changes"].values())
        assert result["context_lifetime"]["context_freed"] == 0
        assert not result["resources_after"]["release_failed"]
        families[family] = dict(
            status="GPU_EVIDENCE_VERIFIED",
            cpu_tests=cpu_count,
            frame_cases=case_count,
            rows=rows,
            resource_lifetime=result["context_lifetime"],
            resource_policy=result["resource_policy"],
            variations=result["visual_changes"],
        )
    report = dict(
        status="GPU_EVIDENCE_VERIFIED",
        job=6669909,
        remote_archive_sha256=expected,
        slurm_log_sha256=sha(REMOTE / "slurm-6669909.log"),
        verified_file_count=len(hashes),
        verified_hashes=hashes,
        families=families,
        semantic_execution="DELEGATED_CPU",
        source_mujoco_pixels_claim="Visual-tier residuals retained; exact comparison is candidate vs unchanged frozen renderer configured from the pinned SWM oracle.",
    )
    (HERE / "verified-gpu.json").write_text(json.dumps(report, indent=2) + "\n")
    print("OGB_REMOTE_GPU_EVIDENCE_VERIFIED", len(hashes), "files", "35 cases", "93 CPU test executions / 92 unique")


if __name__ == "__main__":
    main()
