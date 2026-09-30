"""Independently join immutable campaign images by exact full-State witness.

Run on a CPU compute node. Incomplete or mismatched inputs never become zero
errors, and a partial run never earns an acceptance label.
"""

import argparse
import csv
import hashlib
import html
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROLES = ("source_cpu", "source_gpu", "ours_cpu", "ours_gpu")
PAIRS = (
    ("ours_cpu", "source_cpu"),
    ("ours_gpu", "source_gpu"),
    ("ours_gpu", "source_cpu"),
    ("ours_cpu", "source_gpu"),
    ("source_cpu", "source_gpu"),
    ("ours_cpu", "ours_gpu"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, np.float64)
    if not len(a):
        return dict(n=0, mean=None, sd=None, max=None)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0, max=float(a.max()))


def read_role(path, role, env, required):
    if path is None:
        return dict(status="NOT_APPLICABLE_OR_UNAVAILABLE", frames={})
    path = Path(path)
    cp = path / "campaign.json"
    if not cp.exists():
        return dict(status="PENDING", frames={}, path=str(path))
    try:
        campaign = json.loads(cp.read_text())
    except json.JSONDecodeError:
        return dict(status="PENDING_METADATA_WRITE", frames={}, path=str(path))
    assert campaign["registration"] == env and campaign["role"] == role
    record = dict(
        status=campaign["status"],
        frames={},
        path=str(path),
        campaign_sha256=digest(cp),
        backend=campaign["backend"],
        profile=campaign["profile"],
        job_id=campaign["job_id"],
    )
    if campaign["status"] != "COMPLETE":
        record["reason"] = campaign.get("reason", campaign.get("traceback", ""))
        return record
    source = role.startswith("source")
    data = json.loads((path / "result.json").read_text()) if source else campaign
    if source:
        assert data["status"] == "COMPLETE"
        record["result_sha256"] = digest(path / "result.json")
        record["sampling_limitation"] = data.get("sampling_limitation")
        record["gl_vendor"] = campaign.get("gl_vendor")
        record["gl_renderer"] = campaign.get("gl_renderer")
    frames = data["frames"]
    if not frames:
        record["status"] = "NO_RGB"
        record["reason"] = data.get("unavailable", {})
        return record
    if len(frames) != required:
        record["status"] = "INCOMPLETE_FRAME_COUNT"
        return record
    times = []
    for frame in frames:
        idx = frame["index"]
        assert idx not in record["frames"]
        item = dict(frame)
        item["image_path"] = str(path / frame["oracle_file" if source else "image_file"])
        item["image_hash"] = frame["oracle_sha256" if source else "image_sha256"]
        samples = frame["original_timing" if source else "timing"]["samples_ms"]
        assert len(samples) == 1, "One observation per frame required for this campaign"
        times.extend(samples)
        record["frames"][idx] = item
    record["timing_ms"] = stats(times)
    record["unique_states"] = len({f["state_hash"] for f in frames})
    return record


def load_image(frame):
    return _load_image(frame["image_path"], frame["image_hash"])


@lru_cache(maxsize=4000)
def _load_image(path, expected_hash):
    a = np.load(path, allow_pickle=False)
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] == 3
    assert hashlib.sha256(a.tobytes()).hexdigest() == expected_hash
    return a


def compare(left, right):
    if left["status"] != "COMPLETE" or right["status"] != "COMPLETE":
        return dict(status="UNAVAILABLE", left=left["status"], right=right["status"], frames=[])
    lf, rf = left["frames"], right["frames"]
    assert lf.keys() == rf.keys()
    differing = [i for i in lf if any(lf[i][k] != rf[i][k] for k in ("state_hash", "case", "seed", "step"))]
    if differing:
        return dict(status="NONCOMPARABLE_STATE", mismatched_frames=len(differing), frames=[])
    rows = []
    for i in lf:
        a, b = load_image(lf[i]), load_image(rf[i])
        assert a.shape == b.shape
        mse = float(np.mean(((a.astype(np.float64) - b.astype(np.float64)) / 255.0) ** 2))
        rows.append(
            dict(
                index=i,
                case=lf[i]["case"],
                seed=lf[i]["seed"],
                step=lf[i]["step"],
                state_hash=lf[i]["state_hash"],
                mse=mse,
            )
        )
    values = stats([r["mse"] for r in rows])
    return dict(
        status="MEASURED",
        statistics=values,
        frames=rows,
        all_frames_le_001=values["max"] <= 0.001,
        all_frames_le_0005=values["max"] <= 0.0005,
        worst=max(rows, key=lambda r: r["mse"]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    assert len(manifest["registrations"]) == 42
    summaries, details, per_frame, gallery = [], [], [], []
    assert 0 <= args.shard < args.shards
    for index, (env, paths) in enumerate(manifest["registrations"].items()):
        if index % args.shards != args.shard:
            continue
        _load_image.cache_clear()
        roles = {r: read_role(paths.get(r), r, env, args.frames) for r in ROLES}
        row = dict(env=env, scope=paths.get("scope", "matched full-State snapshots"))
        detail = dict(env=env, roles={}, comparisons={})
        for r, v in roles.items():
            detail["roles"][r] = {k: x for k, x in v.items() if k != "frames"}
            row[r + "_status"] = v["status"]
            row[r + "_backend"] = v.get("backend", "")
            row[r + "_unique_states"] = v.get("unique_states")
            timing = v.get("timing_ms", {})
            for stat in ("n", "mean", "sd"):
                row[r + "_time_ms_" + stat] = timing.get(stat)
        for l, r in PAIRS:
            label = l + "_vs_" + r
            value = compare(roles[l], roles[r])
            detail["comparisons"][label] = {k: v for k, v in value.items() if k != "frames"}
            row[label + "_status"] = value["status"]
            for stat in ("n", "mean", "sd", "max"):
                row[label + "_mse_" + stat] = value.get("statistics", {}).get(stat)
            for f in value["frames"]:
                per_frame.append(dict(env=env, comparison=label, **f))
        available = [
            detail["comparisons"]["ours_cpu_vs_source_cpu"],
            detail["comparisons"]["ours_gpu_vs_source_gpu"]
            if paths.get("source_gpu") is not None
            else detail["comparisons"]["ours_gpu_vs_source_cpu"],
        ]
        row["cpu_and_gpu_max_mse_le_001"] = (
            "PASS"
            if all(v.get("all_frames_le_001") for v in available)
            else "FAIL"
            if any(v.get("all_frames_le_001") is False for v in available)
            else "UNESTABLISHED"
        )
        row["cpu_and_gpu_max_mse_le_0005"] = (
            "PASS"
            if all(v.get("all_frames_le_0005") for v in available)
            else "FAIL"
            if any(v.get("all_frames_le_0005") is False for v in available)
            else "UNESTABLISHED"
        )
        if roles["source_cpu"]["status"] == "NO_RGB":
            row["cpu_and_gpu_max_mse_le_001"] = "NO_RGB"
            row["cpu_and_gpu_max_mse_le_0005"] = "NO_RGB"
        # One honest worst-frame witness per measured registration, source left.
        key = "ours_gpu_vs_source_gpu" if roles["source_gpu"]["status"] == "COMPLETE" else "ours_gpu_vs_source_cpu"
        pair = detail["comparisons"][key]
        if pair["status"] == "MEASURED":
            from PIL import Image

            idx = pair["worst"]["index"]
            right_role = key.split("_vs_")[1]
            ours, source = load_image(roles["ours_gpu"]["frames"][idx]), load_image(roles[right_role]["frames"][idx])
            name = env.replace("/", "_") + ".png"
            Image.fromarray(np.concatenate((source, ours), axis=1)).save(args.output / name)
            gallery.append(
                f'<h2>{html.escape(env)}</h2><p>{html.escape(key)}; frame {idx}; MSE {pair["statistics"]["max"]:.9g}. Source left, ours right.</p><img src="{name}" style="max-width:100%">'
            )
        summaries.append(row)
        details.append(detail)
        print(env, row["cpu_and_gpu_max_mse_le_001"], flush=True)
    for filename, rows in (("renderer_comparison_42.csv", summaries), ("per_frame_errors.csv", per_frame)):
        if rows:
            with (args.output / filename).open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    report = dict(
        manifest=manifest,
        manifest_sha256=digest(args.manifest),
        registrations=details,
        shard=args.shard,
        shards=args.shards,
        mse="mean((uint8_A / 255 - uint8_B / 255)**2) over RGB pixels",
        sd="Sample SD across distinct measured frames (ddof=1)",
        acceptance="All 1000 frames must satisfy threshold on both ours CPU and ours GPU; missing/noncomparable is unestablished.",
        timing="Warm public render; one synchronized sample per frame, excludes reset/step/state extraction and external image save. GPU output upload/readback internal to a backend is included; external host conversion is excluded. Source render returns host RGB; source Craftax render reads cached pixel observation.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "images.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Fresh matched-state rendering</title>' + "\n".join(gallery)
    )


if __name__ == "__main__":
    main()
