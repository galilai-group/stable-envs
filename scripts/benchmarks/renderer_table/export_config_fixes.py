"""Independently verify raw captures and export the configuration-fix comparison."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image


def mse(a, b):
    return float(np.mean(((a.astype(np.float64) - b.astype(np.float64)) / 255) ** 2))


def sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    records, images, comparisons = {}, {}, 0
    for path in args.captures:
        record = json.loads((path / "result.json").read_text())
        assert record["status"] == "PASS"
        key = (record["registration"], record["device"])
        assert key not in records
        records[key] = record
        for frame in record["frames"]:
            arrays = {}
            for role in ["new", "old", "source_cpu", "source_gpu"]:
                a = np.load(path / f"frame-{frame['index']:04d}-{role}.npy", allow_pickle=False)
                assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] == 3
                assert sha(a) == frame["hashes"][role]
                arrays[role] = a
            for side in ["new", "old"]:
                for source in ["source_cpu", "source_gpu"]:
                    measured = mse(arrays[side], arrays[source])
                    assert abs(measured - frame[f"{side}_vs_{source}"]) < 1e-15
                    comparisons += 1
            if record["name"] == "OGBCube":
                np.testing.assert_array_equal(arrays["new"], arrays["old"])
            images[(*key, frame["index"])] = arrays
    with args.baseline_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    summary, sections = [], []
    for row in rows:
        registration = row["environment"]
        gpu = records.get((registration, "cuda"))
        if gpu is None or gpu["name"] == "OGBCube":
            continue
        assert (
            abs(
                max(f["old_vs_source_gpu"] for f in gpu["frames"])
                - float(row["ours_gpu_vs_source_gpu_max_normalized_mse"])
            )
            < 1e-14
        )
        cpu = records.get((registration, "cpu"))
        pairs = {"ours_gpu_vs_source_cpu": [], "ours_gpu_vs_source_gpu": []}
        if cpu:
            assert len(cpu["frames"]) == len(gpu["frames"])
            pairs.update(ours_cpu_vs_source_cpu=[], ours_cpu_vs_source_gpu=[], ours_cpu_vs_ours_gpu=[])
        for frame in gpu["frames"]:
            index = frame["index"]
            g = images[registration, "cuda", index]
            for source in ["source_cpu", "source_gpu"]:
                pairs[f"ours_gpu_vs_{source}"].append(mse(g["new"], g[source]))
            if cpu:
                other = next(f for f in cpu["frames"] if f["index"] == index)
                assert other["state_hash"] == frame["state_hash"]
                c = images[registration, "cpu", index]
                for source in ["source_cpu", "source_gpu"]:
                    np.testing.assert_array_equal(c[source], g[source])
                    pairs[f"ours_cpu_vs_{source}"].append(mse(c["new"], c[source]))
                pairs["ours_cpu_vs_ours_gpu"].append(mse(c["new"], g["new"]))
        for pair, values in pairs.items():
            row[pair + "_max_normalized_mse"] = max(values)
            row[pair + "_frames"] = len(values)
        source_gap = max(
            mse(
                images[registration, "cuda", f["index"]]["source_cpu"],
                images[registration, "cuda", f["index"]]["source_gpu"],
            )
            for f in gpu["frames"]
        )
        assert abs(source_gap - float(row["source_cpu_vs_source_gpu_max_normalized_mse"])) < 1e-14
        row["notes"] += (
            " Candidate re-rendered after source-configuration corrections on 2026-09-28; immutable source frames reused. See renderer_configuration_fixes.md. No pixel-parity promotion."
        )
        entry = dict(
            registration=registration,
            frames=len(gpu["frames"]),
            old_gpu_vs_source_gpu_max=max(f["old_vs_source_gpu"] for f in gpu["frames"]),
            new_gpu_vs_source_gpu_max=max(pairs["ours_gpu_vs_source_gpu"]),
            comparisons={k: dict(maximum=max(v), mean=float(np.mean(v))) for k, v in pairs.items()},
        )
        summary.append(entry)
        old_worst = max(gpu["frames"], key=lambda f: f["old_vs_source_gpu"])
        new_worst = max(gpu["frames"], key=lambda f: f["new_vs_source_gpu"])
        selected = {old_worst["index"]: old_worst, new_worst["index"]: new_worst}
        for index, frame in selected.items():
            g = images[registration, "cuda", index]
            roles = {
                "Source CPU": g["source_cpu"],
                "Source GPU": g["source_gpu"],
                "Before GPU": g["old"],
                "After GPU": g["new"],
            }
            if cpu:
                roles["After CPU"] = images[registration, "cpu", index]["new"]
            tiles = []
            for title, a in roles.items():
                filename = f"{gpu['name']}-{index:04d}-{title.replace(' ', '_')}.png"
                Image.fromarray(a).save(args.output / "images" / filename)
                np.testing.assert_array_equal(np.array(Image.open(args.output / "images" / filename)), a)
                tiles.append(
                    f'<figure><figcaption>{title}</figcaption><img src="images/{filename}" alt="{html.escape(registration)} {title}"></figure>'
                )
            sections.append(
                f'<section><h2>{html.escape(registration)}</h2><p>Seed {frame["seed"]}, case {frame["case"]}, step {frame["step"]}; frame {index}. GPU vs source GPU MSE: {frame["old_vs_source_gpu"]:.9g} → {frame["new_vs_source_gpu"]:.9g}.</p><div class="images">{"".join(tiles)}</div></section>'
            )
    assert len(summary) == 3
    with (args.output / "renderer_differences_42_environments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "evidence.json").write_text(json.dumps(list(records.values()), indent=2) + "\n")
    (args.output / "local-validation.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                candidate_captures=sum(len(r["frames"]) for r in records.values()),
                independent_mse_comparisons=comparisons,
                arrays_hashed=4 * sum(len(r["frames"]) for r in records.values()),
                cube_control_frames=len(records["swm/OGBCube-v0", "cuda"]["frames"]),
                baseline_csv_sha256=hashlib.sha256(args.baseline_csv.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    (args.output / "images.html").write_text(
        '<!doctype html><html><meta charset="utf-8"><title>Renderer configuration corrections</title><style>body{font:16px system-ui;margin:32px;background:#f3f3f3;color:#222}.images{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0}img{width:280px}figcaption{margin:8px 0}section{margin:32px 0}</style><h1>Renderer configuration corrections</h1><p>Matched archived States and appearances. Source frames are the unchanged, isolated CPU/GPU captures. Candidate renders are newly measured. These are measured residuals, not claims of pixel parity. MSE uses RGB values normalized to [0,1].</p>'
        + "".join(sections)
        + "</html>"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
