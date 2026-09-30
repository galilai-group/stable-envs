"""Assemble non-overlapping independently checked CSV export shards."""

import argparse
import csv
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("shards", type=Path, nargs="+")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    reports = [json.loads((s / "report.json").read_text()) for s in a.shards]
    assert len({r["manifest_sha256"] for r in reports}) == 1
    assert len({r["shards"] for r in reports}) == 1
    assert sorted(r["shard"] for r in reports) == list(range(reports[0]["shards"]))
    combined = dict(reports[0])
    combined.pop("shard")
    combined.pop("shards")
    combined["export_shards"] = list(map(str, a.shards))
    combined["registrations"] = [r for report in reports for r in report["registrations"]]
    registrations = combined["manifest"]["registrations"]
    assert len(combined["registrations"]) == len(registrations) == 42
    assert {r["env"] for r in combined["registrations"]} == set(registrations)
    combined["registrations"].sort(key=lambda r: r["env"])
    for filename in ("renderer_comparison_42.csv", "per_frame_errors.csv"):
        header = None
        with (a.output / filename).open("w", newline="") as output:
            writer = None
            rows = []
            for shard in a.shards:
                path = shard / filename
                if not path.exists():
                    continue
                with path.open() as source:
                    reader = csv.DictReader(source)
                    if header is None:
                        header = reader.fieldnames
                        writer = csv.DictWriter(output, fieldnames=header)
                        writer.writeheader()
                    assert header == reader.fieldnames
                    if filename == "renderer_comparison_42.csv":
                        rows.extend(reader)
                    else:
                        writer.writerows(reader)
            if rows:
                assert len(rows) == 42 and {r["env"] for r in rows} == set(registrations)
                writer.writerows(sorted(rows, key=lambda r: r["env"]))
    (a.output / "report.json").write_text(json.dumps(combined, indent=2) + "\n")
    html = []
    for shard in a.shards:
        html.append((shard / "images.html").read_text().split("</title>", 1)[-1])
        for path in shard.glob("*.png"):
            assert not (a.output / path.name).exists()
            shutil.copyfile(path, a.output / path.name)
    (a.output / "images.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Matched-state renderer comparisons</title>' + "\n".join(html)
    )


if __name__ == "__main__":
    main()
