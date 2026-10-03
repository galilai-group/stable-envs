"""Close all five snapshot candidates only after independent terminal device gates."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DOMAINS = ("cartpole", "pendulum", "acrobot", "mountain_car", "mountain_car_continuous")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("terminal", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--array", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    terminal = {}
    for line in args.terminal.read_text().splitlines():
        if line.strip():
            row = line.split("|")
            assert len(row) == 6, row
            terminal[row[0]] = row
    source_hashes = None
    for i, domain in enumerate(DOMAINS):
        report = json.loads((args.results / f"{domain}.json").read_text())
        row = terminal[f"{args.array}_{i}"]
        assert row[2:4] == ["COMPLETED", "0:0"]
        assert report["domain"] == domain and report["job"] == row[1]
        assert report["host"].split(".")[0] == row[5]
        assert report["status"] == "PASS"
        if source_hashes is None:
            source_hashes = report["source_sha256"]
        assert report["source_sha256"] == source_hashes
        suites = list(ET.parse(args.results / f"{domain}-tests.xml").getroot().iter("testsuite"))
        assert len(suites) == 1 and int(suites[0].get("tests")) == 147
        assert all(int(suites[0].get(k)) == 0 for k in ("failures", "errors", "skipped"))
    args.output.mkdir(parents=True)
    rows = []
    for domain in DOMAINS:
        target = args.output / f"{domain}.csv"
        subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_classic_snapshot.py")),
                        str(args.results / f"{domain}.json"), str(target), "--source", str(args.source)], check=True)
        assert target.read_bytes() == (args.results / f"{domain}.csv").read_bytes()
        rows.extend(csv.DictReader(target.open()))
    assert len(rows) == 320
    with (args.output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "acceptance.json").write_text(json.dumps(dict(status="PASS", array=args.array,
        domains=list(DOMAINS), gates_per_registration=28, tests_per_registration=147,
        paired_rows=320, source_sha256=source_hashes,
        scope="exact CPU/CUDA candidate vs frozen library; not upstream parity or speedup"), indent=2) + "\n")
    inputs = [args.terminal, *(args.results / f"{d}.{ext}" for d in DOMAINS for ext in ("json", "csv")),
              *(args.results / f"{d}-tests.xml" for d in DOMAINS)]
    # Paths are recorded explicitly so the closure can move with its parent.
    files = [*inputs, *args.output.iterdir()]
    root = args.output.parent
    (args.output / "SHA256SUMS.json").write_text(json.dumps(
        {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)},
        indent=2) + "\n")
    print("Closed five independent CPU/CUDA gates; 320 paired rows")


if __name__ == "__main__":
    main()
