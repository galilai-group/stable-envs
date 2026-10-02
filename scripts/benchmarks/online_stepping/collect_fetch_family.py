"""Select complete registration sweeps without splicing timings across hosts.

A task-level run can time out after completing earlier registrations. Keep the
raw report unchanged; derived reports explicitly identify their source/status.
The existing strict validator must accept all cells before a report is retained.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    validator = Path(__file__).with_name("summarize_fetch_family.py")
    accepted = {}
    attempts = []
    for source in args.inputs:
        raw = json.loads(source.read_text())
        for suffix in raw["configuration"]["suffixes"]:
            name = "Fetch" + raw["task"] + ("" if suffix == "plain" else suffix)
            if name in accepted:
                continue
            d = dict(raw)
            d["configuration"] = dict(raw["configuration"], suffixes=[suffix])
            d["gates"] = [g for g in raw["gates"] if g["env"] == name]
            d["timings"] = [r for r in raw["timings"] if r["env"] == name]
            d["derived_from"] = dict(
                path=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                original_status=raw["status"],
                rule="Complete registration only; all rows from one source report/host.",
            )
            d["status"] = "PASS"  # Retained only if every required cell validates below.
            candidate = args.output / (name + ".json")
            csv = args.output / (name + ".csv")
            candidate.write_text(json.dumps(d, indent=2) + "\n")
            result = subprocess.run(
                [sys.executable, str(validator), str(candidate), "--output", str(csv)],
                capture_output=True,
                text=True,
            )
            attempts.append(dict(source=str(source), env=name, complete=result.returncode == 0))
            if result.returncode:
                candidate.unlink()
                csv.unlink(missing_ok=True)
                continue
            csv.unlink()
            accepted[name] = candidate
    (args.output / "collection.json").write_text(json.dumps(attempts, indent=2) + "\n")
    expected = {
        "Fetch" + task + suffix
        for task in ("Reach", "Push", "Slide", "PickAndPlace")
        for suffix in ("", "Dense", "Dict")
    }
    assert set(accepted) == expected, f"Incomplete: {sorted(expected - set(accepted))}"
    subprocess.run(
        [
            sys.executable,
            str(validator),
            *map(str, accepted.values()),
            "--output",
            str(args.output / "comparison.csv"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
