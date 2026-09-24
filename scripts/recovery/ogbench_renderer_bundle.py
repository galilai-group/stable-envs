"""Build a complete tracked-tree gate bundle; never submit jobs or edit an oracle."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ORACLE = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/semantic_recovery_evidence/ogb_renderer_closeout"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--oracle-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    out = args.output.resolve()
    out.mkdir()
    # The earlier selective archive omitted transitive test-harness fixtures.
    # Archive the entire committed tree so test imports see their recorded data.
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(out / "source.tar.gz"), head], cwd=ROOT, check=True)
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar.gz",
            "-o",
            str(out / "oracle.tar.gz"),
            ORACLE,
            "stable_worldmodel",
            "pyproject.toml",
        ],
        cwd=args.oracle_repo,
        check=True,
    )
    script = subprocess.check_output(
        ["git", "show", head + ":scripts/recovery/ogbench_renderer_gate.sbatch"], cwd=ROOT
    )
    (out / "gate.sbatch").write_bytes(script)
    manifest = json.loads((EVIDENCE / "gpu-bundle.json").read_text())
    manifest.update(
        local_evidence=head,
        source_archive_sha256=sha(out / "source.tar.gz"),
        oracle_archive_sha256=sha(out / "oracle.tar.gz"),
        source_archive_paths=["Entire tracked tree at " + head],
        remote_directory="/oscar/scratch/sboughan/se-semantic-recovery/" + out.name,
        replaces_failed_job=6669720,
        submitted=False,
        review_required_before_resubmission=True,
        correction="Include transitive test-harness fixture dependencies by archiving the entire tracked tree.",
    )
    for rel in manifest["scripts"]:
        data = subprocess.check_output(["git", "show", head + ":" + rel], cwd=ROOT)
        manifest["scripts"][rel] = hashlib.sha256(data).hexdigest()
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha(out / n)}  {n}\n" for n in ["source.tar.gz", "oracle.tar.gz", "gate.sbatch", "bundle.json"])
    )
    shutil.copyfile(out / "bundle.json", EVIDENCE / "corrected-bundle.json")
    shutil.copyfile(out / "SHA256SUMS", EVIDENCE / "corrected-bundle-SHA256SUMS")
    print(out)
    print((out / "SHA256SUMS").read_text(), end="")


if __name__ == "__main__":
    main()
