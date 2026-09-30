"""Restore or verify the raw arrays preserved in the renderer recovery release.

Download renderer-evidence-20260930.tar.gz from the release linked in
docs/research/github_backup_20260930/backup.json, then run this script with that
archive. Existing matching files are retained; conflicting files are refused.
"""
import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "docs/research/github_backup_20260930/evidence-files.json").read_text())
    seen = set()
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in manifest or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            expected = manifest[member.name]
            if member.size != expected["bytes"]:
                raise ValueError(f"Wrong size: {member.name}")
            target = (root / member.name).resolve()
            target.relative_to(root)
            temporary = None
            output = None
            try:
                if not args.verify_only and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".restore-")
                    output = os.fdopen(fd, "wb")
                digest = hashlib.sha256()
                with archive.extractfile(member) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        if output is not None:
                            output.write(chunk)
                if output is not None:
                    output.close()
                if digest.hexdigest() != expected["sha256"]:
                    raise ValueError(f"Hash mismatch: {member.name}")
                if not args.verify_only:
                    if temporary:
                        # Hard-link creation refuses to overwrite a concurrent file.
                        os.link(temporary, target)
                    else:
                        with target.open("rb") as file:
                            actual = hashlib.file_digest(file, "sha256").hexdigest()
                        if actual != expected["sha256"]:
                            raise ValueError(f"Refusing to overwrite conflicting file: {target}")
                seen.add(member.name)
            finally:
                if output is not None:
                    output.close()
                if temporary:
                    Path(temporary).unlink(missing_ok=True)
    if seen != manifest.keys():
        raise ValueError(f"Missing archive members: {sorted(manifest.keys() - seen)}")
    print(f"{'Verified' if args.verify_only else 'Restored/verified'} {len(seen)} raw evidence files")


if __name__ == "__main__":
    main()
