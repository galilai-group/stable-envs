"""Rebuild this bounded campaign from the recorded candidate/source file hashes.

No remote mutation or submission. Requires the matching checkout and upstream
source tree; rejects any drift and never overwrites an existing output directory.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    candidate = json.loads((evidence / 'candidate_manifest.json').read_text())
    source = json.loads((evidence / 'source_manifest.json').read_text())
    groups = ((args.checkout, 'code', candidate), (args.source, 'source', source))
    for root, _, manifest in groups:
        for rel, digest in manifest['files'].items():
            assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
    args.output.mkdir(parents=True, exist_ok=False)
    for root, prefix, manifest in groups:
        for rel in manifest['files']:
            destination = args.output / prefix / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / rel, destination)
    for name in ('candidate_manifest.json', 'source_manifest.json', 'run.sh'):
        shutil.copyfile(evidence / name, args.output / name)
    archive = args.output / 'bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for prefix in ('code', 'source'):
            for path in sorted((args.output / prefix).rglob('*')):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(args.output)), recursive=False)
        for name in ('candidate_manifest.json', 'source_manifest.json', 'run.sh'):
            tar.add(args.output / name, arcname=name, recursive=False)
    (args.output / 'bundle-sha256.json').write_text(json.dumps({
        'bundle.tar.gz': hashlib.sha256(archive.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
