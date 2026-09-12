"""Fetch the official LeWM Reacher dataset and put it where the swm eval config expects it.

``scripts/plan/config/reacher.yaml`` names ``dmc/reacher_random.h5`` -- a path under the
stable_worldmodel dataset cache, not a HuggingFace id, so nothing downloads it automatically. The
data itself is published as ``quentinll/lewm-reacher``, the dataset the LeWorldModel paper used for
this task, packaged the same way PushT's is: one zstd-compressed blob whose ``.tar`` in the name is
historical. ``scripts/benchmark/convert.py`` decompresses PushT's the same way.

Downloading the published dataset rather than collecting a fresh one with
``scripts/data/collect_reacher.py`` is deliberate: the recorded baselines were measured on this
data, and a locally collected substitute would change the episodes, the start states and the action
statistics all at once, so a miss would have three explanations instead of none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "quentinll/lewm-reacher"
FILE = "reacher.tar.zst"


def sniff(p: Path) -> str:
    """What did we actually get? Name it from the bytes, not from the filename."""
    with p.open("rb") as f:
        head = f.read(512)
    if head[:4] == b"\x89HDF":
        return "hdf5"
    if head[257:262] == b"ustar":
        return "tar"
    if head[:4] == b"\x28\xb5\x2f\xfd":
        return "zstd"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="stable_worldmodel cache root")
    ap.add_argument("--work", type=Path, required=True, help="scratch space for the compressed blob")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    args.work.mkdir(parents=True, exist_ok=True)
    dest_dir = args.cache / "datasets" / "dmc"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "reacher_random.h5"
    if dest.exists():
        print(f"{dest} already present ({dest.stat().st_size / 2**30:.1f} GiB); nothing to do")
        return

    print(f"downloading {REPO}/{FILE} -> {args.work}", flush=True)
    blob = Path(hf_hub_download(repo_id=REPO, filename=FILE, repo_type="dataset",
                                local_dir=str(args.work)))
    print(f"got {blob} ({blob.stat().st_size / 2**30:.1f} GiB), kind={sniff(blob)}", flush=True)

    tmp = args.work / "reacher.decompressed"
    print(f"decompressing -> {tmp}", flush=True)
    try:
        import zstandard as zstd
        with blob.open("rb") as fin, tmp.open("wb") as fout:
            zstd.ZstdDecompressor().copy_stream(fin, fout, write_size=2 ** 24)
    except ImportError:
        rc = subprocess.run(["zstd", "-d", "-f", "-o", str(tmp), str(blob)]).returncode
        if rc != 0:
            sys.exit("zstd decompression failed and the zstandard package is not installed")

    kind = sniff(tmp)
    print(f"decompressed {tmp.stat().st_size / 2**30:.1f} GiB, kind={kind}", flush=True)
    if kind == "hdf5":
        shutil.move(str(tmp), str(dest))
    elif kind == "tar":
        # A tar means the archive holds a directory (a .lance table, most likely). Unpack beside the
        # target and report what came out rather than guessing at a rename.
        out = args.work / "reacher_unpacked"
        out.mkdir(exist_ok=True)
        subprocess.run(["tar", "-xf", str(tmp), "-C", str(out)], check=True)
        print("archive contents:", [str(p.relative_to(out)) for p in sorted(out.rglob("*"))][:20])
        sys.exit("archive is a tar: inspect the listing above and place the payload by hand")
    else:
        sys.exit(f"decompressed payload is {kind}, not an HDF5 file or a tar")

    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 24), b""):
            h.update(chunk)
    meta = {"repo": REPO, "file": FILE, "dest": str(dest),
            "bytes": dest.stat().st_size, "sha256": h.hexdigest()}
    (dest_dir / "reacher_random.provenance.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    os.remove(blob)


if __name__ == "__main__":
    main()
