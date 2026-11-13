"""Print every cell of the given job folders that is not ok (error, oom, invalid, unsupported or no rate), one per line.

usage: python3 check_cells.py FOLDER...   (FOLDER = results/RUN/TAG of pod_queue.sh). Capped source cells are listed
as capped. A folder without online or offline results is listed as empty.
"""

import glob
import json
import os
import sys

for folder in sys.argv[1:]:
    files = sorted(glob.glob(f"{folder}/offline/*/*.json") + glob.glob(f"{folder}/online/*/*.json"))
    if not files:
        print(f"{folder}: empty")
    for f in files:
        rel = os.path.relpath(f, os.path.dirname(os.path.dirname(folder.rstrip("/"))))
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"{rel}: unreadable {e}")
            continue
        if "arms" in d:
            items = d["arms"].items()
        else:
            items = []
            for env, r in d.get("envs", {}).items():
                pts = r.get("points")
                items += [(f"{env}/b{p.get('batch')}", p) for p in pts] if pts else [(env, r)]
        for k, r in items:
            st = r.get("status", "ok")
            if st != "ok":
                print(f"{rel} {k}: {st} {str(r.get('error') or r.get('note') or r.get('unverified') or '')[:200]}")
